use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub general: GeneralConfig,
    pub files: FilesConfig,
    pub similarity: SimilarityConfig,
    pub embedding: EmbeddingConfig,
    pub output: OutputConfig,
    pub progress: ProgressConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GeneralConfig {
    pub max_size: usize,
    pub max_file_size: usize,
    pub max_lines: usize,
    pub max_files: usize,
    pub recursive: bool,
    pub force: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FilesConfig {
    pub file_types: Vec<String>,
    pub exclude: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SimilarityConfig {
    pub mode: String,
    pub threshold: f32,
    pub top_k: usize,
    pub min_tokens: usize,
    pub cross_file_only: bool,
    pub include_snippets: bool,
    pub band_bits: usize,
    // Pre-filtering pipeline
    pub enable_pre_filtering: bool,
    pub size_ratio_max: f32,
    pub pre_token_threshold: f32,
    pub ast_type_matching: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingConfig {
    pub model: String,
    pub batch_size: usize,
    pub chunk_batch_size: usize,
    pub k: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m: usize,
    pub verify_min_jaccard: f32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OutputConfig {
    pub folder: String,
    pub analysis_file: String,
    pub similarity_file: String,
    pub graph_file: String,
    pub api_backend_file: String,
    pub api_frontend_file: String,
    pub api_map_file: String,
    pub suspects_file: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProgressConfig {
    pub verbosity: String,
    pub show_progress_bars: bool,
    pub print_similarity_pairs: bool,
    pub print_limit: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                max_size: 2048,
                max_file_size: 100_000_000,
                max_lines: 0,
                max_files: 20000,
                recursive: false,
                force: false,
            },
            files: FilesConfig {
                file_types: vec![
                    "rs".to_string(), "js".to_string(), "jsx".to_string(), "ts".to_string(), "tsx".to_string(),
                    "py".to_string(), "go".to_string(), "java".to_string(), "c".to_string(), "cpp".to_string(),
                    "cc".to_string(), "cxx".to_string(), "cs".to_string(), "sh".to_string(), "bash".to_string(),
                    "ps1".to_string(), "html".to_string(), "css".to_string(), "json".to_string(), "yml".to_string(),
                    "yaml".to_string(), "toml".to_string(), "xml".to_string(), "sql".to_string(),
                    "dockerfile".to_string(), "md".to_string()
                ],
                exclude: vec![
                    "node_modules".to_string(),
                    "target".to_string(),
                    "dist".to_string(),
                    "build".to_string(),
                    ".git".to_string(),
                    "*.lock".to_string(),
                    "package-lock.json".to_string(),
                ],
            },
            similarity: SimilarityConfig {
                mode: "token".to_string(),
                threshold: 0.86,
                top_k: 300,
                min_tokens: 6,
                cross_file_only: true,
                include_snippets: false,
                band_bits: 8,
                enable_pre_filtering: true,
                size_ratio_max: 5.0,
                pre_token_threshold: 0.1,
                ast_type_matching: true,
            },
            embedding: EmbeddingConfig {
                model: "jina-embeddings-v2-base-code".to_string(),
                batch_size: 32,
                chunk_batch_size: 5000,
                k: 15,
                ef_construction: 200,
                ef_search: 96,
                m: 16,
                verify_min_jaccard: 0.20,
            },
            output: OutputConfig {
                folder: "./_analysisjsons/".to_string(),
                analysis_file: "/analysis.json".to_string(),
                similarity_file: "similarity.json".to_string(),
                graph_file: "/graph.json".to_string(),
                api_backend_file: "/api_backend.json".to_string(),
                api_frontend_file: "/api_frontend.json".to_string(),
                api_map_file: "/api_map.json".to_string(),
                suspects_file: "/suspects.json".to_string(),
            },
            progress: ProgressConfig {
                verbosity: "normal".to_string(),
                show_progress_bars: true,
                print_similarity_pairs: false,
                print_limit: 10,
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        // Try config file locations in order
        let config_paths = get_config_paths();
        
        for path in &config_paths {
            if path.exists() {
                let content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read config file: {}", path.display()))?;
                
                let config: Config = toml::from_str(&content)
                    .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
                
                return Ok(config);
            }
        }
        
        // No config file found, use defaults
        Ok(Config::default())
    }
    
    pub fn save_default() -> Result<PathBuf> {
        let config_path = get_default_config_path()?;
        
        // Create config directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let default_config = Config::default();
        let toml_content = toml::to_string_pretty(&default_config)?;
        
        fs::write(&config_path, toml_content)?;
        Ok(config_path)
    }
}

fn get_config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    
    // Current directory
    paths.push(PathBuf::from("./resonates.toml"));
    
    // User config directory
    if let Ok(config_dir) = get_default_config_path() {
        paths.push(config_dir);
    }
    
    paths
}

fn get_default_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?
        .join("resonates");
    
    Ok(config_dir.join("resonates.toml"))
}