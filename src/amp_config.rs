use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AmpConfig {
    pub general: GeneralConfig,
    pub files: FilesConfig,
    pub chou_talalay: ChouTalayConfig,
    pub embedding: EmbeddingConfig,
    pub output: OutputConfig,
    pub progress: ProgressConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GeneralConfig {
    pub max_size: usize,
    pub max_file_size: usize,
    pub max_lines: usize,
    pub recursive: bool,
    pub force: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FilesConfig {
    pub file_types: Vec<String>,
    pub exclude: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChouTalayConfig {
    /// Token score threshold for 50% detection rate (Dx1 in CI formula)
    pub token_dx: f32,
    /// Embedding score threshold for 50% detection rate (Dx2 in CI formula)  
    pub embed_dx: f32,
    /// Minimum token overlap to consider
    pub min_token_overlap: usize,
    /// Maximum similarity pairs to return
    pub top_k: usize,
    /// Only analyze cross-file similarities
    pub cross_file_only: bool,
    /// CI threshold cutoffs for synergy classification
    pub high_synergy_cutoff: f32,     // CI < this = high synergy
    pub synergistic_cutoff: f32,      // CI < this = synergistic
    pub additive_max: f32,            // CI < this = additive
    pub antagonistic_max: f32,        // CI < this = antagonistic (else conflicting)
    /// Logarithmic transformation parameters
    pub use_log_transform: bool,
    pub log_coefficient: f32,         // For log(similarity) * coefficient transformation
    pub auto_optimize_log_coeff: bool, // Find optimal coefficient automatically
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingConfig {
    pub model: String,
    pub batch_size: usize,
    pub chunk_batch_size: usize,
    pub k: usize,                     // HNSW neighbors per point
    pub ef_construction: usize,       // HNSW ef_construction
    pub ef_search: usize,             // HNSW ef during search
    pub m: usize,                     // HNSW max connections
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OutputConfig {
    pub folder: String,
    pub json_file: String,
    pub csv_file: String,
    pub enable_csv_export: bool,
    pub include_snippets: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProgressConfig {
    pub verbosity: String,
    pub show_progress_bars: bool,
    pub print_ci_distribution: bool,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                max_size: 2048,
                max_file_size: 100_000_000,
                max_lines: 0,
                recursive: true,
                force: false,
            },
            files: FilesConfig {
                file_types: vec![
                    // Core languages
                    "rs".to_string(), "js".to_string(), "jsx".to_string(), "ts".to_string(), "tsx".to_string(),
                    "py".to_string(), "go".to_string(), "java".to_string(), "c".to_string(), "cpp".to_string(),
                    "cc".to_string(), "cxx".to_string(), "cs".to_string(), 
                    // Scripts
                    "sh".to_string(), "bash".to_string(), "ps1".to_string(),
                    // Web
                    "html".to_string(), "css".to_string(), "scss".to_string(), "sass".to_string(),
                    // Data
                    "json".to_string(), "yml".to_string(), "yaml".to_string(), "toml".to_string(), "xml".to_string(),
                    // Database
                    "sql".to_string(),
                    // Config/Docker
                    "dockerfile".to_string(), "md".to_string(), "txt".to_string(),
                ],
                exclude: vec![
                    "node_modules".to_string(),
                    "target".to_string(),
                    "dist".to_string(),
                    "build".to_string(),
                    ".git".to_string(),
                    "*.lock".to_string(),
                    "package-lock.json".to_string(),
                    "Cargo.lock".to_string(),
                    ".vscode".to_string(),
                    ".idea".to_string(),
                ],
            },
            chou_talalay: ChouTalayConfig {
                token_dx: 0.3,              // 30% Jaccard for 50% detection (conservative)
                embed_dx: 0.7,              // 70% cosine for 50% detection (strict)
                min_token_overlap: 3,
                top_k: 1000,
                cross_file_only: true,
                high_synergy_cutoff: 0.5,   // CI < 0.5 = excellent synergy
                synergistic_cutoff: 0.8,    // CI < 0.8 = good synergy  
                additive_max: 1.2,          // CI < 1.2 = additive (expected)
                antagonistic_max: 2.0,      // CI < 2.0 = antagonistic (problematic)
                use_log_transform: false,   // Enable log transformation of similarity scores
                log_coefficient: -0.333,    // 1/3 coefficient as discovered in systems bio
                auto_optimize_log_coeff: true, // Find optimal coefficient from data
            },
            embedding: EmbeddingConfig {
                model: "jina-embeddings-v2-base-code".to_string(),
                batch_size: 64,
                chunk_batch_size: 5000,
                k: 50,
                ef_construction: 200,
                ef_search: 100,
                m: 32,
            },
            output: OutputConfig {
                folder: "./_amp_analysis/".to_string(),
                json_file: "chou_talalay_analysis.json".to_string(),
                csv_file: "chou_talalay_analysis.csv".to_string(),
                enable_csv_export: true,
                include_snippets: true,
            },
            progress: ProgressConfig {
                verbosity: "normal".to_string(),
                show_progress_bars: true,
                print_ci_distribution: true,
            },
        }
    }
}

impl AmpConfig {
    pub fn load() -> Result<Self> {
        let config_paths = get_config_paths();
        
        for path in &config_paths {
            if path.exists() {
                let content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read amp config file: {}", path.display()))?;
                
                let config: AmpConfig = toml::from_str(&content)
                    .with_context(|| format!("Failed to parse amp config file: {}", path.display()))?;
                
                return Ok(config);
            }
        }
        
        Ok(AmpConfig::default())
    }
    
    pub fn save_default() -> Result<PathBuf> {
        let config_path = get_default_config_path()?;
        
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let default_config = AmpConfig::default();
        let toml_content = toml::to_string_pretty(&default_config)?;
        
        fs::write(&config_path, toml_content)?;
        Ok(config_path)
    }
}

fn get_config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    
    // Current directory
    paths.push(PathBuf::from("./amp_resonates.toml"));
    
    // User config directory  
    if let Ok(config_dir) = get_default_config_path() {
        paths.push(config_dir);
    }
    
    paths
}

fn get_default_config_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?
        .join("amp_resonates");
    
    Ok(config_dir.join("amp_resonates.toml"))
}
