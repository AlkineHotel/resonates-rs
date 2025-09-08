use clap::{Parser, ValueEnum};
use anyhow::{Result, Context};
use std::fs;
use walkdir::WalkDir;

mod similarity;
mod embedder_fast;
mod ann_hnsw;
mod graph;
mod api;
mod cluster;
mod config;
mod filter_pipeline;
mod amp_similarity;
mod splitter_gemi;

use similarity::RawChunk;
use amp_similarity::{
    analyze_chou_talalay_similarity, 
    export_ci_analysis_csv,
    ChouTalalaySimilarityParams
};
use splitter_gemi::{Splitter, CharCounter};

#[derive(Parser, Debug)]
#[command(name = "amp-resonates")]
#[command(version = "1.0")]
#[command(about = "Chou-Talalay Inspired Code Similarity Analysis")]
struct Args {
    /// Paths to analyze (space-separated, or use '-' for stdin)
    #[arg(short, long, default_value = "./")]
    path: Vec<String>,

    /// Maximum chunk size (characters)
    #[arg(long, default_value_t = 2048)]
    max_size: usize,

    /// Output folder for analysis results
    #[arg(short, long, default_value = "./_amp_analysis/")]
    output_folder: String,

    /// File extensions to analyze (comma-separated)
    #[arg(long, default_value = "rs,js,jsx,ts,tsx,py,go,java,c,cpp,cc,cxx,cs,sh,bash,ps1,html,css,json,yml,yaml,xml")]
    file_types: String,

    /// Recursive analysis
    #[arg(short, long, default_value_t = true)]
    recursive: bool,

    /// Token Dx (score for 50% detection rate)
    #[arg(long, default_value_t = 0.3)]
    token_dx: f32,

    /// Embedding Dx (score for 50% detection rate)  
    #[arg(long, default_value_t = 0.7)]
    embed_dx: f32,

    /// Minimum token overlap to consider
    #[arg(long, default_value_t = 3)]
    min_token_overlap: usize,

    /// Maximum pairs to analyze
    #[arg(long, default_value_t = 1000)]
    top_k: usize,

    /// FastEmbed model name
    #[arg(long, default_value = "jina-embeddings-v2-base-code")]
    embedder_model: String,

    /// Only analyze cross-file similarities
    #[arg(long, default_value_t = true)]
    cross_file_only: bool,

    /// Export detailed CSV analysis
    #[arg(long, default_value_t = true)]
    export_csv: bool,

    /// Language hint for parsing
    #[arg(short, long, value_enum, default_value_t = Language::Auto)]
    language: Language,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Language {
    Auto,
    Rust,
    JavaScript,
    TypeScript,
    Python,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Bash,
    PowerShell,
    Html,
    Css,
    Json,
    Yaml,
    Xml,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Handle paths (support multiple paths and stdin)
    let paths = if args.path.is_empty() || (args.path.len() == 1 && args.path[0] == "-") {
        // Read from stdin
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        let paths: Result<Vec<String>, _> = stdin.lock().lines().collect();
        match paths {
            Ok(p) => p,
            Err(e) => return Err(anyhow::anyhow!("Failed to read paths from stdin: {}", e)),
        }
    } else {
        args.path.clone()
    };

    // Default to current directory if no paths provided
    let paths = if paths.is_empty() { vec!["./".to_string()] } else { paths };

    // Create output directory
    fs::create_dir_all(&args.output_folder)
        .context("Failed to create output directory")?;

    println!("🧬 Amp-Resonates: Chou-Talalay Code Similarity Analysis");
    if paths.len() == 1 {
        println!("📂 Analyzing: {}", paths[0]);
    } else {
        println!("📂 Analyzing {} paths", paths.len());
    }
    println!("🎯 Token Dx: {:.3}, Embed Dx: {:.3}", args.token_dx, args.embed_dx);

    // Collect files from all paths
    let file_extensions: Vec<&str> = args.file_types.split(',').collect();
    let mut all_files = Vec::new();
    
    for path in &paths {
        let walker = if args.recursive {
            WalkDir::new(path).follow_links(false)
        } else {
            WalkDir::new(path).max_depth(1).follow_links(false)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Some(ext) = entry_path.extension().and_then(|e| e.to_str()) {
                    if file_extensions.contains(&ext) {
                        all_files.push(entry_path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    if all_files.is_empty() {
        println!("❌ No files found with extensions: {}", args.file_types);
        return Ok(());
    }

    println!("📄 Found {} files", all_files.len());

    // Process files into chunks using proper tree-sitter
    let mut all_chunks = Vec::new();
    let mut chunk_id = 0;

    for file_path in all_files {
        println!("🔍 Processing: {}", file_path);
        
        let content = fs::read(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        // Get language for tree-sitter
        let language = get_language(&args.language, &file_path);
        
        // Use Gemini's splitter (fixed version)
        let splitter = Splitter::new(language, CharCounter)
            .map_err(|e| anyhow::anyhow!("Failed to create splitter: {}", e))?
            .with_max_size(args.max_size);

        let chunks = splitter
            .split(&content)
            .map_err(|e| anyhow::anyhow!("Failed to split file into chunks: {}", e))?;

        for chunk in chunks {
            let chunk_text = String::from_utf8_lossy(&content[chunk.range.start_byte..chunk.range.end_byte]).to_string();
            
            if chunk_text.trim().len() > 50 { // skip tiny chunks
                all_chunks.push(RawChunk {
                    id: chunk_id,
                    file_path: file_path.clone(),
                    subtree_description: chunk.subtree,
                    start_line: chunk.range.start_point.row + 1,
                    end_line: chunk.range.end_point.row + 1,
                    size: chunk.size,
                    text: chunk_text,
                });
                chunk_id += 1;
            }
        }
    }

    println!("🧩 Generated {} chunks", all_chunks.len());

    // Configure Chou-Talalay parameters
    let params = ChouTalalaySimilarityParams {
        token_dx: args.token_dx,
        embed_dx: args.embed_dx,
        min_token_overlap: args.min_token_overlap,
        top_k: args.top_k,
        embedder_model: args.embedder_model,
        cross_file_only: args.cross_file_only,
        ann_k: 50,
        ann_ef: 200,
        ann_m: 32,
        ann_ef_search: 100,
    };

    // Run Chou-Talalay analysis
    println!("🧬 Running Chou-Talalay similarity analysis...");
    let report = analyze_chou_talalay_similarity(all_chunks, params)?;

    // Output results
    let json_path = format!("{}/chou_talalay_analysis.json", args.output_folder);
    let json_content = serde_json::to_string_pretty(&report)?;
    fs::write(&json_path, json_content)
        .context("Failed to write JSON analysis")?;

    if args.export_csv {
        let csv_path = format!("{}/chou_talalay_analysis.csv", args.output_folder);
        export_ci_analysis_csv(&report, &csv_path)?;
        println!("📊 CSV exported: {}", csv_path);
    }

    // Print summary
    println!("\n🧬 Chou-Talalay Analysis Complete!");
    println!("📊 Total chunks: {}", report.total_chunks);
    println!("🔗 Pairs analyzed: {}", report.total_pairs_analyzed);
    println!("📈 Mean CI: {:.3}", report.synergy_distribution.mean_ci);
    println!("📊 Median CI: {:.3}", report.synergy_distribution.median_ci);
    println!("\n🎯 Synergy Distribution:");
    println!("   🔥 High Synergy (CI < 0.5): {}", report.synergy_distribution.high_synergy_count);
    println!("   ✨ Synergistic (CI 0.5-0.8): {}", report.synergy_distribution.synergistic_count);
    println!("   ➕ Additive (CI 0.8-1.2): {}", report.synergy_distribution.additive_count);
    println!("   ⚡ Antagonistic (CI 1.2-2.0): {}", report.synergy_distribution.antagonistic_count);
    println!("   💥 Conflicting (CI > 2.0): {}", report.synergy_distribution.conflicting_count);

    println!("\n📄 Results saved: {}", json_path);

    Ok(())
}

fn get_language(hint: &Language, file_path: &str) -> tree_sitter::Language {
    use std::path::Path;
    
    unsafe {
        match hint {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::Go => tree_sitter_go::LANGUAGE.into(),
            Language::Java => tree_sitter_java::LANGUAGE.into(),
            Language::C => tree_sitter_c::LANGUAGE.into(),
            Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Language::Bash => tree_sitter_bash::LANGUAGE.into(),
            Language::PowerShell => tree_sitter_powershell::LANGUAGE.into(),
            Language::Html => tree_sitter_html::LANGUAGE.into(),
            Language::Css => tree_sitter_css::LANGUAGE.into(),
            Language::Json => tree_sitter_json::LANGUAGE.into(),
            Language::Yaml => tree_sitter_yaml::LANGUAGE.into(),
            Language::Xml => tree_sitter_xml::LANGUAGE_XML.into(),
            Language::Auto => {
                match Path::new(file_path).extension().and_then(|ext| ext.to_str()).unwrap_or("") {
                    "rs" => tree_sitter_rust::LANGUAGE.into(),
                    "js" | "jsx" => tree_sitter_javascript::LANGUAGE.into(),
                    "ts" | "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
                    "py" => tree_sitter_python::LANGUAGE.into(),
                    "go" => tree_sitter_go::LANGUAGE.into(),
                    "java" => tree_sitter_java::LANGUAGE.into(),
                    "c" => tree_sitter_c::LANGUAGE.into(),
                    "cpp" | "cc" | "cxx" => tree_sitter_cpp::LANGUAGE.into(),
                    "cs" => tree_sitter_c_sharp::LANGUAGE.into(),
                    "sh" | "bash" => tree_sitter_bash::LANGUAGE.into(),
                    "ps1" => tree_sitter_powershell::LANGUAGE.into(),
                    "html" => tree_sitter_html::LANGUAGE.into(),
                    "css" => tree_sitter_css::LANGUAGE.into(),
                    "json" => tree_sitter_json::LANGUAGE.into(),
                    "yml" | "yaml" => tree_sitter_yaml::LANGUAGE.into(),
                    "xml" => tree_sitter_xml::LANGUAGE_XML.into(),
                    _ => tree_sitter_rust::LANGUAGE.into(), // fallback
                }
            }
        }
    }
}
