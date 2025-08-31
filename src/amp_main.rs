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

use similarity::RawChunk;
use amp_similarity::{
    analyze_chou_talalay_similarity, 
    export_ci_analysis_csv,
    ChouTalalaySimilarityParams,
    ChouTalalaySimilarityReport
};

#[derive(Parser, Debug)]
#[command(name = "amp-resonates")]
#[command(version = "1.0")]
#[command(about = "Chou-Talalay Inspired Code Similarity Analysis")]
struct Args {
    /// Path to analyze
    #[arg(short, long, default_value = "./")]
    path: String,

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

    /// Language to analyze
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

    // Create output directory
    fs::create_dir_all(&args.output_folder)
        .context("Failed to create output directory")?;

    println!("🧬 Amp-Resonates: Chou-Talalay Code Similarity Analysis");
    println!("📂 Analyzing: {}", args.path);
    println!("🎯 Token Dx: {:.3}, Embed Dx: {:.3}", args.token_dx, args.embed_dx);

    // Collect files
    let file_extensions: Vec<&str> = args.file_types.split(',').collect();
    let mut files = Vec::new();
    
    let walker = if args.recursive {
        WalkDir::new(&args.path)
    } else {
        WalkDir::new(&args.path).max_depth(1)
    };

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if file_extensions.contains(&ext) {
                    files.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    if files.is_empty() {
        println!("❌ No files found with extensions: {}", args.file_types);
        return Ok(());
    }

    println!("📄 Found {} files", files.len());

    // Process files into chunks
    let mut all_chunks = Vec::new();
    let mut chunk_id = 0;

    for file_path in files {
        println!("🔍 Processing: {}", file_path);
        
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        // Simple chunking for now (you can enhance with tree-sitter later)
        let lines: Vec<&str> = content.lines().collect();
        let chunk_size_lines = args.max_size / 50; // rough estimate
        
        for (i, chunk_lines) in lines.chunks(chunk_size_lines).enumerate() {
            let chunk_text = chunk_lines.join("\n");
            let start_line = i * chunk_size_lines + 1;
            let end_line = start_line + chunk_lines.len() - 1;
            
            if chunk_text.trim().len() > 50 { // skip tiny chunks
                all_chunks.push(RawChunk {
                    id: chunk_id,
                    file_path: file_path.clone(),
                    subtree_description: format!("lines_{}_to_{}", start_line, end_line),
                    start_line,
                    end_line,
                    size: chunk_text.len(),
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
