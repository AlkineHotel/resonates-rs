use anyhow::Result;
use ahash::{AHashMap, AHashSet};
use serde::Serialize;
use rayon::prelude::*;

use crate::similarity::{RawChunk, ChunkRef};
use crate::embedder_fast::FastEmbedder;
use crate::ann_hnsw::topk_pairs_hnsw_cosine;

/// Chou-Talalay inspired similarity analysis
/// CI = TokenScore/TokenDx + EmbedScore/EmbedDx + (TokenScore*EmbedScore)/(TokenDx*EmbedDx)
/// Where CI < 1 indicates synergistic similarity (better together than apart)

#[derive(Clone, Serialize, Debug)]
pub struct ChouTalalaySimilarityPair {
    pub chunk_a: ChunkRef,
    pub chunk_b: ChunkRef,
    pub token_score: f32,           // Jaccard similarity
    pub embedding_score: f32,       // Cosine similarity
    pub combination_index: f32,     // CI score (lower = more synergistic)
    pub token_dx: f32,              // Token threshold for 50% detection
    pub embed_dx: f32,              // Embedding threshold for 50% detection
    pub synergy_class: SynergyClass,
    pub overlap_tokens: usize,
    pub union_tokens: usize,
    pub same_file: bool,
    pub common_path_prefix: usize,
}

#[derive(Clone, Serialize, Debug)]
pub enum SynergyClass {
    HighSynergy,    // CI < 0.5 - token+embedding work extremely well together
    Synergistic,    // CI 0.5-0.8 - moderate synergy
    Additive,       // CI 0.8-1.2 - methods work independently  
    Antagonistic,   // CI 1.2-2.0 - methods interfere with each other
    Conflicting,    // CI > 2.0 - methods give contradictory results
}

#[derive(Clone, Serialize, Debug)]
pub struct ChouTalalaySimilarityReport {
    pub total_chunks: usize,
    pub total_pairs_analyzed: usize,
    pub synergy_distribution: SynergyDistribution,
    pub pairs: Vec<ChouTalalaySimilarityPair>,
    pub analysis_params: ChouTalalaySimilarityParams,
}

#[derive(Clone, Serialize, Debug)]
pub struct SynergyDistribution {
    pub high_synergy_count: usize,
    pub synergistic_count: usize,
    pub additive_count: usize,
    pub antagonistic_count: usize,
    pub conflicting_count: usize,
    pub mean_ci: f32,
    pub median_ci: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChouTalalaySimilarityParams {
    pub token_dx: f32,              // Token score for 50% detection rate
    pub embed_dx: f32,              // Embedding score for 50% detection rate
    pub min_token_overlap: usize,   // Minimum token overlap to consider
    pub top_k: usize,               // Maximum pairs to return
    pub embedder_model: String,     // FastEmbed model name
    pub cross_file_only: bool,      // Only analyze cross-file similarities
    pub ann_k: usize,               // HNSW neighbors per point
    pub ann_ef: usize,              // HNSW ef_construction
    pub ann_m: usize,               // HNSW max connections
    pub ann_ef_search: usize,       // HNSW search ef
}

impl Default for ChouTalalaySimilarityParams {
    fn default() -> Self {
        Self {
            token_dx: 0.3,              // 30% Jaccard for 50% detection
            embed_dx: 0.7,              // 70% cosine for 50% detection  
            min_token_overlap: 3,
            top_k: 1000,
            embedder_model: "jina-embeddings-v2-base-code".to_string(),
            cross_file_only: true,
            ann_k: 50,
            ann_ef: 200,
            ann_m: 32,
            ann_ef_search: 100,
        }
    }
}

/// Main entry point for Chou-Talalay similarity analysis
pub fn analyze_chou_talalay_similarity(
    raw_chunks: Vec<RawChunk>,
    params: ChouTalalaySimilarityParams,
) -> Result<ChouTalalaySimilarityReport> {
    if raw_chunks.len() < 2 {
        return Ok(ChouTalalaySimilarityReport {
            total_chunks: raw_chunks.len(),
            total_pairs_analyzed: 0,
            synergy_distribution: SynergyDistribution::empty(),
            pairs: vec![],
            analysis_params: params,
        });
    }

    // Step 1: Calculate token similarities for all pairs
    let token_similarities = calculate_token_similarities(&raw_chunks, &params)?;
    
    // Step 2: Calculate embedding similarities using HNSW
    let embedding_similarities = calculate_embedding_similarities(&raw_chunks, &params)?;
    
    // Step 3: Combine using Chou-Talalay CI formula
    let mut ci_pairs = calculate_combination_indices(
        &raw_chunks,
        token_similarities,
        embedding_similarities,
        &params,
    )?;
    
    // Step 4: Sort by CI (lower = more synergistic) and truncate
    ci_pairs.par_sort_unstable_by(|a, b| a.combination_index.partial_cmp(&b.combination_index).unwrap());
    if params.top_k > 0 && ci_pairs.len() > params.top_k {
        ci_pairs.truncate(params.top_k);
    }
    
    // Step 5: Generate distribution statistics
    let synergy_distribution = calculate_synergy_distribution(&ci_pairs);
    
    Ok(ChouTalalaySimilarityReport {
        total_chunks: raw_chunks.len(),
        total_pairs_analyzed: ci_pairs.len(),
        synergy_distribution,
        pairs: ci_pairs,
        analysis_params: params,
    })
}

fn calculate_token_similarities(
    chunks: &[RawChunk],
    params: &ChouTalalaySimilarityParams,
) -> Result<AHashMap<(usize, usize), (f32, usize, usize)>> {
    use regex::Regex;
    
    let ident_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
    let stop_words = crate::similarity::common_stops();
    
    // Tokenize all chunks
    let tokenized: Vec<(usize, AHashSet<String>)> = chunks
        .par_iter()
        .map(|chunk| {
            let mut tokens = AHashSet::new();
            for m in ident_re.find_iter(&chunk.text) {
                let token = m.as_str().to_ascii_lowercase();
                for part in split_identifier(&token) {
                    if part.len() >= 2 && !stop_words.contains(part.as_str()) {
                        tokens.insert(part);
                    }
                }
            }
            (chunk.id, tokens)
        })
        .collect();
    
    // Calculate pairwise Jaccard similarities
    let mut similarities = AHashMap::new();
    for i in 0..tokenized.len() {
        for j in (i + 1)..tokenized.len() {
            let (id_a, tokens_a) = &tokenized[i];
            let (id_b, tokens_b) = &tokenized[j];
            
            if params.cross_file_only && chunks[i].file_path == chunks[j].file_path {
                continue;
            }
            
            let intersection = tokens_a.intersection(tokens_b).count();
            let union = tokens_a.len() + tokens_b.len() - intersection;
            
            if intersection >= params.min_token_overlap && union > 0 {
                let jaccard = intersection as f32 / union as f32;
                similarities.insert((*id_a, *id_b), (jaccard, intersection, union));
            }
        }
    }
    
    Ok(similarities)
}

fn calculate_embedding_similarities(
    chunks: &[RawChunk],
    params: &ChouTalalaySimilarityParams,
) -> Result<AHashMap<(usize, usize), f32>> {
    // Create embedder
    let mut embedder = FastEmbedder::new(&params.embedder_model)?;
    
    // Prepare embedding inputs
    let embedding_inputs: Vec<(usize, String)> = chunks
        .iter()
        .map(|chunk| (chunk.id, snippet_for_embedding(&chunk.text, 4096)))
        .collect();
    
    // Get embeddings
    let embedding_map = embedder.embed(&embedding_inputs, 64)?;
    
    // Convert to vectors in order
    let mut vectors = Vec::new();
    let mut id_order = Vec::new();
    for (id, _) in &embedding_inputs {
        if let Some(vec) = embedding_map.get(id) {
            vectors.push(vec.clone());
            id_order.push(*id);
        }
    }
    
    if vectors.is_empty() {
        return Ok(AHashMap::new());
    }
    
    // Use HNSW to find similar pairs
    let hnsw_pairs = topk_pairs_hnsw_cosine(
        &vectors,
        params.ann_k,
        params.ann_m,
        params.ann_ef,
        params.ann_ef_search,
        None,
    )?;
    
    // Convert back to chunk IDs
    let mut similarities = AHashMap::new();
    for (i, j, cosine_sim) in hnsw_pairs {
        let id_a = id_order[i];
        let id_b = id_order[j];
        
        // Only keep cross-file if required
        if params.cross_file_only {
            let chunk_a = chunks.iter().find(|c| c.id == id_a).unwrap();
            let chunk_b = chunks.iter().find(|c| c.id == id_b).unwrap();
            if chunk_a.file_path == chunk_b.file_path {
                continue;
            }
        }
        
        similarities.insert((id_a, id_b), cosine_sim);
    }
    
    Ok(similarities)
}

fn calculate_combination_indices(
    chunks: &[RawChunk],
    token_similarities: AHashMap<(usize, usize), (f32, usize, usize)>,
    embedding_similarities: AHashMap<(usize, usize), f32>,
    params: &ChouTalalaySimilarityParams,
) -> Result<Vec<ChouTalalaySimilarityPair>> {
    let chunk_map: AHashMap<usize, &RawChunk> = chunks.iter().map(|c| (c.id, c)).collect();
    
    let mut ci_pairs = Vec::new();
    
    // Find pairs that have both token and embedding similarities
    for ((id_a, id_b), (token_score, overlap, union)) in token_similarities {
        if let Some(&embedding_score) = embedding_similarities.get(&(id_a, id_b)) {
            let chunk_a = chunk_map.get(&id_a).unwrap();
            let chunk_b = chunk_map.get(&id_b).unwrap();
            
            // Apply Chou-Talalay CI formula
            // CI = D1/Dx1 + D2/Dx2 + (D1*D2)/(Dx1*Dx2)
            let token_term = token_score / params.token_dx;
            let embed_term = embedding_score / params.embed_dx;
            let synergy_term = (token_score * embedding_score) / (params.token_dx * params.embed_dx);
            
            let combination_index = token_term + embed_term + synergy_term;
            
            let synergy_class = classify_synergy(combination_index);
            
            ci_pairs.push(ChouTalalaySimilarityPair {
                chunk_a: ChunkRef {
                    file_path: chunk_a.file_path.clone(),
                    subtree_description: chunk_a.subtree_description.clone(),
                    start_line: chunk_a.start_line,
                    end_line: chunk_a.end_line,
                    size: chunk_a.size,
                    snippet: Some(snippet(&chunk_a.text)),
                },
                chunk_b: ChunkRef {
                    file_path: chunk_b.file_path.clone(),
                    subtree_description: chunk_b.subtree_description.clone(),
                    start_line: chunk_b.start_line,
                    end_line: chunk_b.end_line,
                    size: chunk_b.size,
                    snippet: Some(snippet(&chunk_b.text)),
                },
                token_score,
                embedding_score,
                combination_index,
                token_dx: params.token_dx,
                embed_dx: params.embed_dx,
                synergy_class,
                overlap_tokens: overlap,
                union_tokens: union,
                same_file: chunk_a.file_path == chunk_b.file_path,
                common_path_prefix: crate::similarity::common_dir_prefix_len(&chunk_a.file_path, &chunk_b.file_path),
            });
        }
    }
    
    Ok(ci_pairs)
}

fn classify_synergy(ci: f32) -> SynergyClass {
    match ci {
        ci if ci < 0.5 => SynergyClass::HighSynergy,
        ci if ci < 0.8 => SynergyClass::Synergistic,
        ci if ci < 1.2 => SynergyClass::Additive,
        ci if ci < 2.0 => SynergyClass::Antagonistic,
        _ => SynergyClass::Conflicting,
    }
}

fn calculate_synergy_distribution(pairs: &[ChouTalalaySimilarityPair]) -> SynergyDistribution {
    let mut high_synergy = 0;
    let mut synergistic = 0;
    let mut additive = 0;
    let mut antagonistic = 0;
    let mut conflicting = 0;
    
    let mut ci_values: Vec<f32> = pairs.iter().map(|p| p.combination_index).collect();
    
    for pair in pairs {
        match pair.synergy_class {
            SynergyClass::HighSynergy => high_synergy += 1,
            SynergyClass::Synergistic => synergistic += 1,
            SynergyClass::Additive => additive += 1,
            SynergyClass::Antagonistic => antagonistic += 1,
            SynergyClass::Conflicting => conflicting += 1,
        }
    }
    
    ci_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ci = ci_values.iter().sum::<f32>() / ci_values.len() as f32;
    let median_ci = if ci_values.is_empty() { 0.0 } else { ci_values[ci_values.len() / 2] };
    
    SynergyDistribution {
        high_synergy_count: high_synergy,
        synergistic_count: synergistic,
        additive_count: additive,
        antagonistic_count: antagonistic,
        conflicting_count: conflicting,
        mean_ci,
        median_ci,
    }
}

impl SynergyDistribution {
    fn empty() -> Self {
        Self {
            high_synergy_count: 0,
            synergistic_count: 0,
            additive_count: 0,
            antagonistic_count: 0,
            conflicting_count: 0,
            mean_ci: 0.0,
            median_ci: 0.0,
        }
    }
}

fn split_identifier(s: &str) -> Vec<String> {
    let snake: Vec<&str> = s.split('_').filter(|p| !p.is_empty()).collect();
    let mut out: Vec<String> = Vec::new();
    for part in snake {
        let mut buf = String::new();
        for (i, ch) in part.chars().enumerate() {
            if i > 0 && ch.is_ascii_uppercase() {
                if !buf.is_empty() { 
                    out.push(buf.to_ascii_lowercase()); 
                }
                buf = ch.to_string();
            } else {
                buf.push(ch);
            }
        }
        if !buf.is_empty() {
            out.push(buf.to_ascii_lowercase());
        }
    }
    out
}

fn snippet(s: &str) -> String {
    const MAX: usize = 800;
    if s.len() <= MAX { s.to_string() } else { s[..MAX].to_string() }
}

fn snippet_for_embedding(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { s[..max].to_string() }
}

/// CSV output for detailed analysis
pub fn export_ci_analysis_csv(
    report: &ChouTalalaySimilarityReport,
    output_path: &str,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(output_path)?;
    
    // CSV header
    writeln!(file, "chunk_a_file,chunk_a_start,chunk_a_end,chunk_b_file,chunk_b_start,chunk_b_end,token_score,embedding_score,combination_index,synergy_class,overlap_tokens,union_tokens,same_file,common_path_prefix")?;
    
    // CSV data
    for pair in &report.pairs {
        writeln!(
            file,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:?},{},{},{},{}",
            pair.chunk_a.file_path,
            pair.chunk_a.start_line,
            pair.chunk_a.end_line,
            pair.chunk_b.file_path,
            pair.chunk_b.start_line,
            pair.chunk_b.end_line,
            pair.token_score,
            pair.embedding_score,
            pair.combination_index,
            pair.synergy_class,
            pair.overlap_tokens,
            pair.union_tokens,
            pair.same_file,
            pair.common_path_prefix,
        )?;
    }
    
    Ok(())
}

/// Logarithmic transformation utilities (for future optimization)
pub mod log_transforms {
    /// Apply logarithmic transformation to similarity scores
    /// Based on the user's systems bio clustering approach
    pub fn log_transform_similarity(score: f32, coefficient: f32) -> f32 {
        if score <= 0.0 { return 0.0; }
        (score.ln() * coefficient).max(-2.0).min(2.0)
    }
    
    /// Find optimal coefficient for logarithmic transformation
    /// Target: bring similarity scores into [-2, 2] range for cosine analysis
    pub fn find_optimal_log_coefficient(scores: &[f32]) -> f32 {
        if scores.is_empty() { return -0.333; }
        
        let max_log = scores.iter()
            .filter(|&&s| s > 0.0)
            .map(|&s| s.ln())
            .fold(f32::NEG_INFINITY, f32::max);
        
        if max_log <= 0.0 { return -0.333; }
        
        // Target: max_log * coeff ≈ 2.0
        let optimal = 2.0 / max_log;
        
        // Round to something close to 1/3 if it's close
        if (optimal - (-0.333)).abs() < 0.1 {
            -0.333
        } else {
            optimal
        }
    }
}
