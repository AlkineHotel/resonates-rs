use anyhow::Result;
use crate::similarity::RawChunk;
use crate::config::SimilarityConfig;
use ahash::AHashSet;
use regex::Regex;
use std::path::Path;

pub struct FilterPipeline {
    config: SimilarityConfig,
    // Pre-compiled regex for AST type extraction
    ast_type_regex: Regex,
}

#[derive(Debug, Clone)]
pub struct FilterStats {
    pub total_pairs: usize,
    pub after_size_filter: usize,
    pub after_directory_filter: usize,
    pub after_ast_filter: usize,
    pub after_token_filter: usize,
    pub final_candidates: usize,
}

impl FilterPipeline {
    pub fn new(config: SimilarityConfig) -> Result<Self> {
        // Regex to extract primary AST node type (first word before any details)
        let ast_type_regex = Regex::new(r"^(\w+)")?;
        
        Ok(Self {
            config,
            ast_type_regex,
        })
    }
    
    /// Multi-stage filtering pipeline that eliminates obvious non-matches
    pub fn filter_chunk_pairs<'a>(&self, chunks: &'a [RawChunk]) -> Result<(Vec<(usize, usize)>, FilterStats)> {
        if !self.config.enable_pre_filtering {
            // No filtering - return all pairs (expensive!)
            let total_pairs = (chunks.len() * (chunks.len() - 1)) / 2;
            let pairs: Vec<(usize, usize)> = (0..chunks.len())
                .flat_map(|i| ((i+1)..chunks.len()).map(move |j| (i, j)))
                .collect();
            
            let stats = FilterStats {
                total_pairs,
                after_size_filter: total_pairs,
                after_directory_filter: total_pairs,
                after_ast_filter: total_pairs,
                after_token_filter: total_pairs,
                final_candidates: total_pairs,
            };
            return Ok((pairs, stats));
        }
        
        let mut stats = FilterStats {
            total_pairs: 0,
            after_size_filter: 0,
            after_directory_filter: 0,
            after_ast_filter: 0,
            after_token_filter: 0,
            final_candidates: 0,
        };
        
        let mut candidates = Vec::new();
        
        // Generate all possible pairs
        for i in 0..chunks.len() {
            for j in (i+1)..chunks.len() {
                stats.total_pairs += 1;
                
                let chunk_a = &chunks[i];
                let chunk_b = &chunks[j];
                
                // Stage 1: Size ratio filter (super cheap)
                if !self.size_compatible(chunk_a, chunk_b) {
                    continue;
                }
                stats.after_size_filter += 1;
                
                // Stage 2: Directory proximity filter (cheap string ops)
                if self.config.cross_file_only && chunk_a.file_path == chunk_b.file_path {
                    continue;
                }
                if !self.directory_compatible(chunk_a, chunk_b) {
                    continue;
                }
                stats.after_directory_filter += 1;
                
                // Stage 3: AST type compatibility (regex match)
                if self.config.ast_type_matching && !self.ast_type_compatible(chunk_a, chunk_b) {
                    continue;
                }
                stats.after_ast_filter += 1;
                
                // Stage 4: Liberal token overlap (expensive but catches edge cases)
                if !self.token_overlap_check(chunk_a, chunk_b) {
                    continue;
                }
                stats.after_token_filter += 1;
                
                // Survived all filters - candidate for embedding comparison
                candidates.push((i, j));
            }
        }
        
        stats.final_candidates = candidates.len();
        Ok((candidates, stats))
    }
    
    fn size_compatible(&self, a: &RawChunk, b: &RawChunk) -> bool {
        let size_a = a.size as f32;
        let size_b = b.size as f32;
        let ratio_max = self.config.size_ratio_max;
        
        // Skip if size difference exceeds threshold
        size_a <= size_b * ratio_max && size_b <= size_a * ratio_max
    }
    
    fn directory_compatible(&self, a: &RawChunk, b: &RawChunk) -> bool {
        // For now, allow all cross-file comparisons
        // Could add directory distance scoring here
        true
    }
    
    fn ast_type_compatible(&self, a: &RawChunk, b: &RawChunk) -> bool {
        let type_a = self.extract_ast_type(&a.subtree_description);
        let type_b = self.extract_ast_type(&b.subtree_description);
        
        // Allow comparison within compatible categories
        self.are_ast_types_compatible(&type_a, &type_b)
    }
    
    fn extract_ast_type(&self, description: &str) -> String {
        self.ast_type_regex
            .captures(description)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().to_lowercase())
            .unwrap_or_else(|| "unknown".to_string())
    }
    
    fn are_ast_types_compatible(&self, type_a: &str, type_b: &str) -> bool {
        // Define compatible AST node type groups
        let function_types: AHashSet<&str> = ["function", "method", "fn", "impl"].into_iter().collect();
        let struct_types: AHashSet<&str> = ["struct", "class", "interface", "enum"].into_iter().collect();
        let import_types: AHashSet<&str> = ["import", "use", "include", "require"].into_iter().collect();
        let comment_types: AHashSet<&str> = ["comment", "block_comment", "line_comment"].into_iter().collect();
        
        // Same type always compatible
        if type_a == type_b {
            return true;
        }
        
        // Check if both belong to same category
        if function_types.contains(type_a) && function_types.contains(type_b) {
            return true;
        }
        if struct_types.contains(type_a) && struct_types.contains(type_b) {
            return true;
        }
        if import_types.contains(type_a) && import_types.contains(type_b) {
            return true;
        }
        if comment_types.contains(type_a) && comment_types.contains(type_b) {
            return true;
        }
        
        false
    }
    
    fn token_overlap_check(&self, a: &RawChunk, b: &RawChunk) -> bool {
        // Quick and dirty token overlap using simple whitespace splitting
        // This is intentionally liberal (low threshold) to catch edge cases
        let tokens_a: AHashSet<&str> = a.text
            .split_whitespace()
            .filter(|t| t.len() > 2) // Skip tiny tokens
            .collect();
        
        let tokens_b: AHashSet<&str> = b.text
            .split_whitespace()
            .filter(|t| t.len() > 2)
            .collect();
        
        if tokens_a.is_empty() || tokens_b.is_empty() {
            return false;
        }
        
        let intersection: AHashSet<_> = tokens_a.intersection(&tokens_b).collect();
        let union_size = tokens_a.len() + tokens_b.len() - intersection.len();
        
        let jaccard = intersection.len() as f32 / union_size as f32;
        jaccard >= self.config.pre_token_threshold
    }
}

impl FilterStats {
    pub fn print_summary(&self) {
        let pct = |after: usize, before: usize| -> f32 {
            if before == 0 { 0.0 } else { (after as f32 / before as f32) * 100.0 }
        };
        
        println!("🔍 Filter Pipeline Summary:");
        println!("  Total possible pairs: {}", self.total_pairs);
        println!("  After size filter: {} ({:.1}%)", self.after_size_filter, pct(self.after_size_filter, self.total_pairs));
        println!("  After directory filter: {} ({:.1}%)", self.after_directory_filter, self.after_size_filter);
        println!("  After AST type filter: {} ({:.1}%)", self.after_ast_filter, self.after_directory_filter);
        println!("  After token filter: {} ({:.1}%)", self.after_token_filter, self.after_ast_filter);
        println!("  Final candidates: {} ({:.1}%)", self.final_candidates, pct(self.final_candidates, self.total_pairs));
        
        if self.total_pairs > 0 {
            let reduction = 100.0 - pct(self.final_candidates, self.total_pairs);
            println!("  📉 Total reduction: {:.1}%", reduction);
        }
    }
}