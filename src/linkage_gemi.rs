use crate::api_gemi::{BackendEndpoint, FrontendApiCall};
use crate::embedder_fast::{FastEmbedder, resolve_model};
use anyhow::{Result, anyhow};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ApiLinkage {
    pub frontend_call: FrontendApiCall,
    pub backend_endpoint: BackendEndpoint,
    pub similarity_score: f32,
    pub linkage_type: LinkageType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LinkageType {
    ExactMatch,
    SemanticMatch,
    PartialMatch,
}

pub async fn find_api_linkages(
    frontend_calls: Vec<FrontendApiCall>,
    backend_endpoints: Vec<BackendEndpoint>,
    embedder_cmd: &str,
    similarity_threshold: f32,
) -> Result<Vec<ApiLinkage>> {
    let mut linkages = Vec::new();

    // Initialize embedder
    let model_name = embedder_cmd.strip_prefix("fastembed:").unwrap_or("jina-embeddings-v2-base-code");
    let mut embedder = FastEmbedder::new(model_name)?;

    // Generate embeddings for backend handlers
    let mut backend_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut backend_texts: Vec<(usize, String)> = Vec::new();
    for (i, endpoint) in backend_endpoints.iter().enumerate() {
        backend_texts.push((i, format!("{} {} {}", endpoint.method, endpoint.path, endpoint.handler)));
    }
    if !backend_texts.is_empty() {
        let embeddings = embedder.embed(&backend_texts, backend_texts.len())?;
        for (idx, emb) in embeddings {
            backend_embeddings.insert(idx, emb);
        }
    }

    // Generate embeddings for frontend calls
    let mut frontend_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut frontend_texts: Vec<(usize, String)> = Vec::new();
    for (i, call) in frontend_calls.iter().enumerate() {
        frontend_texts.push((i, format!("{} {}", call.path, call.context_snippet)));
    }
    if !frontend_texts.is_empty() {
        let embeddings = embedder.embed(&frontend_texts, frontend_texts.len())?;
        for (idx, emb) in embeddings {
            frontend_embeddings.insert(idx, emb);
        }
    }

    // Compare frontend calls to backend endpoints
    for (fe_idx, fe_call) in frontend_calls.iter().enumerate() {
        for (be_idx, be_endpoint) in backend_endpoints.iter().enumerate() {
            let mut linkage_type = LinkageType::SemanticMatch;
            let mut score = 0.0;

            // 1. Exact Path Match (highest confidence)
            if fe_call.path == be_endpoint.path {
                linkage_type = LinkageType::ExactMatch;
                score = 1.0; // Perfect match
            }

            // 2. Semantic Match (using embeddings)
            if score < 1.0 {
                if let (Some(fe_emb), Some(be_emb)) = (
                    frontend_embeddings.get(&fe_idx),
                    backend_embeddings.get(&be_idx),
                ) {
                    let cosine_sim = cosine_similarity(fe_emb, be_emb);
                    if cosine_sim >= similarity_threshold {
                        score = cosine_sim;
                        linkage_type = LinkageType::SemanticMatch;
                    }
                }
            }

            // 3. Partial Path Match (e.g., /users/123 vs /users/:id)
            // This is more complex and might require regex or pattern matching
            // For now, we'll keep it simple and focus on exact and semantic.
            // If a semantic match is found, it's generally better than a partial path match.
            if score < similarity_threshold && fe_call.path.contains(&be_endpoint.path) || be_endpoint.path.contains(&fe_call.path) {
                // Simple heuristic for partial match, could be improved
                score = 0.5; // Assign a lower score for partial matches
                linkage_type = LinkageType::PartialMatch;
            }

            if score > 0.0 {
                linkages.push(ApiLinkage {
                    frontend_call: fe_call.clone(),
                    backend_endpoint: be_endpoint.clone(),
                    similarity_score: score,
                    linkage_type,
                });
            }
        }
    }

    // Sort linkages by similarity score (highest first)
    linkages.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(linkages)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
