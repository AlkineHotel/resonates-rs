use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct FastEmbedder {
    model_name: String,
    inner: TextEmbedding,
}

impl FastEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let model = resolve_model(model_name)?;
        let cache_dir = get_global_cache_dir()?;
        let inner = TextEmbedding::try_new(
            InitOptions::new(model)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )?;
        Ok(Self {
            model_name: model_name.to_string(),
            inner,
        })
    }

    pub fn embed(&mut self, ids_and_texts: &[(usize, String)], batch: usize) -> Result<HashMap<usize, Vec<f32>>> {
        let mut out: HashMap<usize, Vec<f32>> = HashMap::with_capacity(ids_and_texts.len());
        for chunk in ids_and_texts.chunks(batch.max(1)) {
            let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
            let vecs = self.inner.embed(texts, None)?;
            for ((id, _), v) in chunk.iter().zip(vecs.into_iter()) {
                // Fastembed returns already L2 normalized vectors
                out.insert(*id, v);
            }
        }
        Ok(out)
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}   



fn get_global_cache_dir() -> Result<PathBuf> {
    // Try env var first, then fall back to platform-appropriate cache dir
    if let Ok(cache_dir) = std::env::var("FASTEMBED_CACHE_DIR") {
        return Ok(PathBuf::from(cache_dir));
    }
    
    // Platform-specific cache directories
    let cache_dir = if cfg!(target_os = "windows") {
        std::env::var("LOCALAPPDATA")
            .or_else(|_| std::env::var("APPDATA"))
            .map(|dir| PathBuf::from(dir).join("fastembed"))
            .unwrap_or_else(|_| PathBuf::from("C:\\ProgramData\\fastembed"))
    } else if cfg!(target_os = "macos") {
        std::env::var("HOME")
            .map(|dir| PathBuf::from(dir).join("Library").join("Caches").join("fastembed"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/fastembed"))
    } else {
        // Linux/Unix
        std::env::var("XDG_CACHE_HOME")
            .map(|dir| PathBuf::from(dir).join("fastembed"))
            .or_else(|_| std::env::var("HOME").map(|dir| PathBuf::from(dir).join(".cache").join("fastembed")))
            .unwrap_or_else(|_| PathBuf::from("/tmp/fastembed"))
    };
    
    // Ensure directory exists
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)?;
    }
    
    Ok(cache_dir)
}

pub fn resolve_model(model_name: &str) -> Result<EmbeddingModel> {
    let key = model_name.to_ascii_lowercase();
    let model = match key.as_str() {
        // Good CPU defaults
        "bge-small-en-v1.5" | "baai/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "bge-base-en-v1.5" | "baai/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "jina-embeddings-v2-base-code" | "jinaai/jina-embeddings-v2-base-code" => EmbeddingModel::JinaEmbeddingsV2BaseCode,
        // Fallback to a compact strong model
        _ => EmbeddingModel::BGESmallENV15,
    };
    Ok(model)
}
