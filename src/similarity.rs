use anyhow::{anyhow, Result};
use ahash::{AHashMap, AHashSet};
use anndists::dist::DistCosine;
use hnsw_rs::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;

// -------------------------- Public API types --------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarityMode {
    None,
    Token,
    Embedding,
}

#[derive(Clone, Debug)]
pub struct RawChunk {
    pub id: usize,
    pub file_path: String,
    pub subtree_description: String,
    pub start_line: usize,
    pub end_line: usize,
    pub size: usize,
    pub text: String,
}

#[derive(Clone, Serialize)]
pub struct SimilarityReport {
    pub method: String,
    pub threshold: f32,
    pub top_k: usize,
    pub total_chunks: usize,
    pub total_pairs: usize,
    pub pairs: Vec<SimilarityPair>,
}

#[derive(Clone, Serialize)]
pub struct SimilarityPair {
    pub score: f32,
    pub method: String,
    pub a: ChunkRef,
    pub b: ChunkRef,
    pub overlap_tokens: usize,
    pub union_tokens: usize,
    pub same_file: bool,
    pub common_path_prefix: usize,
}

#[derive(Clone, Serialize)]
pub struct ChunkRef {
    pub file_path: String,
    pub subtree_description: String,
    pub start_line: usize,
    pub end_line: usize,
    pub size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

pub struct SimilarityParams<'a> {
    pub mode: SimilarityMode,
    pub threshold: f32,          // token: Jaccard; embedding: cosine
    pub top_k: usize,

    // Token mode
    pub band_bits: usize,
    pub min_tokens: usize,

    // Output controls
    pub include_snippets: bool,
    pub cross_file_only: bool,

    // Embeddings
    pub embedder_cmd: Option<&'a str>, // "fastembed:<model>" or external command

    // HNSW (embedding mode)
    pub ann_k: usize,           // neighbors per point
    pub ann_ef: usize,          // ef_construction
    pub ann_m: usize,           // max_nb_connection (M)
    pub ann_ef_search: usize,   // ef during search

    // Two-stage verification
    pub verify_min_jaccard: f32,
}

// -------------------------- Entry point --------------------------

pub fn run_similarity(
    raw_chunks: Vec<RawChunk>,
    params: SimilarityParams<'_>,
) -> Result<SimilarityReport> {
    match params.mode {
        SimilarityMode::None => Ok(SimilarityReport {
            method: "none".into(),
            threshold: params.threshold,
            top_k: params.top_k,
            total_chunks: raw_chunks.len(),
            total_pairs: 0,
            pairs: vec![],
        }),
        SimilarityMode::Token => token_similarity(raw_chunks, params),
        SimilarityMode::Embedding => embedding_similarity(raw_chunks, params),
    }
}

// -------------------------- Token mode --------------------------

#[derive(Clone, Debug)]
struct TokenizedChunk {
    id: usize,
    file_path: String,
    subtree_description: String,
    start_line: usize,
    end_line: usize,
    size: usize,
    tokens: AHashSet<String>,
    simhash: u64,
}

fn token_similarity(raw_chunks: Vec<RawChunk>, params: SimilarityParams<'_>) -> Result<SimilarityReport> {
    let id_to_snippet: AHashMap<usize, String> = raw_chunks
        .iter()
        .map(|c| (c.id, snippet(&c.text)))
        .collect();

    let tokenized = tokenize_chunks(raw_chunks, params.min_tokens);
    let candidates = band_candidates(&tokenized, params.band_bits);

    let mut pairs: Vec<SimilarityPair> = candidates
        .into_par_iter()
        .filter_map(|(i, j)| {
            let a = &tokenized[i];
            let b = &tokenized[j];
            if params.cross_file_only && a.file_path == b.file_path {
                return None;
            }
            let (inter, union) = jaccard_counts(&a.tokens, &b.tokens);
            if union == 0 {
                return None;
            }
            let score = inter as f32 / union as f32;
            if score >= params.threshold {
                let a_snip = params.include_snippets.then(|| id_to_snippet.get(&a.id).cloned()).flatten();
                let b_snip = params.include_snippets.then(|| id_to_snippet.get(&b.id).cloned()).flatten();
                Some(SimilarityPair {
                    score,
                    method: "token".into(),
                    overlap_tokens: inter,
                    union_tokens: union,
                    same_file: a.file_path == b.file_path,
                    common_path_prefix: common_dir_prefix_len(&a.file_path, &b.file_path),
                    a: ChunkRef {
                        file_path: a.file_path.clone(),
                        subtree_description: a.subtree_description.clone(),
                        start_line: a.start_line,
                        end_line: a.end_line,
                        size: a.size,
                        snippet: a_snip,
                    },
                    b: ChunkRef {
                        file_path: b.file_path.clone(),
                        subtree_description: b.subtree_description.clone(),
                        start_line: b.start_line,
                        end_line: b.end_line,
                        size: b.size,
                        snippet: b_snip,
                    },
                })
            } else {
                None
            }
        })
        .collect();

    pairs.par_sort_unstable_by(|x, y| y.score.partial_cmp(&x.score).unwrap());
    if params.top_k > 0 && pairs.len() > params.top_k {
        pairs.truncate(params.top_k);
    }

    Ok(SimilarityReport {
        method: "token".into(),
        threshold: params.threshold,
        top_k: params.top_k,
        total_chunks: tokenized.len(),
        total_pairs: pairs.len(),
        pairs,
    })
}

fn tokenize_chunks(chunks: Vec<RawChunk>, min_tokens: usize) -> Vec<TokenizedChunk> {
    let ident_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
    let stop: AHashSet<&'static str> = common_stops();

    chunks
        .into_par_iter()
        .filter_map(|c| {
            let mut toks: Vec<String> = Vec::new();
            for m in ident_re.find_iter(&c.text) {
                let tok = m.as_str().to_ascii_lowercase();
                for part in split_ident(&tok) {
                    if part.len() >= 2 && !stop.contains(part.as_str()) {
                        toks.push(part);
                    }
                }
            }
            if toks.len() < min_tokens {
                return None;
            }
            let set: AHashSet<String> = toks.into_iter().collect();
            let simhash = simhash64(&set);
            Some(TokenizedChunk {
                id: c.id,
                file_path: c.file_path,
                subtree_description: c.subtree_description,
                start_line: c.start_line,
                end_line: c.end_line,
                size: c.size,
                tokens: set,
                simhash,
            })
        })
        .collect()
}

// SimHash banding over 64 bits. band_bits must divide 64; otherwise we fall back to 8.
fn band_candidates(chunks: &[TokenizedChunk], band_bits_in: usize) -> Vec<(usize, usize)> {
    let band_bits = match band_bits_in {
        1 | 2 | 4 | 8 | 16 | 32 | 64 => band_bits_in,
        _ => 8,
    };
    let bands = (64 / band_bits).max(1);
    let mut buckets: Vec<AHashMap<u64, Vec<usize>>> = (0..bands).map(|_| AHashMap::default()).collect();

    for (idx, c) in chunks.iter().enumerate() {
        for b in 0..bands {
            let shift = b * band_bits;
            let mask: u64 = if band_bits == 64 { u64::MAX } else { (1u64 << band_bits) - 1 };
            let key = (c.simhash >> shift) & mask;
            buckets[b].entry(key).or_default().push(idx);
        }
    }

    let mut seen: AHashSet<(usize, usize)> = AHashSet::new();
    let mut pairs = Vec::new();
    for b in 0..bands {
        for v in buckets[b].values() {
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    let a = v[i];
                    let bb = v[j];
                    let (x, y) = if a < bb { (a, bb) } else { (bb, a) };
                    if seen.insert((x, y)) {
                        pairs.push((x, y));
                    }
                }
            }
        }
    }
    pairs
}

// -------------------------- Embedding mode (with hnsw_rs) --------------------------

fn embedding_similarity(raw_chunks: Vec<RawChunk>, params: SimilarityParams<'_>) -> Result<SimilarityReport> {
    if params.embedder_cmd.is_none() {
        return Err(anyhow!("embedding mode requires --embedder-cmd, e.g. fastembed:bge-small-en-v1.5"));
    }

    // Token sets for verification pass
    let id_to_tokens: AHashMap<usize, AHashSet<String>> = {
        let ident_re = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
        let stop = common_stops();
        raw_chunks
            .par_iter()
            .map(|c| {
                let mut toks: Vec<String> = Vec::new();
                for m in ident_re.find_iter(&c.text) {
                    let tok = m.as_str().to_ascii_lowercase();
                    for part in split_ident(&tok) {
                        if part.len() >= 2 && !stop.contains(part.as_str()) {
                            toks.push(part);
                        }
                    }
                }
                (c.id, toks.into_iter().collect())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    };

    // Build id -> index mapping for metadata access
    let id_to_idx: AHashMap<usize, usize> = raw_chunks.iter().enumerate().map(|(i, c)| (c.id, i)).collect();

    // Get embeddings
    let (vectors, id_order): (Vec<Vec<f32>>, Vec<usize>) = get_embeddings(&raw_chunks, params.embedder_cmd.unwrap())?;

    if vectors.is_empty() {
        return Ok(SimilarityReport {
            method: "embedding".into(),
            threshold: params.threshold,
            top_k: params.top_k,
            total_chunks: raw_chunks.len(),
            total_pairs: 0,
            pairs: vec![],
        });
    }

    // HNSW candidate neighbors (cosine sim)
    let hnsw_pairs = hnsw_topk_pairs_cosine(
        &vectors,
        params.ann_k,
        params.ann_m.max(32),
        params.ann_ef.max(200),
        params.ann_ef_search.max(params.ann_k + 1).max(64),
        None,
    )?;

    // Convert positions back to chunk indices, filter, verify, and build final pairs
    let mut pairs: Vec<SimilarityPair> = hnsw_pairs
        .into_par_iter()
        .filter_map(|(ai, bi, cos)| {
            if cos < params.threshold || !cos.is_finite() {
                return None;
            }
            let ida = id_order[ai];
            let idb = id_order[bi];
            let ia = *id_to_idx.get(&ida)?;
            let ib = *id_to_idx.get(&idb)?;
            let a = &raw_chunks[ia];
            let b = &raw_chunks[ib];

            if params.cross_file_only && a.file_path == b.file_path {
                return None;
            }

            // Two-stage verify using token Jaccard (light)
            let (inter, union) = jaccard_counts(
                id_to_tokens.get(&ida).unwrap_or(&AHashSet::new()),
                id_to_tokens.get(&idb).unwrap_or(&AHashSet::new()),
            );
            let jac = if union == 0 { 0.0 } else { inter as f32 / union as f32 };
            if jac < params.verify_min_jaccard {
                return None;
            }

            Some(SimilarityPair {
                score: cos,
                method: "embedding".into(),
                overlap_tokens: inter,
                union_tokens: union,
                same_file: a.file_path == b.file_path,
                common_path_prefix: common_dir_prefix_len(&a.file_path, &b.file_path),
                a: ChunkRef {
                    file_path: a.file_path.clone(),
                    subtree_description: a.subtree_description.clone(),
                    start_line: a.start_line,
                    end_line: a.end_line,
                    size: a.size,
                    snippet: params.include_snippets.then(|| snippet(&a.text)),
                },
                b: ChunkRef {
                    file_path: b.file_path.clone(),
                    subtree_description: b.subtree_description.clone(),
                    start_line: b.start_line,
                    end_line: b.end_line,
                    size: b.size,
                    snippet: params.include_snippets.then(|| snippet(&b.text)),
                },
            })
        })
        .collect();

    pairs.par_sort_unstable_by(|x, y| y.score.partial_cmp(&x.score).unwrap());
    if params.top_k > 0 && pairs.len() > params.top_k {
        pairs.truncate(params.top_k);
    }

    Ok(SimilarityReport {
        method: "embedding".into(),
        threshold: params.threshold,
        top_k: params.top_k,
        total_chunks: raw_chunks.len(),
        total_pairs: pairs.len(),
        pairs,
    })
}

// -------------------------- Embedding + HNSW helpers --------------------------

fn get_embeddings(chunks: &[RawChunk], embedder_cmd: &str) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    // Build text payload (shortened per chunk for speed)
    let items: Vec<(usize, String)> = chunks
        .iter()
        .map(|c| (c.id, snippet_for_embed(&c.text, 4096)))
        .collect();

    // fastembed path 
    if let Some(model) = embedder_cmd.strip_prefix("fastembed:") {
        let mut embedder = crate::embedder_fast::FastEmbedder::new(model)?;
        let map = embedder.embed(&items, 1024)?;
        // Keep original chunk order; skip missing
        let mut vecs = Vec::new();
        let mut ids = Vec::new();
        for (id, _) in items.iter() {
            if let Some(v) = map.get(id) {
                vecs.push(v.clone()); // fastembed vectors are already L2-normalized
                ids.push(*id);
            }
        }
        return Ok((vecs, ids));
    }

    // External subprocess path: send JSON in, read JSON out
    use std::io::Write;
    #[derive(serde::Serialize)]
    struct EmbIn<'a> { id: usize, text: &'a str }
    #[derive(serde::Deserialize)]
    struct EmbOut { id: usize, vec: Vec<f32> }

    let payload: Vec<EmbIn> = items.iter().map(|(id, text)| EmbIn { id: *id, text }).collect();
    let json = serde_json::to_string(&payload)?;
    let mut child = std::process::Command::new("sh")
        .arg("-lc")
        .arg(embedder_cmd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    child.stdin.as_mut().ok_or_else(|| anyhow!("failed to open stdin for embedder"))?
        .write_all(json.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(anyhow!("Embedder command failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let outs: Vec<EmbOut> = serde_json::from_slice(&output.stdout)?;
    let mut by_id: AHashMap<usize, Vec<f32>> = AHashMap::default();
    for EmbOut { id, vec } in outs {
        by_id.insert(id, l2_normalize(vec));
    }

    let mut vecs = Vec::new();
    let mut ids = Vec::new();
    for (id, _) in items.iter() {
        if let Some(v) = by_id.get(id) {
            vecs.push(v.clone());
            ids.push(*id);
        }
    }
    Ok((vecs, ids))
}

fn hnsw_topk_pairs_cosine(
    vectors: &[Vec<f32>],
    k: usize,
    max_nb_connection: usize,
    ef_construction: usize,
    ef_search: usize,
    max_layer: Option<usize>,
) -> Result<Vec<(usize, usize, f32)>> {
    if vectors.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let dim = vectors[0].len();
    if dim == 0 {
        return Err(anyhow!("Vectors must have non-zero dimension"));
    }
    if !vectors.iter().all(|v| v.len() == dim) {
        return Err(anyhow!("All vectors must share the same dimension"));
    }
    let n = vectors.len();

    let nb_layer = max_layer.unwrap_or_else(|| {
        let approx = (n as f32).ln().trunc() as usize;
        approx.min(16).max(1)
    });

    let mut hnsw = Hnsw::<f32, DistCosine>::new(
        max_nb_connection.max(8), // M
        n,                        // capacity
        nb_layer.min(16),         // layers
        ef_construction.max(50),  // ef_construction
        DistCosine {},
    );

    // Use stable backing so we can hand out &[f32]
    let backing: Vec<Vec<f32>> = vectors.to_vec();
    let to_insert: Vec<(&Vec<f32>, usize)> = backing
        .iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    // Build
    hnsw.parallel_insert(&to_insert);

    // Search same set, k+1 to skip self
    let knbn = k + 1;
    let ef_arg = ef_search.max(knbn);
    let results = hnsw.parallel_search(&backing, knbn, ef_arg);

    // Dedup unordered pairs i<j, convert distance to cosine similarity
    let mut seen: AHashSet<(usize, usize)> = AHashSet::new();
    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
    for (i, neighs) in results.into_iter().enumerate() {
        for nn in neighs {
            let j = nn.d_id;
            if j == i {
                continue;
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            if seen.insert((a, b)) {
                let sim = 1.0f32 - nn.distance as f32;
                if sim.is_finite() {
                    pairs.push((a, b, sim));
                }
            }
        }
    }

    pairs.par_sort_unstable_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    Ok(pairs)
}

// -------------------------- Utilities --------------------------

fn snippet(s: &str) -> String {
    const MAX: usize = 800;
    if s.len() <= MAX { s.to_string() } else { s[..MAX].to_string() }
}

fn snippet_for_embed(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { s[..max].to_string() }
}

fn split_ident(s: &str) -> Vec<String> {
    let snake: Vec<&str> = s.split('_').filter(|p| !p.is_empty()).collect();
    let mut out: Vec<String> = Vec::new();
    for part in snake {
        let mut buf = String::new();
        for (i, ch) in part.chars().enumerate() {
            if i > 0 && ch.is_ascii_uppercase() {
                if !buf.is_empty() { out.push(buf.to_ascii_lowercase()); }
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

fn simhash64(tokens: &AHashSet<String>) -> u64 {
    // Simple SimHash with ahash
    let mut v = [0i64; 64];
    for t in tokens {
        let hv = ahash::RandomState::with_seeds(1, 2, 3, 4).hash_one(t) as u64;
        for i in 0..64 {
            if (hv >> i) & 1 == 1 {
                v[i] += 1;
            } else {
                v[i] -= 1;
            }
        }
    }
    let mut out: u64 = 0;
    for i in 0..64 {
        if v[i] >= 0 { out |= 1u64 << i; }
    }
    out
}

fn jaccard_counts(a: &AHashSet<String>, b: &AHashSet<String>) -> (usize, usize) {
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    (inter, union)
}

pub fn common_dir_prefix_len(a: &str, b: &str) -> usize {
    let asplit: Vec<&str> = a.split(&['/', '\\'][..]).collect();
    let bsplit: Vec<&str> = b.split(&['/', '\\'][..]).collect();
    let mut n = 0;
    while n < asplit.len() && n < bsplit.len() && asplit[n] == bsplit[n] {
        n += 1;
    }
    n
}

pub fn common_stops() -> AHashSet<&'static str> {
    [
        "let","const","var","fn","function","return","if","else","for","while","do","switch","case","break","continue",
        "class","struct","impl","pub","mod","use","mut","ref","type","enum","match","as","in","of","new","this","super",
        "true","false","null","undefined","await","async","yield","try","catch","finally","throw","static","get","set",
        "import","export","from","package","private","protected","public","interface","extends","implements","with"
    ].into_iter().collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut s = 0f32;
    for i in 0..len {
        s += a[i] * b[i];
    }
    s
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let mut ss = 0f32;
    for &x in &v { ss += x * x; }
    let n = ss.sqrt();
    if n > 0.0 {
        for mut x in &mut v { *x /= n; }
    }
    v
}