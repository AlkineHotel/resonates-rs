use ahash::AHashSet;
use anndists::dist::DistCosine;
use hnsw_rs::prelude::*;
use rayon::prelude::*;

/// Build an HNSW on embedding vectors and return unique top-k neighbor pairs
/// as (i, j, cosine_similarity) with i < j.
/// - vectors: all same dimension, L2-normalized or not; distance is cosine (angular)
/// - k: neighbors per point (excluding self)
/// - max_nb_connection (M): 32 or 32 are typical; default 32 on 32-bit boxes
/// - ef_construction: 200–800 typical; 400 default works well
/// - ef_search (ef_arg): >= k; often between k and M (e.g., max(k+1, 96))
/// - max_layer: ≤ 16; None => min(16, ln(n))
pub fn topk_pairs_hnsw_cosine(
    vectors: &[Vec<f32>],
    k: usize,
    max_nb_connection: usize,
    ef_construction: usize,
    ef_search: usize,
    max_layer: Option<usize>,
) -> anyhow::Result<Vec<(usize, usize, f32)>> {
    if vectors.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let dim = vectors[0].len();
    if dim == 0 {
        anyhow::bail!("Vectors must have non-zero dimension");
    }
    if !vectors.iter().all(|v| v.len() == dim) {
        anyhow::bail!("All vectors must share the same dimension");
    }
    let n = vectors.len();

    let nb_layer = max_layer.unwrap_or_else(|| {
        let approx = (n as f32).ln().trunc() as usize;
        approx.min(16).max(1)
    });

    let hnsw = Hnsw::<f32, DistCosine>::new(
        max_nb_connection.max(8),
        n,
        nb_layer,
        ef_construction.max(50),
        DistCosine {},
    );

    // Parallel insert expects (&Vec<f32>, label_id)
    let backing: Vec<Vec<f32>> = vectors.to_vec();
    let to_insert: Vec<(&Vec<f32>, usize)> = backing
        .iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();

    let to_insert_search: Vec<Vec<f32>> = backing.iter().cloned().collect();

   
    hnsw.parallel_insert(&to_insert);

    // Query same set; fetch k+1 to skip self
    let knbn = k + 1;
    let ef_arg = ef_search.max(knbn);

    let results = hnsw.parallel_search(&to_insert_search, knbn, ef_arg);

    // Unique unordered pairs with cosine similarity (1 - distance)
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