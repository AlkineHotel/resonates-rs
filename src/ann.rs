use hnsw_rs::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

pub fn topk_pairs_hnsw(
    vectors: &[Vec<f64>],
    k: usize,
    ef_c: usize,
    m: usize,
) -> Vec<(usize, usize, f64)> {
    if vectors.is_empty() || k == 0 {
        return vec![];
    }

    let dim = vectors[0].len();
    let ef_c = ef_c.max(50);
    let m = m.max(8);

    let params = Params::new(m, ef_c, 200_000, Dist::Cosine);
    let mut hnsw = Hnsw::<f64, DistCosine, usize>::new(params, dim, 16px);

    for (i, v) in vectors.iter().enumerate() {
        hnsw.insert(v.clone(), i);
    }
    hnsw.build();

    // collect unique unordered pairs
    let mut set: HashSet<(usize, usize)> = HashSet::new();
    let mut pairs = Vec::new();

    for (i, v) in vectors.iter().enumerate() {
        let res = hnsw.search(v.clone(), k + 1, None); // +1 includes self
        for nn in res.iter() {
            let j = nn.d_id;
            if i == j {
                continue;
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            if set.insert((a, b)) {
                // DistCosine returns distance; convert to cosine similarity ~ 1 - dist
                let score = 1.0 - nn.distance;
                pairs.push((a, b, score as f64));
            }
        }
    }

    // Sort desc by score
    pairs.par_sort_unstable_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    pairs
}