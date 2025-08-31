use ahash::{AHashMap, AHashSet};
use serde::Serialize;

use crate::similarity::SimilarityPair;

#[derive(Serialize, Clone)]
pub struct ClusterRef {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Serialize, Clone)]
pub struct Cluster {
    pub id: usize,
    pub members: Vec<ClusterRef>,
    pub size: usize,
    pub dominant_dir: String,
}

#[derive(Serialize, Clone)]
pub struct SuspectsReport {
    pub clusters: Vec<Cluster>,
    pub misplaced: Vec<ClusterRef>,
    pub notes: String,
}

pub fn clusters_and_suspects(pairs: &[SimilarityPair]) -> SuspectsReport {
    // Build union-find over chunk keys
    fn key(r: &super::similarity::ChunkRef) -> String {
        format!("{}:{}-{}", r.file_path, r.start_line, r.end_line)
    }

    let mut parent: AHashMap<String, String> = AHashMap::new();
    let mut rank: AHashMap<String, usize> = AHashMap::new();

    fn find(x: &str, parent: &mut AHashMap<String, String>) -> String {
        let p = parent.get(x).cloned().unwrap_or_else(|| x.to_string());
        if p != x {
            let r = find(&p, parent);
            parent.insert(x.to_string(), r.clone());
            r
        } else {
            p
        }
    }
    fn union(a: &str, b: &str, parent: &mut AHashMap<String, String>, rank: &mut AHashMap<String, usize>) {
        let mut ra = find(a, parent);
        let mut rb = find(b, parent);
        if ra == rb {
            return;
        }
        let rka = *rank.get(&ra).unwrap_or(&0);
        let rkb = *rank.get(&rb).unwrap_or(&0);
        if rka < rkb {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent.insert(rb.clone(), ra.clone());
        if rka == rkb {
            rank.insert(ra, rka + 1);
        }
    }

    let mut keys: AHashSet<String> = AHashSet::new();
    for p in pairs {
        let ka = key(&p.a);
        let kb = key(&p.b);
        keys.insert(ka.clone());
        keys.insert(kb.clone());
        parent.entry(ka.clone()).or_insert(ka.clone());
        parent.entry(kb.clone()).or_insert(kb.clone());
        union(&ka, &kb, &mut parent, &mut rank);
    }

    // Collect components
    let mut comps: AHashMap<String, Vec<ClusterRef>> = AHashMap::new();
    for p in pairs {
        for r in [&p.a, &p.b] {
            let k = key(r);
            let root = find(&k, &mut parent);
            comps.entry(root).or_default().push(ClusterRef {
                file_path: r.file_path.clone(),
                start_line: r.start_line,
                end_line: r.end_line,
            });
        }
    }

    // Build clusters with dominant dir
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut misplaced: Vec<ClusterRef> = Vec::new();
    for (i, (_root, members)) in comps.into_iter().enumerate() {
        // Determine dominant directory by first 2 path segments
        let mut count: AHashMap<String, usize> = AHashMap::new();
        for m in members.iter() {
            let dir = top_dir(&m.file_path);
            *count.entry(dir).or_default() += 1;
        }
        let dominant_dir = count
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(d, _)| d)
            .unwrap_or_else(|| "".into());

        // Misplaced = members not in dominant_dir
        for m in members.iter() {
            if top_dir(&m.file_path) != dominant_dir {
                misplaced.push(m.clone());
            }
        }

        clusters.push(Cluster {
            id: i,
            size: members.len(),
            members,
            dominant_dir,
        });
    }

    SuspectsReport {
        clusters,
        misplaced,
        notes: "Clusters built from similarity pairs (token+embedding). Misplaced = outside cluster's dominant directory."
            .into(),
    }
}

fn top_dir(path: &str) -> String {
    let parts: Vec<&str> = path.split(&['/', '\\'][..]).collect();
    let n = parts.len();
    if n >= 2 {
        format!("{}/{}", parts[0], parts[1])
    } else if n >= 1 {
        parts[0].to_string()
    } else {
        "".into()
    }
}