use ahash::AHashMap;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use regex::Regex;
use serde::Serialize;

#[derive(Clone)]
pub struct FileBlob {
    pub path: String,
    pub text: String,
}

#[derive(Serialize, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub path: String,
    pub degree: usize,
    pub community: usize,
}

#[derive(Serialize, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub kind: String, // "import"
}

#[derive(Serialize, Clone)]
pub struct GraphReport {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

pub fn build_graph(files: &[FileBlob]) -> GraphReport {
    let mut g: Graph<&str, &str> = Graph::new();
    let mut idx_map: AHashMap<String, NodeIndex> = AHashMap::new();

    for f in files {
        let idx = g.add_node(f.path.as_str());
        idx_map.insert(f.path.clone(), idx);
    }

    let (re_import_ts, re_require, re_use_rs) = build_regexes();

    for f in files {
        let from = idx_map.get(&f.path).copied().unwrap();
        for target in extract_imports(&f.path, &f.text, &re_import_ts, &re_require, &re_use_rs) {
            if let Some(to) = idx_map.get(&target).copied() {
                g.add_edge(from, to, "import");
            }
        }
    }

    // degree + simple label propagation for communities
    let degrees: AHashMap<NodeIndex, usize> = g
        .node_indices()
        .map(|n| (n, g.neighbors(n).count()))
        .collect();

    let communities = label_propagation(&g, 5);

    let mut nodes = Vec::new();
    for (i, n) in g.node_indices().enumerate() {
        let path = g[n].to_string();
        let degree = degrees.get(&n).copied().unwrap_or(0);
        let community = communities.get(&n).copied().unwrap_or(0);
        nodes.push(GraphNode {
            id: i,
            path,
            degree,
            community,
        });
    }

    let mut edges = Vec::new();
    for e in g.edge_indices() {
        let (a, b) = g.edge_endpoints(e).unwrap();
        edges.push(GraphEdge {
            from: a.index(),
            to: b.index(),
            kind: "import".into(),
        });
    }

    GraphReport { nodes, edges }
}

fn build_regexes() -> (Regex, Regex, Regex) {
    // JS/TS imports and requires
    let re_import_ts = Regex::new(r#"(?m)^\s*import\s+.*?from\s+['"]([^'"]+)['"]"#).unwrap();
    let re_require = Regex::new(r#"(?m)require\(\s*['"]([^'"]+)['"]\s*\)"#).unwrap();
    // Rust use lines (very rough)
    let re_use_rs = Regex::new(r#"(?m)^\s*use\s+([a-zA-Z0-9_:\{\} ,]+);"#).unwrap();
    (re_import_ts, re_require, re_use_rs)
}

fn extract_imports(path: &str, text: &str, re_import: &Regex, re_require: &Regex, re_use_rs: &Regex) -> Vec<String> {
    let mut out = Vec::new();
    // Only link to local files (heuristic: starts with ./ or ../ or absolute-like within repo)
    for cap in re_import.captures_iter(text) {
        let raw = cap.get(1).unwrap().as_str().to_string();
        if is_local(&raw) {
            if let Some(resolved) = resolve_local(path, &raw) {
                out.push(resolved);
            }
        }
    }
    for cap in re_require.captures_iter(text) {
        let raw = cap.get(1).unwrap().as_str().to_string();
        if is_local(&raw) {
            if let Some(resolved) = resolve_local(path, &raw) {
                out.push(resolved);
            }
        }
    }
    // Rust: skip external crates, match self/super/mod or relative module paths
    for cap in re_use_rs.captures_iter(text) {
        let raw = cap.get(1).unwrap().as_str().trim().to_string();
        if raw.starts_with("crate") || raw.starts_with("self") || raw.starts_with("super") {
            // we can't resolve to file reliably without the Cargo tree; keep heuristic: link to same dir module files
            if let Some(dir) = std::path::Path::new(path).parent() {
                let dir = dir.to_string_lossy().to_string();
                if let Some(first) = raw.split("::").last() {
                    let cand = format!("{}/{}.rs", dir, first.replace(['{','}',' '], ""));
                    out.push(cand);
                }
            }
        }
    }
    out
}

fn is_local(s: &str) -> bool {
    s.starts_with("./") || s.starts_with("../") || s.starts_with("/")
}

fn resolve_local(from_path: &str, import_path: &str) -> Option<String> {
    use std::path::{Path, PathBuf};
    let base = Path::new(from_path).parent()?;
    let mut cand = PathBuf::from(base);
    cand.push(import_path);
    // try exact, then add common extensions and index files
    let tries = [
        cand.clone(),
        cand.with_extension("ts"),
        cand.with_extension("tsx"),
        cand.with_extension("js"),
        cand.with_extension("jsx"),
        {
            let mut t = cand.clone();
            t.push("index.ts");
            t
        },
        {
            let mut t = cand.clone();
            t.push("index.tsx");
            t
        },
        {
            let mut t = cand.clone();
            t.push("index.js");
            t
        },
        {
            let mut t = cand.clone();
            t.push("mod.rs");
            t
        },
    ];
    for t in tries {
        if t.exists() {
            return Some(t.to_string_lossy().to_string());
        }
    }
    None
}

// Very simple synchronous label propagation for communities
fn label_propagation(g: &petgraph::Graph<&str, &str>, iters: usize) -> AHashMap<petgraph::graph::NodeIndex, usize> {
    let mut label: AHashMap<_, _> = g.node_indices().map(|n| (n, n.index())).collect();
    for _ in 0..iters {
        for n in g.node_indices() {
            // pick most common neighbor label
            let mut counts: AHashMap<usize, usize> = AHashMap::new();
            for nb in g.neighbors(n) {
                let l = label.get(&nb).copied().unwrap_or(nb.index());
                *counts.entry(l).or_insert(0) += 1;
            }
            if let Some((&best, _)) = counts.iter().max_by_key(|(_, c)| *c) {
                label.insert(n, best);
            }
        }
    }
    label
}