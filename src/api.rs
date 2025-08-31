use regex::Regex;
use serde::Serialize;

#[derive(Clone)]
pub struct FileBlob {
    pub path: String,
    pub text: String,
}

#[derive(Serialize, Clone)]
pub struct BackendRoute {
    pub file_path: String,
    pub method: String,
    pub path: String,
    pub framework: String,
}

#[derive(Serialize, Clone)]
pub struct FrontendCall {
    pub file_path: String,
    pub method: String,
    pub url: String,
    pub client: String,
}

#[derive(Serialize, Clone)]
pub struct ApiMap {
    pub matches: Vec<ApiMatch>,
    pub orphans_frontend: Vec<FrontendCall>,
    pub orphans_backend: Vec<BackendRoute>,
}

#[derive(Serialize, Clone)]
pub struct ApiMatch {
    pub method: String,
    pub pattern: String,
    pub backends: Vec<BackendRoute>,
    pub frontends: Vec<FrontendCall>,
}

pub fn extract_backend(files: &[FileBlob]) -> Vec<BackendRoute> {
    let re_express = Regex::new(r#"(?m)\b(?:app|router)\.(get|post|put|patch|delete|options|head)\s*\(\s*['"`]([^'"`]+)['"`]"#).unwrap();
    let re_axum = Regex::new(r#"(?m)\.route\(\s*["']([^"']+)["']\s*,\s*(get|post|put|patch|delete)\b"#).unwrap();
    let re_actix = Regex::new(r#"(?m)#\s*\[\s*(get|post|put|patch|delete)\s*\(\s*["']([^"']+)["']\s*\)\s*\]"#).unwrap();

    let mut out = Vec::new();
    for f in files {
        let text = &f.text;
        for cap in re_express.captures_iter(text) {
            out.push(BackendRoute {
                file_path: f.path.clone(),
                method: cap[1].to_uppercase(),
                path: cap[2].to_string(),
                framework: "express".into(),
            });
        }
        for cap in re_axum.captures_iter(text) {
            out.push(BackendRoute {
                file_path: f.path.clone(),
                method: cap[2].to_uppercase(),
                path: cap[1].to_string(),
                framework: "axum".into(),
            });
        }
        for cap in re_actix.captures_iter(text) {
            out.push(BackendRoute {
                file_path: f.path.clone(),
                method: cap[1].to_uppercase(),
                path: cap[2].to_string(),
                framework: "actix-web".into(),
            });
        }
    }
    out
}

pub fn extract_frontend(files: &[FileBlob]) -> Vec<FrontendCall> {
    let re_fetch = Regex::new(r#"(?m)\bfetch\s*\(\s*['"`]([^'"`]+)['"`]\s*(?:,\s*\{\s*[^}]*\bmethod\s*:\s*['"`]([A-Za-z]+)['"`][^}]*\})?"#).unwrap();
    let re_axios_short = Regex::new(r#"(?m)\baxios\.(get|post|put|patch|delete)\s*\(\s*['"`]([^'"`]+)['"`]"#).unwrap();
    let re_axios_cfg = Regex::new(r#"(?m)\baxios\s*\(\s*\{\s*[^}]*\bmethod\s*:\s*['"`]([A-Za-z]+)['"`][^}]*\burl\s*:\s*['"`]([^'"`]+)['"`][^}]*\}\s*\)"#).unwrap();

    let mut out = Vec::new();
    for f in files {
        let text = &f.text;
        for cap in re_fetch.captures_iter(text) {
            let m = cap.get(2).map(|m| m.as_str().to_uppercase()).unwrap_or("GET".into());
            out.push(FrontendCall {
                file_path: f.path.clone(),
                method: m,
                url: cap[1].to_string(),
                client: "fetch".into(),
            });
        }
        for cap in re_axios_short.captures_iter(text) {
            out.push(FrontendCall {
                file_path: f.path.clone(),
                method: cap[1].to_uppercase(),
                url: cap[2].to_string(),
                client: "axios".into(),
            });
        }
        for cap in re_axios_cfg.captures_iter(text) {
            out.push(FrontendCall {
                file_path: f.path.clone(),
                method: cap[1].to_uppercase(),
                url: cap[2].to_string(),
                client: "axios".into(),
            });
        }
    }
    out
}

pub fn map_api(mut fe: Vec<FrontendCall>, mut be: Vec<BackendRoute>) -> ApiMap {
    // Normalize paths: strip protocol/host, keep path, remove trailing slashes
    for f in fe.iter_mut() {
        f.url = normalize_url(&f.url);
    }
    for b in be.iter_mut() {
        b.path = normalize_url(&b.path);
    }

    // Group by method+normalized path
    use std::collections::BTreeMap;
    let mut by_key: BTreeMap<(String, String), (Vec<FrontendCall>, Vec<BackendRoute>)> = BTreeMap::new();

    for f in fe.iter().cloned() {
        let key = (f.method.clone(), f.url.clone());
        by_key.entry(key).or_default().0.push(f);
    }
    for b in be.iter().cloned() {
        let key = (b.method.clone(), b.path.clone());
        by_key.entry(key).or_default().1.push(b);
    }

    let mut matches = Vec::new();
    let mut orphans_frontend = Vec::new();
    let mut orphans_backend = Vec::new();

    for ((method, path), (fes, bes)) in by_key.into_iter() {
        if !fes.is_empty() && !bes.is_empty() {
            matches.push(ApiMatch {
                method,
                pattern: path,
                backends: bes,
                frontends: fes,
            });
        } else if !fes.is_empty() {
            orphans_frontend.extend(fes);
        } else if !bes.is_empty() {
            orphans_backend.extend(bes);
        }
    }

    ApiMap {
        matches,
        orphans_frontend,
        orphans_backend,
    }
}

fn normalize_url(s: &str) -> String {
    // Remove origin, keep path/query; trim trailing slash; ignore http(s)
    let mut t = s.to_string();
    if let Ok(url) = url::Url::parse(&t) {
        t = url.path().to_string();
    }
    if t.starts_with("//") {
        t = t.trim_start_matches("//").to_string();
    }
    // remove trailing slash except root
    if t.len() > 1 && t.ends_with('/') {
        t.pop();
    }
    t
}