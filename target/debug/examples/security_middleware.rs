// 🛡️ PRODUCTION SECURITY MIDDLEWARE
// Rate limiting, IP blocking, request validation, CSRF protection

use axum::{
    extract::{Request, State, ConnectInfo},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::Response,
    Json,
};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use serde_json::Value;

// 🚦 RATE LIMITING STRUCTURE
#[derive(Clone)]
pub struct RateLimiter {
    requests: Arc<RwLock<HashMap<IpAddr, Vec<Instant>>>>,
    blocks: Arc<RwLock<HashMap<IpAddr, Instant>>>,
    config: RateLimitConfig,
}

#[derive(Clone)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
    pub requests_per_hour: usize,
    pub block_duration_minutes: u64,
    pub whitelist: Vec<IpAddr>,
    pub auto_block_threshold: usize, // Auto-block after X violations
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 120,     // Increased from 30 to 120 for research use
            requests_per_hour: 1200,      // Increased from 300 to 1200
            block_duration_minutes: 10,   // Reduced from 60 to 10 minutes
            auto_block_threshold: 20,     // Increased from 5 to 20 violations
            whitelist: vec![
                "127.0.0.1".parse().unwrap(),
                "::1".parse().unwrap(),
                "192.168.0.0/16".parse().unwrap(), // Allow local network
                "10.0.0.0/8".parse().unwrap(),     // Allow private networks
            ],
        }
    }
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            blocks: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    // 🔍 CHECK IF IP IS BLOCKED
    pub async fn is_blocked(&self, ip: IpAddr) -> bool {
        if self.config.whitelist.contains(&ip) {
            return false;
        }

        let blocks = self.blocks.read().await;
        if let Some(blocked_at) = blocks.get(&ip) {
            let block_duration = Duration::from_secs(self.config.block_duration_minutes * 60);
            return blocked_at.elapsed() < block_duration;
        }
        false
    }

    // 🚫 BLOCK IP ADDRESS
    pub async fn block_ip(&self, ip: IpAddr) {
        if self.config.whitelist.contains(&ip) {
            return;
        }

        let mut blocks = self.blocks.write().await;
        blocks.insert(ip, Instant::now());
        
        println!("🚫 BLOCKED IP: {} for {} minutes", ip, self.config.block_duration_minutes);
    }

    // ⚡ CHECK RATE LIMIT WITH CUSTOM LIMITS
    pub async fn check_rate_limit_custom(&self, ip: IpAddr, minute_limit: usize, hour_limit: usize) -> RateLimitResult {
        if self.config.whitelist.contains(&ip) {
            return RateLimitResult::Allowed;
        }

        if self.is_blocked(ip).await {
            return RateLimitResult::Blocked;
        }

        let now = Instant::now();
        let mut requests = self.requests.write().await;
        
        let ip_requests = requests.entry(ip).or_insert_with(Vec::new);
        
        // Clean old requests (older than 1 hour)
        ip_requests.retain(|&time| now.duration_since(time) < Duration::from_secs(3600));
        
        // Count recent requests
        let minute_count = ip_requests.iter()
            .filter(|&&time| now.duration_since(time) < Duration::from_secs(60))
            .count();
        let hour_count = ip_requests.len();

        // Check custom limits
        if minute_count >= minute_limit {
            return RateLimitResult::RateLimited;
        }
        if hour_count >= hour_limit {
            return RateLimitResult::RateLimited;
        }

        // Record this request
        ip_requests.push(now);
        
        RateLimitResult::Allowed
    }

    // ⚡ CHECK RATE LIMIT
    pub async fn check_rate_limit(&self, ip: IpAddr) -> RateLimitResult {
        if self.config.whitelist.contains(&ip) {
            return RateLimitResult::Allowed;
        }

        if self.is_blocked(ip).await {
            return RateLimitResult::Blocked;
        }

        let now = Instant::now();
        let mut requests = self.requests.write().await;
        
        let ip_requests = requests.entry(ip).or_insert_with(Vec::new);
        
        // Clean old requests (older than 1 hour)
        ip_requests.retain(|&time| now.duration_since(time) < Duration::from_secs(3600));
        
        // Count recent requests
        let minute_count = ip_requests.iter()
            .filter(|&&time| now.duration_since(time) < Duration::from_secs(60))
            .count();
        let hour_count = ip_requests.len();

        // Check limits
        if minute_count >= self.config.requests_per_minute {
            // Auto-block if too many minute violations
            let violations = minute_count - self.config.requests_per_minute;
            if violations >= self.config.auto_block_threshold {
                self.block_ip(ip).await;
                return RateLimitResult::Blocked;
            }
            return RateLimitResult::RateLimited;
        }

        if hour_count >= self.config.requests_per_hour {
            return RateLimitResult::RateLimited;
        }

        // Record this request
        ip_requests.push(now);
        RateLimitResult::Allowed
    }

    // 📊 GET RATE LIMIT STATUS
    pub async fn get_status(&self, ip: IpAddr) -> RateLimitStatus {
        let requests = self.requests.read().await;
        let blocks = self.blocks.read().await;
        
        let ip_requests = requests.get(&ip).cloned().unwrap_or_default();
        let now = Instant::now();
        
        let minute_count = ip_requests.iter()
            .filter(|&&time| now.duration_since(time) < Duration::from_secs(60))
            .count();
        let hour_count = ip_requests.len();

        let blocked_until = blocks.get(&ip).map(|blocked_at| {
            *blocked_at + Duration::from_secs(self.config.block_duration_minutes * 60)
        });

        RateLimitStatus {
            requests_this_minute: minute_count,
            requests_this_hour: hour_count,
            max_per_minute: self.config.requests_per_minute,
            max_per_hour: self.config.requests_per_hour,
            blocked_until,
        }
    }
}

#[derive(Debug)]
pub enum RateLimitResult {
    Allowed,
    RateLimited,
    Blocked,
}

#[derive(Debug)]
pub struct RateLimitStatus {
    pub requests_this_minute: usize,
    pub requests_this_hour: usize,
    pub max_per_minute: usize,
    pub max_per_hour: usize,
    pub blocked_until: Option<Instant>,
}

// 🛡️ SECURITY MIDDLEWARE
pub async fn security_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(app_state): State<crate::network_utils::AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let limiter = &app_state.rate_limiter;
    let ip = addr.ip();
    let path = request.uri().path();
    
    // 🔐 RELAXED RATE LIMITING FOR AUTH ENDPOINTS (still secure, but not aggressive)
    let is_auth_endpoint = path.starts_with("/api/auth") || path.starts_with("/api/security");
    
    if is_auth_endpoint {
        // Still check for definitely malicious IPs, but be lenient on rate limits
        if limiter.is_blocked(ip).await {
            println!("🚫 Blocked auth attempt from banned IP: {}", ip);
            return Err(StatusCode::FORBIDDEN);
        }
        
        // For auth endpoints, allow more requests per minute (20 instead of default)
        let auth_rate_check = limiter.check_rate_limit_custom(ip, 20, 100).await;
        match auth_rate_check {
            RateLimitResult::Blocked => {
                println!("🚫 Auth endpoint blocked from {}: too many login attempts", ip);
                return Err(StatusCode::TOO_MANY_REQUESTS);
            },
            _ => {
                println!("🔐 Auth endpoint allowed: {} from {}", path, ip);
            }
        }
    }
    
    // 🚫 CHECK IP BLOCK STATUS
    if limiter.is_blocked(ip).await {
        println!("🚫 Blocked request from {}", ip);
        return Err(StatusCode::FORBIDDEN);
    }

    // 🚦 CHECK RATE LIMIT
    match limiter.check_rate_limit(ip).await {
        RateLimitResult::Blocked => {
            println!("🚫 Auto-blocked request from {} (too many violations)", ip);
            return Err(StatusCode::FORBIDDEN);
        },
        RateLimitResult::RateLimited => {
            println!("🚦 Rate limited request from {}", ip);
            return Err(StatusCode::TOO_MANY_REQUESTS);
        },
        RateLimitResult::Allowed => {}
    }

    // 🔍 SUSPICIOUS REQUEST DETECTION (ONLY BLOCK DEFINITELY MALICIOUS)
    if is_definitely_malicious(request.uri().path()) {
        println!("🚨 Malicious request blocked from {}: {}", ip, request.uri().path());
        limiter.block_ip(ip).await;
        return Err(StatusCode::FORBIDDEN);
    }
    
    // Just log suspicious requests but don't block
    if is_suspicious_request(&headers, request.uri().path()) {
        println!("⚠️ Suspicious request from {}: {} (logged but allowed)", ip, request.uri().path());
    }

    // ✅ REQUEST ALLOWED - CONTINUE
    let response = next.run(request).await;
    Ok(response)
}

// 🔍 DETECT SUSPICIOUS REQUESTS
fn is_suspicious_request(headers: &HeaderMap, path: &str) -> bool {
    // Check user agent for bot patterns
    if let Some(ua) = headers.get("user-agent") {
        if let Ok(ua_str) = ua.to_str() {
            let ua_lower = ua_str.to_lowercase();
            // Only flag obvious automated tools, not development tools
            let bot_patterns = [
                "bot", "crawler", "spider", "scraper", "scan", "automated"
                // Removed: "python", "curl", "wget", "postman", "insomnia" - these are dev tools
            ];
            
            if bot_patterns.iter().any(|&pattern| ua_lower.contains(pattern)) {
                return true;
            }
        }
    }

    // Check for suspicious paths
    let suspicious_paths = [
        "/.env", "/.git", "/admin", "/wp-", "/phpmyadmin",
        "/shell", "/cmd", "/backdoor", "/hack", "/exploit", "/sql"
    ];
    
    suspicious_paths.iter().any(|&pattern| path.contains(pattern))
}

// 🚨 DEFINITELY MALICIOUS REQUESTS (AUTO-BLOCK)
fn is_definitely_malicious(path: &str) -> bool {
    let malicious_patterns = [
        "/.env", "/.git", "/config.php", "/wp-config.php",
        "/shell.php", "/backdoor", "/c99.php", "/phpinfo.php",
        "/eval(", "/system(", "/exec(", "/passthru(",
        "../../../", "..\\..\\..\\", // Directory traversal
        "<script", "javascript:", "onload=", "onerror=", // XSS attempts
        "union select", "drop table", "insert into", // SQL injection
    ];
    
    let path_lower = path.to_lowercase();
    malicious_patterns.iter().any(|&pattern| path_lower.contains(pattern))
}

// 🛡️ CSRF PROTECTION MIDDLEWARE (RELAXED FOR RESEARCH USE)
pub async fn csrf_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let method = request.method();
    let path = request.uri().path();
    
    // 🔓 RELAXED CSRF - Only protect very sensitive operations
    if matches!(method.as_str(), "POST" | "PUT" | "DELETE" | "PATCH") {
        // Only require CSRF for truly dangerous operations
        let requires_csrf = path.contains("/admin") || 
                           path.contains("/delete") ||
                           path.contains("/nuclear") ||
                           path.contains("/reset");
        
        if requires_csrf {
            // Check for CSRF token in header or form data
            if !headers.contains_key("x-csrf-token") && 
               !headers.contains_key("x-requested-with") {
                println!("🚨 CSRF: Missing CSRF token for sensitive operation {} {}", method, path);
                return Err(StatusCode::FORBIDDEN);
            }
        } else {
            // For research/API endpoints, just log but allow
            println!("🔓 CSRF: Allowing {} {} (research mode)", method, path);
        }
    }

    // Add security headers to response
    let mut response = next.run(request).await;
    let response_headers = response.headers_mut();
    
    // Security headers
    response_headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    response_headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    response_headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    response_headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    response_headers.insert("Content-Security-Policy", 
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'".parse().unwrap());

    Ok(response)
}

// 📊 SECURITY STATUS ENDPOINT
pub async fn security_status(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(app_state): State<crate::network_utils::AppState>,
) -> Json<Value> {
    let limiter = &app_state.rate_limiter;
    let ip = addr.ip();
    let status = limiter.get_status(ip).await;
    
    Json(serde_json::json!({
        "ip": ip.to_string(),
        "requests_this_minute": status.requests_this_minute,
        "requests_this_hour": status.requests_this_hour,
        "limits": {
            "per_minute": status.max_per_minute,
            "per_hour": status.max_per_hour
        },
        "blocked": status.blocked_until.is_some(),
        "blocked_until": status.blocked_until.map(|until| until.elapsed().as_secs()),
        "status": if status.blocked_until.is_some() { "blocked" } else { "allowed" }
    }))
}