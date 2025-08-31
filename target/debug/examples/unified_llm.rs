use crate::config_types::{ContinueConfig, ContinueModel};
use reqwest;
use serde::{de::EnumAccess, Deserialize, Serialize};
use std::collections::{self, HashMap};
use std::env;
use crate::simple_config::{SimpleConfig, ModelStatus};
use std::path::PathBuf;
use tokio::fs;
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpointCache {
    model_endpoints: HashMap<String, String>, // model_name -> working_endpoint
    last_tested: HashMap<String, u64>, // model_name -> timestamp
    failed_endpoints: HashMap<String, Vec<String>>, // model_name -> [failed_endpoints]
}

impl ModelEndpointCache {
    fn new() -> Self {
        Self {
            model_endpoints: HashMap::new(),
            last_tested: HashMap::new(),
            failed_endpoints: HashMap::new(),
        }
    }

    async fn load_from_file() -> Result<Self, Box<dyn std::error::Error>> {
        let cache_path = Self::get_cache_path();
        
        if cache_path.exists() {
            let contents = fs::read_to_string(&cache_path).await?;
            let cache: ModelEndpointCache = serde_yaml::from_str(&contents)?;
            println!("📂 Loaded endpoint cache with {} models", cache.model_endpoints.len());
            Ok(cache)
        } else {
            println!("📂 No endpoint cache found, starting fresh");
            Ok(Self::new())
        }
    }

    async fn save_to_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        let cache_path = Self::get_cache_path();
        
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        let yaml_content = serde_yaml::to_string(self)?;
        fs::write(&cache_path, yaml_content).await?;
        println!("💾 Saved endpoint cache with {} models", self.model_endpoints.len());
        Ok(())
    }

    fn get_cache_path() -> PathBuf {
        PathBuf::from("model_endpoint_cache.yaml")
    }

    fn get_working_endpoint(&self, model_name: &str) -> Option<&String> {
        self.model_endpoints.get(model_name)
    }

    fn record_working_endpoint(&mut self, model_name: &str, endpoint: &str) {
        self.model_endpoints.insert(model_name.to_string(), endpoint.to_string());
        self.last_tested.insert(model_name.to_string(), std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());
        println!("✅ Cached working endpoint for {}: {}", model_name, endpoint);
    }

    fn record_failed_endpoint(&mut self, model_name: &str, endpoint: &str) {
        self.failed_endpoints
            .entry(model_name.to_string())
            .or_insert_with(Vec::new)
            .push(endpoint.to_string());
    }

    fn should_retest(&self, model_name: &str) -> bool {
        match self.last_tested.get(model_name) {
            Some(last_test) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                now - last_test > 86400 // Retest after 24 hours instead of 1 hour
            },
            None => true
        }
    }
}

// 🚀 TODO: EXTRACT INTO `ollama-conductor` CRATE
//
// This file contains the foundation for what should become the most badass
// Ollama orchestration crate in the Rust ecosystem. Current Ollama crates are
// just boring HTTP wrappers. We're building:
//
// 🎯 FEATURES FOR FUTURE CRATE:
// - Environment-driven dynamic discovery (no hardcoded hosts)
// - Health scoring & latency optimization
// - Parallel endpoint probing with capability detection
// - Multi-model drumcircle conversations (38 models at once!)
// - Reactive architecture (backend changes = instant frontend updates)
//
// 📦 FUTURE: `cargo install ollama-conductor && ollama-conductor serve`
//
// For now: SHIP THE FEATURE! Impress people in 2 days! Refactor later! 🔥

#[derive(Debug, Clone)]
pub struct UnifiedLLMService {
    config: ContinueConfig,
    client: reqwest::Client,
    api_keys: HashMap<String, String>,
    endpoint_cache: Option<std::sync::Arc<tokio::sync::RwLock<ModelEndpointCache>>>,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model_name: String,
    pub prompt: String,
    pub system_prompt: Option<String>,
}

impl UnifiedLLMService {
    pub fn new() -> Self {
        // Default empty config - will be loaded from YAML
        let config = ContinueConfig {
            name: "Default".to_string(),
            version: "1.0.0".to_string(),
            models: vec![],
        };

        Self {
            config,
            client: reqwest::Client::new(),
            api_keys: HashMap::new(),
            endpoint_cache: None,
        }
    }

    pub async fn new_with_cache() -> Result<Self, Box<dyn std::error::Error>> {
        let config = ContinueConfig {
            name: "Default".to_string(),
            version: "1.0.0".to_string(),
            models: vec![],
        };

        let cache = ModelEndpointCache::load_from_file().await?;

        Ok(Self {
            config,
            client: reqwest::Client::new(),
            api_keys: HashMap::new(),
            endpoint_cache: Some(std::sync::Arc::new(tokio::sync::RwLock::new(cache))),
        })
    }

    pub fn load_from_yaml(&mut self, yaml_content: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.config = ContinueConfig::from_yaml(yaml_content)?;
        println!("📋 Loaded {} models from config", self.config.models.len());

        // Group by provider for display
        let providers = self.config.get_models_by_provider();
        for (provider, models) in providers {
            println!("  🔌 {}: {} models", provider, models.len());
        }

        Ok(())
    }

    pub fn update_api_key(&mut self, provider: &str, api_key: &str) {
        self.api_keys
            .insert(provider.to_string(), api_key.to_string());
        println!("🔑 Updated API key for provider: {}", provider);
    }

    pub fn get_models_by_provider(&self) -> HashMap<String, Vec<&ContinueModel>> {
        self.config.get_models_by_provider()
    }

    pub fn get_model_by_name(&self, name: &str) -> Option<&ContinueModel> {
        self.config.models.iter().find(|m| m.name == name)
    }

    pub fn get_provider_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        let providers = self.get_models_by_provider();

        for (provider, models) in providers {
            let has_api_key = self.api_keys.contains_key(&provider);

            stats.insert(provider.clone(), serde_json::json!({
                "model_count": models.len(),
                "has_api_key": has_api_key,
                "status": if provider == "ollama" || has_api_key { "ready" } else { "needs_api_key" },
                "models": models.iter().map(|m| &m.name).collect::<Vec<_>>()
            }));
        }

        stats
    }

    pub async fn discover_available_models(
        &self,
    ) -> Result<HashMap<String, Vec<String>>, Box<dyn std::error::Error>> {
        let mut discovered = HashMap::new();

        // Discover Google/Gemini models
        if let Some(api_key) = self
            .api_keys
            .get("google")
            .or_else(|| self.api_keys.get("gemini"))
        {
            match self.discover_google_models(api_key).await {
                Ok(models) => {
                    println!("🔮 [DISCOVERY] Found {} Google models", models.len());
                    discovered.insert("google".to_string(), models);
                }
                Err(e) => println!("❌ [DISCOVERY] Google models failed: {}", e),
            }
        }

        // Discover OpenAI models
        if let Some(api_key) = self.api_keys.get("openai") {
            match self.discover_openai_models(api_key).await {
                Ok(models) => {
                    println!("🤖 [DISCOVERY] Found {} OpenAI models", models.len());
                    discovered.insert("openai".to_string(), models);
                }
                Err(e) => println!("❌ [DISCOVERY] OpenAI models failed: {}", e),
            }
        }

        // Discover Ollama models from configured bases AND environment/defaults
        let mut ollama_bases: Vec<String> = self
            .config
            .models
            .iter()
            .filter(|m| m.provider == "ollama")
            .filter_map(|m| m.api_base.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // 🔥 ADD AUTOMATIC OLLAMA HOST DISCOVERY!
        // Even if no Ollama models are configured, try common hosts
        if ollama_bases.is_empty() {
            // Try environment variables first
            if let Ok(env_host) = env::var("OLLAMA_HOST") {
                ollama_bases.push(env_host);
            } else if let Ok(env_host) = env::var("ollama_host") {
                ollama_bases.push(env_host);
            } else {
                // Default fallback hosts to try
                ollama_bases.extend(vec![
                    "http://localhost:11434".to_string(),
                    "http://127.0.0.1:11434".to_string(),
                    "http://192.168.4.97:11434".to_string(), // Your specific host
                ]);
            }
            println!(
                "🔍 [DISCOVERY] No configured Ollama hosts found, trying default discovery..."
            );
        }

        for base_url in ollama_bases {
            match self.discover_ollama_models(&base_url).await {
                Ok(models) => {
                    println!(
                        "🦙 [DISCOVERY] Found {} Ollama models at {}",
                        models.len(),
                        base_url
                    );
                    discovered
                        .entry("ollama".to_string())
                        .or_insert_with(Vec::new)
                        .extend(models);
                }
                Err(e) => println!("❌ [DISCOVERY] Ollama at {} failed: {}", base_url, e),
            }
        }

        Ok(discovered)
    }

    async fn discover_google_models(
        &self,
        api_key: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct GoogleModelsResponse {
            models: Vec<GoogleModel>,
        }

        #[derive(Deserialize)]
        struct GoogleModel {
            name: String,
            #[serde(rename = "displayName")]
            display_name: Option<String>,
            description: Option<String>,
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1/models?key={}",
            api_key
        );

        let response = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Google API error: {}", response.status()).into());
        }

        let google_response: GoogleModelsResponse = response.json().await?;

        let models = google_response
            .models
            .into_iter()
            .map(|m| m.name.replace("models/", "")) // Remove "models/" prefix
            .filter(|name| name.contains("gemini")) // Only Gemini models for inference
            .collect();

        Ok(models)
    }

    async fn discover_openai_models(
        &self,
        api_key: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct OpenAIModelsResponse {
            data: Vec<OpenAIModel>,
        }

        #[derive(Deserialize)]
        struct OpenAIModel {
            id: String,
            object: String,
        }

        let response = self
            .client
            .get("https://api.openai.com/v1/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("OpenAI API error: {}", response.status()).into());
        }

        let openai_response: OpenAIModelsResponse = response.json().await?;

        let models = openai_response
            .data
            .into_iter()
            .filter(|m| m.object == "model")
            .map(|m| m.id)
            .filter(|id| id.starts_with("gpt-") || id.starts_with("o1-")) // Only chat models
            .collect();

        Ok(models)
    }

    async fn discover_ollama_models(
        &self,
        base_url: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct OllamaTagsResponse {
            models: Vec<OllamaModel>,
        }

        #[derive(Deserialize)]
        struct OllamaModel {
            name: String,
            modified_at: String,
            size: u64,
        }

        let url = format!("{}/api/tags", base_url.trim_end_matches('/'));

        let response = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Ollama API error: {}", response.status()).into());
        }

        let ollama_response: OllamaTagsResponse = response.json().await?;

        let models = ollama_response.models.into_iter().map(|m| m.name).collect();

        Ok(models)
    }

    pub async fn send_to_model(
        &self,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // 🔥 NUKE CONFIG DEPENDENCY! TRY OLLAMA FIRST FOR ANY MODEL NAME!

        // Step 1: Always try Ollama first (most common case)
        if let Ok(response) = self.try_direct_ollama_call(request).await {
            return Ok(response);
        }

        // Step 2: Only fall back to config if Ollama fails
        if let Some(model) = self.get_model_by_name(&request.model_name) {
            println!(
                "🤖 [{}] Fallback to config: {} ({}): {}",
                model.provider.to_uppercase(),
                model.name,
                model.model,
                &request.prompt[..std::cmp::min(50, request.prompt.len())]
            );

            let start_time = std::time::Instant::now();

            let result = match model.provider.as_str() {
                "anthropic" => {
                    println!("🧠 [ANTHROPIC] Using Claude model: {}", model.model);
                    self.send_to_anthropic(model, request).await
                }
                "google" | "gemini" => {
                    println!("🔮 [GEMINI] Using Google model: {}", model.model);
                    self.send_to_gemini(model, request).await
                }
                "ollama" => {
                    println!(
                        "🦙 [OLLAMA] Using local model: {} at {}",
                        model.model,
                        model.api_base.as_ref().unwrap_or(&"unknown".to_string())
                    );
                    self.send_to_ollama(model, request).await
                }
                "openai" => {
                    println!("🤖 [OPENAI] Using GPT model: {}", model.model);
                    self.send_to_openai(model, request).await
                }
                _ => Err(format!("Provider '{}' not supported yet", model.provider).into()),
            };

            let duration = start_time.elapsed();

            match &result {
                Ok(response) => {
                    println!(
                        "✅ [{}] {} responded in {:.2}s: {}",
                        model.provider.to_uppercase(),
                        model.name,
                        duration.as_secs_f32(),
                        if response.len() > 500 {
                            format!(
                                "{}... [TRUNCATED - {} total chars]",
                                &response[..500],
                                response.len()
                            )
                        } else {
                            response.clone()
                        }
                    );
                }
                Err(e) => {
                    println!(
                        "❌ [{}] {} failed after {:.2}s: {}",
                        model.provider.to_uppercase(),
                        model.name,
                        duration.as_secs_f32(),
                        e
                    );
                }
            }

            result
        } else {
            // Model not in config - try direct Ollama call for discovered models
            println!(
                "🔍 Model '{}' not in config, trying direct Ollama call",
                request.model_name
            );

            // Get Ollama API base from env var ONLY - nuke config dependency!
            let ollama_api_base = env::var("OLLAMA_HOST")
                .or_else(|_| env::var("ollama_host"))
                .unwrap_or_else(|_| "http://localhost:11434".to_string());

            println!(
                "🦙 [OLLAMA-DIRECT] Using {} at {} (config bypassed)",
                request.model_name, ollama_api_base
            );
            // Create a temporary model struct for discovered Ollama models
            let temp_model = ContinueModel {
                name: request.model_name.clone(),
                model: request.model_name.clone(),
                provider_options: None,
                chat_options: None,
                provider: "ollama".to_string(),
                api_base: Some(ollama_api_base.as_str().to_string()),
                api_key: None,
                // context_length: Some(4096),
                capabilities: Some(vec!["chat".to_string()]),
                default_completion_options: None,
                prompt_templates: None,
            };

            let start_time = std::time::Instant::now();
            let result = self.send_to_ollama(&temp_model, request).await;
            let duration = start_time.elapsed();

            match &result {
                Ok(response) => {
                    println!(
                        "✅ [OLLAMA-DISCOVERED] {} responded in {:.2}s: {}",
                        request.model_name,
                        duration.as_secs_f32(),
                        &response[..std::cmp::min(100, response.len())]
                    );
                }
                Err(e) => {
                    println!(
                        "❌ [OLLAMA-DISCOVERED] {} failed after {:.2}s: {}",
                        request.model_name,
                        duration.as_secs_f32(),
                        e
                    );
                }
            }

            result
        }
    }

    // 🔥 NEW: DIRECT OLLAMA CALL WITHOUT CONFIG DEPENDENCY!
    async fn try_direct_ollama_call(
        &self,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Get Ollama API base from env var or default
        let ollama_api_base = env::var("OLLAMA_HOST")
            .or_else(|_| env::var("ollama_host"))
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        println!(
            "🦙 [OLLAMA-DIRECT] Trying {} at {} (bypassing config)",
            request.model_name, ollama_api_base
        );

        // Create temp model for direct call
        let temp_model = ContinueModel {
            name: request.model_name.clone(),
            model: request.model_name.clone(),
            provider_options: None,
            chat_options: None,
            provider: "ollama".to_string(),
            api_base: Some(ollama_api_base),
            api_key: None,
            capabilities: Some(vec!["chat".to_string()]),
            default_completion_options: None,
            prompt_templates: None,
        };

        // Use existing Ollama send logic
        self.send_to_ollama(&temp_model, request).await
    }

    async fn send_to_ollama(
        &self,
        model: &ContinueModel,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let base_url = model
            .api_base
            .as_ref()
            .ok_or("Ollama model missing apiBase")?;

        let system_prompt = request
            .system_prompt
            .clone()
            .or_else(|| model.chat_options.as_ref()?.base_system_message.clone());

        // 🚀 SIMPLE CONFIG-BASED MODEL TESTING
        let mut config = SimpleConfig::load();
        
        // Check if we need to test this model
        if !config.should_test_model(&model.model) {
            if let Some(tested_model) = config.tested_models.get(&model.model) {
                if matches!(tested_model.status, ModelStatus::Working) {
                    println!("✅ Using known working model: {} at {}", model.model, tested_model.endpoint);
                    
                    // Try the known working endpoint directly
                    match self
                        .try_ollama_endpoint(
                            base_url,
                            "/api/generate",
                            &model.model,
                            &request.prompt,
                            system_prompt.as_ref(),
                        )
                        .await
                    {
                        Ok(response) => return Ok(response),
                        Err(e) => {
                            println!("❌ Known working endpoint failed, will retest: {}", e);
                            // Continue to testing below
                        }
                    }
                }
            }
        }

        // 🔬 METHODICAL ONE-TIME ENDPOINT PROFILING
        println!("🔬 [PROFILING] Starting endpoint discovery for {}", model.model);
        
        let endpoints_to_try = vec![
            "/api/generate",        // Standard Ollama
            "/api/chat",           // Ollama chat mode
            "/v1/chat/completions", // OpenAI-compatible 
            "/v1/completions",     // OpenAI-compatible completions
            "/v1/generate",        // Some custom endpoints
            "/generate",           // Bare endpoint
            "/chat",              // Bare chat endpoint
        ];

        for endpoint in endpoints_to_try {
            println!("🔍 [PROFILING] Testing {} with {}", model.model, endpoint);
            
            let start_time = std::time::Instant::now();
            match self
                .try_ollama_endpoint(
                    base_url,
                    endpoint,
                    &model.model,
                    &request.prompt,
                    system_prompt.as_ref(),
                )
                .await
            {
                Ok(response) => {
                    let duration = start_time.elapsed();
                    println!("✅ [PROFILED] {} works with {} ({}ms)", 
                             model.model, endpoint, duration.as_millis());
                    
                    // 📝 Save successful test to simple config
                    config.record_model_test(&model.model, endpoint, true, duration.as_millis() as u32);
                    
                    return Ok(response);
                }
                Err(e) => {
                    let duration = start_time.elapsed(); 
                    println!("❌ [PROFILED] {} failed with {} after {}ms: {}", 
                             model.model, endpoint, duration.as_millis(), e);
                    
                    // 📝 Save failed test to simple config (only on last endpoint)
                    if endpoint == "/chat" { // Last endpoint in the updated list
                        config.record_model_test(&model.model, endpoint, false, duration.as_millis() as u32);
                    }
                    
                    continue;
                }
            }
        }

        Err("All Ollama endpoints failed".into())
    }

    // 🔥 TRY A SPECIFIC OLLAMA ENDPOINT
    async fn try_ollama_endpoint(
        &self,
        base_url: &str,
        endpoint: &str,
        model_name: &str,
        prompt: &str,
        system_prompt: Option<&String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);

        // Different request formats for different endpoints
        let request_body = if endpoint.contains("v1") || endpoint.contains("chat/completions") {
            // OpenAI-compatible format
            serde_json::json!({
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.unwrap_or(&"You are a helpful assistant.".to_string())
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": false
            })
        } else {
            // Ollama native format
            serde_json::json!({
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": false
            })
        };

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(10)) // Shorter timeout for testing
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()).into());
        }

        // 🔥 PROPERLY AWAIT JSON PARSING!
        let json_response: serde_json::Value = response.json().await?;

        // Try to parse different response formats
        // Extract response from various possible formats
        if let Some(response_str) = json_response.get("response").and_then(|v| v.as_str()) {
            return Ok(response_str.to_string());
        } else if let Some(choices) = json_response.get("choices").and_then(|v| v.as_array()) {
            if let Some(message) = choices
                .first()
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
            {
                return Ok(message.to_string());
            }
        }

        // Fallback: if JSON parsing worked but no expected fields, return raw JSON as string
        Ok(json_response.to_string())
    }

    async fn send_to_gemini(
        &self,
        model: &ContinueModel,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let api_key = self
            .api_keys
            .get("google")
            .or_else(|| self.api_keys.get("gemini"))
            .ok_or("No Google/Gemini API key configured")?;

        #[derive(Serialize)]
        struct GeminiRequest {
            contents: Vec<GeminiContent>,
            #[serde(rename = "systemInstruction")]
            system_instruction: Option<GeminiContent>,
        }

        #[derive(Serialize)]
        struct GeminiContent {
            parts: Vec<GeminiPart>,
        }

        #[derive(Serialize)]
        struct GeminiPart {
            text: String,
        }

        #[derive(Deserialize)]
        struct GeminiResponse {
            candidates: Option<Vec<GeminiCandidate>>,
        }

        #[derive(Deserialize)]
        struct GeminiCandidate {
            content: Option<GeminiResponseContent>,
        }

        #[derive(Deserialize)]
        struct GeminiResponseContent {
            parts: Vec<GeminiResponsePart>,
        }

        #[derive(Deserialize)]
        struct GeminiResponsePart {
            text: String,
        }

        let system_instruction = request
            .system_prompt
            .clone()
            .or_else(|| model.chat_options.as_ref()?.base_system_message.clone())
            .map(|text| GeminiContent {
                parts: vec![GeminiPart { text }],
            });

        let gemini_request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart {
                    text: request.prompt.clone(),
                }],
            }],
            system_instruction,
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model.model, api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&gemini_request)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Gemini API error: {}", error_text).into());
        }

        let gemini_response: GeminiResponse = response.json().await?;

        let response_text = gemini_response
            .candidates
            .and_then(|candidates| candidates.into_iter().next())
            .and_then(|candidate| candidate.content)
            .and_then(|content| content.parts.into_iter().next())
            .map(|part| part.text)
            .unwrap_or_else(|| "No response generated".to_string());

        Ok(response_text)
    }

    async fn send_to_anthropic(
        &self,
        model: &ContinueModel,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let api_key = self
            .api_keys
            .get("anthropic")
            .ok_or("No Anthropic API key configured")?;

        println!(
            "🧠 [ANTHROPIC DEBUG] API key present: {}...",
            &api_key[..std::cmp::min(8, api_key.len())]
        );
        println!("🧠 [ANTHROPIC DEBUG] Model: {}", model.model);
        println!(
            "🧠 [ANTHROPIC DEBUG] Prompt: {}",
            &request.prompt[..std::cmp::min(100, request.prompt.len())]
        );

        #[derive(Serialize)]
        struct AnthropicRequest {
            model: String,
            max_tokens: u32,
            messages: Vec<AnthropicMessage>,
            #[serde(skip_serializing_if = "Option::is_none")]
            system: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
        }

        #[derive(Serialize)]
        struct AnthropicMessage {
            role: String,
            content: Vec<AnthropicMessageContent>,
        }

        #[derive(Serialize)]
        struct AnthropicMessageContent {
            #[serde(rename = "type")]
            content_type: String,
            text: String,
        }

        #[derive(Deserialize)]
        struct AnthropicResponse {
            content: Vec<AnthropicResponseContent>,
            #[serde(default)]
            usage: Option<serde_json::Value>,
        }

        #[derive(Deserialize)]
        struct AnthropicResponseContent {
            text: String,
            #[serde(rename = "type")]
            content_type: Option<String>,
        }

        let system_prompt = request
            .system_prompt
            .clone()
            .or_else(|| model.chat_options.as_ref()?.base_system_message.clone());

        let anthropic_request = AnthropicRequest {
            model: model.model.clone(),
            max_tokens: model
                .default_completion_options
                .as_ref()
                .and_then(|opts| opts.max_tokens)
                .unwrap_or(1000),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicMessageContent {
                    content_type: "text".to_string(),
                    text: request.prompt.clone(),
                }],
            }],
            system: system_prompt,
            temperature: model
                .default_completion_options
                .as_ref()
                .and_then(|opts| opts.temperature),
        };

        let default_base_url = "https://api.anthropic.com/v1".to_string();
        let base_url = model.api_base.as_ref().unwrap_or(&default_base_url);
        let url = format!("{}/messages", base_url.trim_end_matches('/'));

        // Get Anthropic API version from config or use stable version
        let anthropic_version = model
            .provider_options
            .as_ref()
            .and_then(|opts| opts.anthropic_version.as_ref())
            .map(|v| v.as_str())
            .unwrap_or("2023-06-01"); // Anthropic keeps this stable across all models

        println!(
            "🧠 [ANTHROPIC DEBUG] Using {} with API version: {}",
            model.model, anthropic_version
        );
        println!("🧠 [ANTHROPIC DEBUG] URL: {}", url);
        println!(
            "🧠 [ANTHROPIC DEBUG] Request: {}",
            serde_json::to_string_pretty(&anthropic_request)?
        );

        let mut request_builder = self
            .client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", anthropic_version)
            .header("content-type", "application/json")
            .json(&anthropic_request)
            .timeout(std::time::Duration::from_secs(60));

        // Add any custom headers from config
        if let Some(provider_options) = &model.provider_options {
            if let Some(headers) = &provider_options.headers {
                for (key, value) in headers {
                    request_builder = request_builder.header(key, value);
                }
            }
        }

        let response = request_builder.send().await?;

        let status = response.status();
        println!("🧠 [ANTHROPIC DEBUG] Response status: {}", status);

        if !status.is_success() {
            let error_text = response.text().await?;
            println!("❌ [ANTHROPIC DEBUG] Error response: {}", error_text);
            return Err(format!("Anthropic API error ({}): {}", status, error_text).into());
        }

        let response_text = response.text().await?;
        println!(
            "🧠 [ANTHROPIC DEBUG] Raw response: {}",
            &response_text[..std::cmp::min(200, response_text.len())]
        );

        let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)?;

        let response_text = anthropic_response
            .content
            .into_iter()
            .next()
            .map(|content| content.text)
            .unwrap_or_else(|| "No response generated".to_string());

        Ok(response_text)
    }

    async fn send_to_openai(
        &self,
        model: &ContinueModel,
        request: &ChatRequest,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let api_key = self
            .api_keys
            .get("openai")
            .ok_or("No OpenAI API key configured")?;

        #[derive(Serialize)]
        struct OpenAIRequest {
            model: String,
            messages: Vec<OpenAIMessage>,
            max_tokens: Option<u32>,
            temperature: Option<f32>,
        }

        #[derive(Serialize)]
        struct OpenAIMessage {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<OpenAIChoice>,
        }

        #[derive(Deserialize)]
        struct OpenAIChoice {
            message: OpenAIResponseMessage,
        }

        #[derive(Deserialize)]
        struct OpenAIResponseMessage {
            content: String,
        }

        let mut messages = Vec::new();

        // Add system message if present
        if let Some(system_prompt) = request
            .system_prompt
            .clone()
            .or_else(|| model.chat_options.as_ref()?.base_system_message.clone())
        {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system_prompt,
            });
        }

        // Add user message
        messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        let openai_request = OpenAIRequest {
            model: model.model.clone(),
            messages,
            max_tokens: model
                .default_completion_options
                .as_ref()
                .and_then(|opts| opts.max_tokens),
            temperature: model
                .default_completion_options
                .as_ref()
                .and_then(|opts| opts.temperature),
        };

        let default_base_url = "https://api.openai.com/v1".to_string();
        let base_url = model.api_base.as_ref().unwrap_or(&default_base_url);
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        println!("🤖 [OPENAI] Using {} at {}", model.model, base_url);

        let mut request_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .timeout(std::time::Duration::from_secs(60));

        // Add any custom headers from config
        if let Some(provider_options) = &model.provider_options {
            if let Some(headers) = &provider_options.headers {
                for (key, value) in headers {
                    request_builder = request_builder.header(key, value);
                }
            }
        }

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("OpenAI API error: {}", error_text).into());
        }

        let openai_response: OpenAIResponse = response.json().await?;

        let response_text = openai_response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .unwrap_or_else(|| "No response generated".to_string());

        Ok(response_text)
    }
}
