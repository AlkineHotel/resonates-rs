// main.rs. -0.9127
// REMEMBER: DO NOT HARDCODE. IF ANYTHING MAKES SENSE TO BE AN OPTION. MAKE IT AN OPTION AND PUT IT IN THE CONFIG TAB
// main.response_msg
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod types;
mod network_utils; // 🌐 CONSOLIDATED NETWORK UTILITIES (websocket + console_broadcast)
// mod websocket; // 🌐 CONSOLIDATED INTO network_utils
mod discovery;
mod ollama;
mod qwen_drumcircle;
mod input_sanitizer; // 🛡️ INPUT SANITIZATION MODULE
mod simple_config; // 📝 SIMPLE UNIVERSAL CONFIG
mod config_types; // ⚙️ CONSOLIDATED CONFIG TYPES (continue_config)
// mod continue_config; // ⚙️ CONSOLIDATED INTO config_types
mod unified_llm;
mod conversation;
mod conversation_service;
mod config_persistence;
mod phi_detection;
mod live_react_engine; // Add this line to your imports
mod true_quantum_engine; // 🌊 TRUE QUANTUM CONSCIOUSNESS ENGINE
mod quantum_handlers; // 🌊 QUANTUM CONSCIOUSNESS ENDPOINTS
mod session_handlers; // 💾 SESSION PERSISTENCE ENDPOINTS
mod threaded_conversation; // 🧵 THREADED MODEL CONVERSATIONS
mod web_search; // 🌐 WEB CONTEXT INJECTION
mod security_manager; // 🛡️ DUAL-MODE SECURITY & HONEYPOTS
mod security_middleware; // 🛡️ PRODUCTION SECURITY MIDDLEWARE
mod auth_system; // 🔐 SECURE AUTHENTICATION SYSTEM
// mod console_broadcast; // 📝 CONSOLIDATED INTO network_utils
mod model_testing; // 🧪 COMPREHENSIVE MODEL TESTING ENGINE
mod hybrid_database; // 🗄️ SQLITE + YAML HYBRID DATABASE
mod zombie_endpoints; // 💀 EXTRACTED DEAD CODE (PHI/Python endpoints)
mod belongsInAConfigFile; // 📁 HARDCODED DATA THAT SHOULD BE IN CONFIG FILES
mod server_setup; //  ROUTER CONFIGURATION & SERVER STARTUP
mod state_manager; // 💾 STATE SNAPSHOT & PERSISTENCE
mod config_handlers; // 📋 CONFIG ENDPOINT HANDLERS
mod llm_handlers; // 🤖 LLM/CHAT/AI ENDPOINT HANDLERS
mod model_testing_handlers; // 🧪 MODEL TESTING ENDPOINT HANDLERS

use tower_sessions::{MemoryStore, SessionManagerLayer, Session};

use axum::{
    routing::{get, post, delete},
    Router,
    Json,
    extract::{Path, State, ConnectInfo},
    middleware,
    http::{HeaderValue, Method, StatusCode, HeaderMap, HeaderName},
};
use tower::ServiceBuilder;
use config_types::ContinueModel;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use std::sync::Mutex; // 🥁 For drumcircle parallel chaos
use tokio::sync::RwLock; // 🧵 For threading sequential reads
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use uuid::Uuid;
use chrono; // 🕐 For timestamps in testing
// use futures_util::future::FutureExt; // Unused

// �️ CONSOLE LOGGING HELPER FOR MINICONSOLE
fn send_console_log(tx: &broadcast::Sender<types::Message>, level: &str, message: &str, source: &str) {
    let console_msg = types::Message::ConsoleLog {
        id: Uuid::new_v4(),
        level: level.to_string(),
        message: message.to_string(),
        source: source.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_millis() as u64,
    };
    
    // Send to WebSocket (ignore errors if no one is listening)
    let _ = tx.send(console_msg);
}

// 🔥 ENHANCED PRINTLN MACRO THAT SENDS TO MINICONSOLE
macro_rules! console_log {
    ($tx:expr, $level:expr, $source:expr, $($arg:tt)*) => {
        {
            let msg = format!($($arg)*);
            println!("{}", msg); // Still print to stdout
            send_console_log($tx, $level, &msg, $source);
        }
    };
}

// �🔥 CONDITIONAL CONCURRENCY TYPES
#[derive(Debug, Clone)]
struct ConversationState {
    prompt: String,
    responses: Vec<ModelResponse>,
    round: u32,
    conversation_type: ConversationType,
}

#[derive(Debug, Clone)]
struct ModelResponse {
    model_name: String,
    content: String,
    timestamp: u64,
    round: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum ConversationType {
    Drumcircle, // Arc<Mutex<T>> - All models write/read simultaneously  
    Threading,  // Arc<RwLock<T>> - Sequential with many readers, one writer
}

// 🔥 THE SEE-SAW: Choose concurrency primitive based on conversation type
#[derive(Clone)]
enum ConversationContainer {
    DrumcircleChaos(Arc<Mutex<ConversationState>>), // Parallel mayhem
    ThreadingOrder(Arc<RwLock<ConversationState>>), // Sequential civilization 
}

impl ConversationContainer {
    fn new_drumcircle(prompt: String) -> Self {
        Self::DrumcircleChaos(Arc::new(Mutex::new(ConversationState {
            prompt,
            responses: Vec::new(),
            round: 1,
            conversation_type: ConversationType::Drumcircle,
        })))
    }
    
    fn new_threading(prompt: String) -> Self {
        Self::ThreadingOrder(Arc::new(RwLock::new(ConversationState {
            prompt,
            responses: Vec::new(),
            round: 1,
            conversation_type: ConversationType::Threading,
        })))
    }
}

// 🖥️ CONSOLE BROADCAST HELPER - PRINTS TO CONSOLE AND MINICONSOLE!
fn broadcast_console_log(
    tx: &broadcast::Sender<types::Message>,
    level: &str,
    message: &str,
    source: &str,
) {
    // Print to PowerShell console (keeps existing behavior)
    match level {
        "error" => println!("❌ {}", message),
        "warn" => println!("⚠️ {}", message),
        "info" => println!("ℹ️ {}", message),
        "debug" => println!("🔍 {}", message),
        "success" => println!("✅ {}", message),
        _ => println!("{}", message),
    }
    
    // Broadcast to WebSocket (feeds miniconsole)
    let console_msg = types::Message::ConsoleLog {
        id: Uuid::new_v4(),
        level: level.to_string(),
        message: message.to_string(),
        source: source.to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    };
    
    let _ = tx.send(console_msg);
}

// 🔥 MACRO FOR EASY CONSOLE BROADCASTING
macro_rules! broadcast_log {
    ($tx:expr, $level:expr, $source:expr, $($arg:tt)*) => {
        broadcast_console_log($tx, $level, &format!($($arg)*), $source)
    };
}

use crate::types::Config;
use crate::network_utils::{websocket_handler, AppState};
use crate::model_testing::ModelTestingEngine;

#[derive(Clone)]
struct ApiKeyUpdate {
    gemini: Option<String>,
    ollama_host: Option<String>,
}

// 🪟 TAURI WINDOW MANAGEMENT - FLOATING MODEL TESTING WINDOW
#[tauri::command]
async fn open_model_testing_window(app: tauri::AppHandle) -> Result<(), String> {
    println!("🪟 Opening model testing window...");
    
    // Tauri v2 window creation
    match tauri::WebviewWindowBuilder::new(&app, "model-testing", tauri::WebviewUrl::App("/model-testing".into()))
        .title("🦙 Ollama Model Controller Testing")
        .inner_size(700.0, 500.0)
        .min_inner_size(500.0, 350.0)
        .resizable(true)
        .center()
        .focused(false) // 🎯 DON'T STEAL FOCUS!
        .always_on_top(false) // User can configure this later
        .decorations(true) // Keep window controls
        .build() {
            Ok(_) => {
                println!("✅ Model testing window created successfully");
                Ok(())
            }
            Err(e) => {
                println!("❌ Failed to create testing window: {}", e);
                Err(e.to_string())
            }
        }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![open_model_testing_window])
        .setup(|_app| {
            // Start the axum server in a background task with proper async runtime
            std::thread::spawn(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                if let Err(e) = rt.block_on(start_server()) {
                    eprintln!("❌ Server failed to start: {}", e);
                } else {
                    eprintln!("✅ Server started successfully");
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn start_server() -> Result<(), Box<dyn std::error::Error>> {
    println!("🥸 Initializing server...");
    tracing_subscriber::fmt::init();
    
    let (tx, _rx) = broadcast::channel(100);
    let device_id = Uuid::new_v4();
    
    // Initialize config manager and load persisted config
    broadcast_log!(&tx, "info", "rust", "📁 Initializing config persistence...");
    let config_manager = config_persistence::ConfigManager::new()?;
    let app_config = config_manager.load_app_config();
    
    // Convert persisted config to runtime config
    broadcast_log!(&tx, "info", "rust", "🗄️ BYPASSING Database initialization (quick fix)...");
    // let db_instance = hybrid_database::HybridModelDatabase::new().await?;
    // let db = Arc::new(Mutex::new(db_instance));
    
    broadcast_log!(&tx, "info", "rust", "💾 BYPASSING State Manager initialization...");
    // Create dummy state manager - we'll fix this properly later
    // let state_db = hybrid_database::HybridModelDatabase::new().await?;
    // let state_manager = state_manager::StateManager::new(Arc::new(tokio::sync::Mutex::new(state_db)));
    // state_manager.init_schema().await?; // Ensure the table exists
    broadcast_log!(&tx, "success", "rust", "✅ Database bypass mode active - app will run without DB.");
    let mut runtime_threading_config = app_config.threading_config.clone();
    // 🔥 ENSURE WEB_CONTEXT IS LOADED FROM PERSISTENCE INTO THE CORRECT NESTED LOCATION!
    runtime_threading_config.context_injection.web_context = app_config.web_context.clone();
    
    let config = Arc::new(tokio::sync::RwLock::new(types::Config {
        device_name: app_config.device_name.clone(),
        threading_config: runtime_threading_config, // 🔥 USE THE CORRECTED THREADING CONFIG!
        port: app_config.port,
        debug_verbosity: app_config.debug_verbosity.clone(), // 🔥 LOAD FROM PERSISTENCE!
        llm_content_handling: app_config.llm_content_handling.clone(), // 🔥 LOAD LLM HANDLING!
        live_mode: app_config.live_mode,
        accept_html: app_config.accept_html,
        endpoint_management: app_config.endpoint_management.clone(), // 🔥 LOAD CONFIGURABLE ENDPOINTS!
    }));
    
    println!("📋 Device ID: {}", device_id);
    println!("🔧 Loaded config - Device: {}, Port: {}", app_config.device_name, app_config.port);
    
    // Load YAML config if available
    let mut unified_llm_service = unified_llm::UnifiedLLMService::new();
    if let Some(yaml_content) = config_manager.load_yaml_config() {
        broadcast_log!(&tx, "info", "rust", "📄 Loaded YAML config ({} chars)", yaml_content.len());
        // Load config into unified LLM service
        if let Err(e) = unified_llm_service.load_from_yaml(&yaml_content) {
            broadcast_log!(&tx, "error", "rust", "❌ Failed to load LLM config: {}", e);
        } else {
            broadcast_log!(&tx, "success", "rust", "✅ LLM models loaded from config");
        }
    } else {
        broadcast_log!(&tx, "warn", "rust", "⚠️ No YAML config found - models will be empty until config is uploaded");
    }
    
    broadcast_log!(&tx, "info", "rust", "🔍 Starting discovery service...");
    let discovery = {
        let config = config.read().await;
        match discovery::DiscoveryService::new(
            device_id,
            config.device_name.clone(),
            config.port,
        ) {
            Ok(d) => d,
            Err(e) => {
                broadcast_log!(&tx, "error", "rust", "⚠️ Discovery service failed: {}", e);
                return Err(e);
            }
        }
    };
    
    if let Err(e) = discovery.start(config.read().await.port).await {
        broadcast_log!(&tx, "warn", "rust", "⚠️ Discovery start failed: {}", e);
        // Continue anyway - discovery is optional
    }
    
    // 🧵 Get threading config for AppState construction
    let threading_config = config.read().await.threading_config.clone();
    
    // 🏗️ CREATE APPSTATE USING EXTRACTED FUNCTION (BYPASSING STATE MANAGER FOR NOW)
    let app_state = server_setup::create_app_state(
        tx.clone(),
        device_id,
        config.clone(),
        discovery,
        unified_llm_service,
        config_manager,
        threading_config,
        // state_manager, // COMMENTED OUT FOR BYPASS
    );
    
    //  START SERVER USING EXTRACTED FUNCTION
    server_setup::start_server(app_state, config.clone(), &tx).await.map_err(|e| e.into())
}
#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct ComponentComputeRequest {
    component_name: String,
    props: serde_json::Value,
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct ComponentComputeResponse {
    computed_props: serde_json::Value,
    status: String,
    warnings: Option<Vec<String>>,
}

struct LiveReactCompute {}

impl LiveReactCompute {
    fn new() -> Self {
        LiveReactCompute {}
    }

    async fn compute_component_props(&self, req: ComponentComputeRequest) -> ComponentComputeResponse {
        // Simple mock implementation: echo back props with a computed flag and component name
        let mut merged = req.props.clone();
        if let Some(map) = merged.as_object_mut() {
            map.insert("computed".to_string(), serde_json::json!(true));
            map.insert("component".to_string(), serde_json::json!(req.component_name));
        }

        ComponentComputeResponse {
            computed_props: merged,
            status: "ok".to_string(),
            warnings: None,
        }
    }
}

async fn live_compute_props(
    State(_state): State<AppState>,
    Json(request): Json<ComponentComputeRequest>,
) -> Json<ComponentComputeResponse> {
    let compute_engine = LiveReactCompute::new();
    let response = compute_engine.compute_component_props(request).await;
    Json(response)
}

#[derive(Deserialize)]
struct DrumcircleRequest {
    prompt: String,
    include_gemini: Option<bool>,
    debate_mode: Option<bool>, // 🔥 NEW: true = sequential debate, false = parallel prompting
}

#[derive(Deserialize)]
struct ConfigurableDrumcircleRequest {
    prompt: String,
    role_mapping: std::collections::HashMap<String, String>, // role -> model_name mapping
    task_type: Option<String>, // "code", "analysis", "research", "debate"
    max_rounds: Option<u32>,
    include_gemini: Option<bool>,
}

#[derive(Serialize)]
struct DrumcircleModel {
    name: String,
    provider: String,
    available: bool,
    description: String,
}

#[derive(Deserialize)]
struct ApiKeysRequest {
    gemini: Option<String>,
    openai: Option<String>,
    anthropic: Option<String>,
}

#[derive(Deserialize)]
struct ContinueConfigRequest {
    yaml_content: String,
}

#[derive(Deserialize)]
struct ChatIndividualRequest {
    model_name: String,
    prompt: String,
    system_prompt: Option<String>,
    use_context: Option<bool>,
    is_html: Option<bool>, // 🔥 ADD HTML SUPPORT!
    web_context: Option<crate::types::WebContextConfig>, // 🌐 WEB CONTEXT INJECTION!
}

#[derive(Deserialize)]
struct ContextToggleRequest {
    enabled: bool,
}

#[derive(Deserialize)]
struct UpdateTestSettingsRequest {
    retest_passed_after_hours: Option<u32>,
    retest_failed_after_hours: Option<u32>,
    timeout_seconds: Option<u32>,
    max_concurrent_tests: Option<u32>,
    default_test_prompt: Option<String>,
    quality_threshold: Option<f32>,
}



/// 🦙 Discover available Ollama models by querying the Ollama API
async fn discover_ollama_models(tx: &broadcast::Sender<types::Message>) -> Result<Vec<crate::config_types::ContinueModel>, Box<dyn std::error::Error>> {
    use std::time::Duration;
    
    // Try common Ollama endpoints
    let ollama_hosts = vec![
        "http://localhost:11434",
        "http://127.0.0.1:11434", 
        "http://192.168.4.97:11434", // Your specific Ollama host
    ];
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    
    for host in ollama_hosts {
        println!("🔍 Checking Ollama at: {}", host);
        
        match client.get(&format!("{}/api/tags", host)).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let tags_response: serde_json::Value = response.json().await?;
                    
                    // 🔍 DEBUG: Print the actual JSON structure
                    println!("🔍 Raw Ollama API response: {}", serde_json::to_string_pretty(&tags_response)?);
                    
                    if let Some(models_array) = tags_response["models"].as_array() {
                        let mut ollama_models = Vec::new();
                        
                        broadcast_log!(tx, "info", "rust", "📊 Found {} models in Ollama response", models_array.len());
                        
                        for model in models_array {
                            if let Some(name) = model["name"].as_str() {
                                // Extract useful model info
                                let size = model["size"].as_u64().unwrap_or(0);
                                let family = model["details"]["family"].as_str().unwrap_or("unknown");
                                let param_size = model["details"]["parameter_size"].as_str().unwrap_or("unknown");
                                let quantization = model["details"]["quantization_level"].as_str().unwrap_or("unknown");
                                
                                broadcast_log!(tx, "info", "rust", "🦙 Found model: {} ({}, {}, {})", name, family, param_size, quantization);
                                
                                ollama_models.push(crate::config_types::ContinueModel {
                                    name: name.to_string(),
                                    model: name.to_string(), 
                                    provider: "ollama".to_string(),
                                    api_base: Some(host.to_string()),
                                    api_key: None,
                                    chat_options: Some(crate::config_types::ChatOptions {
                                        base_system_message: Some("You are a helpful assistant.".to_string()),
                                    }),
                                    default_completion_options: Some(crate::config_types::CompletionOptions {
                                        context_length: Some(4096),
                                        max_tokens: Some(2048),
                                        temperature: Some(0.7),
                                        top_p: Some(0.9),
                                        top_k: Some(40),
                                        stop: Some(vec![]),
                                    }),
                                    capabilities: Some(vec!["chat".to_string(), "autocomplete".to_string()]),
                                    prompt_templates: None,
                                    provider_options: Some(crate::config_types::ProviderOptions {
                                        anthropic_version: None,
                                        headers: None,
                                        custom_config: Some({
                                            let mut config = std::collections::HashMap::new();
                                            config.insert("model_family".to_string(), serde_json::Value::String(family.to_string()));
                                            config.insert("parameter_size".to_string(), serde_json::Value::String(param_size.to_string()));
                                            config.insert("quantization".to_string(), serde_json::Value::String(quantization.to_string()));
                                            config.insert("size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(size)));
                                            config
                                        }),
                                    }),
                                });
                            }
                        }
                        
                        broadcast_log!(tx, "success", "rust", "✅ Successfully discovered {} Ollama models from {}", ollama_models.len(), host);
                        return Ok(ollama_models);
                    } else {
                        println!("⚠️ No 'models' array found in Ollama response from {}", host);
                        println!("🔍 Available keys in response: {:?}", tags_response.as_object().map(|o| o.keys().collect::<Vec<_>>()));
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to connect to Ollama at {}: {}", host, e);
                continue;
            }
        }
    }
    
    Err("No Ollama instance found on any common ports".into())
}

async fn get_conversation_history(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Json<serde_json::Value> {
    let history = state.conversation_manager
        .get_conversation_history(&model_name)
        .await;
    
    Json(serde_json::json!({
        "model_name": model_name,
        "messages": history,
        "message_count": history.len()
    }))
}

async fn set_conversation_context(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
    Json(request): Json<ContextToggleRequest>,
) -> Json<serde_json::Value> {
    state.conversation_manager
        .set_context_enabled(&model_name, request.enabled)
        .await;
    
    Json(serde_json::json!({
        "model_name": model_name,
        "context_enabled": request.enabled,
        "status": "updated"
    }))
}

async fn clear_conversation(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Json<serde_json::Value> {
    state.conversation_manager
        .clear_conversation(&model_name)
        .await;
    
    Json(serde_json::json!({
        "model_name": model_name,
        "status": "cleared"
    }))
}

// Config persistence endpoints

#[derive(serde::Deserialize)]
struct SaveConfigRequest {
    config: crate::types::Config,  // The old simple config
    api_keys: ApiKeyConfig,
    ui_preferences: UiPreferences, 
    llm_config_yaml: String,
}

#[derive(serde::Deserialize)]
struct ApiKeyConfig {
    gemini: String,
    openai: String,
    anthropic: String,
}

#[derive(serde::Deserialize)]
struct UiPreferences {
    selected_model: String,
    context_enabled: bool,
    max_context_messages: usize,
    console_max_entries: usize,
    active_tab: String,
    show_history: bool,
    // Drumcircle settings (optional for backwards compatibility)
    selected_drumcircle_models: Option<Vec<String>>,
    drumcircle_task_type: Option<String>,
    drumcircle_rounds: Option<u32>,
    include_gemini: Option<bool>,
}

#[derive(serde::Deserialize)]
struct SaveYamlRequest {
    yaml_content: String,
}




// 🌐 WEB SCRAPING ENDPOINTS - TAURI CHROMIUM POWER!

#[derive(serde::Deserialize)]
struct WebScrapeRequest {
    url: String,
    extract_text_only: Option<bool>,
    disable_truncation: Option<bool>,
}

#[derive(serde::Deserialize)]
struct WebSearchRequest {
    query: String,
    max_results: Option<u32>,
    search_engine: Option<String>, // "duckduckgo", "google", "bing"
}

async fn scrape_web_content(
    State(state): State<AppState>,
    Json(request): Json<WebScrapeRequest>,
) -> Json<serde_json::Value> {
    println!("🌐 Direct web scraping: {}", request.url);
    
    // Use the web search service for direct URL scraping
    let config = state.config.read().await;
    let mut web_service = web_search::WebSearchService::new(config.threading_config.context_injection.web_context.clone());
    
    // Apply truncation settings if specified
    if let Some(disable_truncation) = request.disable_truncation {
        if disable_truncation {
            // Temporarily increase max_content_length for this request
            web_service.set_max_content_length(100000); // 100KB instead of default ~2KB
            println!("🔓 Truncation disabled - allowing up to 100KB content");
        }
    }
    
    match web_service.scrape_url_directly(&request.url).await {
        Ok(content) => {
            let processed_content = if request.extract_text_only.unwrap_or(true) {
                // Extract just text, remove HTML tags
                content
            } else {
                // Return full content including HTML structure
                content
            };
            
            Json(serde_json::json!({
                "status": "success",
                "url": request.url,
                "content": processed_content,
                "content_length": processed_content.len(),
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            }))
        }
        Err(e) => {
            println!("❌ Failed to scrape {}: {}", request.url, e);
            Json(serde_json::json!({
                "status": "error",
                "url": request.url,
                "message": format!("Failed to scrape URL: {}", e)
            }))
        }
    }
}

async fn search_and_scrape(
    State(state): State<AppState>,
    Json(request): Json<WebSearchRequest>,
) -> Json<serde_json::Value> {
    println!("🔍 Web search and scrape: {}", request.query);
    
    // Use the web search service for search + scraping
    let config = state.config.read().await;
    let web_service = web_search::WebSearchService::new(config.threading_config.context_injection.web_context.clone());
    
    match web_service.search_and_scrape_content(&request.query).await {
        Ok(result) => {
            Json(serde_json::json!({
                "status": "success",
                "query": request.query,
                "results_count": result.total_results,
                "provider": result.provider_used,
                "processing_time_ms": result.processing_time_ms,
                "formatted_context": result.formatted_context,
                "raw_results": result.results
            }))
        }
        Err(e) => {
            println!("❌ Failed to search for '{}': {}", request.query, e);
            Json(serde_json::json!({
                "status": "error",
                "query": request.query,
                "message": format!("Search failed: {}", e)
            }))
        }
    }
}

// 💀 PHI DETECTION ENDPOINTS - MOVED TO zombie_endpoints.rs

// 💀 MEDICAL TRANSFORM - MOVED TO zombie_endpoints.rs

// 💀 PYTHON INTEGRATION - MOVED TO zombie_endpoints.rs

// 💀 get_python_status - MOVED TO zombie_endpoints.rs

// 📁 AI MODEL PROCESSING - MOVED TO belongsInAConfigFile.rs


// 💾 SESSION PERSISTENCE ENDPOINTS - MOVED TO session_handlers.rs

// 🌊 QUANTUM CONSCIOUSNESS ENGINE HANDLERS - MOVED TO quantum_handlers.rs

// 🧵 THREADED CONVERSATION ENDPOINTS

#[derive(serde::Deserialize)]
struct StartThreadRequest {
    initial_prompt: String,
    participating_models: Vec<String>,
    max_rounds: Option<u32>,
}

#[derive(serde::Deserialize)]
struct ThreadResponseRequest {
    model_name: String,
    model_provider: String,
    content: String,
    is_html: Option<bool>,
    responding_to_model: Option<String>,
}

async fn start_threaded_conversation(
    State(state): State<AppState>,
    Json(request): Json<StartThreadRequest>,
) -> Json<serde_json::Value> {
    let config = state.config.read().await;
    let debug = &config.debug_verbosity;
    
    if debug.should_log(crate::types::DebugLevel::Normal) {
        println!("🧵 Starting threaded conversation:");
        println!("   📝 Prompt: {}", debug.format_content(&request.initial_prompt, "message_content"));
        println!("   🤖 Models: {:?}", request.participating_models);
        println!("   🔄 Max rounds: {:?}", request.max_rounds);
    }
    drop(config);

    let thread_id = state.threaded_conversation_manager
        .start_thread(
            request.initial_prompt.clone(),
            request.participating_models.clone(),
            request.max_rounds,
        )
        .await;

    // Send thread start message via WebSocket
    let thread_start_msg = types::Message::ThreadStart {
        id: Uuid::new_v4(),
        thread_id,
        initial_prompt: request.initial_prompt.clone(),
        participating_models: request.participating_models.clone(),
        max_rounds: request.max_rounds.unwrap_or(3),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    let _ = state.tx.send(thread_start_msg);

    // 🔥 NOW ACTUALLY START THE CONVERSATION!
    // Spawn a task to run the threaded conversation
    let state_clone = state.clone();
    let initial_prompt = request.initial_prompt.clone();
    let models = request.participating_models.clone();
    let max_rounds = request.max_rounds.unwrap_or(3);
    

    tokio::spawn(async move {
        run_threaded_conversation(state_clone, thread_id, initial_prompt, models, max_rounds).await;
    });

    Json(serde_json::json!({
        "status": "thread_started",
        "thread_id": thread_id,
        "message": "Threaded conversation started successfully"
    }))
}

// 🧵 UNIFIED THREADED CONVERSATION RUNNER WITH SEE-SAW CONCURRENCY
async fn run_threaded_conversation(
    state: AppState,
    thread_id: Uuid,
    initial_prompt: String,
    participating_models: Vec<String>,
    max_rounds: u32,
) {
    println!("🧵 [THREAD {}] Starting conversation with {} models for {} rounds using SEE-SAW CONCURRENCY", 
             thread_id.to_string()[..8].to_string(), 
             participating_models.len(), 
             max_rounds);
    // 🔥 USE SEE-SAW PATTERN FOR THREADING
    let conversation = ConversationContainer::new_threading(initial_prompt.clone());
    
    for round in 1..=max_rounds {
        println!("🧵 [THREAD {}] Round {}/{} - ENGAGING SEE-SAW!", thread_id.to_string()[..8].to_string(), round, max_rounds);
        
        // Call all models for this round using see-saw concurrency
        match call_models_with_seesaw_concurrency(&state, conversation.clone(), participating_models.clone()).await {
            Ok(responses) => {
                println!("✅ [THREAD {}] Round {} complete: {} responses", 
                         thread_id.to_string()[..8].to_string(), round, responses.len());
                
                // Send WebSocket messages for each response
                for response in responses {
                    let model_msg = types::Message::ModelMessage {
                        id: Uuid::new_v4(),
                        model_name: response.model_name.clone(),
                        model_provider: "unified".to_string(),
                        content: response.content,
                        is_html: false,
                        thread_id,
                        parent_message_id: None,
                        responding_to_model: None, // Threading doesn't track individual responses
                        conversation_round: round,
                        timestamp: response.timestamp,
                    };
                    let _ = state.tx.send(model_msg);
                }
            }
            Err(e) => {
                println!("❌ [THREAD {}] Round {} failed: {}", thread_id.to_string()[..8].to_string(), round, e);
            }
        }
        
        // Only delay between rounds if user wants it (and make it configurable!)
        if round < max_rounds {
            // TODO: Make this configurable via ThreadingConfig
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Minimal delay
        }
    }
    
    // Mark thread as complete
    let _ = state.threaded_conversation_manager.complete_thread(thread_id).await;
    
    // Send completion message
    let complete_msg = types::Message::ThreadComplete {
        id: Uuid::new_v4(),
        thread_id,
        total_messages: (participating_models.len() * max_rounds as usize) as u32,
        final_round: max_rounds,
        summary: None,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    let _ = state.tx.send(complete_msg);
    
    println!("🏁 [THREAD {}] Conversation complete using SEE-SAW POWER!", thread_id.to_string()[..8].to_string());
}

// 🥸 UNIFIED MODEL CALLER WITH CONDITIONAL CONCURRENCY
async fn call_models_with_seesaw_concurrency(
    state: &AppState,
    conversation: ConversationContainer,
    models: Vec<String>,
) -> Result<Vec<ModelResponse>, String> {
    match conversation {
        // 🥁 DRUMCIRCLE: Arc<Mutex<T>> - All models go crazy simultaneously
        ConversationContainer::DrumcircleChaos(shared_state) => {
            println!("🥁 DRUMCIRCLE MODE: Arc<Mutex<T>> parallel chaos!");
            
            // All models start simultaneously, each writes to shared state
            let futures: Vec<_> = models.into_iter().map(|model_name| {
                let shared_state = shared_state.clone();
                let state = state.clone();
                
                async move {
                    // Read the current prompt (multiple readers can do this safely)
                    let prompt = {
                        let state_guard = shared_state.lock().unwrap();
                        state_guard.prompt.clone()
                    };
                    
                    // Call model
                    match call_model_via_unified_service(&state, &model_name, &prompt).await {
                        Ok(response) => {
                            // Write response back to shared state (mutex ensures safety)
                            let current_timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                            let current_round = {
                                let mut state_guard = shared_state.lock().unwrap();
                                let round = state_guard.round;
                                state_guard.responses.push(ModelResponse {
                                    model_name: model_name.clone(),
                                    content: response.clone(),
                                    timestamp: current_timestamp,
                                    round,
                                });
                                round
                            };
                            Ok(ModelResponse {
                                model_name,
                                content: response,
                                timestamp: current_timestamp,
                                round: current_round,
                            })
                        }
                        Err(e) => Err(format!("Model {} failed: {}", model_name, e))
                    }
                }
            }).collect();
            
            // Wait for all the chaos to settle
            let results = futures_util::future::join_all(futures).await;
            let responses: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
            
            println!("🥁 Drumcircle complete: {} responses", responses.len());
            Ok(responses)
        }
        
        // 🧵 THREADING: Arc<RwLock<T>> - Sequential with many readers
        ConversationContainer::ThreadingOrder(shared_state) => {
            println!("🧵 THREADING MODE: Arc<RwLock<T>> sequential civilization!");
            
            let mut all_responses = Vec::new();
            
            // Sequential model calls - each model sees previous conversation
            for model_name in models {
                // Many readers can see the conversation history simultaneously
                let conversation_context = {
                    let state_guard = shared_state.read().await;
                    format!("{}\n\nConversation so far:\n{}", 
                        state_guard.prompt,
                        state_guard.responses.iter()
                            .map(|r| format!("**{}**: {}", r.model_name, r.content))
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    )
                };
                
                // Call model with conversation context
                match call_model_via_unified_service(state, &model_name, &conversation_context).await {
                    Ok(response) => {
                        let model_response = ModelResponse {
                            model_name: model_name.clone(),
                            content: response.clone(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            round: 1,
                        };
                        
                        // One writer at a time adds to conversation
                        {
                            let mut state_guard = shared_state.write().await;
                            state_guard.responses.push(model_response.clone());
                        }
                        
                        all_responses.push(model_response);
                        println!("✅ {} added to threading conversation", model_name);
                    }
                    Err(e) => {
                        println!("❌ {} failed in threading: {}", model_name, e);
                    }
                }
            }
            
            println!("🧵 Threading complete: {} responses", all_responses.len());
            Ok(all_responses)
        }
    }
}
async fn call_model_via_unified_service(
    state: &AppState,
    model_name: &str,
    prompt: &str,
) -> Result<String, String> {
    // Try unified LLM service first
    {
        let unified_llm = state.unified_llm.read().await;
        if let Some(_model) = unified_llm.get_model_by_name(model_name) {
            let chat_request = crate::unified_llm::ChatRequest {
                model_name: model_name.to_string(),
                prompt: prompt.to_string(),
                system_prompt: Some("You are participating in a threaded conversation with other AI models. Be thoughtful and engage constructively.".to_string()),
            };
            
            match unified_llm.send_to_model(&chat_request).await {
                Ok(response) => return Ok(response),
                Err(e) => broadcast_log!(&state.tx, "warn", "rust", "⚠️ Unified LLM failed for {}: {}", model_name, e),
            }
        }
    }
    
    // Fall back to Ollama service
    if let Some(personality) = state.ollama.get_personalities()
        .iter()
        .find(|p| p.name == model_name) {
        
        let config = state.config.read().await;
        let debug_config = Some(&config.debug_verbosity);
        
        match state.ollama.send_to_llm(personality, prompt, debug_config).await {
            Ok(response) => return Ok(response),
            Err(e) => return Err(format!("Ollama failed: {}", e)),
        }
    }
    
    Err(format!("Model '{}' not found in any service", model_name))
}

async fn get_thread_info(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Json<serde_json::Value> {
    let thread_uuid = match Uuid::parse_str(&thread_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return Json(serde_json::json!({
                "status": "error",
                "message": "Invalid thread ID format"
            }));
        }
    };

    match state.threaded_conversation_manager.get_thread(thread_uuid).await {
        Some(thread) => Json(serde_json::json!({
            "status": "success",
            "thread": {
                "id": thread.id,
                "initial_prompt": thread.initial_prompt,
                "participating_models": thread.participating_models,
                "messages": thread.messages,
                "current_round": thread.current_round,
                "max_rounds": thread.max_rounds,
                "is_active": thread.is_active,
                "created_at": thread.created_at,
                "last_activity": thread.last_activity
            }
        })),
        None => Json(serde_json::json!({
            "status": "error",
            "message": "Thread not found"
        }))
    }
}

async fn add_thread_response(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Json(request): Json<ThreadResponseRequest>,
) -> Json<serde_json::Value> {
    let thread_uuid = match Uuid::parse_str(&thread_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return Json(serde_json::json!({
                "status": "error",
                "message": "Invalid thread ID format"
            }));
        }
    };

    let config = state.config.read().await;
    let debug = &config.debug_verbosity;
    
    if debug.should_log(crate::types::DebugLevel::Normal) {
        println!("🧵 Adding response to thread {}:", thread_uuid);
        println!("   🤖 Model: {} ({})", request.model_name, request.model_provider);
        println!("   📝 Content: {}", debug.format_content(&request.content, "message_content"));
    }
    drop(config);

    match state.threaded_conversation_manager.add_message(
        thread_uuid,
        request.model_name.clone(),
        request.model_provider.clone(),
        request.content.clone(),
        request.is_html.unwrap_or(false),
        None, // parent_message_id - TODO: implement proper threading
        request.responding_to_model.clone(),
        Vec::new(), // context_used - TODO: implement context injection
    ).await {
        Ok(message_id) => {
            // Send model message via WebSocket
            let model_msg = types::Message::ModelMessage {
                id: message_id,
                model_name: request.model_name,
                model_provider: request.model_provider,
                content: request.content,
                is_html: request.is_html.unwrap_or(false),
                thread_id: thread_uuid,
                parent_message_id: None,
                responding_to_model: request.responding_to_model,
                conversation_round: 0, // TODO: get actual round
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };
            let _ = state.tx.send(model_msg);

            // Check if round should advance
            let _ = state.threaded_conversation_manager.maybe_advance_round(thread_uuid).await;

            Json(serde_json::json!({
                "status": "success",
                "message_id": message_id,
                "thread_id": thread_uuid
            }))
        },
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to add message: {}", e)
        }))
    }
}

async fn get_active_threads(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let threads = state.threaded_conversation_manager.get_active_threads().await;
    
    Json(serde_json::json!({
        "status": "success",
        "active_threads": threads.iter().map(|thread| {
            serde_json::json!({
                "id": thread.id,
                "initial_prompt": thread.initial_prompt,
                "participating_models": thread.participating_models,
                "message_count": thread.messages.len(),
                "current_round": thread.current_round,
                "max_rounds": thread.max_rounds,
                "is_active": thread.is_active,
                "created_at": thread.created_at,
                "last_activity": thread.last_activity
            })
        }).collect::<Vec<_>>()
    }))
}

// 🌊 QUANTUM CONSCIOUSNESS ENDPOINTS - MOVED TO quantum_handlers.rs

// 🔐 SESSION-BASED AUTH ENDPOINTS

async fn auth_login(
    session: Session,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let password = payload.get("password")
        .and_then(|p| p.as_str())
        .unwrap_or("");
    
    println!("🔐 Auth login - password: '{}'", password);
    
    let role = match password {
        "alkine-admin-2025" => "Admin",
        "friend-demo-access" => "Friend", 
        _ => {
            println!("❌ Invalid password: '{}'", password);
            return Json(serde_json::json!({
                "success": false,
                "error": "Invalid credentials"
            }))
        }
    };
    
    println!("✅ Role determined: {}", role);
    
    // Store in session instead of localStorage
    if let Err(e) = session.insert("user_role", role).await {
        println!("❌ Session role insert failed: {:?}", e);
        return Json(serde_json::json!({
            "success": false,
            "error": "Session error"
        }));
    }
    
    if let Err(e) = session.insert("authenticated", true).await {
        println!("❌ Session auth insert failed: {:?}", e);
        return Json(serde_json::json!({
            "success": false,
            "error": "Session error"
        }));
    }
    
    println!("✅ Session stored - role: {}", role);
    
    Json(serde_json::json!({
        "success": true,
        "role": role,
        "token": "session-based"
    }))
}

async fn auth_logout(session: Session) -> Json<serde_json::Value> {
    let _ = session.flush().await;
    Json(serde_json::json!({
        "success": true
    }))
}

async fn security_status(session: Session) -> Json<serde_json::Value> {
    let authenticated = match session.get::<bool>("authenticated").await {
        Ok(Some(val)) => val,
        _ => false,
    };
    
    let role = match session.get::<String>("user_role").await {
        Ok(Some(val)) => val,
        _ => String::new(),
    };
    
    println!("🔍 Security status check - authenticated: {}, role: '{}'", authenticated,role);
    
    Json(serde_json::json!({
        "authenticated": authenticated,
        "role": role
    }))
}

// 📝 SIMPLE CONFIG ENDPOINTS

