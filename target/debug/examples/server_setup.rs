// SERVER_SETUP.RS
// Extracted server configuration and router setup from main.rs
// This contains all the router definitions, AppState construction, and server startup logic

use std::sync::Arc;
use axum::{
    extract::State,
    http::Method,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use tower_sessions::{MemoryStore, SessionManagerLayer, Session};
use tower_http::cors::{Any, CorsLayer};
use crate::{
    network_utils::{AppState, websocket_handler},
    config_persistence::ConfigManager,
    types::{Config, Message},
    discovery,
    ollama,
    unified_llm::UnifiedLLMService,
    conversation,
    threaded_conversation,
    security_middleware,
    auth_system,
    security_manager,
    zombie_endpoints,
    belongsInAConfigFile,
    config_handlers,
    llm_handlers,
    model_testing_handlers,
    quantum_handlers,
    session_handlers,
    state_manager::StateManager,
};

// 🏗️ APPSTATE CONSTRUCTION
pub fn create_app_state(
    tx: tokio::sync::broadcast::Sender<Message>,
    device_id: uuid::Uuid,
    config: Arc<tokio::sync::RwLock<Config>>,
    discovery: discovery::DiscoveryService,
    unified_llm_service: UnifiedLLMService,
    config_manager: ConfigManager,
    threading_config: crate::types::ThreadingConfig,
    // state_manager: StateManager, // COMMENTED OUT FOR BYPASS
    
) -> AppState {
    // 🛡️ Initialize security middleware
    crate::network_utils::broadcast_log(&tx, "info", "rust", "🛡️ Initializing security middleware...");
    let rate_limit_config = security_middleware::RateLimitConfig {
        requests_per_minute: 200,    // Much higher for dev work
        requests_per_hour: 2000,     // Much higher limits
        block_duration_minutes: 10,  // Shorter blocks
        auto_block_threshold: 50,    // Much higher threshold
        whitelist: vec![
            "127.0.0.1".parse().unwrap(),
            "::1".parse().unwrap(),
            "192.168.4.97".parse().unwrap(), // Your Ollama host
            "::ffff:127.0.0.1".parse().unwrap(), // IPv6 localhost
        ],
    };
    let rate_limiter = Arc::new(security_middleware::RateLimiter::new(rate_limit_config));

    // 🔐 Initialize authentication system
    crate::network_utils::broadcast_log(&tx, "info", "rust", "🔐 Initializing authentication system...");
    let auth_config = auth_system::AuthConfig::default(); // Use config from file in production
    let auth_manager = Arc::new(auth_system::AuthManager::new(auth_config));

    // 🧵 Initialize threaded conversation manager
    let threaded_conversation_manager = threaded_conversation::ThreadedConversationManager::new(threading_config);

    AppState {
        tx: tx.clone(),
        device_id,
        config: config.clone(),
        discovery: Arc::new(discovery),
        ollama: Arc::new(ollama::OllamaService::new_with_config(
            Some("192.168.4.97:11434".to_string()), // Your actual Ollama host
            std::env::var("GEMINI_API_KEY").ok(),
        )),
        unified_llm: Arc::new(tokio::sync::RwLock::new(unified_llm_service)),
        conversation_manager: Arc::new(conversation::ConversationManager::new()),
        config_manager: Arc::new(config_manager),
        threaded_conversation_manager: Arc::new(threaded_conversation_manager), // 🧵 NEW!
        rate_limiter: rate_limiter.clone(), // 🛡️ RATE LIMITER
        auth_manager: auth_manager.clone(), // 🔐 AUTH MANAGER
        // state_manager: Arc::new(state_manager), // 💾 STATE MANAGER - BYPASSED
    }
}

// 🌐 ROUTER CONFIGURATION
pub fn create_router() -> Router<AppState> {
    Router::new()
        .route("/ws", get(websocket_handler))
        .route("/config", get(config_handlers::get_config).post(config_handlers::update_config))
        .route("/peers", get(config_handlers::get_peers))
        .route("/drumcircle", post(llm_handlers::start_drumcircle))
        .route("/drumcircle-config", post(llm_handlers::start_configurable_drumcircle))
        .route("/drumcircle-models", get(llm_handlers::get_drumcircle_models))
        .route("/qwen-personalities", get(llm_handlers::get_qwen_personalities))
        .route("/llm-providers", get(llm_handlers::get_llm_providers))
        .route("/api-keys", post(llm_handlers::update_api_keys))
        .route("/load-continue-config", post(config_handlers::load_continue_config))
        .route("/chat-individual", post(llm_handlers::chat_individual))
        .route("/available-models", get(llm_handlers::get_available_models))
        // .route("/test-ollama-models", post(test_ollama_models_v2)) // 🦙 TEMPORARILY DISABLED
        .route("/conversation-history/:model_name", get(crate::get_conversation_history))
        .route("/conversation-context/:model_name", post(crate::set_conversation_context))
        .route("/conversation-clear/:model_name", post(crate::clear_conversation))
        .route("/config-save", post(config_handlers::save_config_to_disk))
        // TODO: Fix handler signature mismatch for load_config_from_disk
        // .route("/config-load", get(config_handlers::load_config_from_disk))
        .route("/yaml-config-save", post(config_handlers::save_yaml_config))
        .route("/yaml-config-load", get(config_handlers::load_yaml_config))
        // 🌐 WEB SCRAPING ENDPOINTS - TAURI CHROMIUM POWER!
        .route("/web-scrape", post(crate::scrape_web_content))
        .route("/web-search", post(crate::search_and_scrape))
        // 💀 ZOMBIE ENDPOINTS - EXTRACTED DEAD CODE
        .route("/phi-detect", post(zombie_endpoints::detect_phi_in_text))
        .route("/medical-transform", post(zombie_endpoints::transform_medical_data))
        .route("/python-execute", post(zombie_endpoints::execute_python_script))
        .route("/python-status", get(zombie_endpoints::get_python_status))
        .route("/ai-model-process", post(belongsInAConfigFile::process_with_ai_model))
        .route("/python-install-deps", post(zombie_endpoints::install_python_packages))
        // 💾 SESSION PERSISTENCE ENDPOINTS
        .route("/session-save", post(session_handlers::save_session_to_disk))
        .route("/session-load/:session_id", get(session_handlers::load_session_from_disk))
        .route("/session-list", get(session_handlers::list_available_sessions))
        .route("/session-delete/:session_id", delete(session_handlers::delete_session_from_disk))
        .route("/live-compute-props", post(crate::live_compute_props)) // Add this route to your router
        // 🌊 QUANTUM CONSCIOUSNESS ROUTES
        .route("/quantum/start", post(quantum_handlers::start_quantum_evolution))
        .route("/quantum/state", get(quantum_handlers::get_quantum_state))
        .route("/quantum/component", post(quantum_handlers::add_quantum_component))
        .route("/quantum/interact", post(quantum_handlers::quantum_interaction))
        // 🧵 THREADED CONVERSATION ROUTES
        .route("/thread/start", post(crate::start_threaded_conversation))
        .route("/thread/:thread_id", get(crate::get_thread_info))
        .route("/thread/:thread_id/respond", post(crate::add_thread_response))
        .route("/threads/active", get(crate::get_active_threads))
        // 🧪 MODEL TESTING ROUTES - TEMPORARILY DISABLED FOR COMPILATION
        // TODO: Fix handler signatures for these routes
        // .route("/test-ollama-models", post(model_testing_handlers::test_ollama_models_v2)) // 🦙 MODEL TESTING ENDPOINT!
        // .route("/test-ollama-models-v1", post(model_testing_handlers::test_ollama_models)) // 🦙 LEGACY TESTING ENDPOINT
        // .route("/model-test-report", get(model_testing_handlers::get_model_test_report))
        // .route("/model-test-settings", post(model_testing_handlers::update_model_test_settings))
        // .route("/clear-model-tests", post(model_testing_handlers::clear_model_test_results))
        // 🛡️ SECURITY & ALKINEHOTEL ROUTES
        .route("/api/security/mode", get(security_manager::security_mode_handler))
        .route("/api/security/authenticate", post(security_manager::authenticate_handler))
        .route("/api/security/logout", post(security_manager::logout_handler))
        
        // 🏨 ALKINEHOTEL TRAPS - Order matters! Specific routes first
        .route("/shell", get(security_manager::alkinehotel_shell_page))
        .route("/admin", get(security_manager::alkinehotel_admin_page))
        .route("/wp-admin", get(security_manager::alkinehotel_wordpress_page))
        .route("/phpmyadmin", get(security_manager::alkinehotel_db_page))
        .route("/panel", get(security_manager::alkinehotel_panel_page))
        .route("/login", get(security_manager::alkinehotel_login_page))
        .route("/dashboard", get(security_manager::alkinehotel_dashboard_page))
        
        // 🏨 ALKINEHOTEL API HANDLERS
        .route("/alkinehotel/shell/:session_id", post(security_manager::alkinehotel_shell_handler))
        .route("/sandbox/:session_id/:endpoint", post(security_manager::sandbox_api_handler))
        
        //  AUTHENTICATION ENDPOINTS  
        .route("/api/auth/login", post(auth_system::login_endpoint))
        .route("/api/auth/logout", post(auth_system::logout_endpoint))
        
        // 🛡️ SECURITY STATUS ENDPOINT
        .route("/api/security/status", get(security_middleware::security_status))
        
        // 📝 SIMPLE CONFIG ENDPOINTS
        .route("/simple-config", get(config_handlers::get_simple_config))
        .route("/simple-config/clear", post(config_handlers::clear_simple_config))
        
        // 🌐 CORS - JUST MAKE IT WORK
        .layer(CorsLayer::very_permissive())
        // 🍪 SESSION MANAGEMENT
        .layer(SessionManagerLayer::new(MemoryStore::default()))
        // 🛡️ APPLY SECURITY MIDDLEWARE (ORDER MATTERS!)
        // .layer(middleware::from_fn_with_state(app_state.clone(), security_middleware::security_middleware))
        // .layer(middleware::from_fn(security_middleware::csrf_middleware))
        // 🔐 AUTH MIDDLEWARE (DISABLED FOR DEV - ENABLE IN PRODUCTION)
        // // .layer(middleware::from_fn_with_state(app_state.clone(), auth_system::auth_middleware))
}

//  SERVER STARTUP
pub async fn start_server(
    app_state: AppState,
    config: Arc<tokio::sync::RwLock<Config>>,
    tx: &tokio::sync::broadcast::Sender<Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router().with_state(app_state);
    
    let port = config.read().await.port;
    let addr = format!("0.0.0.0:{}", port);
    
    crate::network_utils::broadcast_log(&tx, "info", "rust", &format!("🥸 Starting server on {}", addr));
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    crate::network_utils::broadcast_log(&tx, "success", "rust", &format!("✅ Server listening on {}", addr));
    
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await?;
    Ok(())
}