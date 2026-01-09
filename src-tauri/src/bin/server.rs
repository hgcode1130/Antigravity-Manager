use antigravity_tools_lib::proxy::{
    AxumServer, ProxyConfig, TokenManager, ProxyAuthMode,
    monitor::ProxyMonitor,
    config::{UpstreamProxyConfig, ExperimentalConfig, ZaiConfig}, 
    security::ProxySecurityConfig,
};
use std::env;
use std::sync::Arc;
use std::path::PathBuf;
use std::fs;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize simple logging (since we might not have access to the full AppHandle logger)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Starting Antigravity-Manager Standalone Server...");

    // 1. Setup Data Directory
    let data_dir = PathBuf::from("data");
    if !data_dir.exists() {
        fs::create_dir_all(&data_dir)?;
    }
    
    // 2. Load Accounts from Env
    if let Ok(accounts_json) = env::var("ACCOUNTS_JSON") {
        tracing::info!("Found ACCOUNTS_JSON, loading accounts...");
        let accounts_dir = data_dir.join("accounts");
        if !accounts_dir.exists() {
            fs::create_dir_all(&accounts_dir)?;
        }

        let accounts: serde_json::Value = serde_json::from_str(&accounts_json)
            .expect("Failed to parse ACCOUNTS_JSON");

        if let Some(arr) = accounts.as_array() {
            for acc in arr {
                if let Some(id) = acc.get("id").and_then(|s| s.as_str()) {
                    let path = accounts_dir.join(format!("{}.json", id));
                    let content = serde_json::to_string_pretty(acc)?;
                    fs::write(path, content)?;
                    tracing::info!("Restored account: {}", id);
                }
            }
        }
    } else {
        tracing::warn!("ACCOUNTS_JSON not found. Starting with empty/existing accounts.");
    }

    // 3. Initialize TokenManager
    let token_manager = Arc::new(TokenManager::new(data_dir.clone()));
    let count = token_manager.load_accounts().await
        .map_err(|e| format!("Failed to load accounts: {}", e))?;
    tracing::info!("Loaded {} accounts", count);

    // 4. Configure Proxy
    let host = "0.0.0.0".to_string(); // Always listen on all interfaces for Docker
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8045".to_string())
        .parse::<u16>()?;
    
    let api_key = env::var("API_KEY").unwrap_or_else(|_| {
        let key = format!("sk-{}", uuid::Uuid::new_v4().simple());
        tracing::warn!("API_KEY not set, generated temporary key: {}", key);
        key
    });

    // Extract other configs from Env or defaults
    // Note: In a real expanded version, we'd map all ProxyConfig fields from Env.
    // For now, we focus on the essentials.
    
    let upstream_proxy = UpstreamProxyConfig {
        enabled: false, 
        url: "".to_string() 
        // TODO: Map HTTP_PROXY/HTTPS_PROXY if needed
    };

    let security_config = ProxySecurityConfig {
        auth_mode: ProxyAuthMode::Strict, // Always strict for public deployment
        api_key: api_key.clone(),
    };

    let zai_config = ZaiConfig::default(); // Defaults for now
    let monitor = Arc::new(ProxyMonitor::new());
    let experimental_config = ExperimentalConfig::default();
    let custom_mapping = std::collections::HashMap::new();

    tracing::info!("Server listening on {}:{}", host, port);
    tracing::info!("Access Key: {}...", &api_key[0..6.min(api_key.len())]);

    let (server, handle) = AxumServer::start(
        host,
        port,
        token_manager,
        custom_mapping,
        120, // Request Timeout
        upstream_proxy,
        security_config,
        zai_config,
        monitor,
        experimental_config
    ).await.map_err(|e| format!("Server start failed: {}", e))?;

    // Graceful Shutdown
    match signal::ctrl_c().await {
        Ok(()) => {
            tracing::info!("Shutdown signal received");
        },
        Err(err) => {
            tracing::error!("Unable to listen for shutdown signal: {}", err);
        }
    }

    server.stop();
    handle.await?;
    tracing::info!("Server stopped successfully");

    Ok(())
}
