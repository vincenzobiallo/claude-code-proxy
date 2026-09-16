//! Web dashboard: static page plus small JSON APIs it reads on load. Live
//! monitor data is served by the existing loopback-restricted `/monitor`
//! endpoint; the page polls that directly instead of a second live channel.
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

use crate::{anthropic::json_error, registry::Registry};

const DASHBOARD_HTML: &str = include_str!("dashboard.html");

// Real provider logos, background removed (see the processing script used to
// generate these - flood-filled from the source marketing assets, not
// scraped at runtime). No source image exists for "opencode"; it keeps its
// monogram badge in the frontend instead of a logo here.
const LOGO_ANTHROPIC: &[u8] = include_bytes!("assets/logos/anthropic.png");
const LOGO_CODEX: &[u8] = include_bytes!("assets/logos/codex.png");
const LOGO_CURSOR: &[u8] = include_bytes!("assets/logos/cursor.png");
const LOGO_KIMI: &[u8] = include_bytes!("assets/logos/kimi.png");
const LOGO_GROK: &[u8] = include_bytes!("assets/logos/grok.png");

#[derive(Clone)]
pub struct DashboardState {
    pub registry: Arc<Registry>,
}

pub fn router(registry: Arc<Registry>) -> axum::Router {
    axum::Router::new()
        .route("/dashboard", get(dashboard_page))
        .route("/dashboard/api/registry", get(dashboard_registry))
        .route("/dashboard/api/auth", get(dashboard_auth))
        .route("/dashboard/api/config", get(dashboard_config))
        .route("/dashboard/logos/{file}", get(dashboard_logo))
        .with_state(DashboardState { registry })
}

async fn dashboard_logo(
    Path(file): Path<String>,
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Response {
    if let Err(response) = require_loopback(peer) {
        return response;
    }
    let provider = file.strip_suffix(".png").unwrap_or(file.as_str());
    let bytes: &'static [u8] = match provider {
        "anthropic" => LOGO_ANTHROPIC,
        "codex" => LOGO_CODEX,
        "cursor" => LOGO_CURSOR,
        "kimi" => LOGO_KIMI,
        "grok" => LOGO_GROK,
        _ => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                "No logo for this provider",
            );
        }
    };
    (
        [(http::header::CONTENT_TYPE, "image/png")],
        [(http::header::CACHE_CONTROL, "max-age=86400")],
        bytes,
    )
        .into_response()
}

fn require_loopback(
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Result<(), Response> {
    let Some(axum::Extension(axum::extract::ConnectInfo(peer))) = peer else {
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Dashboard access requires a local connection",
        ));
    };
    if !peer.ip().to_canonical().is_loopback() {
        return Err(json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Dashboard access requires a local connection",
        ));
    }
    Ok(())
}

async fn dashboard_page(
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Response {
    if let Err(response) = require_loopback(peer) {
        return response;
    }
    Html(DASHBOARD_HTML).into_response()
}

/// `{ "codex": ["gpt-5.6-sol", ...], "anthropic": ["opus", ...], ... }` - the
/// same grouping `claude-code-proxy models` prints, for the setup page's
/// checklist and modelPicker.options generator.
async fn dashboard_registry(
    State(state): State<DashboardState>,
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Response {
    if let Err(response) = require_loopback(peer) {
        return response;
    }
    Json(state.registry.grouped_models()).into_response()
}

#[derive(Serialize)]
struct AuthStatus {
    provider: String,
    connected: bool,
    detail: Option<String>,
}

async fn dashboard_auth(
    State(state): State<DashboardState>,
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
) -> Response {
    if let Err(response) = require_loopback(peer) {
        return response;
    }
    let mut out = Vec::new();
    for name in state.registry.list_provider_names() {
        let Some(provider) = state.registry.provider(&name) else {
            continue;
        };
        let (connected, detail) = match provider.cli().status() {
            Ok(()) => (true, None),
            Err(err) => (false, Some(err.to_string())),
        };
        out.push(AuthStatus {
            provider: name,
            connected,
            detail,
        });
    }
    Json(out).into_response()
}

async fn dashboard_config(
    peer: Option<axum::Extension<axum::extract::ConnectInfo<std::net::SocketAddr>>>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(response) = require_loopback(peer) {
        return response;
    }
    let cfg = crate::config::load_config();
    let overrides = crate::config::config_override_summary_lines(&cfg);
    // `--port` on the CLI can override the config-derived default without
    // changing it, so the config's own port can be stale here. The `Host`
    // header the client actually dialed is what's really listening.
    let port = headers
        .get(http::header::HOST)
        .and_then(|value| value.to_str().ok())
        .and_then(|host| host.rsplit_once(':'))
        .and_then(|(_, port)| port.parse::<u16>().ok())
        .unwrap_or(cfg.port);
    let setup_text = setup_text(port, &crate::registry::Registry::with_default_alias());
    Json(json!({ "overrides": overrides, "setup_text": setup_text })).into_response()
}

/// Copy-paste block for pointing Claude Code at this proxy, shown as a static
/// reference on the setup page.
///
/// Deliberately does NOT set `ANTHROPIC_MODEL`/`ANTHROPIC_SMALL_FAST_MODEL` or
/// `ANTHROPIC_AUTH_TOKEN`: this fork's whole point is adding other providers'
/// models as extra `/model` picker entries via `modelPicker.options`, without
/// touching Claude Code's native Opus/Sonnet/Haiku/Fable slots. Setting an
/// auth token here overrides the Claude subscription login and breaks the
/// Claude route with a 401 - see the project README.
pub fn setup_text(port: u16, registry: &Registry) -> String {
    let grouped = registry.grouped_models();
    let model_summary = ["codex", "kimi", "cursor"]
        .into_iter()
        .filter_map(|provider| {
            grouped
                .get(provider)
                .map(|models| format!("{provider}: {} models", models.len()))
        })
        .collect::<Vec<_>>()
        .join("  ");
    [
        format!("Logs: {}", crate::paths::log_file().display()),
        format!("Config: {}", crate::paths::config_dir().display()),
        format!("Providers: {model_summary}"),
        String::new(),
        format!("export ANTHROPIC_BASE_URL=\"http://localhost:{port}\""),
        String::new(),
        "Do not set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY - either one overrides"
            .to_string(),
        "the Claude subscription login and the Claude route returns 401.".to_string(),
        String::new(),
        "Do not set ANTHROPIC_MODEL or ANTHROPIC_SMALL_FAST_MODEL either - that".to_string(),
        "overrides Claude Code's native Opus/Sonnet/Haiku/Fable slots. Instead, pick".to_string(),
        "your models above and copy the generated JSON into modelPicker.options in"
            .to_string(),
        "~/.claude/settings.json, leaving replaceBuiltInOptions unset so the native"
            .to_string(),
        "slots stay untouched.".to_string(),
        String::new(),
        "Optional: export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1".to_string(),
    ]
    .join("\n")
}
