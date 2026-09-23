//! Live-discovered Anthropic model catalog, additive on top of the curated
//! `registry::ANTHROPIC_STYLE_ALIASES` aliases. Refreshed only on demand
//! (the monitor's "r" key in the Models view - see `providers::refresh_model_catalogs`).
//!
//! This proxy never stores an Anthropic credential on disk (see the module
//! doc comment on `providers::anthropic`). To refresh outside of a client
//! request, it keeps - in memory only, for this process's lifetime - the
//! auth headers of the most recent request Claude Code sent through it
//! (`remember_auth`), and reuses those for `GET /v1/models`.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Duration;

use http::{HeaderMap, HeaderName};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The only request headers `remember_auth` keeps: what `GET /v1/models`
/// needs to authenticate as the same Claude subscription/API key.
const AUTH_HEADERS: [&str; 4] = ["authorization", "x-api-key", "anthropic-version", "anthropic-beta"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicModelEntry {
    pub id: String,
    pub display_name: Option<String>,
    pub created_at: Option<String>,
}

pub struct AnthropicModelCatalog {
    entries: RwLock<Vec<AnthropicModelEntry>>,
    last_auth: Mutex<Option<HeaderMap>>,
    refreshing: AtomicBool,
}

impl AnthropicModelCatalog {
    fn with_entries(entries: Vec<AnthropicModelEntry>) -> Self {
        Self {
            entries: RwLock::new(entries),
            last_auth: Mutex::new(None),
            refreshing: AtomicBool::new(false),
        }
    }

    fn seeded() -> Self {
        Self::with_entries(load_cache_from_disk())
    }

    /// Keeps the auth headers of a request Claude Code just sent (memory
    /// only), so a later on-demand refresh can authenticate as the same
    /// account. Requests without any credential are ignored, so they never
    /// wipe a good one.
    pub fn remember_auth(&self, headers: &HeaderMap) {
        let mut kept = HeaderMap::new();
        for name in AUTH_HEADERS {
            for value in headers.get_all(name) {
                kept.append(HeaderName::from_static(name), value.clone());
            }
        }
        if kept.contains_key("authorization") || kept.contains_key("x-api-key") {
            *self.last_auth.lock().expect("auth lock poisoned") = Some(kept);
        }
    }

    fn last_auth(&self) -> Option<HeaderMap> {
        self.last_auth.lock().expect("auth lock poisoned").clone()
    }

    pub fn snapshot(&self) -> Vec<AnthropicModelEntry> {
        self.entries.read().expect("catalog lock poisoned").clone()
    }

    pub fn snapshot_ids(&self) -> Vec<String> {
        self.snapshot().into_iter().map(|entry| entry.id).collect()
    }

    pub fn display_name_for(&self, model: &str) -> Option<String> {
        self.entries
            .read()
            .expect("catalog lock poisoned")
            .iter()
            .find(|entry| entry.id == model)
            .and_then(|entry| entry.display_name.clone())
    }

    pub fn created_at_for(&self, model: &str) -> Option<String> {
        self.entries
            .read()
            .expect("catalog lock poisoned")
            .iter()
            .find(|entry| entry.id == model)
            .and_then(|entry| entry.created_at.clone())
    }

    /// Unions `fetched` into the existing state (never removes an entry) and
    /// persists the result to disk - same additive philosophy as the Codex
    /// catalog, and for the same reason: a bad/partial live response should
    /// never make previously-known models disappear. Returns the number of
    /// models known afterwards.
    fn merge_and_persist(&self, fetched: Vec<AnthropicModelEntry>) -> usize {
        let snapshot = self.merge(fetched);
        let _ = persist_cache_to_disk(&snapshot);
        snapshot.len()
    }

    /// The in-memory half of `merge_and_persist`; returns the merged entries.
    fn merge(&self, fetched: Vec<AnthropicModelEntry>) -> Vec<AnthropicModelEntry> {
        let mut entries = self.entries.write().expect("catalog lock poisoned");
        for entry in fetched {
            match entries.iter_mut().find(|existing| existing.id == entry.id) {
                Some(existing) => {
                    if entry.display_name.is_some() {
                        existing.display_name = entry.display_name;
                    }
                    if entry.created_at.is_some() {
                        existing.created_at = entry.created_at;
                    }
                }
                None => entries.push(entry),
            }
        }
        entries.clone()
    }

    /// Claims the right to refresh: `Some` for exactly one caller at a
    /// time, so repeated key presses don't stack upstream fetches. The claim
    /// is released when the guard drops - including when the future is
    /// cancelled mid-fetch.
    fn try_begin_refresh(&self) -> Option<RefreshGuard<'_>> {
        if self.refreshing.swap(true, Ordering::AcqRel) {
            None
        } else {
            Some(RefreshGuard(&self.refreshing))
        }
    }

    /// Fetches `GET /v1/models` with the remembered auth headers and merges
    /// the result. Returns the number of models known afterwards.
    pub async fn refresh(&self) -> anyhow::Result<usize> {
        let Some(_guard) = self.try_begin_refresh() else {
            anyhow::bail!("refresh already in progress");
        };
        let Some(auth) = self.last_auth() else {
            anyhow::bail!("no Claude Code request seen yet - send one message, then retry");
        };
        let fetched = fetch_live_anthropic_models(&crate::config::anthropic_base_url(), &auth).await?;
        Ok(self.merge_and_persist(fetched))
    }
}

struct RefreshGuard<'a>(&'a AtomicBool);

impl Drop for RefreshGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn load_cache_from_disk() -> Vec<AnthropicModelEntry> {
    std::fs::read_to_string(crate::paths::anthropic_model_catalog_cache_file())
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn persist_cache_to_disk(entries: &[AnthropicModelEntry]) -> anyhow::Result<()> {
    crate::paths::write_atomic(
        &crate::paths::anthropic_model_catalog_cache_file(),
        serde_json::to_string_pretty(entries)?.as_bytes(),
    )
}

pub fn global() -> &'static AnthropicModelCatalog {
    static CATALOG: OnceLock<AnthropicModelCatalog> = OnceLock::new();
    CATALOG.get_or_init(AnthropicModelCatalog::seeded)
}

/// Anthropic's real, documented `/v1/models` response shape:
/// `{"data": [{"id","display_name","created_at","type":"model"}], ...}`.
fn parse_models(payload: &Value) -> Vec<AnthropicModelEntry> {
    let Some(items) = payload.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?;
            Some(AnthropicModelEntry {
                id: id.to_string(),
                display_name: item
                    .get("display_name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                created_at: item
                    .get("created_at")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build anthropic model-catalog client")
    })
}

/// `GET {base_url}/v1/models` with `auth` (the `AUTH_HEADERS` subset kept by
/// `remember_auth`).
async fn fetch_live_anthropic_models(
    base_url: &str,
    auth: &HeaderMap,
) -> anyhow::Result<Vec<AnthropicModelEntry>> {
    // The endpoint paginates (default page size 20); ask for the documented
    // maximum so one request covers the whole catalog.
    let url = format!("{}/v1/models?limit=1000", base_url.trim_end_matches('/'));
    let payload: Value = http_client()
        .get(&url)
        .headers(auth.clone())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(parse_models(&payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_models_reads_the_documented_response_shape() {
        let payload = serde_json::json!({
            "data": [
                {
                    "id": "claude-opus-5-5-20260915",
                    "display_name": "Claude Opus 5.5",
                    "created_at": "2026-09-15T00:00:00Z",
                    "type": "model"
                },
                { "id": "claude-sonnet-5-20260910", "type": "model" }
            ]
        });
        let parsed = parse_models(&payload);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "claude-opus-5-5-20260915");
        assert_eq!(parsed[0].display_name.as_deref(), Some("Claude Opus 5.5"));
        assert_eq!(parsed[1].display_name, None);
    }

    #[test]
    fn parse_models_returns_empty_without_a_data_array() {
        assert!(parse_models(&serde_json::json!({ "error": "nope" })).is_empty());
    }

    #[test]
    fn merge_is_additive_and_updates_known_fields_in_place() {
        let catalog = AnthropicModelCatalog::with_entries(vec![AnthropicModelEntry {
            id: "claude-opus-5-5-20260915".to_string(),
            display_name: None,
            created_at: None,
        }]);
        catalog.merge(vec![AnthropicModelEntry {
            id: "claude-opus-5-5-20260915".to_string(),
            display_name: Some("Claude Opus 5.5".to_string()),
            created_at: Some("2026-09-15T00:00:00Z".to_string()),
        }]);
        assert_eq!(catalog.snapshot().len(), 1);
        assert_eq!(
            catalog.display_name_for("claude-opus-5-5-20260915").as_deref(),
            Some("Claude Opus 5.5")
        );
    }

    #[test]
    fn only_one_concurrent_refresh_is_granted() {
        let catalog = AnthropicModelCatalog::with_entries(Vec::new());
        let guard = catalog.try_begin_refresh();
        assert!(guard.is_some());
        assert!(catalog.try_begin_refresh().is_none());
        drop(guard);
        assert!(catalog.try_begin_refresh().is_some());
    }

    #[test]
    fn remember_auth_keeps_only_auth_headers_and_ignores_unauthenticated_requests() {
        let catalog = AnthropicModelCatalog::with_entries(Vec::new());
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("anthropic-beta", "oauth-2025-04-20".parse().unwrap());
        headers.insert("user-agent", "claude-cli".parse().unwrap());
        headers.insert("host", "localhost".parse().unwrap());
        catalog.remember_auth(&headers);

        let mut anonymous = HeaderMap::new();
        anonymous.insert("anthropic-version", "2023-06-01".parse().unwrap());
        catalog.remember_auth(&anonymous);

        let kept = catalog.last_auth().unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(kept["authorization"], "Bearer secret");
        assert_eq!(kept["anthropic-beta"], "oauth-2025-04-20");
    }

    #[tokio::test]
    async fn refresh_without_a_seen_request_explains_why() {
        let catalog = AnthropicModelCatalog::with_entries(Vec::new());
        let error = catalog.refresh().await.unwrap_err();
        assert!(error.to_string().contains("no Claude Code request seen yet"));
    }
}
