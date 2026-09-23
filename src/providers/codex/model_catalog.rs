//! Live-discovered Codex model catalog, additive on top of the hardcoded
//! `ALLOWED_MODELS` allowlist (`translate::model_allowlist`). The ChatGPT
//! backend exposes `GET .../codex/models` as the per-account model catalog -
//! undocumented by OpenAI, but the same sibling-endpoint shape as the
//! `/alpha/search` endpoint `client.rs::search_endpoint` already talks to.
//! Refreshed only on demand (the monitor's "r" key in the Models view, or
//! `ccp models --refresh`); on a failed fetch (offline, not authenticated,
//! endpoint shape changes) the proxy just keeps using whatever it already
//! knew - the static seed, and/or a previous successful fetch persisted to
//! disk.
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::auth::constants::{CODEX_API_ENDPOINT, ORIGINATOR};
use super::auth::manager::CodexAuthManager;
use super::auth::token_store::file_store;
use super::translate::model_allowlist::ALLOWED_MODELS;
use crate::config;

/// Mirrors the pre-existing hardcoded match in `model_allowlist::uses_responses_lite`
/// - the seed data for models we already knew about before any live fetch.
const RESPONSES_LITE_SEED: &[&str] = &["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexModelEntry {
    pub id: String,
    #[serde(default)]
    pub use_responses_lite: bool,
}

/// One entry as parsed from a live fetch. Unlike `CodexModelEntry`, the
/// lite-lane flag is optional: an endpoint response that simply doesn't
/// mention it must not flip a known lite-only model (e.g. `gpt-5.6-luna`)
/// back to the full lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedCodexModel {
    pub id: String,
    pub use_responses_lite: Option<bool>,
}

struct CatalogState {
    entries: Vec<CodexModelEntry>,
}

pub struct CodexModelCatalog {
    state: RwLock<CatalogState>,
}

impl CodexModelCatalog {
    fn seeded() -> Self {
        let mut entries: Vec<CodexModelEntry> = ALLOWED_MODELS
            .iter()
            .map(|id| CodexModelEntry {
                id: (*id).to_string(),
                use_responses_lite: RESPONSES_LITE_SEED.contains(id),
            })
            .collect();
        for cached in load_cache_from_disk() {
            // The seed always wins for models it knows: a cache written by
            // an older build may carry a wrong lite flag for them.
            if !cached.id.ends_with("-fast") && !entries.iter().any(|entry| entry.id == cached.id) {
                entries.push(cached);
            }
        }
        Self {
            state: RwLock::new(CatalogState { entries }),
        }
    }

    pub fn snapshot(&self) -> Vec<CodexModelEntry> {
        self.state.read().expect("catalog lock poisoned").entries.clone()
    }

    pub fn snapshot_ids(&self) -> Vec<String> {
        self.snapshot().into_iter().map(|entry| entry.id).collect()
    }

    pub fn contains(&self, model: &str) -> bool {
        self.state
            .read()
            .expect("catalog lock poisoned")
            .entries
            .iter()
            .any(|entry| entry.id == model)
    }

    /// Defaults to `false` (full lane) for anything never seen - the safer
    /// failure mode, since a full-lane request to a lite-only model fails
    /// visibly instead of silently misrouting known-good traffic.
    pub fn uses_responses_lite(&self, model: &str) -> bool {
        self.state
            .read()
            .expect("catalog lock poisoned")
            .entries
            .iter()
            .find(|entry| entry.id == model)
            .is_some_and(|entry| entry.use_responses_lite)
    }

    /// Unions `fetched` into the existing state (never removes an entry) and
    /// persists the result to disk. A live catalog that's shorter than what
    /// we already knew is treated as a partial/odd response, not a signal to
    /// forget models - see the module doc comment.
    pub fn merge_and_persist(&self, fetched: Vec<FetchedCodexModel>) {
        let snapshot = self.merge(fetched);
        let _ = persist_cache_to_disk(&snapshot);
    }

    /// The in-memory half of `merge_and_persist` (no disk write, so tests
    /// can use it without touching the real cache); returns the merged
    /// entries.
    pub(crate) fn merge(&self, fetched: Vec<FetchedCodexModel>) -> Vec<CodexModelEntry> {
        let mut state = self.state.write().expect("catalog lock poisoned");
        for entry in fetched {
            merge_entry(&mut state.entries, entry);
        }
        state.entries.clone()
    }
}

/// Only an explicit flag in the live response overrides what we already
/// know; an absent flag keeps the existing value (or `false` for a brand new
/// model - see `uses_responses_lite`).
fn merge_entry(entries: &mut Vec<CodexModelEntry>, incoming: FetchedCodexModel) {
    match entries.iter_mut().find(|entry| entry.id == incoming.id) {
        Some(existing) => {
            if let Some(flag) = incoming.use_responses_lite {
                existing.use_responses_lite = flag;
            }
        }
        None => entries.push(CodexModelEntry {
            id: incoming.id,
            use_responses_lite: incoming.use_responses_lite.unwrap_or(false),
        }),
    }
}

fn load_cache_from_disk() -> Vec<CodexModelEntry> {
    std::fs::read_to_string(crate::paths::codex_model_catalog_cache_file())
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn persist_cache_to_disk(entries: &[CodexModelEntry]) -> anyhow::Result<()> {
    crate::paths::write_atomic(
        &crate::paths::codex_model_catalog_cache_file(),
        serde_json::to_string_pretty(entries)?.as_bytes(),
    )
}

pub fn global() -> &'static CodexModelCatalog {
    static CATALOG: OnceLock<CodexModelCatalog> = OnceLock::new();
    CATALOG.get_or_init(CodexModelCatalog::seeded)
}

/// The ChatGPT backend now 400s this endpoint without a `client_version`
/// query parameter (observed live, not documented by OpenAI).
fn models_endpoint() -> String {
    let base_url = config::codex_base_url(CODEX_API_ENDPOINT);
    let base_url = base_url.trim_end_matches('/');
    let root = match base_url.strip_suffix("/responses") {
        Some(api_root) => format!("{api_root}/models"),
        None => format!("{base_url}/models"),
    };
    format!("{root}?client_version={}", env!("CARGO_PKG_VERSION"))
}

/// Accepts a bare array or a `{"data": [...]}` / `{"models": [...]}` wrapper;
/// per-entry id comes from `slug`, else `id` (never a display-ish field like
/// `name`, which would become a routable, permanently cached "model"), and
/// the lite-lane flag from the first of `use_responses_lite`/
/// `useResponsesLite`/`responses_lite` that's present (`None` otherwise).
/// Entries missing an id, `-fast` ids (a proxy-side alias, never a real
/// model) and entries the backend marks hidden are skipped rather than
/// failing the whole fetch, since this is a reverse-engineered, undocumented
/// endpoint.
fn parse_models(payload: &Value) -> Vec<FetchedCodexModel> {
    let items = payload
        .as_array()
        .or_else(|| payload.get("data").and_then(Value::as_array))
        .or_else(|| payload.get("models").and_then(Value::as_array));
    let Some(items) = items else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let id = ["slug", "id"]
                .into_iter()
                .find_map(|key| item.get(key).and_then(Value::as_str))
                .map(str::trim)
                .filter(|id| !id.is_empty() && !id.ends_with("-fast"))?;
            let hidden = item
                .get("visibility")
                .and_then(Value::as_str)
                .is_some_and(|value| matches!(value, "hide" | "hidden"));
            if hidden {
                return None;
            }
            let use_responses_lite = ["use_responses_lite", "useResponsesLite", "responses_lite"]
                .into_iter()
                .find_map(|key| item.get(key).and_then(Value::as_bool));
            Some(FetchedCodexModel {
                id: id.to_string(),
                use_responses_lite,
            })
        })
        .collect()
}

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build codex model-catalog client")
    })
}

async fn fetch_live_codex_models() -> anyhow::Result<Vec<FetchedCodexModel>> {
    let auth_manager = CodexAuthManager::new(file_store());
    let auth = auth_manager.get_auth().await?;
    let mut request = http_client()
        .get(models_endpoint())
        .bearer_auth(&auth.access)
        .header("Accept", "application/json")
        .header("originator", config::codex_originator(ORIGINATOR));
    if let Some(account_id) = auth.account_id.as_deref() {
        request = request.header("ChatGPT-Account-Id", account_id);
    }
    let payload: Value = request.send().await?.error_for_status()?.json().await?;
    Ok(parse_models(&payload))
}

/// One-shot fetch + merge, used by the monitor's "r" key and the
/// `ccp models --refresh` CLI flag. Returns the number of models known after
/// the merge (seed + any newly discovered).
pub async fn refresh_once() -> anyhow::Result<usize> {
    let fetched = fetch_live_codex_models().await?;
    global().merge_and_persist(fetched);
    Ok(global().snapshot_ids().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_only() -> CodexModelCatalog {
        CodexModelCatalog {
            state: RwLock::new(CatalogState {
                entries: ALLOWED_MODELS
                    .iter()
                    .map(|id| CodexModelEntry {
                        id: (*id).to_string(),
                        use_responses_lite: RESPONSES_LITE_SEED.contains(id),
                    })
                    .collect(),
            }),
        }
    }

    #[test]
    fn seed_matches_previous_hardcoded_allowlist_behavior() {
        let catalog = seeded_only();
        assert!(catalog.contains("gpt-5.4"));
        assert!(!catalog.uses_responses_lite("gpt-5.4"));
        assert!(catalog.uses_responses_lite("gpt-5.6-terra"));
        assert!(!catalog.contains("gpt-7"));
    }

    fn fetched(id: &str, use_responses_lite: Option<bool>) -> FetchedCodexModel {
        FetchedCodexModel {
            id: id.to_string(),
            use_responses_lite,
        }
    }

    #[test]
    fn merge_is_additive_and_never_shrinks() {
        let catalog = seeded_only();
        let before = catalog.snapshot_ids().len();
        catalog.merge(vec![fetched("gpt-6-luna", Some(true))]);
        assert_eq!(catalog.snapshot_ids().len(), before + 1);
        assert!(catalog.contains("gpt-6-luna"));
        assert!(catalog.uses_responses_lite("gpt-6-luna"));

        // A second merge with an empty/short fetch must not drop anything.
        catalog.merge(vec![]);
        assert_eq!(catalog.snapshot_ids().len(), before + 1);
    }

    #[test]
    fn merge_without_a_lite_flag_keeps_the_known_lane() {
        let catalog = seeded_only();
        catalog.merge(vec![fetched("gpt-5.6-luna", None), fetched("gpt-6-nova", None)]);
        assert!(catalog.uses_responses_lite("gpt-5.6-luna"));
        assert!(!catalog.uses_responses_lite("gpt-6-nova"));
    }

    #[test]
    fn merge_with_an_explicit_lite_flag_overrides_it() {
        let catalog = seeded_only();
        catalog.merge(vec![fetched("gpt-5.6-terra", Some(false))]);
        assert!(!catalog.uses_responses_lite("gpt-5.6-terra"));
    }

    #[test]
    fn parse_models_accepts_bare_array() {
        let payload = serde_json::json!([
            { "id": "gpt-6-luna", "use_responses_lite": true },
            { "slug": "gpt-6-nova" },
        ]);
        let parsed = parse_models(&payload);
        assert_eq!(
            parsed,
            vec![fetched("gpt-6-luna", Some(true)), fetched("gpt-6-nova", None)]
        );
    }

    #[test]
    fn parse_models_accepts_data_wrapper_and_camel_case_flag() {
        let payload = serde_json::json!({
            "data": [{ "slug": "gpt-6-nova", "useResponsesLite": true }]
        });
        assert_eq!(parse_models(&payload), vec![fetched("gpt-6-nova", Some(true))]);
    }

    #[test]
    fn parse_models_ignores_display_fields_fast_ids_and_hidden_models() {
        let payload = serde_json::json!({
            "models": [
                { "unexpected_field": "x" },
                { "name": "GPT-6 Nova" },
                { "model": "gpt-6-nova" },
                { "slug": "gpt-6-nova-fast" },
                { "slug": "gpt-6-internal", "visibility": "hide" },
                { "slug": "gpt-6-nova", "display_name": "GPT-6 Nova", "visibility": "list" },
            ]
        });
        assert_eq!(parse_models(&payload), vec![fetched("gpt-6-nova", None)]);
    }

    #[test]
    fn parse_models_returns_empty_for_unrecognized_shape() {
        let payload = serde_json::json!({ "unexpected": true });
        assert!(parse_models(&payload).is_empty());
    }
}
