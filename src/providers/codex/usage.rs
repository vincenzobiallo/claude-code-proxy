//! Periodic poll of the ChatGPT account's usage stats (the monthly credit
//! pool), mirroring what `~/.claude/statusbar/codex/refresh.js` already does
//! for the Claude Code status line: same endpoint, same OAuth token. Unlike
//! Anthropic, Codex/OpenAI never sends per-response usage headers, so a
//! side-channel poll is the only way to show this continuously rather than
//! only once a window is fully exhausted (see `events::usage_limit_from_*`).
use std::time::Duration;

use serde_json::Value;

use super::auth::manager::CodexAuthManager;
use super::auth::token_store::file_store;
use crate::monitor::MonitorHandle;

const USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";
// The credit pool this reports moves on a monthly cadence, so there is no
// value in polling as tightly as the per-prompt statusline script does (it
// hits this same endpoint on its own 60s TTL) - a long interval keeps this
// call rare instead of doubling the request rate against an internal,
// undocumented ChatGPT endpoint.
const POLL_INTERVAL: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Copy, Default)]
struct CodexUsage {
    used_percentage: Option<f64>,
    resets_at: Option<u64>,
}

/// `wham/usage` mixes plain JSON numbers and numeric strings for the same
/// shape depending on account/plan type (observed live: `used_percent` comes
/// back as a number, `used`/`limit` on the same object as strings) - accept
/// either instead of silently dropping the field.
fn flexible_f64(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn parse_usage(payload: &Value) -> CodexUsage {
    let monthly = payload
        .pointer("/spend_control/individual_limit")
        .or_else(|| payload.pointer("/spendControl/individualLimit"))
        .or_else(|| payload.pointer("/individual_limit"));
    let used_percentage = monthly.and_then(|entry| {
        flexible_f64(entry.get("used_percent")).or_else(|| flexible_f64(entry.get("usedPercent")))
    });
    let resets_at = monthly
        .and_then(|entry| {
            flexible_f64(entry.get("reset_at")).or_else(|| flexible_f64(entry.get("resetAt")))
        })
        .map(|secs| secs.max(0.0) as u64);
    CodexUsage {
        used_percentage,
        resets_at,
    }
}

async fn fetch_usage(
    client: &reqwest::Client,
    access_token: &str,
    account_id: Option<&str>,
) -> anyhow::Result<CodexUsage> {
    let mut request = client
        .get(USAGE_URL)
        .bearer_auth(access_token)
        .header("Accept", "application/json")
        .header("Origin", "https://chatgpt.com")
        .header("Referer", "https://chatgpt.com/");
    if let Some(account_id) = account_id {
        request = request.header("ChatGPT-Account-Id", account_id);
    }
    let payload: Value = request.send().await?.error_for_status()?.json().await?;
    Ok(parse_usage(&payload))
}

/// Runs until the process exits, polling on `POLL_INTERVAL`. A tick that
/// finds no Codex auth, or whose fetch fails, is silently skipped - the
/// monitor just keeps showing whatever it last had (or nothing, before the
/// first success).
pub async fn run_poller(monitor: MonitorHandle) {
    let auth_manager = CodexAuthManager::new(file_store());
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(client) => client,
        Err(_) => return,
    };
    loop {
        if let Ok(auth) = auth_manager.get_auth().await
            && let Ok(usage) =
                fetch_usage(&client, &auth.access, auth.account_id.as_deref()).await
            && let Some(pct) = usage.used_percentage
        {
            monitor.usage_window_updated("codex", "monthly", pct.clamp(0.0, 100.0), usage.resets_at);
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_numeric_used_percent_with_string_used_and_limit() {
        // Real `wham/usage` shape for a credit-pool ("business") plan: `used`
        // and `limit` are numeric strings, `used_percent` is a plain number.
        let payload = json!({
            "spend_control": {
                "individual_limit": {
                    "used": "931.2319300174713",
                    "limit": "6550",
                    "used_percent": 14,
                    "reset_at": 1790812800u64
                }
            }
        });
        let usage = parse_usage(&payload);
        assert_eq!(usage.used_percentage, Some(14.0));
        assert_eq!(usage.resets_at, Some(1790812800));
    }

    #[test]
    fn parses_camel_case_spend_control() {
        let payload = json!({
            "spendControl": {
                "individualLimit": { "usedPercent": 12.0 }
            }
        });
        let usage = parse_usage(&payload);
        assert_eq!(usage.used_percentage, Some(12.0));
    }

    #[test]
    fn missing_spend_control_yields_none() {
        let usage = parse_usage(&json!({}));
        assert_eq!(usage.used_percentage, None);
        assert_eq!(usage.resets_at, None);
    }
}
