pub mod anthropic;
pub mod codex;
pub mod translate_shared;

/// Refreshes every provider's live model catalog, in parallel, and returns a
/// one-line summary for the monitor (e.g. "anthropic: 9 models · codex:
/// failed (not logged in)"). The only way catalogs refresh - there is no
/// background polling. A failure for one provider never blocks the other.
pub async fn refresh_model_catalogs() -> String {
    let (anthropic, codex) = tokio::join!(
        anthropic::model_catalog::global().refresh(),
        codex::model_catalog::refresh_once(),
    );
    [("anthropic", anthropic), ("codex", codex)]
        .into_iter()
        .map(|(provider, result)| match result {
            Ok(count) => format!("{provider}: {count} models"),
            Err(err) => format!("{provider}: failed ({err})"),
        })
        .collect::<Vec<_>>()
        .join(" \u{b7} ")
}
