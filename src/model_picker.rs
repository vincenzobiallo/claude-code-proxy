//! Persisted selection + ordering behind the terminal Models view's `[v]`
//! toggle, and the merge/backup logic for pushing the result into
//! `modelPicker.options` in `~/.claude/settings.json`. The selection itself
//! lives in our own config dir (`model_picker.json`) so it survives restarts
//! even if the user never applies it to `settings.json`.
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PickerEntry {
    pub model: String,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PickerSelection {
    pub entries: Vec<PickerEntry>,
    /// Mirrors `modelPicker.replaceBuiltInOptions` in `~/.claude/settings.json`:
    /// `false` (the default) keeps Claude Code's native Opus/Sonnet/Haiku/Fable
    /// dropdown entries alongside `entries`; `true` replaces them so only
    /// `entries` shows in `/model`. Only takes effect in `settings.json` once
    /// applied with the TUI's "o" override, same as `entries` itself.
    #[serde(default)]
    pub replace_built_in_options: bool,
}

impl PickerSelection {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Matches by the normalized (1m-suffix-stripped) model id, ignoring
    /// `provider` - entries don't record it (see `registry::ordered_models`,
    /// which re-derives it from the model list), and an entry customized
    /// through `set_alias` may or may not carry the `[1m]` suffix regardless
    /// of `picker_model_field`'s per-provider default.
    pub fn position(&self, provider: &str, model: &str) -> Option<usize> {
        let _ = provider;
        let base = crate::registry::normalize_incoming_model(model);
        self.entries
            .iter()
            .position(|entry| crate::registry::normalize_incoming_model(&entry.model).eq_ignore_ascii_case(&base))
    }

    pub fn is_enabled(&self, provider: &str, model: &str) -> bool {
        self.position(provider, model).is_some()
    }

    pub fn entry_at(&self, provider: &str, model: &str) -> Option<&PickerEntry> {
        self.position(provider, model).map(|index| &self.entries[index])
    }

    /// The alias/description the Models view should show for `provider/model`:
    /// its live picker entry when enabled (`live: true`), else
    /// `curated_default` as a dimmed suggestion (`live: false`), else blank.
    pub fn preview(&self, provider: &str, model: &str) -> (String, String, bool) {
        if let Some(entry) = self.entry_at(provider, model) {
            return (entry.label.clone(), entry.description.clone(), true);
        }
        match curated_default(provider, model) {
            Some((label, description)) => (label.to_string(), description.to_string(), false),
            None => (String::new(), String::new(), false),
        }
    }

    /// Whether `provider/model` currently requests the 1M-context variant:
    /// the stored entry's own field when enabled, otherwise the same
    /// per-provider default `toggle` uses for a brand new entry.
    pub fn use_1m(&self, provider: &str, model: &str) -> bool {
        match self.entry_at(provider, model) {
            Some(entry) => entry.model.to_ascii_lowercase().ends_with("[1m]"),
            None => provider == "codex",
        }
    }

    /// Toggles `provider/model`. Re-enabling a model reuses the label/description
    /// already present in `existing` (the live `settings.json`) if there is a
    /// match, so a description the user hand-edited there survives a
    /// disable/enable round trip instead of getting overwritten with a placeholder.
    pub fn toggle(&mut self, provider: &str, model: &str, existing: &[PickerEntry]) {
        if let Some(index) = self.position(provider, model) {
            self.entries.remove(index);
            return;
        }
        let field = picker_model_field(provider, model);
        let entry = existing
            .iter()
            .find(|entry| entry.model.eq_ignore_ascii_case(&field))
            .cloned()
            .unwrap_or_else(|| {
                let (label, description) = match curated_default(provider, model) {
                    Some((label, description)) => (label.to_string(), description.to_string()),
                    None => (model.to_string(), format!("via {provider}")),
                };
                PickerEntry { model: field, label, description }
            });
        self.entries.push(entry);
    }

    /// Sets (or creates) `provider/model`'s alias (`label`) and whether it
    /// requests the 1M-context variant, driven by the TUI's "a" editor
    /// (`tui::render_alias_editor`). An existing entry keeps its position and
    /// description; a brand new one gets `curated_default`'s description
    /// when there is one, else the same placeholder `toggle` uses.
    pub fn set_alias(&mut self, provider: &str, model: &str, label: String, use_1m: bool) {
        let field = model_field(model, use_1m);
        if let Some(index) = self.position(provider, model) {
            self.entries[index].model = field;
            self.entries[index].label = label;
        } else {
            let description = curated_default(provider, model)
                .map(|(_, description)| description.to_string())
                .unwrap_or_else(|| format!("via {provider}"));
            self.entries.push(PickerEntry { model: field, label, description });
        }
    }

    /// Swaps the entry at `index` with its neighbor `delta` steps away
    /// (`delta` is always +-1 here). No-op if that would go out of bounds.
    pub fn move_by(&mut self, index: usize, delta: i64) {
        let Some(target) = index.checked_add_signed(delta as isize) else {
            return;
        };
        if index >= self.entries.len() || target >= self.entries.len() {
            return;
        }
        self.entries.swap(index, target);
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.entries).unwrap_or_default()
    }

    pub fn from_json(text: &str) -> anyhow::Result<Vec<PickerEntry>> {
        Ok(serde_json::from_str(text)?)
    }
}

/// Every codex model this proxy exposes also has a 1M-token context variant,
/// selected by suffixing the model id with `[1m]` (see
/// `registry::normalize_incoming_model`, which strips it back off before
/// routing). The picker always requests that variant for codex models.
fn picker_model_field(provider: &str, model: &str) -> String {
    model_field(model, provider == "codex")
}

/// Builds the `modelPicker.options[].model` value for `model`, with or
/// without the `[1m]` suffix that requests its 1M-token context variant
/// (see `registry::normalize_incoming_model`, which strips it back off
/// before routing).
fn model_field(model: &str, use_1m: bool) -> String {
    let base = crate::registry::normalize_incoming_model(model);
    if use_1m { format!("{base}[1m]") } else { base }
}

/// Anthropic's own official label/description for its model-family aliases
/// (matches Claude Code's own `/model` picker UI) - used to pre-fill a fresh
/// picker entry instead of a generic placeholder. No equivalent catalog
/// exists for Codex models or for pinned Anthropic snapshots (e.g.
/// `claude-sonnet-5`), which fall back to `format!("via {provider}")`
/// until the user sets one with the TUI's "a" alias editor.
fn curated_default(provider: &str, model: &str) -> Option<(&'static str, &'static str)> {
    if provider != "anthropic" {
        return None;
    }
    match model {
        "sonnet" => Some(("Sonnet", "Sonnet 5 \u{b7} Efficient for routine tasks")),
        "fable" => Some((
            "Fable",
            "Fable 5.1 \u{b7} Most capable for your hardest and longest-running tasks \u{b7} Requires usage credits",
        )),
        "opus" => Some(("Opus", "Opus 5 \u{b7} Best for everyday, complex tasks \u{b7} ~2\u{d7} usage vs Sonnet")),
        "haiku" => Some(("Haiku", "Haiku 4.5 \u{b7} Fastest for quick answers")),
        _ => None,
    }
}

/// Reads the existing `modelPicker.options` array out of `~/.claude/settings.json`,
/// if the file exists and is well-formed. Used both to seed a re-enabled
/// model's label/description from what's already live, and to show the user
/// what an override would replace.
pub fn read_existing_entries(path: &Path) -> Vec<PickerEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    root.get("modelPicker")
        .and_then(|picker| picker.get("options"))
        .and_then(|options| options.as_array())
        .map(|options| {
            options
                .iter()
                .filter_map(|entry| serde_json::from_value(entry.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Overwrites `modelPicker.options` and `modelPicker.replaceBuiltInOptions`
/// in `~/.claude/settings.json`, preserving every other key in the file and
/// leaving a `.bak` copy of the previous contents next to it before writing.
/// Unlike `options`, `replaceBuiltInOptions` is always written explicitly
/// (even `false`) so the TUI's toggle state is never left stale behind a
/// value some earlier, unrelated edit of `settings.json` happened to leave.
pub fn apply_override(
    path: &Path,
    entries: &[PickerEntry],
    replace_built_in_options: bool,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
    let mut root: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| anyhow::anyhow!("{} is not valid JSON: {err}", path.display()))?;
    if !root.is_object() {
        anyhow::bail!("{} does not contain a JSON object at the top level", path.display());
    }
    if path.exists() {
        std::fs::copy(path, path.with_extension("json.bak"))?;
    }
    let picker = root
        .as_object_mut()
        .expect("checked is_object above")
        .entry("modelPicker")
        .or_insert_with(|| serde_json::json!({}));
    if !picker.is_object() {
        *picker = serde_json::json!({});
    }
    let picker = picker.as_object_mut().expect("just ensured it's an object");
    picker.insert("options".to_string(), serde_json::to_value(entries)?);
    picker.insert(
        "replaceBuiltInOptions".to_string(),
        serde_json::json!(replace_built_in_options),
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_adds_then_removes_a_codex_model_with_1m_suffix() {
        let mut selection = PickerSelection::default();
        selection.toggle("codex", "gpt-5.6-terra", &[]);
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(selection.entries[0].model, "gpt-5.6-terra[1m]");
        selection.toggle("codex", "gpt-5.6-terra", &[]);
        assert!(selection.entries.is_empty());
    }

    #[test]
    fn toggle_prefills_curated_label_and_description_for_a_known_alias() {
        let mut selection = PickerSelection::default();
        selection.toggle("anthropic", "sonnet", &[]);
        assert_eq!(selection.entries[0].label, "Sonnet");
        assert!(selection.entries[0].description.starts_with("Sonnet 5"));
    }

    #[test]
    fn set_alias_creates_an_entry_with_the_chosen_label_and_1m_flag() {
        let mut selection = PickerSelection::default();
        selection.set_alias("codex", "gpt-5.6-terra", "Terra".to_string(), false);
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(selection.entries[0].model, "gpt-5.6-terra");
        assert_eq!(selection.entries[0].label, "Terra");
        assert!(selection.is_enabled("codex", "gpt-5.6-terra"));
    }

    #[test]
    fn set_alias_updates_an_existing_entry_in_place_and_keeps_its_description() {
        let mut selection = PickerSelection::default();
        selection.toggle("codex", "gpt-5.6-terra", &[]);
        let original_description = selection.entries[0].description.clone();
        selection.set_alias("codex", "gpt-5.6-terra", "Terra".to_string(), false);
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(selection.entries[0].model, "gpt-5.6-terra");
        assert_eq!(selection.entries[0].label, "Terra");
        assert_eq!(selection.entries[0].description, original_description);
    }

    #[test]
    fn position_matches_regardless_of_a_customized_1m_suffix() {
        let mut selection = PickerSelection::default();
        // A model whose provider default is 1m-on, customized to 1m-off.
        selection.set_alias("codex", "gpt-5.6-terra", "Terra".to_string(), false);
        assert!(selection.is_enabled("codex", "gpt-5.6-terra"));
        assert!(!selection.use_1m("codex", "gpt-5.6-terra"));
    }

    #[test]
    fn toggle_reuses_existing_label_and_description() {
        let existing = vec![PickerEntry {
            model: "gpt-5.6-terra[1m]".to_string(),
            label: "gpt-5.6-terra".to_string(),
            description: "(OpenAI) Balanced agentic coding model for everyday work.".to_string(),
        }];
        let mut selection = PickerSelection::default();
        selection.toggle("codex", "gpt-5.6-terra", &existing);
        assert_eq!(selection.entries[0].description, existing[0].description);
    }

    #[test]
    fn move_by_swaps_adjacent_entries_and_ignores_out_of_bounds() {
        let mut selection = PickerSelection::default();
        selection.toggle("codex", "a", &[]);
        selection.toggle("codex", "b", &[]);
        selection.move_by(0, 1);
        assert_eq!(selection.entries[0].model, "b[1m]");
        assert_eq!(selection.entries[1].model, "a[1m]");
        selection.move_by(0, -1);
        assert_eq!(selection.entries[0].model, "b[1m]");
    }

    #[test]
    fn apply_override_preserves_unrelated_keys_and_writes_a_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{"someOtherSetting": true}"#).unwrap();
        let entries = vec![PickerEntry {
            model: "gpt-5.6-terra[1m]".to_string(),
            label: "gpt-5.6-terra".to_string(),
            description: "d".to_string(),
        }];
        apply_override(&path, &entries, true).unwrap();
        assert!(path.with_extension("json.bak").exists());
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["someOtherSetting"], true);
        assert_eq!(written["modelPicker"]["options"][0]["model"], "gpt-5.6-terra[1m]");
        assert_eq!(written["modelPicker"]["replaceBuiltInOptions"], true);
    }

    #[test]
    fn apply_override_always_writes_replace_built_in_options_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(
            &path,
            r#"{"modelPicker": {"replaceBuiltInOptions": true, "options": []}}"#,
        )
        .unwrap();
        // A later apply with `false` must actually flip it, not just leave
        // the earlier `true` in place because the key was already present.
        apply_override(&path, &[], false).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["modelPicker"]["replaceBuiltInOptions"], false);
    }
}
