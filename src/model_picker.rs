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

    pub fn position(&self, provider: &str, model: &str) -> Option<usize> {
        let field = picker_model_field(provider, model);
        self.entries.iter().position(|entry| entry.model.eq_ignore_ascii_case(&field))
    }

    pub fn is_enabled(&self, provider: &str, model: &str) -> bool {
        self.position(provider, model).is_some()
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
            .unwrap_or_else(|| PickerEntry {
                model: field,
                label: model.to_string(),
                description: format!("via {provider}"),
            });
        self.entries.push(entry);
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
    if provider == "codex" && !model.to_ascii_lowercase().ends_with("[1m]") {
        format!("{model}[1m]")
    } else {
        model.to_string()
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

/// Overwrites `modelPicker.options` in `~/.claude/settings.json` with
/// `entries`, preserving every other key in the file (including
/// `modelPicker.replaceBuiltInOptions`, if set) and leaving a `.bak` copy of
/// the previous contents next to it before writing.
pub fn apply_override(path: &Path, entries: &[PickerEntry]) -> anyhow::Result<()> {
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
    picker
        .as_object_mut()
        .expect("just ensured it's an object")
        .insert("options".to_string(), serde_json::to_value(entries)?);
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
        apply_override(&path, &entries).unwrap();
        assert!(path.with_extension("json.bak").exists());
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["someOtherSetting"], true);
        assert_eq!(written["modelPicker"]["options"][0]["model"], "gpt-5.6-terra[1m]");
    }
}
