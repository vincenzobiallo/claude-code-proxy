use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DirResolverEnv {
    pub platform: String,
    pub env: HashMap<String, String>,
    pub home: String,
}

impl Default for DirResolverEnv {
    fn default() -> Self {
        Self {
            platform: std::env::consts::OS.into(),
            env: std::env::vars().collect(),
            home: std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| "/".to_string()),
        }
    }
}

pub fn resolve_config_dir(deps: &DirResolverEnv) -> PathBuf {
    if let Some(override_dir) = deps.env.get("CCP_CONFIG_DIR") {
        return Path::new(override_dir).to_path_buf();
    }

    if deps.platform == "win32" {
        let appdata = deps
            .env
            .get("APPDATA")
            .cloned()
            .unwrap_or_else(|| format!("{}\\AppData\\Roaming", deps.home));
        return join_with_sep(&appdata, &["claude-code-proxy"], true);
    }

    if deps.platform == "darwin" {
        return join_with_sep(&deps.home, &[".config", "claude-code-proxy"], false);
    }

    let base = deps.env.get("XDG_CONFIG_HOME").cloned().unwrap_or_else(|| {
        join_with_sep(&deps.home, &[".config"], false)
            .to_string_lossy()
            .into_owned()
    });
    join_with_sep(&base, &["claude-code-proxy"], false)
}

pub fn resolve_state_dir(deps: &DirResolverEnv) -> PathBuf {
    if deps.platform == "win32" {
        let local = deps
            .env
            .get("LOCALAPPDATA")
            .cloned()
            .unwrap_or_else(|| format!("{}\\AppData\\Local", deps.home));
        return join_with_sep(&local, &["claude-code-proxy"], true);
    }

    let base = deps.env.get("XDG_STATE_HOME").cloned().unwrap_or_else(|| {
        join_with_sep(&deps.home, &[".local", "state"], false)
            .to_string_lossy()
            .into_owned()
    });
    join_with_sep(&base, &["claude-code-proxy"], false)
}

/// The pre-`resolve_config_dir` location, still read (and cleared on logout)
/// as a fallback. An explicit `CCP_CONFIG_DIR` override disables that
/// fallback by pointing it back at the override itself: otherwise a process
/// isolated to its own config dir (every test does this) would still read -
/// and, on logout or a 401 token refresh, delete - the real user's
/// `~/.config/claude-code-proxy/<provider>/auth.json`.
pub fn legacy_config_dir(deps: &DirResolverEnv) -> PathBuf {
    if deps.env.contains_key("CCP_CONFIG_DIR") {
        return resolve_config_dir(deps);
    }
    join_with_sep(&deps.home, &[".config", "claude-code-proxy"], false)
}

pub fn config_dir() -> PathBuf {
    resolve_config_dir(&DirResolverEnv::default())
}

pub fn state_dir() -> PathBuf {
    resolve_state_dir(&DirResolverEnv::default())
}

pub fn codex_auth_file(deps: &DirResolverEnv) -> PathBuf {
    resolve_config_dir(deps).join("codex").join("auth.json")
}

pub fn log_file() -> PathBuf {
    resolve_state_dir(&DirResolverEnv::default()).join("proxy.log")
}

/// Disk cache of the last live-fetched Codex model catalog, so a restart
/// doesn't lose what was discovered while offline. Regenerable, so it lives
/// under `state_dir()` alongside `log_file()`, not `config_dir()`.
pub fn codex_model_catalog_cache_file() -> PathBuf {
    resolve_state_dir(&DirResolverEnv::default()).join("codex_model_catalog.json")
}

/// Disk cache of the last live-fetched Anthropic model catalog. See
/// `codex_model_catalog_cache_file` for why this lives under `state_dir()`.
pub fn anthropic_model_catalog_cache_file() -> PathBuf {
    resolve_state_dir(&DirResolverEnv::default()).join("anthropic_model_catalog.json")
}

/// Writes `contents` to a sibling temp file, then renames it over `path`, so
/// a crash mid-write never leaves a truncated file behind (which the cache
/// loaders would otherwise silently discard).
pub fn write_atomic(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn provider_auth_file(provider: &str) -> PathBuf {
    let deps = DirResolverEnv::default();
    resolve_config_dir(&deps).join(provider).join("auth.json")
}

pub fn provider_legacy_auth_file(provider: &str) -> PathBuf {
    let deps = DirResolverEnv::default();
    legacy_config_dir(&deps).join(provider).join("auth.json")
}

/// Where the terminal Models view persists its `[v]` selection and ordering,
/// independent of whether that's ever been applied to `settings.json`.
pub fn model_picker_file() -> PathBuf {
    resolve_config_dir(&DirResolverEnv::default()).join("model_picker.json")
}

/// Claude Code's own settings file - not ours, but where `modelPicker.options`
/// lives, so the TUI's "overwrite settings.json" action targets this path.
pub fn claude_settings_file() -> PathBuf {
    Path::new(&DirResolverEnv::default().home)
        .join(".claude")
        .join("settings.json")
}

fn join_with_sep(base: &str, parts: &[&str], win32: bool) -> PathBuf {
    let sep = '/';
    let _ = win32;
    let mut out = String::new();
    for part in std::iter::once(base).chain(parts.iter().copied()) {
        if !out.is_empty() && !out.ends_with(sep) {
            out.push(sep);
        }
        out.push_str(part);
    }
    Path::new(&out).to_path_buf()
}

pub fn resolve_config_dir_for_env(
    platform: &str,
    home: &str,
    env: &HashMap<String, String>,
) -> PathBuf {
    resolve_config_dir(&DirResolverEnv {
        platform: platform.to_string(),
        env: env.clone(),
        home: home.to_string(),
    })
}

pub fn resolve_state_dir_for_env(
    platform: &str,
    home: &str,
    env: &HashMap<String, String>,
) -> PathBuf {
    resolve_state_dir(&DirResolverEnv {
        platform: platform.to_string(),
        env: env.clone(),
        home: home.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deps(env: &[(&str, &str)]) -> DirResolverEnv {
        DirResolverEnv {
            platform: "linux".to_string(),
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            home: "/home/user".to_string(),
        }
    }

    #[test]
    fn legacy_dir_is_under_home_without_an_override() {
        assert_eq!(
            legacy_config_dir(&deps(&[])),
            PathBuf::from("/home/user/.config/claude-code-proxy")
        );
    }

    #[test]
    fn config_dir_override_also_isolates_the_legacy_dir() {
        let deps = deps(&[("CCP_CONFIG_DIR", "/tmp/isolated")]);
        assert_eq!(legacy_config_dir(&deps), PathBuf::from("/tmp/isolated"));
        assert_eq!(legacy_config_dir(&deps), resolve_config_dir(&deps));
    }
}
