//! Prevents two `serve` processes from sharing the same config directory at
//! once. Two instances racing on the same Codex `auth.json` is a real,
//! observed failure mode: each holds its own in-process refresh lock, so
//! nothing stops them from both refreshing the OAuth token around the same
//! time. Refresh tokens rotate on use, so whichever instance loses the race
//! gets a rejected (already-consumed) refresh token, and the refresh
//! failure path in `providers::codex::auth::manager` deletes the stored
//! auth entirely - wiping out the file the winning instance just wrote.
//!
//! The fix is a single exclusive lock file per config directory, held for
//! the lifetime of the process. A second `serve` against the same
//! directory fails fast instead of silently corrupting the first one's
//! session.
use std::fs::File;
use std::path::Path;

/// Held for as long as this process should be considered "the" instance for
/// its config directory. Dropping it (including on normal process exit)
/// releases the OS-level lock automatically.
pub struct InstanceLock {
    _file: File,
}

pub fn acquire(config_dir: &Path) -> Result<InstanceLock, anyhow::Error> {
    std::fs::create_dir_all(config_dir)?;
    let path = config_dir.join(".proxy.lock");
    let file = open_exclusive(&path).map_err(|_| {
        anyhow::anyhow!(
            "Another claude-code-proxy instance is already running against {} \
             (lock file: {}). Stop it first - two instances sharing the same \
             Codex auth file can race on token refresh and wipe each other's login.",
            config_dir.display(),
            path.display()
        )
    })?;
    Ok(InstanceLock { _file: file })
}

#[cfg(windows)]
fn open_exclusive(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    // share_mode(0): no other process may open this file at all - not even
    // for reading - while we hold it open. That's exactly a single-instance
    // lock, no extra crate needed on Windows.
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .share_mode(0)
        .open(path)
}

#[cfg(unix)]
fn open_exclusive(path: &Path) -> std::io::Result<File> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)?;
    // SAFETY: `file`'s fd is valid for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_lock_in_the_same_dir_fails_while_the_first_is_held() {
        let temp = tempfile::tempdir().unwrap();
        let _first = acquire(temp.path()).expect("first lock succeeds");
        let second = acquire(temp.path());
        assert!(second.is_err());
    }

    #[test]
    fn lock_is_reacquirable_after_being_dropped() {
        let temp = tempfile::tempdir().unwrap();
        {
            let _first = acquire(temp.path()).expect("first lock succeeds");
        }
        let second = acquire(temp.path());
        assert!(second.is_ok());
    }

    #[test]
    fn separate_directories_do_not_contend() {
        let temp_a = tempfile::tempdir().unwrap();
        let temp_b = tempfile::tempdir().unwrap();
        let _a = acquire(temp_a.path()).expect("lock a succeeds");
        let _b = acquire(temp_b.path()).expect("lock b succeeds");
    }
}
