//! Optional integration with the `headroom` CLI (a context-optimization
//! proxy for LLM traffic). When present, `serve` starts `headroom proxy` as
//! a sidecar in front of this app's own Anthropic-compatible server, so
//! Claude Code can point at headroom instead and get compression for free.
//! Headroom then forwards everything it doesn't handle itself to this app,
//! which keeps doing its own provider routing.
use std::{
    io,
    net::{TcpStream, ToSocketAddrs},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

pub const DEFAULT_PORT: u16 = 8787;

pub fn port() -> u16 {
    std::env::var("HEADROOM_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// True if something is already accepting connections on `host:port` - used
/// both for the monitor's status dot and for `code` deciding whether to send
/// Claude Code through headroom.
pub fn reachable(host: &str, port: u16) -> bool {
    (host, port)
        .to_socket_addrs()
        .ok()
        .and_then(|mut addrs| addrs.next())
        .is_some_and(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok())
}

/// What the footer's status dot shows. Computed by a background watcher
/// thread (see `spawn_watcher`), never on the render path: `reachable` is a
/// blocking TCP connect, and on Windows a refused localhost connect waits out
/// most of its timeout - doing that on every frame is what made the monitor
/// stutter while headroom was still starting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadroomStatus {
    Starting,
    Ready,
    Down,
}

impl HeadroomStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Ready,
            2 => Self::Down,
            _ => Self::Starting,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::Starting => 0,
            Self::Ready => 1,
            Self::Down => 2,
        }
    }
}

/// The sidecar process we spawned. Killed when the last `HeadroomHandle`
/// clone goes away, not when any single clone is dropped.
struct OwnedChild(Mutex<Child>);

impl OwnedChild {
    fn kill(&self) {
        if let Ok(mut child) = self.0.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn has_exited(&self) -> bool {
        self.0
            .lock()
            .ok()
            .is_none_or(|mut child| !matches!(child.try_wait(), Ok(None)))
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        self.kill();
    }
}

#[derive(Clone)]
pub struct HeadroomHandle {
    /// `None` means this handle wraps a headroom instance we did not spawn
    /// (the user chose "connect to the existing one" on a port conflict) -
    /// we only ever observe it, never kill it.
    child: Option<Arc<OwnedChild>>,
    status: Arc<AtomicU8>,
    pub port: u16,
}

impl HeadroomHandle {
    /// Wraps a headroom instance already running on `port` that this
    /// process didn't start. `shutdown` is a no-op for it.
    pub fn external(port: u16) -> Self {
        Self::with_watcher(None, port)
    }

    fn with_watcher(child: Option<Arc<OwnedChild>>, port: u16) -> Self {
        let status = Arc::new(AtomicU8::new(HeadroomStatus::Starting.as_u8()));
        spawn_watcher(Arc::downgrade(&status), child.as_ref().map(Arc::downgrade), port);
        Self { child, status, port }
    }

    pub fn dashboard_url(&self) -> String {
        format!("http://127.0.0.1:{}/dashboard", self.port)
    }

    /// Last status observed by the background watcher. Never blocks, so it's
    /// safe to call on every frame.
    pub fn status(&self) -> HeadroomStatus {
        HeadroomStatus::from_u8(self.status.load(Ordering::Relaxed))
    }

    /// Kills the sidecar, unless this handle wraps a process we don't own.
    /// Safe to call more than once - killing an already-dead process is a
    /// harmless no-op.
    pub fn shutdown(&self) {
        if let Some(child) = &self.child {
            child.kill();
        }
    }
}

/// Polls headroom's state off the UI thread. Exits on its own once every
/// `HeadroomHandle` clone (and with it the shared status cell) is dropped.
fn spawn_watcher(status: Weak<AtomicU8>, child: Option<Weak<OwnedChild>>, port: u16) {
    let _ = std::thread::Builder::new()
        .name("headroom-watcher".to_string())
        .spawn(move || {
            loop {
                let Some(cell) = status.upgrade() else {
                    return;
                };
                let current = HeadroomStatus::from_u8(cell.load(Ordering::Relaxed));
                let exited = child
                    .as_ref()
                    .is_some_and(|child| child.upgrade().is_none_or(|child| child.has_exited()));
                let next = if exited {
                    HeadroomStatus::Down
                } else if reachable("127.0.0.1", port) {
                    HeadroomStatus::Ready
                } else if child.is_some() && current == HeadroomStatus::Starting {
                    // Our own process is alive but not listening yet: still
                    // importing its ML deps.
                    HeadroomStatus::Starting
                } else {
                    HeadroomStatus::Down
                };
                cell.store(next.as_u8(), Ordering::Relaxed);
                drop(cell);
                let pause = if next == HeadroomStatus::Starting { 500 } else { 2000 };
                std::thread::sleep(Duration::from_millis(pause));
            }
        });
}

/// Starts `headroom proxy` pointed at `upstream_url` (this app's own
/// Anthropic-compatible server) as a detached, lower-priority background
/// process, so its heavy startup (importing tokenizer/ML deps) doesn't
/// compete with the monitor for CPU. Returns `Ok(None)` when the `headroom`
/// binary isn't on PATH; other spawn failures are returned as `Err`.
pub fn spawn(port: u16, upstream_url: &str) -> io::Result<Option<HeadroomHandle>> {
    let mut command = Command::new("headroom");
    command
        .args([
            "proxy",
            "--port",
            &port.to_string(),
            "--anthropic-api-url",
            upstream_url,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_background(&mut command);
    match command.spawn() {
        Ok(child) => Ok(Some(HeadroomHandle::with_watcher(
            Some(Arc::new(OwnedChild(Mutex::new(child)))),
            port,
        ))),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

/// No console window, its own process group (the terminal's Ctrl+C reaches
/// only us - headroom is killed explicitly on shutdown), below-normal
/// priority.
#[cfg(windows)]
fn configure_background(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS);
}

/// Its own process group (terminal signals reach only us), at a lower
/// scheduling priority.
#[cfg(unix)]
fn configure_background(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
    // SAFETY: `nice` is async-signal-safe and only affects the child.
    unsafe {
        command.pre_exec(|| {
            libc::nice(10);
            Ok(())
        });
    }
}

/// Best-effort: kills whatever process is listening on `port`. Shells out to
/// platform tools (`netstat`/`taskkill` on Windows, `lsof`/`kill` on Unix)
/// rather than pulling in a process-inspection crate for this one path.
#[cfg(windows)]
pub fn kill_process_on_port(port: u16) -> io::Result<()> {
    let output = Command::new("netstat").args(["-ano"]).output()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let needle = format!(":{port} ");
    let mut found = false;
    for line in text.lines() {
        if line.contains("LISTENING")
            && line.contains(&needle)
            && let Some(pid) = line.split_whitespace().last()
        {
            let _ = Command::new("taskkill")
                .args(["/PID", pid, "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            found = true;
        }
    }
    if found {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no listener found on port {port}"),
        ))
    }
}

#[cfg(unix)]
pub fn kill_process_on_port(port: u16) -> io::Result<()> {
    let output = Command::new("lsof")
        .args(["-t", "-i", &format!(":{port}")])
        .output()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let mut found = false;
    for pid in text.split_whitespace() {
        let _ = Command::new("kill")
            .args(["-9", pid])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        found = true;
    }
    if found {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no listener found on port {port}"),
        ))
    }
}
