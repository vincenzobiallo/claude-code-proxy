//! Sets this process's console window title and (on Windows) icon, so the
//! terminal identifies itself as "Claude Code Proxy" instead of a generic
//! shell. Best-effort only: a host that ignores these calls just keeps its
//! default title/icon.
const TITLE: &str = "Claude Code Proxy";

#[cfg(windows)]
const ICON_BYTES: &[u8] = include_bytes!("assets/icons/octopus.ico");

pub fn apply() {
    set_title();
    #[cfg(windows)]
    set_icon();
}

/// OSC 0 (set window title) is understood by conhost with VT processing
/// enabled, Windows Terminal, and Unix terminals alike - one escape code
/// covers every platform this app runs on. Skipped when stdout isn't a
/// terminal, so piped/captured output (scripts, tests) stays clean.
fn set_title() {
    use std::io::{IsTerminal, Write};
    let mut stdout = std::io::stdout();
    if !stdout.is_terminal() {
        return;
    }
    print!("\x1b]0;{TITLE}\x07");
    let _ = stdout.flush();
}

#[cfg(windows)]
fn set_icon() {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        HICON, IMAGE_ICON, LR_DEFAULTSIZE, LR_LOADFROMFILE, LoadImageW,
    };

    // `SetConsoleIcon` takes an HICON, not raw bytes, and there's no public
    // API to build one from an in-memory .ico - so the embedded bytes get
    // written to a temp file once per run and loaded from there via
    // `LoadImageW`, which understands the .ico container format directly.
    let path = std::env::temp_dir().join("claude-code-proxy-octopus.ico");
    if std::fs::write(&path, ICON_BYTES).is_err() {
        return;
    }
    let wide_path: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let kernel32_name: Vec<u16> = "kernel32.dll\0".encode_utf16().collect();

    // SAFETY: `wide_path` is a valid null-terminated UTF-16 path to a file
    // this function just wrote. `SetConsoleIcon` is a genuine kernel32
    // export (confirmed via `dumpbin /exports kernel32.dll`) that isn't in
    // the import library windows-sys links against, so it's resolved by
    // hand via `GetProcAddress` on kernel32 (already loaded in every
    // process) instead of a static `#[link]` declaration. The transmute
    // matches the documented `BOOL SetConsoleIcon(HICON)` signature.
    unsafe {
        let hicon: HICON = LoadImageW(
            std::ptr::null_mut(),
            wide_path.as_ptr(),
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        ) as HICON;
        if hicon.is_null() {
            return;
        }
        let kernel32 = GetModuleHandleW(kernel32_name.as_ptr());
        if kernel32.is_null() {
            return;
        }
        let Some(proc) = GetProcAddress(kernel32, b"SetConsoleIcon\0".as_ptr()) else {
            return;
        };
        let set_console_icon: unsafe extern "system" fn(HICON) -> i32 = std::mem::transmute(proc);
        set_console_icon(hicon);
    }
}
