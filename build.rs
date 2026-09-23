fn main() {
    #[cfg(windows)]
    {
        // Embeds the red-octopus icon as a PE resource so `ccp.exe` shows it
        // in Explorer/taskbar/Alt-Tab, and so a Windows Terminal profile
        // whose commandline points at this exe (no explicit "icon" set)
        // picks it up automatically for the tab, same as `claude.exe` does.
        let mut res = winresource::WindowsResource::new();
        res.set_icon("src/assets/icons/octopus.ico");
        res.compile().expect("failed to embed Windows icon resource");
    }
}
