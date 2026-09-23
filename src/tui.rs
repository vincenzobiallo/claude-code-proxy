//! Minimal terminal dashboard: current sessions, active requests, cumulative
//! token consumption, provider auth status (with login from the monitor),
//! and the list of available models. Runs in-process alongside the server
//! (see `main.rs::run_serve_with_tui`) and reads the same `MonitorHandle`
//! and `Registry` the HTML dashboard uses over HTTP - the web UI stays
//! reachable at `/dashboard` the whole time this is on screen.
use std::{
    collections::BTreeMap,
    io::{self, Stdout},
    sync::mpsc,
    time::Duration,
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Cell, Clear, Padding, Paragraph, Row, Table, Wrap},
};
use tokio::sync::oneshot;

use crate::{
    headroom::{HeadroomHandle, HeadroomStatus},
    model_picker::{self, PickerSelection},
    monitor::{ActiveRequest, MonitorHandle, MonitorState, SessionSummary, UsageWindow},
    paths,
    registry::Registry,
};

// Same palette as the HTML dashboard's `:root` custom properties
// (dashboard.html) - `--accent` there is the dashboard's red, so this stays
// visually consistent between the web UI and the terminal one.
// Claude Code's own status line uses ANSI bright red (`\x1b[91m`) for its
// "near limit" warning, not a fixed hex - it renders as whatever the user's
// terminal theme defines for that code. `Color::LightRed` is ratatui's name
// for that same ANSI 91, so this matches Claude Code exactly instead of
// approximating it with an arbitrary RGB.
const ACCENT: Color = Color::LightRed;
const ACCENT_FG: Color = Color::Rgb(24, 10, 9); // dark text for contrast on the accent background
const TEXT: Color = Color::Rgb(236, 236, 238); // --text #ececee
const HEADING: Color = Color::Rgb(215, 215, 220); // --heading #d7d7dc
const BORDER: Color = Color::Rgb(51, 51, 56); // --border #333338
const BG: Color = Color::Rgb(19, 19, 21); // --bg #131315
const PANEL: Color = Color::Rgb(25, 25, 28); // --panel #19191c
const PANEL_ALT: Color = Color::Rgb(32, 32, 36); // --panel-alt #202024
const OK: Color = Color::Rgb(123, 196, 127); // --ok #7bc47f
const BAD: Color = Color::Rgb(239, 91, 82); // --bad #ef5b52
const WARN: Color = Color::Rgb(223, 168, 63); // --warn #dfa83f
const INFO: Color = Color::Rgb(126, 163, 217); // --info #7ea3d9
const PURPLE: Color = Color::Rgb(201, 127, 214); // .status-compacting #c97fd6
const DIM: Color = Color::Rgb(156, 156, 164); // --dim #9c9ca4

// A terminal can't render the dashboard's real provider logos (PNGs), so a
// colored bullet stands in for one: coral for Anthropic (its brand clay
// color), teal-green for Codex/OpenAI.
const ANTHROPIC_COLOR: Color = Color::Rgb(217, 119, 87);
const CODEX_COLOR: Color = Color::Rgb(65, 200, 165);

fn provider_color(name: &str) -> Color {
    match name {
        "anthropic" => ANTHROPIC_COLOR,
        "codex" => CODEX_COLOR,
        _ => DIM,
    }
}

fn provider_dot(name: &str) -> Span<'static> {
    Span::styled("\u{25cf} ", Style::default().fg(provider_color(name)))
}

const SPINNER_FRAMES: [char; 10] = ['\u{280b}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283c}', '\u{2834}', '\u{2826}', '\u{2827}', '\u{2807}', '\u{280f}'];

/// A wall-clock-driven animation frame, not a stored counter - the TUI
/// redraws on its own poll cadence (see `run_events`), so sampling the clock
/// each render is simpler than threading extra state through `MonitorApp`.
fn spinner_frame() -> char {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    SPINNER_FRAMES[((millis / 100) % SPINNER_FRAMES.len() as u128) as usize]
}

/// The registry key is the internal id ("anthropic", "codex"); the product
/// people actually recognize is "Claude" (Anthropic) and "Codex" (OpenAI).
/// Only used for display - lookups still go through the raw registry key.
fn provider_label(name: &str) -> &str {
    match name {
        "anthropic" => "claude",
        other => other,
    }
}

/// A rounded, dark panel block with an accent border/title when it holds the
/// interactive list for the current view, or a dim border otherwise -
/// mirrors the HTML dashboard's palette.
fn panel(title: String, focused: bool) -> Block<'static> {
    let color = if focused { ACCENT } else { BORDER };
    Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(if focused { ACCENT } else { HEADING })
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(color))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(PANEL).fg(TEXT))
}

fn header_row(cells: &[&str]) -> Row<'static> {
    Row::new(
        cells
            .iter()
            .map(|cell| Cell::from(Span::styled(cell.to_string(), Style::default().fg(ACCENT))))
            .collect::<Vec<_>>(),
    )
    .style(Style::default().add_modifier(Modifier::BOLD))
}

pub struct MonitorUiConfig<'a> {
    pub listen_url: String,
    pub registry: &'a Registry,
    pub headroom: Option<HeadroomHandle>,
    /// The proxy's own runtime: the "r" model refresh runs there, so the
    /// shared HTTP clients' pooled connections always belong to a runtime
    /// that outlives them.
    pub runtime: tokio::runtime::Handle,
    pub shutdown: Option<oneshot::Sender<()>>,
    pub shutdown_complete: Option<mpsc::Receiver<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorExit {
    ShutdownComplete,
    ForceQuit,
}

pub fn run_monitor(
    handle: MonitorHandle,
    config: MonitorUiConfig<'_>,
) -> Result<MonitorExit, anyhow::Error> {
    let mut terminal = setup_terminal()?;
    let _guard = TerminalGuard;
    let mut app = MonitorApp {
        listen_url: config.listen_url,
        registry: config.registry,
        headroom: config.headroom,
        phase: MonitorPhase::Running,
        view: View::Activity,
        show_help: false,
        session_offset: 0,
        provider_offset: 0,
        model_offset: 0,
        model_scroll: 0,
        picker: PickerSelection::load(&paths::model_picker_file()),
        show_picker_json: false,
        confirm_override: false,
        picker_error: None,
        picker_status: None,
        alias_editor: None,
        model_refresh: ModelRefresh::new(config.runtime),
        shutdown: config.shutdown,
        shutdown_complete: config.shutdown_complete,
    };

    let run_result = run_events(&mut terminal, &handle, &mut app);
    if run_result.is_err() {
        app.begin_shutdown();
        let state = handle.snapshot();
        let _ = terminal.draw(|frame| render(frame, &mut app, &state));
        app.wait_for_shutdown_completion();
    }
    let cursor_result = terminal.show_cursor();
    let exit = run_result?;
    cursor_result?;
    Ok(exit)
}

fn run_events(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    handle: &MonitorHandle,
    app: &mut MonitorApp<'_>,
) -> Result<MonitorExit, anyhow::Error> {
    loop {
        app.model_refresh.poll();
        let state = handle.snapshot();
        app.clamp_selection(&state);
        terminal.draw(|frame| render(frame, app, &state))?;
        if app.shutdown_is_complete() {
            return Ok(MonitorExit::ShutdownComplete);
        }
        if event::poll(Duration::from_millis(250))?
            && let Event::Key(key) = event::read()?
            // Windows' Console API reports both key-down and key-up as
            // separate `Event::Key`s (unlike most Unix terminals, which only
            // ever send press). Without this filter every key press fires
            // its action twice there - e.g. one `Tab` skips a whole view.
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if app.handle_ctrl_c() {
                        return Ok(MonitorExit::ForceQuit);
                    }
                }
                _ if app.phase == MonitorPhase::ShuttingDown => {}
                KeyCode::Esc => {
                    if app.alias_editor.is_some() {
                        app.alias_editor = None;
                    } else if app.confirm_override {
                        app.confirm_override = false;
                    } else if app.show_picker_json {
                        app.show_picker_json = false;
                        app.picker_error = None;
                    } else if app.show_help {
                        app.show_help = false;
                    } else if app.phase == MonitorPhase::ConfirmingShutdown {
                        app.phase = MonitorPhase::Running;
                    }
                }
                _ if app.alias_editor.is_some() => app.handle_alias_editor_key(key.code),
                KeyCode::Char('y') if app.confirm_override => {
                    app.apply_picker_override();
                    app.confirm_override = false;
                }
                KeyCode::Char('n') if app.confirm_override => {
                    app.confirm_override = false;
                }
                _ if app.confirm_override => {}
                KeyCode::Char('y') if app.phase == MonitorPhase::ConfirmingShutdown => {
                    app.begin_shutdown()
                }
                KeyCode::Char('n') if app.phase == MonitorPhase::ConfirmingShutdown => {
                    app.phase = MonitorPhase::Running
                }
                _ if app.phase == MonitorPhase::ConfirmingShutdown => {}
                KeyCode::Char('?') => app.show_help = !app.show_help,
                KeyCode::Char('q') => app.phase = MonitorPhase::ConfirmingShutdown,
                KeyCode::Tab => app.view = app.view.next(),
                KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
                KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
                KeyCode::Char('l') if app.view == View::Providers => {
                    if let Some(name) = app.selected_provider_name() {
                        run_login_flow(terminal, app.registry, &name)?;
                    }
                }
                KeyCode::Char(' ') if app.view == View::Models && !app.show_picker_json => {
                    app.toggle_selected_model();
                }
                KeyCode::Char('a') if app.view == View::Models && !app.show_picker_json => {
                    app.open_alias_editor();
                }
                KeyCode::Char('r') if app.view == View::Models && !app.show_picker_json => {
                    app.model_refresh.start();
                }
                KeyCode::Char('b') if app.view == View::Models && !app.show_picker_json => {
                    app.toggle_replace_built_in_options();
                }
                KeyCode::Char('[') if app.view == View::Models && !app.show_picker_json => {
                    app.move_selected_model(-1);
                }
                KeyCode::Char(']') if app.view == View::Models && !app.show_picker_json => {
                    app.move_selected_model(1);
                }
                KeyCode::Char('p') if app.view == View::Models => {
                    app.show_picker_json = !app.show_picker_json;
                    app.picker_error = None;
                }
                KeyCode::Char('e') if app.show_picker_json => {
                    let json = app.picker.to_json_pretty();
                    if let Some(text) = run_json_editor_flow(terminal, &json)? {
                        match PickerSelection::from_json(&text) {
                            Ok(entries) => {
                                app.picker.entries = entries;
                                let _ = app.picker.save(&paths::model_picker_file());
                                app.picker_error = None;
                                app.picker_status = Some("Edited JSON applied.".to_string());
                            }
                            Err(err) => app.picker_error = Some(format!("Invalid JSON: {err}")),
                        }
                    }
                }
                KeyCode::Char('o') if app.show_picker_json => {
                    app.confirm_override = true;
                }
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MonitorPhase {
    Running,
    ConfirmingShutdown,
    ShuttingDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Activity,
    Providers,
    Models,
}

impl View {
    const ALL: [View; 3] = [View::Activity, View::Providers, View::Models];

    fn next(self) -> Self {
        match self {
            View::Activity => View::Providers,
            View::Providers => View::Models,
            View::Models => View::Activity,
        }
    }

    fn label(self) -> &'static str {
        match self {
            View::Activity => "Activity",
            View::Providers => "Providers",
            View::Models => "Models",
        }
    }
}

struct MonitorApp<'a> {
    listen_url: String,
    registry: &'a Registry,
    headroom: Option<HeadroomHandle>,
    phase: MonitorPhase,
    view: View,
    show_help: bool,
    session_offset: usize,
    provider_offset: usize,
    model_offset: usize,
    model_scroll: usize,
    picker: PickerSelection,
    show_picker_json: bool,
    confirm_override: bool,
    picker_error: Option<String>,
    picker_status: Option<String>,
    alias_editor: Option<AliasEditorState>,
    model_refresh: ModelRefresh,
    shutdown: Option<oneshot::Sender<()>>,
    shutdown_complete: Option<mpsc::Receiver<()>>,
}

/// The Models view's "r" refresh of every provider's live model catalog.
/// Spawned on the proxy's runtime so the UI loop never waits on the network;
/// `poll` picks the summary up once it's done.
struct ModelRefresh {
    runtime: tokio::runtime::Handle,
    pending: Option<mpsc::Receiver<String>>,
    last: Option<String>,
}

impl ModelRefresh {
    fn new(runtime: tokio::runtime::Handle) -> Self {
        Self {
            runtime,
            pending: None,
            last: None,
        }
    }

    fn start(&mut self) {
        if self.pending.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.runtime.spawn(async move {
            let _ = tx.send(crate::providers::refresh_model_catalogs().await);
        });
        self.pending = Some(rx);
    }

    fn poll(&mut self) {
        let Some(rx) = &self.pending else {
            return;
        };
        match rx.try_recv() {
            Ok(summary) => {
                self.last = Some(summary);
                self.pending = None;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.last = Some("refresh failed (worker exited)".to_string());
                self.pending = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn status(&self) -> Option<String> {
        if self.pending.is_some() {
            Some(format!("{} refreshing models...", spinner_frame()))
        } else {
            self.last.clone()
        }
    }
}

/// State for the "a" alias editor popup (Models view): the alias/label and
/// whether to request the 1M-context (`[1m]`) variant for the model under
/// the cursor when it was opened.
struct AliasEditorState {
    provider: String,
    model: String,
    label: String,
    use_1m: bool,
}

impl<'a> MonitorApp<'a> {
    /// Ctrl+C never shuts down on a single, unconfirmed press - a stray key
    /// combo (or a copy that lands wrong in a Windows console) would otherwise
    /// kill live sessions with no warning. First press asks for confirmation,
    /// same prompt as `q`; a second press while that prompt (or an
    /// already-running graceful shutdown) is up confirms/forces it, so
    /// "ctrl+c ctrl+c" is still the fast path when a graceful shutdown hangs
    /// on a stuck request.
    fn handle_ctrl_c(&mut self) -> bool {
        match self.phase {
            MonitorPhase::ShuttingDown => true,
            MonitorPhase::ConfirmingShutdown => {
                self.begin_shutdown();
                false
            }
            MonitorPhase::Running => {
                self.phase = MonitorPhase::ConfirmingShutdown;
                false
            }
        }
    }

    fn begin_shutdown(&mut self) {
        if self.phase == MonitorPhase::ShuttingDown {
            return;
        }
        self.phase = MonitorPhase::ShuttingDown;
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    fn shutdown_is_complete(&self) -> bool {
        let Some(shutdown_complete) = &self.shutdown_complete else {
            return self.phase == MonitorPhase::ShuttingDown;
        };
        match shutdown_complete.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => true,
            Err(mpsc::TryRecvError::Empty) => false,
        }
    }

    fn wait_for_shutdown_completion(&self) {
        if !self.shutdown_is_complete()
            && let Some(shutdown_complete) = &self.shutdown_complete
        {
            let _ = shutdown_complete.recv();
        }
    }

    fn clamp_selection(&mut self, state: &MonitorState) {
        self.session_offset = self.session_offset.min(state.sessions.len().saturating_sub(1));
        self.provider_offset = self
            .provider_offset
            .min(self.registry.list_provider_names().len().saturating_sub(1));
        let model_count: usize = self.registry.grouped_models().values().map(Vec::len).sum();
        self.model_offset = self.model_offset.min(model_count.saturating_sub(1));
    }

    fn move_selection(&mut self, delta: i64) {
        let offset = match self.view {
            View::Activity => &mut self.session_offset,
            View::Providers => &mut self.provider_offset,
            View::Models => &mut self.model_offset,
        };
        *offset = offset.saturating_add_signed(delta as isize);
    }

    fn selected_provider_name(&self) -> Option<String> {
        self.registry
            .list_provider_names()
            .get(self.provider_offset)
            .cloned()
    }

    fn toggle_selected_model(&mut self) {
        let ordered = ordered_models(self.registry, &self.picker);
        let Some((provider, model)) = ordered.get(self.model_offset).cloned() else {
            return;
        };
        let existing = model_picker::read_existing_entries(&paths::claude_settings_file());
        self.picker.toggle(&provider, &model, &existing);
        let _ = self.picker.save(&paths::model_picker_file());
    }

    /// Opens the "a" alias editor for the model under the cursor, pre-filled
    /// with its current alias (or the curated suggestion, or the bare model
    /// id) and its current 1M-context choice.
    fn open_alias_editor(&mut self) {
        let ordered = ordered_models(self.registry, &self.picker);
        let Some((provider, model)) = ordered.get(self.model_offset).cloned() else {
            return;
        };
        let (preview_label, _description, _live) = self.picker.preview(&provider, &model);
        let label = if preview_label.is_empty() { model.clone() } else { preview_label };
        let use_1m = self.picker.use_1m(&provider, &model);
        self.alias_editor = Some(AliasEditorState { provider, model, label, use_1m });
    }

    /// Routes a keypress to the open alias editor. `Esc` (cancel, handled
    /// alongside the app's other popups in `run_events`) is the only key not
    /// funneled through here.
    fn handle_alias_editor_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Enter => {
                let Some(editor) = self.alias_editor.take() else { return };
                let label = if editor.label.trim().is_empty() {
                    editor.model.clone()
                } else {
                    editor.label.trim().to_string()
                };
                self.picker.set_alias(&editor.provider, &editor.model, label, editor.use_1m);
                let _ = self.picker.save(&paths::model_picker_file());
            }
            KeyCode::Tab => {
                if let Some(editor) = self.alias_editor.as_mut() {
                    editor.use_1m = !editor.use_1m;
                }
            }
            KeyCode::Backspace => {
                if let Some(editor) = self.alias_editor.as_mut() {
                    editor.label.pop();
                }
            }
            KeyCode::Char(ch) => {
                if let Some(editor) = self.alias_editor.as_mut() {
                    editor.label.push(ch);
                }
            }
            _ => {}
        }
    }

    /// Reordering only makes sense while the cursor sits on an already-enabled
    /// row - `ordered_models` always places those first, in `picker.entries`
    /// order, so that's exactly the rows with index `< picker.entries.len()`.
    fn move_selected_model(&mut self, delta: i64) {
        if self.model_offset >= self.picker.entries.len() {
            return;
        }
        self.picker.move_by(self.model_offset, delta);
        let _ = self.picker.save(&paths::model_picker_file());
        let last = self.picker.entries.len().saturating_sub(1) as i64;
        self.model_offset = (self.model_offset as i64 + delta).clamp(0, last) as usize;
    }

    fn apply_picker_override(&mut self) {
        match model_picker::apply_override(
            &paths::claude_settings_file(),
            &self.picker.entries,
            self.picker.replace_built_in_options,
        ) {
            Ok(()) => {
                self.picker_status = Some(format!(
                    "Wrote {} entries to {} (replaceBuiltInOptions: {})",
                    self.picker.entries.len(),
                    paths::claude_settings_file().display(),
                    self.picker.replace_built_in_options
                ));
                self.picker_error = None;
            }
            Err(err) => self.picker_error = Some(err.to_string()),
        }
    }

    /// Toggles whether an override (`o`) replaces Claude Code's native
    /// Opus/Sonnet/Haiku/Fable `/model` entries or just adds `entries`
    /// alongside them (`modelPicker.replaceBuiltInOptions`).
    fn toggle_replace_built_in_options(&mut self) {
        self.picker.replace_built_in_options = !self.picker.replace_built_in_options;
        let _ = self.picker.save(&paths::model_picker_file());
    }
}

/// Models listed enabled-first (in the user's chosen `picker` order), then the
/// rest grouped by provider - shared by rendering and by the toggle/reorder
/// actions so "the row under the cursor" means the same thing everywhere.
fn ordered_models(registry: &Registry, picker: &PickerSelection) -> Vec<(String, String)> {
    let grouped = registry.grouped_models();
    let mut enabled = Vec::new();
    for entry in &picker.entries {
        let base = crate::registry::normalize_incoming_model(&entry.model);
        if let Some(provider) = grouped
            .iter()
            .find(|(_, models)| models.iter().any(|model| model == &base))
            .map(|(provider, _)| provider.clone())
        {
            enabled.push((provider, base));
        }
    }
    let mut rest = Vec::new();
    for (provider, models) in &grouped {
        for model in models {
            if !enabled.iter().any(|(p, m)| p == provider && m == model) {
                rest.push((provider.clone(), model.clone()));
            }
        }
    }
    enabled.into_iter().chain(rest).collect()
}

impl Drop for MonitorApp<'_> {
    fn drop(&mut self) {
        self.begin_shutdown();
    }
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>, anyhow::Error> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

/// Suspends the alternate screen to run a provider's (blocking, printing)
/// login flow as a normal foreground process, then restores the TUI.
/// Necessary because raw mode/the alt-screen buffer isn't a real terminal as
/// far as `println!`-based flows are concerned - it needs the "real" stdout.
fn run_login_flow(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    registry: &Registry,
    provider_name: &str,
) -> Result<(), anyhow::Error> {
    let Some(provider) = registry.provider(provider_name) else {
        return Ok(());
    };
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    println!("\n--- {provider_name} login ---\n");
    match provider.cli().login() {
        Ok(()) => println!("\nLogin succeeded."),
        Err(err) => println!("\nLogin failed: {err}"),
    }
    println!("\nPress Enter to return to the monitor...");
    let mut discard = String::new();
    let _ = io::stdin().read_line(&mut discard);
    execute!(io::stdout(), EnterAlternateScreen)?;
    enable_raw_mode()?;
    terminal.clear()?;
    Ok(())
}

/// Suspends the alternate screen to let the user hand-edit `json_text` in
/// their own editor ($VISUAL, then $EDITOR, then a platform default), the
/// same way `run_login_flow` suspends it for a blocking login prompt. Returns
/// the file's contents if the editor exited successfully, or `None` if it
/// failed to launch or exited with an error status (previous JSON is kept).
fn run_json_editor_flow(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    json_text: &str,
) -> Result<Option<String>, anyhow::Error> {
    let path = std::env::temp_dir().join(format!("ccp-model-picker-{}.json", std::process::id()));
    std::fs::write(&path, json_text)?;
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;

    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| if cfg!(windows) { "notepad".to_string() } else { "vi".to_string() });
    println!("\nOpening '{editor}' on {}...\n", path.display());
    let mut parts = editor.split_whitespace();
    let result = match parts.next() {
        Some(program) => {
            let mut command = std::process::Command::new(program);
            command.args(parts).arg(&path);
            match command.status() {
                Ok(status) if status.success() => {
                    Some(std::fs::read_to_string(&path).unwrap_or(json_text.to_string()))
                }
                Ok(status) => {
                    println!("Editor exited with {status}; keeping the previous JSON.");
                    None
                }
                Err(err) => {
                    println!("Could not launch '{editor}': {err}");
                    None
                }
            }
        }
        None => None,
    };

    println!("\nPress Enter to return to the monitor...");
    let mut discard = String::new();
    let _ = io::stdin().read_line(&mut discard);
    let _ = std::fs::remove_file(&path);
    execute!(io::stdout(), EnterAlternateScreen)?;
    enable_raw_mode()?;
    terminal.clear()?;
    Ok(result)
}

fn render(frame: &mut ratatui::Frame<'_>, app: &mut MonitorApp<'_>, state: &MonitorState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG)), area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area);

    render_banner(frame, root[0], &*app);
    render_header(frame, root[1], &*app, state);
    match app.view {
        View::Activity => render_activity(frame, root[2], state, app.session_offset),
        View::Providers => render_providers(frame, root[2], &*app),
        View::Models => render_models(
            frame,
            root[2],
            app.registry,
            &app.picker,
            app.model_offset,
            &mut app.model_scroll,
            app.model_refresh.status(),
        ),
    }
    render_footer(frame, root[3], &*app);
    if let Some(editor) = &app.alias_editor {
        render_alias_editor(frame, area, editor);
    } else if app.show_picker_json {
        render_picker_json(frame, area, &*app);
    } else if app.show_help {
        render_help(frame, area);
    }
}

fn render_alias_editor(frame: &mut ratatui::Frame<'_>, area: Rect, editor: &AliasEditorState) {
    let popup = centered_rect(60, 30, area);
    frame.render_widget(Clear, popup);

    let checkbox = if editor.use_1m { "[x]" } else { "[ ]" };
    let lines = vec![
        Line::from(vec![
            Span::styled("Alias: ", Style::default().fg(HEADING)),
            Span::styled(editor.label.clone(), Style::default().fg(TEXT)),
            Span::styled("\u{2588}", Style::default().fg(ACCENT)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{checkbox} "), Style::default().fg(if editor.use_1m { OK } else { DIM })),
            Span::styled("Use 1M context variant ([1m])", Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "enter: save   tab: toggle 1m   esc: cancel",
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().fg(TEXT)).block(panel(
            format!(" Set alias - {}/{} ", editor.provider, editor.model),
            true,
        )),
        popup,
    );
}

fn render_banner(frame: &mut ratatui::Frame<'_>, area: Rect, app: &MonitorApp<'_>) {
    let connected = |provider: &str| {
        app.registry
            .provider(provider)
            .is_some_and(|p| p.cli().status_text().is_ok())
    };
    let provider_status = |provider: &str, label: &str| {
        let color = if connected(provider) { OK } else { BAD };
        vec![
            Span::styled("\u{25cf} ", Style::default().fg(color)),
            Span::styled(label.to_string(), Style::default().fg(TEXT)),
        ]
    };

    let mut providers_line = vec![Span::styled(
        "Providers: ",
        Style::default().fg(HEADING),
    )];
    providers_line.extend(provider_status("anthropic", "Anthropic"));
    providers_line.push(Span::raw("  "));
    providers_line.extend(provider_status("codex", "Codex"));

    // Headroom is a sidecar proxy in front of this one, not an inference
    // provider, so it doesn't belong in the "Providers:" list above - its
    // status dot (and, once ready, its dashboard link) live in the footer
    // instead (see `render_footer` / `headroom_status_spans`).
    let tail_spans = vec![Span::styled(
        "  \u{259d}\u{259d} \u{259d}\u{259d}",
        Style::default().fg(ACCENT),
    )];

    let lines = vec![
        Line::from(vec![
            Span::styled(" \u{2590}\u{259b}\u{2588}\u{2588}\u{2588}\u{259b}\u{2588}   ", Style::default().fg(ACCENT)),
            Span::styled(
                format!("Claude Code Proxy v{}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(
            [Span::styled("\u{259d}\u{259c}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2580}  ", Style::default().fg(ACCENT))]
                .into_iter()
                .chain(providers_line)
                .collect::<Vec<_>>(),
        ),
        Line::from(tail_spans),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_help(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let popup = centered_rect(60, 60, area);
    frame.render_widget(Clear, popup);

    let lines = vec![
        Line::from(Span::styled(
            "Keyboard",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("tab            switch view (Activity / Providers / Models)"),
        Line::from("up/down, j/k   scroll or select in the current view"),
        Line::from("l              (Providers) log in to the selected provider"),
        Line::from("space          (Models) toggle the selected model in/out of modelPicker.options"),
        Line::from("a              (Models) set the selected model's alias + 1M-context variant"),
        Line::from("[ ]            (Models) reorder an enabled model within the list"),
        Line::from("r              (Models) refresh the live model lists of every provider"),
        Line::from("b              (Models) toggle replaceBuiltInOptions (hide the native /model list)"),
        Line::from("p              (Models) preview modelPicker.options JSON"),
        Line::from("e              (JSON preview) edit the JSON in $VISUAL/$EDITOR"),
        Line::from("o              (JSON preview) overwrite ~/.claude/settings.json (confirm y/n)"),
        Line::from("q              ask for confirmation, then shut down the proxy"),
        Line::from("ctrl+c         shut down; press twice to force quit"),
        Line::from("?              toggle this help"),
        Line::from("esc            close this help / cancel shutdown prompt"),
        Line::from(""),
        Line::from(Span::styled(
            "CLI commands",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from("serve [--port N] [--no-monitor] [--no-headroom]   start the proxy"),
        Line::from("code [args...]                     open a claude session against it"),
        Line::from("models [--full]                    list supported models"),
        Line::from("codex auth login|device|status|logout"),
        Line::from("version | --version | -v"),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().fg(TEXT))
            .block(panel(" Help (esc or ? to close) ".to_string(), true)),
        popup,
    );
}

/// Centers a `percent_x` x `percent_y` box within `area` (standard ratatui
/// popup pattern).
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, app: &MonitorApp<'_>, state: &MonitorState) {
    let uptime = format_duration(state.uptime);
    let base = Style::default().fg(ACCENT_FG).bg(ACCENT);
    let mut spans = vec![
        Span::styled(" claude-code-proxy", base.add_modifier(Modifier::BOLD)),
        Span::styled("  ", base),
        Span::styled(app.listen_url.clone(), base),
        Span::styled("  uptime ", base),
        Span::styled(uptime, base.add_modifier(Modifier::BOLD)),
        Span::styled("  sessions ", base),
        Span::styled(state.sessions.len().to_string(), base.add_modifier(Modifier::BOLD)),
        Span::styled("  active ", base),
        Span::styled(state.active.len().to_string(), base.add_modifier(Modifier::BOLD)),
        Span::styled("   ", base),
    ];
    for view in View::ALL {
        let style = if view == app.view {
            Style::default().fg(ACCENT).bg(BG).add_modifier(Modifier::BOLD)
        } else {
            base.add_modifier(Modifier::DIM)
        };
        spans.push(Span::styled(format!(" {} ", view.label()), style));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(base),
        area,
    );
}

fn render_activity(frame: &mut ratatui::Frame<'_>, area: Rect, state: &MonitorState, session_offset: usize) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(28),
            Constraint::Length(8),
        ])
        .split(area);
    render_sessions(frame, rows[0], &state.sessions, session_offset);
    render_active(frame, rows[1], &state.active);
    render_consumption(frame, rows[2], &state.sessions, &state.usage_windows);
}

fn render_sessions(frame: &mut ratatui::Frame<'_>, area: Rect, sessions: &[SessionSummary], offset: usize) {
    let header = header_row(&["SESSION", "PROVIDER / MODEL", "REQ", "IN", "OUT", "RATE", "LAST"]);

    let rows = sessions.iter().skip(offset).map(|session| {
        Row::new(vec![
            Cell::from(truncate(&session.label(), 22)),
            provider_model_cell(&session.provider, &session.model, 26),
            Cell::from(format!("{}/{}", session.request_count, session.active_count)),
            Cell::from(format_tokens(session.input_tokens)),
            Cell::from(format_tokens(session.output_tokens)),
            Cell::from(session.rate().label()),
            Cell::from(status_span(&session.last_status)),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(22),
            Constraint::Min(20),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(12),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(panel(format!(" Sessions ({}) ", sessions.len()), true));
    frame.render_widget(table, area);
}

fn render_active(frame: &mut ratatui::Frame<'_>, area: Rect, active: &[ActiveRequest]) {
    let header = header_row(&[
        "REQUEST",
        "SESSION",
        "PROVIDER / MODEL",
        "STATUS",
        "ELAPSED",
        "IN",
        "OUT",
    ]);

    let rows = active.iter().map(|request| {
        Row::new(vec![
            Cell::from(truncate(&request.request_id, 12)),
            Cell::from(truncate(
                request.session_id.as_deref().unwrap_or("-"),
                18,
            )),
            provider_model_cell(&request.provider, &request.model, 26),
            Cell::from(status_span(request.status.label())),
            Cell::from(format_duration(request.elapsed())),
            Cell::from(request.input_tokens.map(format_tokens).unwrap_or_else(|| "-".into())),
            Cell::from(request.output_tokens.map(format_tokens).unwrap_or_else(|| "-".into())),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(18),
            Constraint::Min(20),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(9),
        ],
    )
    .header(header)
    .block(panel(format!(" Active requests ({}) ", active.len()), false));
    frame.render_widget(table, area);
}

/// Width, in filled/empty block chars, of a usage bar - matches the Claude
/// Code status line's own `makeBar` (`~/.claude/statusbar/lib/format.js`).
const USAGE_BAR_WIDTH: usize = 12;

fn usage_bar_color(pct: f64) -> Color {
    if pct > 80.0 {
        BAD
    } else if pct > 65.0 {
        WARN
    } else {
        OK
    }
}

/// One "label [bar] NN%  resets in Xh" line, mirroring the status line's own
/// 5-hour usage bar so the TUI and `claude`'s own prompt read the same way.
fn usage_bar_line(label: &str, window: &UsageWindow) -> Line<'static> {
    let pct = window.used_percentage.clamp(0.0, 999.0);
    let filled = ((pct / 100.0) * USAGE_BAR_WIDTH as f64)
        .round()
        .clamp(0.0, USAGE_BAR_WIDTH as f64) as usize;
    let bar = "\u{2593}".repeat(filled) + &"\u{2591}".repeat(USAGE_BAR_WIDTH - filled);
    let color = usage_bar_color(pct);
    let mut spans = vec![
        Span::styled(format!("{label:<15}"), Style::default().fg(HEADING)),
        Span::styled(bar, Style::default().fg(color)),
        Span::styled(
            format!(" {pct:>3.0}%"),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(resets_at) = window.resets_at {
        let target = std::time::UNIX_EPOCH + Duration::from_secs(resets_at);
        let text = match target.duration_since(std::time::SystemTime::now()) {
            Ok(remaining) => format!("  resets in {}", format_duration(remaining)),
            Err(_) => "  reset pending".to_string(),
        };
        spans.push(Span::styled(text, Style::default().fg(DIM)));
    }
    Line::from(spans)
}

fn render_consumption(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    sessions: &[SessionSummary],
    usage_windows: &[UsageWindow],
) {
    let mut total_in = 0u64;
    let mut total_out = 0u64;
    let mut by_provider: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for session in sessions {
        total_in = total_in.saturating_add(session.input_tokens);
        total_out = total_out.saturating_add(session.output_tokens);
        let key = session.provider.clone().unwrap_or_else(|| "unknown".into());
        let entry = by_provider.entry(key).or_default();
        entry.0 = entry.0.saturating_add(session.input_tokens);
        entry.1 = entry.1.saturating_add(session.output_tokens);
    }

    let mut lines = vec![Line::from(vec![
        Span::styled("Total   ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(
                "in: {}   out: {}   sessions: {}",
                format_tokens(total_in),
                format_tokens(total_out),
                sessions.len()
            ),
            Style::default().fg(TEXT),
        ),
    ])];
    if !by_provider.is_empty() {
        let breakdown = by_provider
            .iter()
            .map(|(provider, (input, output))| {
                format!(
                    "{}  in: {}  out: {}",
                    provider_label(provider),
                    format_tokens(*input),
                    format_tokens(*output)
                )
            })
            .collect::<Vec<_>>()
            .join("   |   ");
        lines.push(Line::from(Span::styled(breakdown, Style::default().fg(HEADING))));
    }

    let find_window = |provider: &str, window: &str| {
        usage_windows
            .iter()
            .find(|entry| entry.provider == provider && entry.window == window)
    };
    if let Some(window) = find_window("anthropic", "five_hour") {
        lines.push(usage_bar_line("claude 5h", window));
    }
    if let Some(window) = find_window("anthropic", "seven_day") {
        lines.push(usage_bar_line("claude weekly", window));
    }
    if let Some(window) = find_window("codex", "monthly") {
        lines.push(usage_bar_line("codex monthly", window));
    }

    frame.render_widget(
        Paragraph::new(lines).block(panel(" Consumption ".to_string(), false)),
        area,
    );
}

fn render_providers(frame: &mut ratatui::Frame<'_>, area: Rect, app: &MonitorApp<'_>) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(9)])
        .split(area);

    let names = app.registry.list_provider_names();
    let statuses: Vec<(String, bool, String)> = names
        .iter()
        .map(|name| match app.registry.provider(name) {
            Some(provider) => match provider.cli().status_text() {
                Ok(text) => (name.clone(), true, text),
                Err(err) => (name.clone(), false, err.to_string()),
            },
            None => (name.clone(), false, "unknown provider".to_string()),
        })
        .collect();

    let header = header_row(&["PROVIDER", "STATUS", "DETAIL"]);

    let table_rows = statuses.iter().enumerate().map(|(index, (name, connected, detail))| {
        let style = if index == app.provider_offset {
            Style::default().bg(PANEL_ALT).fg(TEXT)
        } else {
            Style::default().fg(TEXT)
        };
        let status = if *connected {
            Span::styled("connected", Style::default().fg(OK))
        } else {
            Span::styled("not authenticated", Style::default().fg(BAD))
        };
        Row::new(vec![
            Cell::from(Line::from(vec![
                provider_dot(name),
                Span::styled(provider_label(name).to_string(), Style::default().fg(TEXT)),
            ])),
            Cell::from(status),
            Cell::from(truncate(detail.lines().next().unwrap_or(""), 60)),
        ])
        .style(style)
    });

    let table = Table::new(
        table_rows,
        [
            Constraint::Length(14),
            Constraint::Length(20),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(panel(" Providers ".to_string(), true));
    frame.render_widget(table, rows[0]);

    let detail_text = statuses
        .get(app.provider_offset)
        .map(|(_, _, detail)| detail.as_str())
        .unwrap_or("-");
    frame.render_widget(
        Paragraph::new(detail_text)
            .style(Style::default().fg(HEADING))
            .block(panel(" Selected provider (l: login) ".to_string(), false)),
        rows[1],
    );
}

/// Adjusts `*scroll` by the minimum amount needed to keep `selected` inside
/// the `visible`-row window - unlike skip-by-offset, the cursor can move
/// freely within an already-visible window without scrolling the list at all.
fn clamp_scroll(scroll: &mut usize, selected: usize, visible: usize, total: usize) {
    if visible == 0 {
        *scroll = 0;
        return;
    }
    if selected < *scroll {
        *scroll = selected;
    } else if selected + 1 > *scroll + visible {
        *scroll = selected + 1 - visible;
    }
    *scroll = (*scroll).min(total.saturating_sub(visible));
}

fn render_models(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    registry: &Registry,
    picker: &PickerSelection,
    offset: usize,
    scroll: &mut usize,
    refresh_status: Option<String>,
) {
    let ordered = ordered_models(registry, picker);
    // Header row (1) + top/bottom borders (2) don't hold list rows.
    let visible = area.height.saturating_sub(3) as usize;
    clamp_scroll(scroll, offset.min(ordered.len().saturating_sub(1)), visible, ordered.len());

    let header = header_row(&["", "PROVIDER", "MODEL", "ALIAS", "DESCRIPTION"]);
    let rows = ordered.iter().enumerate().skip(*scroll).map(|(index, (provider, model))| {
        let enabled = picker.is_enabled(provider, model);
        let mark = if enabled { "[v]" } else { "[ ]" };
        let row_style = if index == offset {
            Style::default().bg(PANEL_ALT).fg(TEXT)
        } else {
            Style::default().fg(TEXT)
        };
        let model_text = if picker.use_1m(provider, model) {
            format!("{model} [1m]")
        } else {
            model.clone()
        };
        let (alias, description, live) = picker.preview(provider, model);
        // A curated default nobody has actually enabled yet is a dimmed
        // suggestion, not a saved value - `preview`'s `live` flag tells them
        // apart so the two don't read the same as an applied setting.
        let alias_color = if live { TEXT } else { DIM };
        let description_color = if live { HEADING } else { DIM };
        Row::new(vec![
            Cell::from(Span::styled(mark, Style::default().fg(if enabled { OK } else { DIM }))),
            Cell::from(Span::styled(provider_label(provider), Style::default().fg(TEXT))),
            Cell::from(Span::styled(model_text, Style::default().fg(HEADING))),
            Cell::from(Span::styled(alias, Style::default().fg(alias_color))),
            Cell::from(Span::styled(description, Style::default().fg(description_color))),
        ])
        .style(row_style)
    });

    let table = Table::new(
        rows,
        [
            Constraint::Length(4),
            Constraint::Length(9),
            Constraint::Length(24),
            Constraint::Length(14),
            Constraint::Min(24),
        ],
    )
    .header(header)
    .block(panel(
        format!(
            " Models ({}, {} enabled{}){} ",
            ordered.len(),
            picker.entries.len(),
            if picker.replace_built_in_options {
                ", replacing native /model options"
            } else {
                ""
            },
            refresh_status
                .map(|status| format!(" \u{b7} {status}"))
                .unwrap_or_default()
        ),
        true,
    ));
    frame.render_widget(table, area);
}

fn render_picker_json(frame: &mut ratatui::Frame<'_>, area: Rect, app: &MonitorApp<'_>) {
    let popup = centered_rect(70, 70, area);
    frame.render_widget(Clear, popup);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("replaceBuiltInOptions: ", Style::default().fg(HEADING)),
        Span::styled(
            app.picker.replace_built_in_options.to_string(),
            Style::default().fg(if app.picker.replace_built_in_options { WARN } else { DIM }),
        ),
        Span::styled("  (b, from the Models list, to toggle)", Style::default().fg(DIM)),
    ]));
    lines.push(Line::from(""));
    if let Some(status) = &app.picker_status {
        lines.push(Line::from(Span::styled(status.clone(), Style::default().fg(OK))));
        lines.push(Line::from(""));
    }
    if let Some(err) = &app.picker_error {
        lines.push(Line::from(Span::styled(format!("Error: {err}"), Style::default().fg(BAD))));
        lines.push(Line::from(""));
    }
    if app.picker.entries.is_empty() {
        lines.push(Line::from(Span::styled(
            "No models enabled - press space on a model in the list to add it.",
            Style::default().fg(DIM),
        )));
    } else {
        for line in app.picker.to_json_pretty().lines() {
            lines.push(Line::from(Span::styled(line.to_string(), Style::default().fg(TEXT))));
        }
    }

    let title = if app.confirm_override {
        format!(
            " Overwrite modelPicker.options in {}? (y/n) ",
            paths::claude_settings_file().display()
        )
    } else {
        " modelPicker.options - e: edit in $EDITOR  o: overwrite settings.json  esc: close ".to_string()
    };

    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().fg(TEXT))
            .wrap(Wrap { trim: false })
            .block(panel(title, true)),
        popup,
    );
}

/// The headroom status dot, "Headroom" label, and - once ready - its
/// dashboard link, all together in one place (the footer). No headroom
/// handle means nothing is shown at all.
fn headroom_status_spans(headroom: Option<&HeadroomHandle>) -> Vec<Span<'static>> {
    let Some(headroom) = headroom else {
        return Vec::new();
    };
    // Cached by headroom's own watcher thread - never a blocking connect here.
    let status = headroom.status();
    let mut spans = vec![Span::raw("  ")];
    match status {
        HeadroomStatus::Ready => spans.push(Span::styled("\u{25cf} ", Style::default().fg(OK))),
        // Still warming up (headroom's own startup - importing its ML deps -
        // takes several seconds): a spinner instead of a flat red dot makes
        // clear this is "starting", not "down".
        HeadroomStatus::Starting => spans.push(Span::styled(
            format!("{} ", spinner_frame()),
            Style::default().fg(WARN),
        )),
        HeadroomStatus::Down => spans.push(Span::styled("\u{25cf} ", Style::default().fg(BAD))),
    }
    spans.push(Span::styled("Headroom", Style::default().fg(TEXT)));
    if status == HeadroomStatus::Ready {
        spans.push(Span::styled(
            format!("  {} ", headroom.dashboard_url()),
            Style::default().fg(DIM),
        ));
    }
    spans
}

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect, app: &MonitorApp<'_>) {
    let text = match app.phase {
        MonitorPhase::ConfirmingShutdown => Line::from(Span::styled(
            " Shut down the proxy? (y/n) ",
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        )),
        MonitorPhase::ShuttingDown => Line::from(Span::styled(
            " Shutting down... ",
            Style::default().fg(WARN),
        )),
        MonitorPhase::Running => {
            let key = |label: &str| Span::styled(label.to_string(), Style::default().fg(ACCENT));
            let desc = |label: &str| Span::styled(label.to_string(), Style::default().fg(DIM));
            let mut spans = vec![
                Span::raw(" "),
                key("?"),
                desc(" help  "),
                key("q"),
                desc(" quit  "),
                key("tab"),
                desc(" view  "),
                key("\u{2191}/\u{2193} j/k"),
                desc(" scroll  "),
            ];
            if app.view == View::Providers {
                spans.push(key("l"));
                spans.push(desc(" login  "));
            }
            if app.view == View::Models {
                spans.push(key("space"));
                spans.push(desc(" toggle  "));
                spans.push(key("a"));
                spans.push(desc(" alias  "));
                spans.push(key("[ ]"));
                spans.push(desc(" reorder  "));
                spans.push(key("r"));
                spans.push(desc(" refresh  "));
                spans.push(key("b"));
                spans.push(desc(" replace built-ins  "));
                spans.push(key("p"));
                spans.push(desc(" json  "));
            }
            spans.extend(headroom_status_spans(app.headroom.as_ref()));
            Line::from(spans)
        }
    };
    frame.render_widget(
        Paragraph::new(text).style(Style::default().bg(BG)),
        area,
    );
}

fn provider_model(provider: &Option<String>, model: &Option<String>) -> String {
    match (provider, model) {
        (Some(provider), Some(model)) => format!("{} / {model}", provider_label(provider)),
        (Some(provider), None) => provider_label(provider).to_string(),
        (None, Some(model)) => model.clone(),
        (None, None) => "-".to_string(),
    }
}

fn provider_model_cell(provider: &Option<String>, model: &Option<String>, max: usize) -> Cell<'static> {
    let text = truncate(&provider_model(provider, model), max);
    match provider {
        Some(name) => Cell::from(Line::from(vec![
            provider_dot(name),
            Span::styled(text, Style::default().fg(TEXT)),
        ])),
        None => Cell::from(Span::styled(text, Style::default().fg(HEADING))),
    }
}

/// Matches the HTML dashboard's `.status-*` classes (dashboard.html) so a
/// given status reads the same color in both UIs.
fn status_span(label: &str) -> Span<'static> {
    let color = match label {
        "completed" => OK,
        "streaming" | "upstream" => INFO,
        "compacting" => PURPLE,
        "failed" => BAD,
        "started" | "selected" => WARN,
        _ => DIM,
    };
    Span::styled(label.to_string(), Style::default().fg(color))
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}m", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    if total_secs >= 3600 {
        format!("{}h{:02}m", total_secs / 3600, (total_secs % 3600) / 60)
    } else if total_secs >= 60 {
        format!("{}m{:02}s", total_secs / 60, total_secs % 60)
    } else {
        format!("{}.{}s", total_secs, duration.subsec_millis() / 100)
    }
}
