use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};
use claude_code_proxy::{
    config, console_branding, headroom, instance_lock, logging,
    monitor::MonitorHandle,
    paths,
    registry::{ANTHROPIC_STYLE_ALIASES, Registry},
    server::{self, ServerConfig},
    tui::{self, MonitorExit, MonitorUiConfig},
};
use std::io::IsTerminal;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Parser)]
#[command(
    name = "claude-code-proxy",
    version = VERSION,
    about = "Anthropic-compatible proxy for Claude Code provider backends",
    disable_version_flag = true
)]
struct Cli {
    #[arg(long = "version", short = 'v', action = ArgAction::SetTrue)]
    version_flag: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print version information
    Version,
    /// Start the proxy server and web dashboard
    Serve {
        #[arg(long)]
        port: Option<u16>,
        #[arg(long = "no-monitor", action = ArgAction::SetTrue)]
        no_monitor: bool,
        #[arg(long = "no-headroom", action = ArgAction::SetTrue)]
        no_headroom: bool,
    },
    /// List supported provider models
    Models {
        #[arg(long)]
        full: bool,
        /// Live-fetch the Codex model catalog before printing, instead of
        /// only showing the static allowlist plus whatever was cached from
        /// a previous run.
        #[arg(long)]
        refresh: bool,
    },
    /// Manage Codex authentication
    Codex {
        #[command(subcommand)]
        command: ProviderGroup,
    },
    /// Open a `claude` session against the running proxy (through headroom
    /// when it's up). Any extra arguments are forwarded to `claude` as-is.
    Code {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Debug, Subcommand)]
enum ProviderGroup {
    Auth {
        #[command(subcommand)]
        command: claude_code_proxy::provider::AuthCommand,
    },
}

fn main() -> Result<()> {
    console_branding::apply();
    let cli = Cli::parse();

    if cli.version_flag {
        println!("claude-code-proxy {}", VERSION);
        return Ok(());
    }

    let commands = cli.command.unwrap_or(Commands::Serve {
        port: None,
        no_monitor: false,
        no_headroom: false,
    });

    match commands {
        Commands::Version => {
            println!("claude-code-proxy {}", VERSION);
            Ok(())
        }
        Commands::Serve {
            port,
            no_monitor,
            no_headroom,
        } => {
            let _instance_lock = instance_lock::acquire(&paths::config_dir())?;
            let bind_address = config::bind_address();
            let effective_port = port.unwrap_or_else(config::port);
            let registry = Registry::with_default_alias();
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let headroom_handle = match resolve_headroom(&bind_address, effective_port, no_headroom)? {
                HeadroomOutcome::Abort => return Ok(()),
                HeadroomOutcome::Start(handle) => handle,
            };
            if should_run_tui(std::io::stdout().is_terminal(), no_monitor) {
                run_serve_with_tui(
                    &runtime,
                    &bind_address,
                    effective_port,
                    &registry,
                    headroom_handle,
                )
            } else {
                print_server_banner(&bind_address, effective_port, &registry, headroom_handle.as_ref());
                runtime
                    .block_on(run_service(
                        ServerConfig {
                            bind_address,
                            port: effective_port,
                            monitor: Some(MonitorHandle::default()),
                        },
                        headroom_handle,
                    ))
                    .map_err(|err| anyhow::anyhow!(err))
            }
        }
        Commands::Models { full, refresh } => {
            if refresh {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                match runtime.block_on(claude_code_proxy::providers::codex::model_catalog::refresh_once())
                {
                    Ok(count) => println!("Refreshed Codex model catalog: {count} known models."),
                    Err(err) => println!(
                        "Codex model catalog refresh failed ({err}); showing cached/static list."
                    ),
                }
            }
            print_models(&Registry::with_default_alias(), full);
            Ok(())
        }
        Commands::Codex { command } => run_provider_cli("codex", command),
        Commands::Code { args } => run_code_session(args),
    }
}

async fn run_service(config: ServerConfig, headroom_handle: Option<headroom::HeadroomHandle>) -> Result<()> {
    let mut signals = ServiceShutdownSignals::new()?;
    let (shutdown, stopped) = tokio::sync::oneshot::channel();
    let server = server::serve_with_shutdown(config, async {
        let _ = stopped.await;
    });
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        signal = signals.recv() => {
            signal?;
            let _ = shutdown.send(());
            tokio::select! {
                result = &mut server => result,
                signal = signals.recv() => {
                    signal?;
                    if let Some(handle) = &headroom_handle {
                        handle.shutdown();
                    }
                    std::process::exit(130);
                }
            }
        }
    }
}

/// Checks that the proxy is listening on `bind_address:port`, then launches
/// an interactive `claude` session pointed at it - through headroom, when
/// headroom is up, so its compression sits in front of every request.
fn run_code_session(args: Vec<String>) -> Result<()> {
    let bind_address = config::bind_address();
    let app_port = config::port();
    if !headroom::reachable(&bind_address, app_port) {
        eprintln!("The proxy isn't running. Start it first with: ccp serve");
        std::process::exit(1);
    }
    let base_url = if headroom::reachable("127.0.0.1", headroom::port()) {
        format!("http://127.0.0.1:{}", headroom::port())
    } else {
        listen_url(&bind_address, app_port)
    };
    let status = std::process::Command::new("claude")
        .env("ANTHROPIC_BASE_URL", base_url)
        .args(&args)
        .status()
        .map_err(|err| anyhow::anyhow!("failed to launch claude: {err}"))?;
    std::process::exit(status.code().unwrap_or(1));
}

enum HeadroomOutcome {
    Start(Option<headroom::HeadroomHandle>),
    Abort,
}

enum HeadroomConflictChoice {
    UseExisting,
    KillAndRestart,
    SkipHeadroom,
    Abort,
}

/// Decides what headroom sidecar (if any) `serve` should run with. If
/// something is already listening on headroom's port, asks the user how to
/// proceed instead of silently reusing or clobbering it - it could just as
/// easily be an unrelated headroom instance (e.g. one backing an editor's
/// headroom MCP integration) as a leftover from a previous `serve`.
fn resolve_headroom(bind_address: &str, port: u16, no_headroom: bool) -> Result<HeadroomOutcome> {
    if no_headroom {
        return Ok(HeadroomOutcome::Start(None));
    }
    let headroom_port = headroom::port();
    let upstream = listen_url(bind_address, port);

    if !headroom::reachable("127.0.0.1", headroom_port) {
        return Ok(HeadroomOutcome::Start(spawn_headroom_or_warn(
            headroom_port,
            &upstream,
        )));
    }

    // The prompt below blocks on a synchronous stdin read. That's fine at a
    // real console, but this command can also run somewhere with no
    // attached input (a task runner, a non-interactive wrapper, this CLI's
    // own tool-exec channel) - there, stdin.is_terminal() is false and
    // read_line() would just hang forever with no way for anyone to answer.
    // Skip straight to a safe default instead of blocking.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        println!(
            "A process is already listening on headroom's port ({headroom_port}), and this \
             isn't an interactive terminal to ask how to proceed. Starting without headroom \
             - rerun `ccp serve` at a real console to choose reuse/kill/abort instead."
        );
        return Ok(HeadroomOutcome::Start(None));
    }

    match prompt_headroom_conflict(headroom_port) {
        HeadroomConflictChoice::UseExisting => {
            println!(
                "Using the existing headroom proxy on port {headroom_port} - \
                 note it may not be routed to this instance ({upstream})."
            );
            Ok(HeadroomOutcome::Start(Some(headroom::HeadroomHandle::external(
                headroom_port,
            ))))
        }
        HeadroomConflictChoice::KillAndRestart => {
            println!("Stopping the process on port {headroom_port}...");
            if let Err(err) = headroom::kill_process_on_port(headroom_port) {
                eprintln!(
                    "Warning: couldn't stop it automatically ({err}). Starting without headroom."
                );
                return Ok(HeadroomOutcome::Start(None));
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
            Ok(HeadroomOutcome::Start(spawn_headroom_or_warn(
                headroom_port,
                &upstream,
            )))
        }
        HeadroomConflictChoice::SkipHeadroom => Ok(HeadroomOutcome::Start(None)),
        HeadroomConflictChoice::Abort => Ok(HeadroomOutcome::Abort),
    }
}

fn spawn_headroom_or_warn(port: u16, upstream: &str) -> Option<headroom::HeadroomHandle> {
    match headroom::spawn(port, upstream) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("Warning: failed to start headroom proxy: {err}");
            None
        }
    }
}

fn prompt_headroom_conflict(port: u16) -> HeadroomConflictChoice {
    println!();
    println!("A process is already listening on headroom's port ({port}).");
    println!("  1) Connect to it (assume it's already routed correctly)");
    println!("  2) Kill it and start a fresh headroom proxy");
    println!("  3) Start the proxy without headroom");
    println!("  4) Abort");
    loop {
        print!("> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(0) | Err(_) => {
                println!("(no input available - starting without headroom)");
                return HeadroomConflictChoice::SkipHeadroom;
            }
            Ok(_) => match input.trim() {
                "1" => return HeadroomConflictChoice::UseExisting,
                "2" => return HeadroomConflictChoice::KillAndRestart,
                "3" => return HeadroomConflictChoice::SkipHeadroom,
                "4" => return HeadroomConflictChoice::Abort,
                _ => println!("Please enter 1, 2, 3, or 4."),
            },
        }
    }
}

#[cfg(unix)]
struct ServiceShutdownSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl ServiceShutdownSignals {
    fn new() -> std::io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
        })
    }

    async fn recv(&mut self) -> std::io::Result<()> {
        tokio::select! {
            _ = self.interrupt.recv() => Ok(()),
            _ = self.terminate.recv() => Ok(()),
        }
    }
}

#[cfg(windows)]
struct ServiceShutdownSignals {
    ctrl_c: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
impl ServiceShutdownSignals {
    fn new() -> std::io::Result<Self> {
        Ok(Self {
            ctrl_c: tokio::signal::windows::ctrl_c()?,
        })
    }

    async fn recv(&mut self) -> std::io::Result<()> {
        let _ = self.ctrl_c.recv().await;
        Ok(())
    }
}

#[cfg(not(any(unix, windows)))]
struct ServiceShutdownSignals;

#[cfg(not(any(unix, windows)))]
impl ServiceShutdownSignals {
    fn new() -> std::io::Result<Self> {
        Ok(Self)
    }

    async fn recv(&mut self) -> std::io::Result<()> {
        tokio::signal::ctrl_c().await
    }
}

/// The live terminal dashboard only makes sense for someone sitting at an
/// interactive terminal; a background/service launch (non-tty stdout) or an
/// explicit `--no-monitor` should leave the HTML dashboard reachable over
/// HTTP without taking over the screen.
fn should_run_tui(stdout_is_tty: bool, no_monitor: bool) -> bool {
    stdout_is_tty && !no_monitor
}

/// Runs the proxy with the terminal dashboard in the foreground. The HTTP
/// server runs on a background task; the TUI drives shutdown by sending on
/// `shutdown_tx` (graceful) or aborting the task directly (force quit on a
/// second ctrl+c). Mirrors `run_service`'s double-signal semantics, but the
/// "signal" here is a key event read by crossterm's raw mode instead of an
/// OS signal, which is what makes it work identically on Windows.
fn run_serve_with_tui(
    runtime: &tokio::runtime::Runtime,
    bind_address: &str,
    port: u16,
    registry: &Registry,
    headroom_handle: Option<headroom::HeadroomHandle>,
) -> Result<()> {
    let _stderr_guard = logging::suppress_stderr();
    let monitor = MonitorHandle::default();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let (shutdown_complete_tx, shutdown_complete_rx) = std::sync::mpsc::channel();
    let listener = runtime.block_on(server::bind_proxy_listener(bind_address, port))?;
    let local_addr = listener.local_addr()?;
    let monitor_listen_url = listen_url(&local_addr.ip().to_string(), local_addr.port());
    let server_monitor = monitor.clone();
    let server_task = runtime.spawn(async move {
        let result = server::serve_listener(listener, Some(server_monitor), async move {
            let _ = shutdown_rx.await;
        })
        .await;
        let _ = shutdown_complete_tx.send(());
        result
    });

    let ui_result = tui::run_monitor(
        monitor,
        MonitorUiConfig {
            listen_url: monitor_listen_url,
            registry,
            headroom: headroom_handle.clone(),
            runtime: runtime.handle().clone(),
            shutdown: Some(shutdown_tx),
            shutdown_complete: Some(shutdown_complete_rx),
        },
    );
    if matches!(&ui_result, Ok(MonitorExit::ForceQuit)) {
        server_task.abort();
        let _ = runtime.block_on(server_task);
        if let Some(handle) = &headroom_handle {
            handle.shutdown();
        }
        std::process::exit(130);
    }
    let server_result = runtime.block_on(server_task)?;
    ui_result?;
    server_result.map_err(|err| anyhow::anyhow!(err))
}

fn run_provider_cli(name: &str, command: ProviderGroup) -> Result<()> {
    let registry = Registry::with_default_alias();
    let provider = registry
        .provider(name)
        .ok_or_else(|| anyhow::anyhow!("unknown provider: {name}"))?;
    let handlers = provider.cli();
    match command {
        ProviderGroup::Auth { command } => match command {
            claude_code_proxy::provider::AuthCommand::Login => {
                if let Err(err) = handlers.login() {
                    eprintln!("{err}");
                    std::process::exit(2);
                }
                Ok(())
            }
            claude_code_proxy::provider::AuthCommand::Device => {
                if let Err(err) = handlers.device() {
                    eprintln!("{err}");
                    std::process::exit(2);
                }
                Ok(())
            }
            claude_code_proxy::provider::AuthCommand::Status => {
                if let Err(err) = handlers.status() {
                    println!("{err}");
                    if err.to_string() == "Not authenticated" {
                        std::process::exit(1);
                    }
                    std::process::exit(2);
                }
                Ok(())
            }
            claude_code_proxy::provider::AuthCommand::Logout => {
                handlers.logout()?;
                Ok(())
            }
        },
    }
}

fn print_models(registry: &Registry, full: bool) {
    let _ = full;
    let grouped = registry.grouped_models();
    for provider in ["codex"] {
        let Some(models) = grouped.get(provider) else {
            continue;
        };
        println!("{provider}: {}", models.join(", "));
    }
}

fn listen_url(bind_address: &str, port: u16) -> String {
    match bind_address.parse::<std::net::IpAddr>() {
        Ok(ip) => format!("http://{}", std::net::SocketAddr::new(ip, port)),
        Err(_) => format!("http://{bind_address}:{port}"),
    }
}

fn print_server_banner(
    bind_address: &str,
    port: u16,
    registry: &Registry,
    headroom_handle: Option<&headroom::HeadroomHandle>,
) {
    println!("Proxy listening on {}", listen_url(bind_address, port));
    println!(
        "Dashboard: {}/dashboard",
        listen_url(bind_address, port)
    );
    if let Some(handle) = headroom_handle {
        println!("Headroom proxy: {}", handle.dashboard_url());
    }
    println!("Logs: {}", paths::log_file().display());
    let cfg = paths::config_dir();
    if cfg.exists() {
        println!("Config: {}", cfg.display());
    }
    print_models(registry, false);
    println!();
    println!("Configure Claude Code:");
    println!("  export ANTHROPIC_BASE_URL=\"http://localhost:{port}\"");
    println!();
    println!("Do not set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY - either one overrides");
    println!("the Claude subscription login and the Claude route returns 401.");
    println!();
    println!("Do not set ANTHROPIC_MODEL or ANTHROPIC_SMALL_FAST_MODEL either - that");
    println!("overrides Claude Code's native Opus/Sonnet/Haiku/Fable slots. Instead, open");
    println!("the dashboard above, pick your models, and copy the generated JSON into");
    println!("modelPicker.options in ~/.claude/settings.json.");
    println!();
    println!("Optional: export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1");
}

#[allow(dead_code)]
fn alias_names() -> usize {
    ANTHROPIC_STYLE_ALIASES.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_serve_runs_tui_on_tty() {
        assert!(should_run_tui(true, false));
    }

    #[test]
    fn no_monitor_skips_tui() {
        assert!(!should_run_tui(true, true));
    }

    #[test]
    fn non_tty_stdout_skips_tui() {
        assert!(!should_run_tui(false, false));
    }

    #[tokio::test]
    async fn shutdown_signal_setup_and_receive_preserve_io_results() {
        fn assert_constructor(_: fn() -> std::io::Result<ServiceShutdownSignals>) {}
        fn assert_io_future<F: std::future::Future<Output = std::io::Result<()>>>(_: &F) {}

        assert_constructor(ServiceShutdownSignals::new);
        let mut signals = ServiceShutdownSignals::new().unwrap();
        let receive = signals.recv();
        assert_io_future(&receive);
    }

    #[test]
    fn listen_url_brackets_ipv6_addresses() {
        assert_eq!(listen_url("::1", 18765), "http://[::1]:18765");
    }
}
