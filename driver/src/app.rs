//! Top-level application entry point.
//!
//! Parses CLI args, loads config, constructs solver, and dispatches to a frontend.
//! The binary's `main()` calls [`CliArgs::parse`] once, then branches on
//! [`Mode::Gui`] (launch a native window) vs the terminal modes handled by
//! [`run`].

use crate::config::{Checkpoint, SimulationConfig, build_nested, merge};
use crate::driver::{Driver, DriverState};
use crate::solver::Solver;
use serde_json::Value;
use std::path::{Path, PathBuf};

// ============================================================================
// Public types
// ============================================================================

/// Frontend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Native GUI window (default when no arguments are given).
    Gui,
    /// Fire-and-forget: create state, run to completion, exit.
    Batch,
    /// Interactive line-oriented REPL.
    Repl,
    /// Terminal UI with three-column layout.
    Tui,
}

/// What the user asked for before we build a driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Print help text and exit.
    Help,
    /// Print the JSON schema and exit.
    Schema,
    /// Print the resolved config (after loading files + overrides) and exit.
    DumpConfig,
    /// Normal operation — build a driver and run in the chosen mode.
    Run,
}

/// Parsed command-line arguments.
#[derive(Debug)]
pub struct CliArgs {
    pub action: Action,
    pub mode: Mode,
    pub checkpoint: Option<PathBuf>,
    pub ron_file: Option<PathBuf>,
    pub overrides: Vec<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            action: Action::Run,
            mode: Mode::Gui,
            checkpoint: None,
            ron_file: None,
            overrides: Vec::new(),
        }
    }
}

impl CliArgs {
    /// Parse from raw CLI args (everything after the program name).
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut result = CliArgs::default();
        let mut iter = args.iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    result.action = Action::Help;
                    return Ok(result);
                }
                "-s" | "--schema" => {
                    result.action = Action::Schema;
                    return Ok(result);
                }
                "-d" | "--dump-config" => {
                    result.action = Action::DumpConfig;
                }
                "--mode" => {
                    let val = iter.next().ok_or("--mode requires a value")?;
                    result.mode = parse_mode(val)?;
                }
                _ if arg.starts_with("--mode=") => {
                    result.mode = parse_mode(arg.strip_prefix("--mode=").unwrap())?;
                }
                _ => classify_positional(&mut result, arg)?,
            }
        }

        Ok(result)
    }

    /// Convenience: parse from `std::env::args()` (skips argv[0]).
    pub fn from_env() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        Self::parse(&args)
    }

    /// The program name, extracted from argv[0].
    pub fn program_name() -> String {
        let raw = std::env::args().next().unwrap_or_else(|| "program".into());
        std::path::Path::new(&raw)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(raw)
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Run a terminal frontend (batch, repl, or tui) using already-parsed CLI args.
///
/// For [`Mode::Gui`] the binary should launch its own window toolkit instead of
/// calling this function.
///
/// Handles [`Action::Help`], [`Action::Schema`], and [`Action::DumpConfig`]
/// internally (they print and exit).
pub fn run<S: Solver + Send + 'static>(cli: CliArgs)
where
    S::State: Send,
    S::Physics: Send,
    S::Initial: Send,
    S::Compute: Send,
{
    if let Err(e) = run_inner::<S>(cli) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn run_inner<S: Solver + Send + 'static>(cli: CliArgs) -> Result<(), String>
where
    S::State: Send,
    S::Physics: Send,
    S::Initial: Send,
    S::Compute: Send,
{
    // Early-exit actions
    match cli.action {
        Action::Help => {
            print_help(&CliArgs::program_name());
            std::process::exit(0);
        }
        Action::Schema => {
            print_schema::<S>();
            std::process::exit(0);
        }
        _ => {}
    }

    // Load config / checkpoint
    let (mut config, mut solver_state, driver_state) = if let Some(ref path) = cli.checkpoint {
        let chkpt = load_checkpoint::<S>(path)?;
        (chkpt.config, Some(chkpt.state), chkpt.driver)
    } else {
        let config = match cli.ron_file {
            Some(ref path) => load_config::<S>(path)?,
            None => SimulationConfig::default(),
        };
        (config, None, DriverState::new())
    };

    apply_overrides(&mut config, &cli.overrides)?;

    if cli.action == Action::DumpConfig {
        dump_config_and_exit(&config);
    }

    // Build solver
    let solver = S::new((
        config.physics.clone(),
        config.initial.clone(),
        config.compute.clone(),
    ));

    if matches!(cli.mode, Mode::Batch) {
        solver_state = Some(solver.initial());
    }

    // Spawn worker and dispatch to frontend
    use crate::frontend;
    use crate::worker;

    let (driver, init_events) = Driver::new(config, solver, solver_state, driver_state);
    let handle = worker::spawn::<S>(driver, init_events);

    match cli.mode {
        Mode::Batch => frontend::cli::run(handle),
        Mode::Repl => frontend::repl::run(handle),
        Mode::Tui => frontend::tui::run(handle, Vec::new()),
        Mode::Gui => {
            unreachable!("Mode::Gui should be handled by the binary, not app::run")
        }
    }

    Ok(())
}

// ============================================================================
// Internal helpers
// ============================================================================

fn parse_mode(s: &str) -> Result<Mode, String> {
    match s {
        "gui" => Ok(Mode::Gui),
        "batch" => Ok(Mode::Batch),
        "repl" => Ok(Mode::Repl),
        "tui" => Ok(Mode::Tui),
        other => Err(format!("unknown mode: {}", other)),
    }
}

fn classify_positional(cli: &mut CliArgs, arg: &str) -> Result<(), String> {
    let path = Path::new(arg);
    match path.extension().and_then(|e| e.to_str()) {
        Some("mpk") => {
            if cli.checkpoint.is_some() {
                return Err("multiple checkpoint arguments not supported".into());
            }
            if cli.ron_file.is_some() {
                return Err("cannot specify both a checkpoint and a .ron config".into());
            }
            cli.checkpoint = Some(path.to_path_buf());
        }
        Some("ron") => {
            if cli.ron_file.is_some() {
                return Err("multiple config files not supported".into());
            }
            if cli.checkpoint.is_some() {
                return Err("cannot specify both a checkpoint and a .ron config".into());
            }
            cli.ron_file = Some(path.to_path_buf());
        }
        _ => cli.overrides.push(arg.into()),
    }
    Ok(())
}

// ============================================================================
// File loading
// ============================================================================

fn load_checkpoint<S: Solver>(
    path: &Path,
) -> Result<Checkpoint<S, S::State, serde_json::Value>, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    rmp_serde::from_slice(&data).map_err(|e| format!("{}: {}", path.display(), e))
}

fn load_config<S: Solver>(path: &Path) -> Result<SimulationConfig<S>, String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    ron::from_str(&source).map_err(|e| format!("{}: {}", path.display(), e))
}

// ============================================================================
// Config overrides
// ============================================================================

/// Parse CLI override args using a prefix-stack state machine.
///
/// Three arg forms:
/// - `--name`    → reset prefix to `[name]` (validated against `top_level_keys`)
/// - `key=value` → set: prefix + key as dot path
/// - bare word   → push onto prefix
///
/// Fully qualified dot paths work with an empty prefix:
///   `physics.gamma=1.6` → `build_nested("physics.gamma", 1.6)`
fn parse_cli_overrides(args: &[String], top_level_keys: &[String]) -> Result<Value, String> {
    let mut overrides = Value::Object(serde_json::Map::new());
    let mut prefix: Vec<String> = Vec::new();

    for arg in args {
        if let Some(name) = arg.strip_prefix("--") {
            let name = name.replace('-', "_");
            if !top_level_keys.contains(&name) {
                return Err(format!(
                    "unknown config section '--{}', expected one of: {}",
                    name,
                    top_level_keys.join(", ")
                ));
            }
            prefix = vec![name];
        } else if let Some((key, val_str)) = arg.split_once('=') {
            let dot_path = if prefix.is_empty() {
                key.to_string()
            } else {
                format!("{}.{}", prefix.join("."), key)
            };
            let entry = build_nested(
                &dot_path,
                serde_json::from_str(val_str).unwrap_or_else(|_| Value::String(val_str.into())),
            );
            overrides = merge(overrides, entry);
        } else {
            prefix.push(arg.clone());
        }
    }

    Ok(overrides)
}

fn apply_overrides<S: Solver>(
    config: &mut SimulationConfig<S>,
    override_args: &[String],
) -> Result<(), String> {
    if override_args.is_empty() {
        return Ok(());
    }
    let base = serde_json::to_value(&*config).map_err(|e| e.to_string())?;
    let top_level_keys: Vec<String> = base
        .as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    let overrides = parse_cli_overrides(override_args, &top_level_keys)?;
    let merged = merge(base, overrides);
    *config = serde_json::from_value(merged).map_err(|e| e.to_string())?;
    Ok(())
}

// ============================================================================
// Help / schema / dump
// ============================================================================

fn print_help(program: &str) {
    const R: &str = "\x1b[0m";
    const B: &str = "\x1b[1m";
    const C: &str = "\x1b[1;36m";
    const G: &str = "\x1b[32m";
    const Y: &str = "\x1b[33m";
    const D: &str = "\x1b[2m";

    println!("{B}USAGE{R}");
    println!("  {C}{program}{R} {D}[OPTIONS]{R} {D}[FILES...]{R} {D}[OVERRIDES...]{R}");
    println!();
    println!("  With no arguments, launches the native GUI window.");
    println!();
    println!("{B}OPTIONS{R}");
    println!("  {G}-h{R}, {G}--help{R}            Show this help message");
    println!("  {G}-s{R}, {G}--schema{R}          Print the config JSON schema");
    println!("  {G}-d{R}, {G}--dump-config{R}     Print the resolved config and exit");
    println!(
        "  {G}--mode{R} {Y}MODE{R}           Frontend mode: {Y}gui{R} {D}(default){R}, {Y}batch{R}, {Y}repl{R}, {Y}tui{R}"
    );
    println!();
    println!("{B}FILES{R}");
    println!("  {C}config.ron{R}            Load a RON config file");
    println!("  {C}chkpt.0004.mpk{R}        Restart from a checkpoint");
    println!();
    println!("{B}OVERRIDES{R}");
    println!("  Config values can be overridden on the command line:");
    println!();
    println!("  {D}# dot-path syntax{R}");
    println!("  {C}{program}{R} physics.gamma=1.6 driver.checkpoint_interval=0.1");
    println!();
    println!("  {D}# section prefix syntax{R}");
    println!("  {C}{program}{R} --physics gamma=1.6 --driver checkpoint_interval=0.1");
}

fn print_schema<S: Solver>() {
    let schema = schemars::schema_for!(SimulationConfig<S>);
    crate::format::print_schema(&schema);
}

fn dump_config_and_exit<S: Solver>(config: &SimulationConfig<S>) -> ! {
    use std::io::IsTerminal;

    let pretty = ron::ser::PrettyConfig::new()
        .depth_limit(8)
        .struct_names(true)
        .enumerate_arrays(false);
    let ron_str = ron::ser::to_string_pretty(config, pretty).unwrap_or_else(|e| {
        eprintln!("error: failed to serialize config: {}", e);
        std::process::exit(1);
    });

    if std::io::stdout().is_terminal() {
        print!("{}", crate::format::highlight_ron(&ron_str));
    } else {
        print!("{}", ron_str);
    }
    std::process::exit(0);
}
