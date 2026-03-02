//! Interactive REPL frontend.
//!
//! Reads commands from stdin, sends them to the driver, prints responses.
//! Uses rustyline for history, line editing, and tab completion.

use crate::command::{Command, Message};
use crate::config::{build_nested, merge};
use crate::worker::DriverHandle;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Editor, Helper};
use serde_json::Value;

const COMMANDS: &[&str] = &[
    "run",
    "pause",
    "step",
    "status",
    "config",
    "schema",
    "set ",
    "create",
    "destroy",
    "checkpoint",
    "write",
    "quit",
    "exit",
];

struct ReplHelper;

impl Helper for ReplHelper {}
impl Validator for ReplHelper {}
impl Highlighter for ReplHelper {}
impl Hinter for ReplHelper {
    type Hint = String;
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = &line[..pos];
        let candidates: Vec<Pair> = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .map(|cmd| Pair {
                display: cmd.to_string(),
                replacement: cmd.to_string(),
            })
            .collect();
        Ok((0, candidates))
    }
}

pub fn run(handle: DriverHandle) {
    let msg_rx = handle.msg_rx;

    // Spawn a thread to drain and print messages
    let printer = std::thread::spawn(move || {
        while let Ok(msg) = msg_rx.recv() {
            match msg {
                Message::StepCompleted { display, .. } => println!("{}", display),
                Message::SimulationDone => println!("simulation done"),
                Message::Status {
                    mode,
                    iteration,
                    time,
                } => println!(
                    "status: {:?} iteration={} time={:.4e}",
                    mode, iteration, time
                ),
                Message::Config(v) => {
                    println!("{}", serde_json::to_string_pretty(&v).unwrap())
                }
                Message::ConfigSections {
                    driver,
                    physics,
                    initial,
                    compute,
                } => {
                    for (name, ron) in [
                        ("driver", driver),
                        ("physics", physics),
                        ("initial", initial),
                        ("compute", compute),
                    ] {
                        println!("{}:", name);
                        print!("{}", crate::format::highlight_ron(&ron));
                        println!();
                    }
                }
                Message::Schema(v) => {
                    println!("{}", serde_json::to_string_pretty(&v).unwrap())
                }
                Message::ConfigUpdated(Ok(())) => println!("config updated"),
                Message::ConfigUpdated(Err(e)) => println!("config update failed: {}", e),
                Message::CheckpointWritten { path } => println!("wrote {}", path),
                Message::ConfigWritten { path } => println!("wrote config to {}", path),
                Message::ConfigLoaded { path } => println!("loaded config from {}", path),
                Message::CheckpointLoaded { path } => println!("loaded checkpoint from {}", path),
                Message::StateCreated => println!("state created"),
                Message::StateDestroyed => println!("state destroyed"),
                Message::StateInfo {
                    driver_state,
                    solver_status,
                } => {
                    println!(
                        "driver: iter={} chkpt={}",
                        driver_state.iteration, driver_state.checkpoint_number
                    );
                    if let Some(v) = solver_status {
                        println!(
                            "solver_status: {}",
                            serde_json::to_string_pretty(&v).unwrap()
                        );
                    } else {
                        println!("solver_status: no state");
                    }
                }
                Message::PlotData { linear, planar } => {
                    if !linear.is_empty() {
                        println!("linear data: {} series", linear.len());
                        for (name, vals) in &linear {
                            println!("  {}: {} points", name, vals.len());
                        }
                    }
                    if !planar.is_empty() {
                        println!("planar data: {} fields", planar.len());
                        for (name, (rows, cols, _)) in &planar {
                            println!("  {}: {}x{}", name, rows, cols);
                        }
                    }
                }
                Message::Error(e) => eprintln!("error: {}", e),
                Message::Finished => {
                    break;
                }
            }
        }
    });

    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();

    let mut rl = Editor::with_config(config).expect("failed to create readline editor");
    rl.set_helper(Some(ReplHelper));

    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(trimmed).ok();

                let cmd = match trimmed {
                    "run" => Command::Run,
                    "pause" => Command::Pause,
                    "step" => Command::Step,
                    "status" => Command::QueryStatus,
                    "config" => Command::QueryConfig,
                    "schema" => Command::QuerySchema,
                    "create" => Command::CreateState,
                    "destroy" => Command::DestroyState,
                    "checkpoint" => Command::Checkpoint,
                    "write" => Command::WriteConfig("config.ron".into()),
                    "quit" | "exit" => Command::Quit,
                    s if s.starts_with("set ") => match parse_set_command(&s[4..]) {
                        Ok(patch) => Command::UpdateConfig(patch),
                        Err(e) => {
                            eprintln!("parse error: {}", e);
                            continue;
                        }
                    },
                    _ => {
                        eprintln!("unknown command: {}", trimmed);
                        eprintln!(
                            "commands: run, pause, step, status, config, schema, set <key>=<value>, create, destroy, checkpoint, write, quit"
                        );
                        continue;
                    }
                };

                let is_quit = matches!(cmd, Command::Quit);
                if handle.cmd_tx.send(cmd).is_err() {
                    break;
                }
                if is_quit {
                    break;
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                let _ = handle.cmd_tx.send(Command::Quit);
                break;
            }
            Err(e) => {
                eprintln!("readline error: {}", e);
                break;
            }
        }
    }

    printer.join().unwrap();
}

/// Parse "key=value" into a JSON merge patch.
fn parse_set_command(s: &str) -> Result<Value, String> {
    let (key, val_str) = s
        .split_once('=')
        .ok_or_else(|| "expected key=value".to_string())?;
    let key = key.trim();
    let val_str = val_str.trim();
    let leaf: Value =
        serde_json::from_str(val_str).unwrap_or_else(|_| Value::String(val_str.into()));
    Ok(merge(
        Value::Object(Default::default()),
        build_nested(key, leaf),
    ))
}
