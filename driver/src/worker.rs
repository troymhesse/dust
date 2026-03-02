//! Worker thread: runs the driver on a background thread, communicating via channels.

use crate::command::{Command, Message};
use crate::driver::Driver;
use crate::solver::Solver;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

/// Handle returned by [`spawn`]. Provides command/message channels
/// and a join handle for the driver thread.
pub struct DriverHandle {
    pub cmd_tx: Sender<Command>,
    pub msg_rx: Receiver<Message>,
    pub thread: std::thread::JoinHandle<()>,
}

/// Spawn a thread that owns the driver and communicates via channels.
pub fn spawn<S: Solver + Send + 'static>(driver: Driver<S>, init_msgs: Vec<Message>) -> DriverHandle
where
    S::State: Send,
    S::Physics: Send,
    S::Initial: Send,
    S::Compute: Send,
{
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
    let (msg_tx, msg_rx) = std::sync::mpsc::channel();

    for msg in init_msgs {
        let _ = msg_tx.send(msg);
    }

    let thread = std::thread::spawn(move || {
        worker_main(driver, cmd_rx, msg_tx);
    });

    DriverHandle {
        cmd_tx,
        msg_rx,
        thread,
    }
}

/// Minimum time to spend stepping before checking for commands.
const STEP_BATCH_DURATION: Duration = Duration::from_millis(30);

fn worker_main<S: Solver>(
    mut driver: Driver<S>,
    cmd_rx: Receiver<Command>,
    msg_tx: Sender<Message>,
) {
    loop {
        if driver.is_running() {
            // Process any pending commands first
            while let Ok(cmd) = cmd_rx.try_recv() {
                let is_quit = matches!(cmd, Command::Quit);
                for msg in driver.accept(cmd) {
                    let _ = msg_tx.send(msg);
                }
                if is_quit {
                    return;
                }
            }

            // If we're still running after processing commands, step in a batch
            if driver.is_running() {
                let deadline = Instant::now() + STEP_BATCH_DURATION;
                while driver.is_running() {
                    for msg in driver.accept(Command::Run) {
                        let _ = msg_tx.send(msg);
                    }
                    if Instant::now() >= deadline {
                        break;
                    }
                }
            }
        } else {
            // Idle: block until a command arrives
            match cmd_rx.recv() {
                Ok(cmd) => {
                    let is_quit = matches!(cmd, Command::Quit);
                    for msg in driver.accept(cmd) {
                        let _ = msg_tx.send(msg);
                    }
                    if is_quit {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    }
}
