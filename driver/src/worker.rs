//! Worker thread: runs the driver on a background thread, communicating via channels.

use crate::command::{Command, Message};
use crate::driver::Driver;
use crate::solver::Solver;
use std::sync::mpsc::{Receiver, Sender};

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

fn worker_main<S: Solver>(
    mut driver: Driver<S>,
    cmd_rx: Receiver<Command>,
    msg_tx: Sender<Message>,
) {
    loop {
        let cmd = if driver.is_running() {
            match cmd_rx.try_recv() {
                Ok(cmd) => cmd,
                Err(std::sync::mpsc::TryRecvError::Empty) => Command::Run,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        } else {
            match cmd_rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        };

        let is_quit = matches!(cmd, Command::Quit);
        for msg in driver.accept(cmd) {
            let _ = msg_tx.send(msg);
        }
        if is_quit {
            break;
        }
    }
}
