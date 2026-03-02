//! Fire-and-forget CLI frontend.
//!
//! Sends `Run`, prints messages to stdout, exits when simulation completes.

use crate::command::{Command, Message};
use crate::worker::DriverHandle;

pub fn run(handle: DriverHandle) {
    handle.cmd_tx.send(Command::Run).unwrap();

    loop {
        match handle.msg_rx.recv() {
            Ok(Message::StepCompleted { display, .. }) => println!("{}", display),
            Ok(Message::CheckpointWritten { path }) => println!("wrote {}", path),
            Ok(Message::SimulationDone) => {
                let _ = handle.cmd_tx.send(Command::Quit);
            }
            Ok(Message::Error(e)) => eprintln!("error: {}", e),
            Ok(Message::Finished) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }

    handle.thread.join().unwrap();
    println!();
}
