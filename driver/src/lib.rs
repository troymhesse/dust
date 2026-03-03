//! Generic simulation driver framework with CLI, REPL, and TUI frontends.

pub mod app;
pub mod command;
pub mod config;
pub mod driver;
pub mod format;
pub mod frontend;
pub mod solver;
pub mod watch;
pub mod worker;

pub use app::{Action, CliArgs, Mode};
pub use config::{build_nested, merge, DriverConfig, SimulationConfig};
pub use driver::{Driver, DriverState};
pub use solver::{PlotData, Solver, StepInfo, Validate};
pub use watch::{Snapshot, Watch};

#[cfg(feature = "gpui")]
pub use frontend::gpui as gpui_frontend;
