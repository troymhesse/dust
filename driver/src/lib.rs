//! Generic simulation driver framework with CLI, REPL, and TUI frontends.

pub mod app;
pub mod command;
pub mod config;
pub mod driver;
pub mod format;
pub mod frontend;
pub mod solver;
pub mod worker;

pub use config::{DriverConfig, SimulationConfig, build_nested, merge};
pub use driver::{Driver, DriverState};
pub use solver::{PlotData, Solver, StepInfo, Validate};
