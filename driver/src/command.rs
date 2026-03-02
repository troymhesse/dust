//! Command/message protocol for the simulation driver.

use crate::driver::DriverState;
use serde_json::Value;
use std::collections::HashMap;

/// Commands sent from a frontend to the driver.
pub enum Command {
    /// Start or resume the simulation loop.
    Run,
    /// Pause after the current iteration completes.
    Pause,
    /// Advance exactly one iteration, then return to idle.
    Step,
    /// Request current status (iteration, time, mode).
    QueryStatus,
    /// Request the current config as a JSON value.
    QueryConfig,
    /// Request the current config as a RON string.
    QueryConfigRon,
    /// Request the JSON schema for the simulation config.
    QuerySchema,
    /// Apply a partial config update (JSON merge patch).
    UpdateConfig(Value),
    /// Replace the full config from a RON string.
    UpdateConfigRon(String),
    /// Update a single config section by name ("driver", "physics", etc.) from a RON string.
    UpdateConfigSection { section: String, ron: String },
    /// Create simulation state from the current initial config.
    CreateState,
    /// Destroy the current simulation state.
    DestroyState,
    /// Write a checkpoint to disk immediately.
    Checkpoint,
    /// Write the current config to a file in RON format.
    WriteConfig(String),
    /// Load a RON config file from disk.
    LoadConfig(String),
    /// Load a checkpoint (.mpk) file from disk.
    LoadCheckpoint(String),
    /// Request plot data (1D series and 2D fields) from products.
    QueryPlotData,
    /// Shut down the driver.
    Quit,
}

/// Messages sent from the driver back to the frontend.
pub enum Message {
    /// Emitted after each completed iteration.
    StepCompleted {
        iteration: i64,
        time: f64,
        seconds: f64,
        display: String,
    },
    /// The simulation ended naturally (finished returned true).
    SimulationDone,
    /// Current driver status.
    Status {
        mode: DriverMode,
        iteration: i64,
        time: f64,
    },
    /// Response to QueryConfig.
    Config(Value),
    /// Response to QueryConfigRon: four RON strings, one per config section.
    ConfigSections {
        driver: String,
        physics: String,
        initial: String,
        compute: String,
    },
    /// Response to QuerySchema.
    Schema(Value),
    /// Acknowledgment of UpdateConfig or UpdateConfigRon.
    ConfigUpdated(Result<(), String>),
    /// A checkpoint was written.
    CheckpointWritten { path: String },
    /// Simulation state was created from initial config.
    StateCreated,
    /// Simulation state was destroyed.
    StateDestroyed,
    /// Driver bookkeeping and physics status for display.
    StateInfo {
        driver_state: DriverState,
        solver_status: Option<Value>,
    },
    /// Config was written to a file.
    ConfigWritten { path: String },
    /// A config file was loaded from disk.
    ConfigLoaded { path: String },
    /// A checkpoint file was loaded from disk.
    CheckpointLoaded { path: String },
    /// Plot data from products: 1D series and 2D fields.
    PlotData {
        linear: HashMap<String, Vec<f64>>,
        planar: HashMap<String, (usize, usize, Vec<f64>)>,
    },
    /// A non-fatal error or warning.
    Error(String),
    /// The driver is shutting down.
    Finished,
}

/// Observable mode of the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverMode {
    /// Waiting for a command.
    Idle,
    /// Actively iterating.
    Running,
}
