//! Command/message protocol for the simulation driver.

use serde_json::Value;

/// Commands sent from a frontend to the driver.
#[derive(Clone)]
pub enum Command {
    /// Start or resume the simulation loop.
    Run,
    /// Pause after the current iteration completes.
    Pause,
    /// Advance exactly one iteration, then return to idle.
    Step,
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
    /// Request the current config as a JSON value.
    QueryConfig,
    /// Request the current config as a RON string.
    QueryConfigRon,
    /// Request the JSON schema for the simulation config.
    QuerySchema,
    /// Shut down the driver.
    Quit,
}

/// Low-frequency events sent from the driver to the frontend.
///
/// These represent discrete state transitions and responses to commands.
/// Continuous data (step progress, plot data) flows through [`crate::watch::Watch`]
/// instead.
pub enum Event {
    /// The simulation ended naturally (finished returned true).
    SimulationDone,
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
    /// Config was written to a file.
    ConfigWritten { path: String },
    /// A config file was loaded from disk.
    ConfigLoaded { path: String },
    /// A checkpoint file was loaded from disk.
    CheckpointLoaded { path: String },
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
