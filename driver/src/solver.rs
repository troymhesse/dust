//! Core simulation trait.

use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;

/// Info passed to `Physics::message` after each step.
pub struct StepInfo {
    pub iteration: i64,
    pub time: f64,
    pub seconds: f64,
}

/// Trait for products types that can provide data for plotting.
pub trait PlotData {
    /// Named 1D series for line plots (e.g. x vs rho).
    fn linear_data(&self) -> HashMap<String, Vec<f64>> {
        HashMap::new()
    }
    /// Named 2D fields for heatmap plots. Each entry is (rows, cols, row-major data).
    fn planar_data(&self) -> HashMap<String, (usize, usize, Vec<f64>)> {
        HashMap::new()
    }
}

/// Trait for validating config values before they are accepted.
pub trait Validate {
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    /// Config paths that should be disabled (non-editable) in the TUI.
    ///
    /// Return dot-separated paths like `"compute.backend.cuda"` for
    /// enum variants unavailable due to feature gates.
    fn disabled_config_paths() -> Vec<String> {
        vec![]
    }
}

/// Solver provides initial conditions and advance function
pub trait Solver {
    type State: Serialize + DeserializeOwned;
    type Products: Serialize + PlotData;
    type Status: Serialize;
    type Physics: Serialize + DeserializeOwned + Default + Clone + JsonSchema + Validate;
    type Initial: Serialize + DeserializeOwned + Default + Clone + JsonSchema + Validate;
    type Compute: Serialize + DeserializeOwned + Default + Clone + JsonSchema + Validate;

    fn new(config: (Self::Physics, Self::Initial, Self::Compute)) -> Self;
    fn initial(&self) -> Self::State;
    fn finished(&self, state: &Self::State) -> bool;
    fn time(&self, state: &Self::State) -> f64;
    fn advance(&self, state: Self::State, dt: f64) -> Self::State;
    fn timestep(&self, state: &Self::State) -> f64;
    fn products(&self, state: &Self::State) -> Self::Products;
    fn status(&self, state: &Self::State) -> Self::Status;
    fn message(&self, state: &Self::State, info: &StepInfo) -> String;
}
