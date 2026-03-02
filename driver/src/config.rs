//! Simulation configuration types.

use crate::driver::DriverState;
use crate::solver::Solver;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// Driver configuration (the `driver` section of the RON config).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct DriverConfig {
    pub checkpoint_interval: Option<f64>,
    pub output_dir: String,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: None,
            output_dir: ".".into(),
        }
    }
}

/// Top-level simulation config parsed from RON.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    default,
    bound(
        serialize = "S::Physics: Serialize, S::Initial: Serialize, S::Compute: Serialize",
        deserialize = "S::Physics: Default + serde::de::DeserializeOwned, S::Initial: Default + serde::de::DeserializeOwned, S::Compute: Default + serde::de::DeserializeOwned"
    )
)]
#[schemars(bound = "S::Physics: JsonSchema + Default, S::Initial: JsonSchema + Default, S::Compute: JsonSchema + Default")]
pub struct SimulationConfig<S: Solver> {
    pub driver: DriverConfig,
    pub physics: S::Physics,
    pub initial: S::Initial,
    pub compute: S::Compute,
}

impl<S: Solver> Default for SimulationConfig<S> {
    fn default() -> Self {
        Self {
            driver: DriverConfig::default(),
            physics: S::Physics::default(),
            initial: S::Initial::default(),
            compute: S::Compute::default(),
        }
    }
}

impl<S: Solver> Clone for SimulationConfig<S> {
    fn clone(&self) -> Self {
        Self {
            driver: self.driver.clone(),
            physics: self.physics.clone(),
            initial: self.initial.clone(),
            compute: self.compute.clone(),
        }
    }
}

/// A complete checkpoint: state + typed config + driver bookkeeping + products.
///
/// The state type `St` is separate from `S::State` so that serialization can
/// use a borrowed `&S::State` while deserialization uses an owned `S::State`.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "St: Serialize, S::Physics: Serialize, S::Initial: Serialize, S::Compute: Serialize, Pr: Serialize",
    deserialize = "St: serde::de::DeserializeOwned, S::Physics: serde::de::DeserializeOwned + Default, S::Initial: serde::de::DeserializeOwned + Default, S::Compute: serde::de::DeserializeOwned + Default, Pr: serde::de::DeserializeOwned + Default",
))]
pub(crate) struct Checkpoint<S: Solver, St, Pr> {
    pub state: St,
    pub config: SimulationConfig<S>,
    pub driver: DriverState,
    pub products: Pr,
}

/// Build a nested JSON object from a dot-separated path and a leaf value.
pub fn build_nested(dot_path: &str, leaf: Value) -> Value {
    let mut result = leaf;
    for seg in dot_path.rsplit('.') {
        let mut map = serde_json::Map::new();
        map.insert(seg.into(), result);
        result = Value::Object(map);
    }
    result
}

/// Recursively merge two JSON values.
///
/// When both are objects whose keys overlap, merge recursively. When both are
/// single-key objects with different uppercase keys (i.e. different enum
/// variants), the override replaces entirely.
pub fn merge(base: Value, over: Value) -> Value {
    match (base, over) {
        (Value::Object(base_map), Value::Object(over_map))
            if is_variant_object(&base_map)
                && is_variant_object(&over_map)
                && base_map.keys().next() != over_map.keys().next() =>
        {
            Value::Object(over_map)
        }
        (Value::Object(mut base_map), Value::Object(over_map)) => {
            for (k, v) in over_map {
                let merged = match base_map.remove(&k) {
                    Some(bv) => merge(bv, v),
                    None => v,
                };
                base_map.insert(k, merged);
            }
            Value::Object(base_map)
        }
        (_, over) => over,
    }
}

fn is_variant_object(map: &serde_json::Map<String, Value>) -> bool {
    map.len() == 1
        && map
            .keys()
            .next()
            .is_some_and(|k| k.starts_with(char::is_uppercase))
}

pub(crate) fn ensure_output_dir(path: &Path) {
    if path != Path::new(".") {
        std::fs::create_dir_all(path).unwrap_or_else(|e| {
            eprintln!(
                "error: failed to create output dir '{}': {}",
                path.display(),
                e
            );
            std::process::exit(1);
        });
    }
}
