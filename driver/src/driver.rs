//! Simulation driver

use crate::command::{Command, DriverMode, Event};
use crate::config::{Checkpoint, SimulationConfig, ensure_output_dir};
use crate::solver::{PlotData, Solver, StepInfo, Validate};
use crate::watch::{Snapshot, Watch};
use serde::{Serialize, de};
use serde_json::Value;
use std::path::Path;

/// A type that skips any serialized value during deserialization.
///
/// Used when loading checkpoints: products are recomputed from state,
/// so we skip deserializing them (they may contain types like `Field`
/// that only implement `Serialize`, not `Deserialize`).
#[derive(Default)]
struct IgnoredProducts;

impl<'de> de::Deserialize<'de> for IgnoredProducts {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        de::IgnoredAny::deserialize(deserializer)?;
        Ok(IgnoredProducts)
    }
}

/// Serializable driver bookkeeping state, stored in checkpoints.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct DriverState {
    pub iteration: i64,
    pub checkpoint_number: usize,
    pub next_checkpoint_time: f64,
}

impl DriverState {
    pub fn new() -> Self {
        DriverState {
            iteration: 0,
            checkpoint_number: 0,
            next_checkpoint_time: 0.0,
        }
    }

    pub fn catch_up_checkpoint_time(&mut self, current_time: f64, interval: Option<f64>) {
        if let Some(interval) = interval {
            while self.next_checkpoint_time <= current_time + 1e-15 {
                self.next_checkpoint_time += interval;
            }
        }
    }
}

/// The simulation driver. Owns all simulation state and processes commands.
pub struct Driver<S: Solver> {
    config: SimulationConfig<S>,
    solver: S,
    state: Option<S::State>,
    driver_state: DriverState,
    mode: DriverMode,
    schema: Value,
    /// Status line from the most recent step.
    status_text: String,
}

impl<S: Solver> Driver<S> {
    /// Create a new driver. Writes an initial checkpoint for fresh starts
    /// and emits any resulting events.
    pub fn new(
        config: SimulationConfig<S>,
        solver: S,
        state: Option<S::State>,
        driver_state: DriverState,
    ) -> (Self, Vec<Event>) {
        let schema = serde_json::to_value(schemars::schema_for!(SimulationConfig<S>)).unwrap();

        let mut driver = Driver {
            config,
            solver,
            state,
            driver_state,
            mode: DriverMode::Idle,
            schema,
            status_text: String::new(),
        };

        let mut events = Vec::new();

        if driver.driver_state.iteration == 0 {
            if let Some(s) = driver.state.take() {
                events.push(driver.write_checkpoint(&s));
                if let Some(interval) = driver.config.driver.checkpoint_interval {
                    driver.driver_state.next_checkpoint_time += interval;
                }
                driver.state = Some(s);
            }
        }

        (driver, events)
    }

    /// Whether the driver is currently running (wants to step continuously).
    pub fn is_running(&self) -> bool {
        self.mode == DriverMode::Running
    }

    /// Write the current observable state into the watch channel.
    pub fn write_snapshot(&self, watch: &Watch<Snapshot>) {
        let (linear, planar) = if let Some(state) = &self.state {
            let prods = self.solver.products(state);
            (prods.linear_data(), prods.planar_data())
        } else {
            (
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            )
        };

        watch.write(Snapshot {
            mode: self.mode,
            has_state: self.state.is_some(),
            iteration: self.driver_state.iteration,
            time: self
                .state
                .as_ref()
                .map(|s| self.solver.time(s))
                .unwrap_or(0.0),
            status_text: self.status_text.clone(),
            linear,
            planar,
        });
    }

    /// Return the JSON schema for the simulation config.
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Process a command, returning any events produced.
    pub fn accept(&mut self, cmd: Command) -> Vec<Event> {
        let mut events = Vec::new();

        match cmd {
            Command::Run => {
                if self.state.is_none() {
                    events.push(Event::Error("no state".into()));
                } else {
                    self.mode = DriverMode::Running;
                    self.step_or_finish(&mut events);
                }
            }
            Command::Pause => {
                self.mode = DriverMode::Idle;
            }
            Command::Step => {
                if let Some(s) = self.state.take() {
                    if !self.solver.finished(&s) {
                        let (s, step_events) = self.do_step(s);
                        events.extend(step_events);
                        self.state = Some(s);
                    } else {
                        self.state = Some(s);
                    }
                    self.mode = DriverMode::Idle;
                } else {
                    events.push(Event::Error("no state".into()));
                }
            }
            Command::UpdateConfig(patch) => {
                events.extend(self.update_config(patch));
            }
            Command::UpdateConfigRon(ron) => {
                events.extend(self.update_config_ron(&ron));
            }
            Command::UpdateConfigSection { section, ron } => {
                events.extend(self.update_config_section(&section, &ron));
            }
            Command::CreateState => {
                if self.state.is_some() {
                    events.push(Event::Error("state already exists".into()));
                } else {
                    let s = self.solver.initial();
                    self.driver_state = DriverState::new();
                    events.push(self.write_checkpoint(&s));
                    self.state = Some(s);
                    events.push(Event::StateCreated);
                }
            }
            Command::DestroyState => {
                self.state = None;
                self.mode = DriverMode::Idle;
                self.status_text.clear();
                events.push(Event::StateDestroyed);
            }
            Command::WriteConfig(path) => {
                events.extend(self.write_config(&path));
            }
            Command::Checkpoint => {
                if let Some(s) = self.state.take() {
                    events.push(self.write_checkpoint(&s));
                    self.state = Some(s);
                } else {
                    events.push(Event::Error("no state".into()));
                }
            }
            Command::LoadConfig(path) => {
                events.extend(self.load_config(&path));
            }
            Command::LoadCheckpoint(path) => {
                events.extend(self.load_checkpoint(&path));
            }
            Command::QueryConfig => {
                events.push(Event::Config(
                    serde_json::to_value(&self.config).unwrap_or_default(),
                ));
            }
            Command::QueryConfigRon => {
                events.push(self.config_sections());
            }
            Command::QuerySchema => {
                events.push(Event::Schema(self.schema.clone()));
            }
            Command::Quit => {
                events.push(Event::Finished);
            }
        }

        events
    }

    // ========================================================================
    // Simulation stepping
    // ========================================================================

    fn step_or_finish(&mut self, events: &mut Vec<Event>) {
        let s = self.state.take().unwrap();
        if !self.solver.finished(&s) {
            let (s, step_events) = self.do_step(s);
            events.extend(step_events);
            self.state = Some(s);
        } else {
            self.state = Some(s);
            events.push(Event::SimulationDone);
            self.mode = DriverMode::Idle;
        }
    }

    fn do_step(&mut self, state: S::State) -> (S::State, Vec<Event>) {
        let mut events = Vec::new();
        self.maybe_write_checkpoint(&state, &mut events);
        let mut dt = self.solver.timestep(&state);
        if let Some(max_dt) = self.time_to_next_checkpoint(self.solver.time(&state)) {
            dt = dt.min(max_dt);
        }
        let (state, seconds) = timed(|| self.solver.advance(state, dt));
        self.driver_state.iteration += 1;

        let info = StepInfo {
            iteration: self.driver_state.iteration,
            time: self.solver.time(&state),
            seconds,
        };
        self.status_text = self.solver.message(&state, &info);

        (state, events)
    }

    // ========================================================================
    // Checkpointing
    // ========================================================================

    fn maybe_write_checkpoint(&mut self, state: &S::State, events: &mut Vec<Event>) {
        if let Some(interval) = self.config.driver.checkpoint_interval {
            let tol = 1e-15;
            if (self.solver.time(state) - self.driver_state.next_checkpoint_time).abs() < tol {
                events.push(self.write_checkpoint(state));
                self.driver_state.next_checkpoint_time += interval;
            }
        }
    }

    fn time_to_next_checkpoint(&self, time: f64) -> Option<f64> {
        self.config
            .driver
            .checkpoint_interval
            .map(|_| self.driver_state.next_checkpoint_time - time)
    }

    fn write_checkpoint(&mut self, state: &S::State) -> Event {
        let output_dir = Path::new(&self.config.driver.output_dir);
        ensure_output_dir(output_dir);

        let name = output_dir.join(format!(
            "chkpt.{:04}.mpk",
            self.driver_state.checkpoint_number
        ));
        self.driver_state.checkpoint_number += 1;

        let chkpt = Checkpoint {
            state,
            config: self.config.clone(),
            driver: self.driver_state.clone(),
            products: self.solver.products(state),
        };

        let result = rmp_serde::to_vec_named(&chkpt)
            .map_err(|e| format!("failed to serialize checkpoint: {}", e))
            .and_then(|data| {
                std::fs::write(&name, &data)
                    .map_err(|e| format!("failed to write {}: {}", name.display(), e))
            });

        match result {
            Ok(()) => Event::CheckpointWritten {
                path: name.display().to_string(),
            },
            Err(e) => Event::Error(e),
        }
    }

    // ========================================================================
    // Config management
    // ========================================================================

    fn update_config(&mut self, patch: Value) -> Vec<Event> {
        if self.state.is_some() {
            if let Some(obj) = patch.as_object() {
                let current = serde_json::to_value(&self.config).unwrap_or_default();
                for section in ["initial", "compute"] {
                    if let Some(incoming) = obj.get(section) {
                        if current.get(section) != Some(incoming) {
                            return vec![Event::ConfigUpdated(Err(format!(
                                "{} config cannot be changed while state exists",
                                section
                            )))];
                        }
                    }
                }
            }
        }

        let has_changes = patch.as_object().map_or(false, |m| !m.is_empty());
        if has_changes {
            let result = try_update_config(&mut self.config, patch);
            if result.is_err() {
                return vec![Event::ConfigUpdated(result)];
            }
        }

        self.solver = self.new_solver();

        if let Some(ref s) = self.state {
            self.driver_state.catch_up_checkpoint_time(
                self.solver.time(s),
                self.config.driver.checkpoint_interval,
            );
        }

        vec![Event::ConfigUpdated(Ok(())), self.config_sections()]
    }

    fn update_config_ron(&mut self, ron: &str) -> Vec<Event> {
        let new_config: SimulationConfig<S> = match ron::from_str(ron) {
            Ok(c) => c,
            Err(e) => return vec![Event::ConfigUpdated(Err(format!("{}", e)))],
        };

        self.config = new_config;
        self.solver = self.new_solver();

        if let Some(ref s) = self.state {
            self.driver_state.catch_up_checkpoint_time(
                self.solver.time(s),
                self.config.driver.checkpoint_interval,
            );
        }

        vec![Event::ConfigUpdated(Ok(())), self.config_sections()]
    }

    fn update_config_section(&mut self, section: &str, ron: &str) -> Vec<Event> {
        let err = |msg: String| vec![Event::ConfigUpdated(Err(msg))];

        match section {
            "driver" => match ron::from_str(ron) {
                Ok(v) => self.config.driver = v,
                Err(e) => return err(format!("driver: {}", e)),
            },
            "physics" => match ron::from_str(ron) {
                Ok(v) => self.config.physics = v,
                Err(e) => return err(format!("physics: {}", e)),
            },
            "initial" => match ron::from_str(ron) {
                Ok(v) => self.config.initial = v,
                Err(e) => return err(format!("initial: {}", e)),
            },
            "compute" => match ron::from_str(ron) {
                Ok(v) => self.config.compute = v,
                Err(e) => return err(format!("compute: {}", e)),
            },
            other => return err(format!("unknown section: {}", other)),
        }

        self.solver = self.new_solver();

        if let Some(ref s) = self.state {
            self.driver_state.catch_up_checkpoint_time(
                self.solver.time(s),
                self.config.driver.checkpoint_interval,
            );
        }

        vec![Event::ConfigUpdated(Ok(())), self.config_sections()]
    }

    fn write_config(&self, path: &str) -> Vec<Event> {
        match ron::ser::to_string_pretty(&self.config, Self::ron_pretty()) {
            Ok(ron_str) => match std::fs::write(path, &ron_str) {
                Ok(()) => vec![Event::ConfigWritten { path: path.into() }],
                Err(e) => vec![Event::Error(format!("write config: {}", e))],
            },
            Err(e) => vec![Event::Error(format!("serialize config: {}", e))],
        }
    }

    fn load_config(&mut self, path: &str) -> Vec<Event> {
        match std::fs::read_to_string(path) {
            Err(e) => vec![Event::Error(format!("read {}: {}", path, e))],
            Ok(source) => match ron::from_str::<SimulationConfig<S>>(&source) {
                Err(e) => vec![Event::Error(format!("parse {}: {}", path, e))],
                Ok(new_config) => {
                    self.config = new_config;
                    self.solver = self.new_solver();
                    vec![
                        Event::ConfigLoaded { path: path.into() },
                        Event::ConfigUpdated(Ok(())),
                        self.config_sections(),
                    ]
                }
            },
        }
    }

    fn load_checkpoint(&mut self, path: &str) -> Vec<Event> {
        match std::fs::read(path) {
            Err(e) => vec![Event::Error(format!("read {}: {}", path, e))],
            Ok(data) => {
                match rmp_serde::from_slice::<Checkpoint<S, S::State, IgnoredProducts>>(&data) {
                    Err(e) => vec![Event::Error(format!("parse {}: {}", path, e))],
                    Ok(chkpt) => {
                        self.config = chkpt.config;
                        self.driver_state = chkpt.driver;
                        self.state = Some(chkpt.state);
                        self.mode = DriverMode::Idle;
                        self.solver = self.new_solver();
                        vec![
                            Event::CheckpointLoaded { path: path.into() },
                            Event::StateCreated,
                            self.config_sections(),
                        ]
                    }
                }
            }
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn config_sections(&self) -> Event {
        let p = Self::ron_pretty();
        Event::ConfigSections {
            driver: ron::ser::to_string_pretty(&self.config.driver, p.clone())
                .unwrap_or_else(|_| "()".into()),
            physics: ron::ser::to_string_pretty(&self.config.physics, p.clone())
                .unwrap_or_else(|_| "()".into()),
            initial: ron::ser::to_string_pretty(&self.config.initial, p.clone())
                .unwrap_or_else(|_| "()".into()),
            compute: ron::ser::to_string_pretty(&self.config.compute, p)
                .unwrap_or_else(|_| "()".into()),
        }
    }

    fn ron_pretty() -> ron::ser::PrettyConfig {
        ron::ser::PrettyConfig::new()
            .depth_limit(8)
            .struct_names(true)
            .enumerate_arrays(false)
    }

    fn new_solver(&self) -> S {
        S::new((
            self.config.physics.clone(),
            self.config.initial.clone(),
            self.config.compute.clone(),
        ))
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    (result, start.elapsed().as_secs_f64())
}

fn try_update_config<S: Solver>(
    config: &mut SimulationConfig<S>,
    patch: Value,
) -> Result<(), String> {
    let base = serde_json::to_value(&*config).map_err(|e| e.to_string())?;
    let merged = crate::config::merge(base, patch);
    let new_config: SimulationConfig<S> =
        serde_json::from_value(merged).map_err(|e| e.to_string())?;
    new_config.physics.validate()?;
    new_config.initial.validate()?;
    new_config.compute.validate()?;
    *config = new_config;
    Ok(())
}
