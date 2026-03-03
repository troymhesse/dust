//! Particle simulation with GPUI visualization.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use driver::command::Command;
use driver::config::SimulationConfig;
use driver::gpui_frontend::{
    self, CreateState, DestroyState, DriverFooter, DriverLog, Quit, SnapshotReader, Step,
    ToggleLog, ToggleRun, WriteCheckpoint, WriteConfig,
};
use driver::worker::DriverHandle;
use driver::{Action, CliArgs, Driver, DriverState, Mode, PlotData, Solver, StepInfo, Validate};
use gpui::{
    App, AppContext as _, Application, Context, Entity, FocusHandle, Focusable,
    InteractiveElement as _, IntoElement, Menu, MenuItem, ParentElement, Render,
    StatefulInteractiveElement as _, Styled, Window, WindowOptions, div, px, size,
};
use gpui_component::{ActiveTheme, Root, h_flex, scroll::ScrollableElement, v_flex};
use gpui_plot::{Plot, PlotStyle, Series, data_range};
use gpui_schema::{NodeFilter, SchemaForm};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ============================================================================
// Config types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct DustPhysics {
    /// Final simulation time
    tfinal: f64,
    /// Time step
    dt: f64,
    /// Gravitational softening length
    softening: f64,
    /// Mass of the central object
    central_mass: f64,
}

impl Default for DustPhysics {
    fn default() -> Self {
        Self {
            tfinal: 10.0,
            dt: 0.001,
            softening: 0.01,
            central_mass: 1.0,
        }
    }
}

impl Validate for DustPhysics {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct DustInitial {
    /// Number of particles
    num_particles: usize,
    /// Initial condition setup
    setup: DustSetup,
}

impl Default for DustInitial {
    fn default() -> Self {
        Self {
            num_particles: 1000,
            setup: DustSetup::Ring,
        }
    }
}

impl Validate for DustInitial {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum DustSetup {
    /// Particles on a circular ring
    Ring,
    /// Particles in a randomized disk
    RandomDisk,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct DustCompute {}

impl Default for DustCompute {
    fn default() -> Self {
        Self {}
    }
}

impl Validate for DustCompute {}

// ============================================================================
// State and Products
// ============================================================================

#[derive(Serialize, Deserialize)]
struct State {
    time: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
}

#[derive(Serialize)]
struct DustProducts {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl PlotData for DustProducts {
    fn linear_data(&self) -> HashMap<String, Vec<f64>> {
        let mut map = HashMap::new();
        map.insert("x".into(), self.x.clone());
        map.insert("y".into(), self.y.clone());
        map
    }
}

#[derive(Debug, Clone, Serialize)]
struct DustStatus {
    time: f64,
    num_particles: usize,
}

// ============================================================================
// Node filter
// ============================================================================

struct DustFilter {
    has_state: bool,
}

impl NodeFilter for DustFilter {
    fn enabled(&self, path: &str) -> bool {
        if self.has_state {
            !path.starts_with("initial.") && !path.starts_with("compute.")
        } else {
            true
        }
    }
}

// ============================================================================
// Solver
// ============================================================================

struct Dust {
    physics: DustPhysics,
    initial: DustInitial,
}

impl Dust {
    /// Compute gravitational acceleration from central mass at origin.
    fn acceleration(&self, x: f64, y: f64) -> (f64, f64) {
        let eps = self.physics.softening;
        let r2 = x * x + y * y + eps * eps;
        let r = r2.sqrt();
        let a = -self.physics.central_mass / (r * r2);
        (a * x, a * y)
    }
}

impl Solver for Dust {
    type State = State;
    type Products = DustProducts;
    type Status = DustStatus;
    type Physics = DustPhysics;
    type Initial = DustInitial;
    type Compute = DustCompute;

    fn new(config: (DustPhysics, DustInitial, DustCompute)) -> Self {
        let (physics, initial, _compute) = config;
        Dust { physics, initial }
    }

    fn initial(&self) -> State {
        let n = self.initial.num_particles;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);

        match self.initial.setup {
            DustSetup::Ring => {
                for i in 0..n {
                    let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                    let r = 1.0;
                    let px = r * theta.cos();
                    let py = r * theta.sin();
                    // Circular orbital velocity: v = sqrt(GM/r)
                    let v = (self.physics.central_mass / r).sqrt();
                    x.push(px);
                    y.push(py);
                    vx.push(-v * theta.sin());
                    vy.push(v * theta.cos());
                }
            }
            DustSetup::RandomDisk => {
                // Simple deterministic pseudo-random using golden ratio
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                for i in 0..n {
                    let r = 0.3 + 0.7 * (i as f64 / n as f64).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * (i as f64 * phi);
                    let px = r * theta.cos();
                    let py = r * theta.sin();
                    let v = (self.physics.central_mass / r).sqrt();
                    x.push(px);
                    y.push(py);
                    vx.push(-v * theta.sin());
                    vy.push(v * theta.cos());
                }
            }
        }

        State {
            time: 0.0,
            x,
            y,
            vx,
            vy,
        }
    }

    fn finished(&self, state: &State) -> bool {
        state.time >= self.physics.tfinal
    }

    fn time(&self, state: &State) -> f64 {
        state.time
    }

    fn timestep(&self, _state: &State) -> f64 {
        self.physics.dt
    }

    fn advance(&self, mut state: State, dt: f64) -> State {
        let n = state.x.len();

        // Leapfrog (kick-drift-kick)
        for i in 0..n {
            let (ax, ay) = self.acceleration(state.x[i], state.y[i]);
            state.vx[i] += 0.5 * dt * ax;
            state.vy[i] += 0.5 * dt * ay;
        }
        for i in 0..n {
            state.x[i] += dt * state.vx[i];
            state.y[i] += dt * state.vy[i];
        }
        for i in 0..n {
            let (ax, ay) = self.acceleration(state.x[i], state.y[i]);
            state.vx[i] += 0.5 * dt * ax;
            state.vy[i] += 0.5 * dt * ay;
        }

        state.time += dt;
        state
    }

    fn products(&self, state: &State) -> DustProducts {
        DustProducts {
            x: state.x.clone(),
            y: state.y.clone(),
        }
    }

    fn status(&self, state: &State) -> DustStatus {
        DustStatus {
            time: state.time,
            num_particles: state.x.len(),
        }
    }

    fn message(&self, _state: &State, info: &StepInfo) -> String {
        format!(
            "[{:06}] t={:.6e}  dt={:.4e}  n={}",
            info.iteration, info.time, info.seconds, self.initial.num_particles,
        )
    }
}

// ============================================================================
// GPUI Application
// ============================================================================

struct DustApp {
    handle: DriverHandle,
    form: Entity<SchemaForm>,
    plot: Entity<Plot>,
    focus_handle: FocusHandle,
    snapshot_reader: SnapshotReader,
    log: DriverLog,
    show_log: Rc<Cell<bool>>,
}

impl Focusable for DustApp {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl DustApp {
    /// Read the latest snapshot, diff it, drain events, and perform
    /// app-specific updates (plot data, config filter).
    fn read_snapshot(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let diff = self.snapshot_reader.update(&self.handle.snapshot);

        // Drain events into the log (generic)
        self.log.drain_events(&self.handle.event_rx);

        // Log iteration messages when iteration advances
        if diff.iteration_advanced && !self.snapshot_reader.status_text().is_empty() {
            self.log
                .log_iteration(self.snapshot_reader.status_text().to_string());
        }

        // App-specific: update plot data (preserves pan/zoom view state)
        if diff.iteration_advanced || diff.state_changed {
            let snap = self.snapshot_reader.snapshot();
            let style = PlotStyle::from_theme(cx.theme());
            let first_data = diff.state_changed && snap.has_state;
            if let (Some(x), Some(y)) = (snap.linear.get("x"), snap.linear.get("y")) {
                self.plot.update(cx, |plot, cx| {
                    plot.set_series(vec![
                        Series::scatter(x.clone(), y.clone()).label("particles"),
                    ]);
                    plot.set_style(style);
                    if first_data {
                        let (xmin, xmax) = data_range(x);
                        let (ymin, ymax) = data_range(y);
                        plot.set_x_range(xmin, xmax);
                        plot.set_y_range(ymin, ymax);
                    }
                    cx.notify();
                });
            } else if !snap.has_state {
                self.plot.update(cx, |plot, cx| {
                    plot.set_series(vec![]);
                    plot.set_style(style);
                    cx.notify();
                });
            }
        }

        // App-specific: update form filter when state existence changes
        if diff.state_changed {
            let has_state = self.snapshot_reader.has_state();
            self.form.update(cx, |form, cx| {
                form.set_filter(DustFilter { has_state }, window, cx);
            });
        }

        window.request_animation_frame();
    }

    fn running(&self) -> bool {
        self.snapshot_reader.running()
    }

    fn has_state(&self) -> bool {
        self.snapshot_reader.has_state()
    }

    fn editing(&self, cx: &Context<Self>) -> bool {
        self.form.read(cx).editing()
    }

    fn send(&self, cmd: Command) {
        let _ = self.handle.cmd_tx.send(cmd);
    }
}

impl Render for DustApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.read_snapshot(window, cx);

        if self.form.read(cx).is_dirty() {
            let value = self.form.read(cx).to_value();
            self.send(Command::UpdateConfig(value));
        }

        let show_log = self.show_log.get();

        // Right panel: either plot or log
        let right_panel = if show_log {
            self.log.render_log_view(cx).into_any_element()
        } else {
            div()
                .flex_1()
                .size_full()
                .child(self.plot.clone())
                .into_any_element()
        };

        // Footer (generic driver widget)
        let cmd_tx = self.handle.cmd_tx.clone();
        let footer = DriverFooter::new(&self.snapshot_reader, &self.log, self.show_log.clone())
            .cmd_tx(cmd_tx)
            .render(cx);

        let focus_handle = self.focus_handle.clone();
        let focus_handle2 = self.focus_handle.clone();

        v_flex()
            .id("dust-app-root")
            .track_focus(&self.focus_handle)
            .size_full()
            .on_action(cx.listener(|this, _: &ToggleRun, _, cx| {
                if this.editing(cx) {
                    return;
                }
                if !this.has_state() {
                    return;
                }
                if this.running() {
                    this.send(Command::Pause);
                } else {
                    this.send(Command::Run);
                }
            }))
            .on_action(cx.listener(|this, _: &Step, _, cx| {
                if this.editing(cx) {
                    return;
                }
                if this.has_state() && !this.running() {
                    this.send(Command::Step);
                }
            }))
            .on_action(cx.listener(|this, _: &CreateState, _, cx| {
                if this.editing(cx) {
                    return;
                }
                if !this.has_state() {
                    this.send(Command::CreateState);
                }
            }))
            .on_action(cx.listener(|this, _: &DestroyState, _, cx| {
                if this.editing(cx) {
                    return;
                }
                if this.has_state() && !this.running() {
                    this.send(Command::DestroyState);
                }
            }))
            .on_action(cx.listener(|this, _: &WriteCheckpoint, _, cx| {
                if this.editing(cx) {
                    return;
                }
                if this.has_state() && !this.running() {
                    this.send(Command::Checkpoint);
                }
            }))
            .on_action(cx.listener(|this, _: &WriteConfig, _, cx| {
                if this.editing(cx) {
                    return;
                }
                this.send(Command::WriteConfig("config.ron".into()));
            }))
            .on_action(cx.listener(|this, _: &ToggleLog, _, cx| {
                if this.editing(cx) {
                    return;
                }
                this.show_log.set(!this.show_log.get());
            }))
            .child(
                // Main content: config panel + plot/log
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        div()
                            .w(px(400.0))
                            .border_r_1()
                            .border_color(cx.theme().border)
                            .overflow_y_scrollbar()
                            .child(self.form.clone()),
                    )
                    .child(
                        div()
                            .id("right-panel")
                            .flex_1()
                            .size_full()
                            .on_click(move |_, window, _cx| {
                                focus_handle.focus(window);
                            })
                            .child(right_panel),
                    ),
            )
            .child(
                div()
                    .id("footer-click-area")
                    .on_click(move |_, window, _cx| {
                        focus_handle2.focus(window);
                    })
                    .child(footer),
            )
    }
}

fn main() {
    let cli = match CliArgs::from_env() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    if cli.mode != Mode::Gui || cli.action != Action::Run {
        driver::app::run::<Dust>(cli);
        return;
    }

    let app = Application::new();

    app.run(move |cx: &mut App| {
        gpui_component::init(cx);
        gpui_schema::init(cx);

        cx.on_action(|_: &Quit, cx| cx.quit());
        gpui_frontend::register_keybindings(cx);

        cx.set_menus(vec![Menu {
            name: "Dust".into(),
            items: vec![MenuItem::action("Quit", Quit)],
        }]);

        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        cx.spawn(async move |cx| {
            let window_options = cx.update(|cx| WindowOptions {
                window_bounds: Some(gpui::WindowBounds::Windowed(gpui::Bounds::centered(
                    None,
                    size(px(1400.0), px(800.0)),
                    cx,
                ))),
                ..Default::default()
            })?;

            let window = cx.open_window(window_options, |window, cx| {
                let config = SimulationConfig::<Dust>::default();
                let solver = Dust::new((
                    config.physics.clone(),
                    config.initial.clone(),
                    config.compute.clone(),
                ));
                let (driver, init_msgs) =
                    Driver::new(config.clone(), solver, None, DriverState::new());
                let handle = driver::worker::spawn(driver, init_msgs);

                let schema = schemars::schema_for!(SimulationConfig<Dust>);
                let value = serde_json::to_value(&config).unwrap();
                let form = cx.new(|cx| {
                    let mut form = SchemaForm::new(&schema, &value, window, cx);
                    form.set_filter(DustFilter { has_state: false }, window, cx);
                    form
                });
                let style = PlotStyle::from_theme(cx.theme());
                let plot = cx.new(|_cx| Plot::new().grid(true).aspect_ratio(1.0).style(style));

                let app_entity = cx.new(|cx| {
                    let focus_handle = cx.focus_handle();
                    focus_handle.focus(window);
                    DustApp {
                        handle,
                        form,
                        plot,
                        focus_handle,
                        snapshot_reader: SnapshotReader::new(),
                        log: DriverLog::new(),
                        show_log: Rc::new(Cell::new(false)),
                    }
                });

                cx.new(|cx| Root::new(app_entity, window, cx))
            })?;

            window.update(cx, |_, window, _| {
                window.activate_window();
            })?;

            cx.update(|cx| cx.activate(true))?;

            Ok::<_, anyhow::Error>(())
        })
        .detach();
    });
}
