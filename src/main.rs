//! Particle simulation with GPUI visualization.

use driver::command::{Command, Message};
use driver::config::SimulationConfig;
use driver::worker::DriverHandle;
use driver::{Driver, DriverState, PlotData, Solver, StepInfo, Validate};
use gpui::{
    App, AppContext as _, Application, Context, Entity, InteractiveElement as _, IntoElement,
    KeyBinding, Menu, MenuItem, ParentElement, Render, StatefulInteractiveElement as _, Styled,
    Window, WindowOptions, actions, div, px, size,
};
use gpui_component::{ActiveTheme, Root, h_flex, scroll::ScrollableElement, v_flex};
use gpui_plot::{Plot, Series};
use gpui_schema::{NodeFilter, SchemaForm};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc::TryRecvError;

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
    /// Central mass strength
    central_mass: f64,
}

impl Default for DustPhysics {
    fn default() -> Self {
        Self {
            tfinal: 100.0,
            dt: 0.001,
            softening: 0.05,
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
    /// Initial configuration
    setup: DustSetup,
}

impl Default for DustInitial {
    fn default() -> Self {
        Self {
            num_particles: 200,
            setup: DustSetup::Ring,
        }
    }
}

impl Validate for DustInitial {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum DustSetup {
    /// Particles arranged in a ring
    Ring,
    /// Particles in a random disk
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
            "[{:06}] t={:.4e} n={}",
            info.iteration, info.time, self.initial.num_particles,
        )
    }
}

// ============================================================================
// GPUI Application
// ============================================================================

actions!(dust, [Quit]);

struct DustApp {
    handle: DriverHandle,
    form: Entity<SchemaForm>,
    plot: Entity<Plot>,
    running: bool,
    has_state: bool,
    status_text: String,
}

impl DustApp {
    fn drain_messages(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        loop {
            match self.handle.msg_rx.try_recv() {
                Ok(Message::StepCompleted { display, .. }) => {
                    self.status_text = display;
                    if self.running {
                        let _ = self.handle.cmd_tx.send(Command::QueryPlotData);
                    }
                }
                Ok(Message::PlotData { linear, .. }) => {
                    if let (Some(x), Some(y)) = (linear.get("x"), linear.get("y")) {
                        let new_plot = Plot::new()
                            .grid(true)
                            .aspect_ratio(1.0)
                            .x_label("x")
                            .y_label("y")
                            .series(Series::scatter(x.clone(), y.clone()).label("particles"));
                        self.plot = cx.new(|_cx| new_plot);
                    }
                    cx.notify();
                }
                Ok(Message::SimulationDone) => {
                    self.running = false;
                    self.status_text = "Simulation done".into();
                    let _ = self.handle.cmd_tx.send(Command::QueryPlotData);
                }
                Ok(Message::StateCreated) => {
                    self.has_state = true;
                    self.status_text = "State created".into();
                    self.form.update(cx, |form, cx| {
                        form.set_filter(DustFilter { has_state: true }, window, cx);
                    });
                    let _ = self.handle.cmd_tx.send(Command::QueryPlotData);
                }
                Ok(Message::StateDestroyed) => {
                    self.has_state = false;
                    self.status_text = "State destroyed".into();
                    self.form.update(cx, |form, cx| {
                        form.set_filter(DustFilter { has_state: false }, window, cx);
                    });
                    self.plot = cx.new(|_cx| Plot::new().grid(true).aspect_ratio(1.0));
                    cx.notify();
                }
                Ok(Message::Error(e)) => {
                    self.status_text = format!("Error: {}", e);
                }
                Ok(Message::ConfigUpdated(result)) => {
                    if let Err(e) = result {
                        self.status_text = format!("Config error: {}", e);
                    }
                }
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn send(&self, cmd: Command) {
        let _ = self.handle.cmd_tx.send(cmd);
    }
}

impl Render for DustApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.drain_messages(window, cx);

        if self.form.read(cx).is_dirty() {
            let value = self.form.read(cx).to_value();
            self.send(Command::UpdateConfig(value));
        }

        let status = self.status_text.clone();
        let running = self.running;

        v_flex()
            .size_full()
            .child(
                // Toolbar
                h_flex()
                    .h(px(36.0))
                    .px_2()
                    .gap_2()
                    .items_center()
                    .border_b_1()
                    .border_color(cx.theme().border)
                    .child(
                        div()
                            .id("play-pause-btn")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .cursor_pointer()
                            .bg(cx.theme().primary)
                            .text_color(cx.theme().primary_foreground)
                            .text_xs()
                            .child(if running { "Pause" } else { "Play" })
                            .on_click(cx.listener(|this, _, _, _cx| {
                                if this.running {
                                    this.running = false;
                                    this.send(Command::Pause);
                                } else {
                                    this.running = true;
                                    this.send(Command::Run);
                                }
                            })),
                    )
                    .child(
                        div()
                            .id("step-btn")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .cursor_pointer()
                            .bg(cx.theme().secondary)
                            .text_color(cx.theme().secondary_foreground)
                            .text_xs()
                            .child("Step")
                            .on_click(cx.listener(|this, _, _, _cx| {
                                this.send(Command::Step);
                                this.send(Command::QueryPlotData);
                            })),
                    )
                    .child(
                        div()
                            .id("create-btn")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .cursor_pointer()
                            .bg(cx.theme().secondary)
                            .text_color(cx.theme().secondary_foreground)
                            .text_xs()
                            .child("Create")
                            .on_click(cx.listener(|this, _, _, _cx| {
                                this.send(Command::CreateState);
                            })),
                    )
                    .child(
                        div()
                            .id("destroy-btn")
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .cursor_pointer()
                            .bg(cx.theme().secondary)
                            .text_color(cx.theme().secondary_foreground)
                            .text_xs()
                            .child("Destroy")
                            .on_click(cx.listener(|this, _, _, _cx| {
                                this.running = false;
                                this.send(Command::DestroyState);
                            })),
                    )
                    .child(
                        div()
                            .pl_4()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(status),
                    ),
            )
            .child(
                // Main content: config panel + plot
                h_flex()
                    .flex_1()
                    .size_full()
                    .child(
                        div()
                            .w(px(400.0))
                            .border_r_1()
                            .border_color(cx.theme().border)
                            .overflow_y_scrollbar()
                            .child(self.form.clone()),
                    )
                    .child(div().flex_1().size_full().child(self.plot.clone())),
            )
    }
}

fn main() {
    let app = Application::new();

    app.run(move |cx: &mut App| {
        gpui_component::init(cx);
        gpui_schema::init(cx);

        cx.on_action(|_: &Quit, cx| cx.quit());
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
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
                let plot = cx.new(|_cx| Plot::new().grid(true).aspect_ratio(1.0));

                let app_entity = cx.new(|_cx| DustApp {
                    handle,
                    form,
                    plot,
                    running: false,
                    has_state: false,
                    status_text: "Ready".into(),
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
