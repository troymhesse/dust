//! Particle simulation with GPUI visualization.

use std::f64::consts::PI;
use std::ops::{Add, Sub, Mul, AddAssign};
use rand::Rng;

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use driver::command::{Command, Event as DriverEvent};
use driver::config::SimulationConfig;
use driver::gpui_frontend::{DriverFooter, DriverLog, Quit, SnapshotReader};
use driver::worker::DriverHandle;
use driver::{Action, CliArgs, Driver, DriverState, Mode, PlotData, Solver, StepInfo, Validate};
use gpui::{
    App, AppContext as _, Application, Context, Entity, FocusHandle, Focusable,
    InteractiveElement as _, IntoElement, KeyDownEvent, Menu, MenuItem, ParentElement, Render,
    StatefulInteractiveElement as _, Styled, Window, WindowOptions, div, px, size,
};
// use gpui_component::switch::{self, Switch};
use gpui_component::{
    ActiveTheme, Root,
    input::{Input, InputState},
    resizable::{h_resizable, resizable_panel},
    scroll::ScrollableElement,
    tab::TabBar,
    v_flex,
};
use gpui_plot::{Plot, PlotStyle, Series, data_range};
use gpui_schema::{NodeFilter, SchemaForm, SchemaFormEvent};
use libm::atan2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ============================================================================
// Left panel tab
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftTab {
    Config,
    Editor,
}

/// Which widget last sent a config update to the driver.
/// Used to avoid echoing changes back to the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigSource {
    Form,
    Editor,
    Driver,
}

// ============================================================================
// Vector types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
struct Vec2 {
    x: f64, 
    y: f64, 
}

impl Default for Vec2 {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
struct Vec3 {
    x: f64, 
    y: f64,
    z: f64, 
}

impl Default for Vec3 {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, vec: Vec3) -> Self {
        Vec3 {
            x: self.x + vec.x, 
            y: self.y + vec.y, 
            z: self.z + vec.z, 
        }
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        };
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, vec: Vec3) -> Self {
        Vec3 {
            x: self.x - vec.x, 
            y: self.y - vec.y, 
            z: self.z - vec.z, 
        }
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, c: f64) -> Self {
        Vec3 {
            x: c * self.x, 
            y: c * self.y, 
            z: c * self.z, 
        }
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: self * vec.x, 
            y: self * vec.y, 
            z: self * vec.z, 
        }
    }
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x: x, y: y, z: z}
    }
    fn dot(&self, vec: Vec3) -> f64 {
        self.x * vec.x + self.y * vec.y + self.z * vec.z
    }
    fn mag(&self) -> f64 {
        (self.dot(*self)).sqrt()
    }
}

// ============================================================================
// Config types
// ============================================================================


#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum CentralObject {
    Single{ mass: f64 },
    Binary{ mass: f64, q: f64, a: f64, e: Vec2, i: f64 },
}


impl Default for CentralObject {
    fn default() -> Self {
        Self::Binary {
            mass: 1.0, 
            q: 0.1, 
            a: 0.5, 
            e: Vec2::default(), 
            i: 0.0,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct DustPhysics {
    /// Initial simulation time
    tstart: f64, 
    /// Final simulation time
    tfinal: f64,
    /// Time step
    dt: f64,
    /// Gravitational softening length
    softening: f64,
    /// Type of central object
    central_object: CentralObject, 
}

impl Default for DustPhysics {
    fn default() -> Self {
        Self {
            tstart: 0.0, 
            tfinal: 10.0,
            dt: 0.001,
            softening: 0.01,
            central_object: CentralObject::default(), 
        }
    }
}

impl Validate for DustPhysics {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct DustInitial {
    /// Number of particles
    num_particles: usize,
    /// Position of center of disk
    disk_center: DiskCenter, 
    /// Initial condition setup
    setup: DustSetup,
}

impl Default for DustInitial {
    fn default() -> Self {
        Self {
            num_particles: 1000,
            disk_center: DiskCenter::Arbitrary { x: 0.0, y: 0.0, z: 0.0 }, 
            setup: DustSetup::RandomDisk { inner_radius: 1.0, outer_radius: 2.0 }
        }
    }
}

impl Validate for DustInitial {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum DustSetup {
    /// Particles on a circular ring
    Ring {radius: f64},
    /// Particles in a randomized disk
    RandomDisk {inner_radius: f64, outer_radius: f64},
    /// Particles in a uniform disk
    UniformDisk {inner_radius: f64, outer_radius: f64}
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum DiskCenter {
    Primary, 
    Secondary, 
    Arbitrary{ x: f64, y: f64, z: f64 }, 
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
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
}

#[derive(Serialize)]
struct DustProducts {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
struct AnomalyParams {
    mass: f64, 
    q: f64, 
    a: f64, 
    e: Vec2, 
    i: f64,
    mean_anom: f64, 
}

impl Default for AnomalyParams {
    fn default() -> Self {
        Self {
            mass: 1.0, 
            q: 0.1, 
            a: 1.0, 
            e: Vec2{ x: 0.0, y: 0.0 }, 
            i: 0.0,
            mean_anom: 0.0, 
        }
    }
}

impl AnomalyParams {
    /// Magnitude of eccentricity vector
    fn e_mag(&self) -> f64 {
        (self.e.x * self.e.x + self.e.y * self.e.y).sqrt()
    }
    /// Mean angular motion of object (angular frequency)
    fn mean_angular_motion(&self) -> f64 {
        (self.mass / self.a / self.a / self.a).sqrt()
    }
    /// Period of object motion
    fn period(&self) -> f64 {
        2. * PI / self.mean_angular_motion()
    }
    /// Kepler's equation
    fn f(&self, ecc_anom: f64) -> f64 {
        ecc_anom - self.e_mag() * ecc_anom.sin() - self.mean_anom
    }
    /// Derivative of Kepler's equation (wrt E)
    fn fprime(&self, ecc_anom: f64) -> f64 {
        1. - self.e_mag() * ecc_anom.cos()
    }
    /// Calculate eccentric anomaly E for object orbit
    fn eccentric_anomaly(&mut self, time_since_periapse: f64) -> f64 {
        let w = self.mean_angular_motion();
        let p = self.period();
        let t = time_since_periapse - p * (time_since_periapse / p).floor();
        self.mean_anom = w * t;
        let ecc_anom = newton_raphson(|ecc_anom| self.f(ecc_anom), |ecc_anom| self.fprime(ecc_anom), self.mean_anom);
        
        ecc_anom
    }
}

fn newton_raphson<F, G>(f: F, fprime: G, mut x: f64) -> f64 
where 
    F: Fn(f64) -> f64, 
    G: Fn(f64) -> f64, 
{
    let mut n = 0;
    while f(x).abs() > 1e-15 {
        x -= f(x) / fprime(x);
        n += 1;
        if n > 10 {
            panic!("newton_raphson: no solution!");
        }
    }
    x
}

/// Position of primary and secondary objects in binary configuration; origin at CM
fn orbital_state(mut p: AnomalyParams, time_since_periapse: f64) -> (Vec3, Vec3) {
    let e_mag = p.e_mag();
    let ecc_anom = p.eccentric_anomaly(time_since_periapse);
    let x = p.a * (ecc_anom.cos() - e_mag);
    let y = p.a * (1. - e_mag * e_mag).sqrt() * ecc_anom.sin();
    let z = 0.0;
    // Rotate orbit based on inclination (i) wrt disk plane
    let x_rot_i = x * p.i.cos() - z * p.i.sin();
    let y_rot_i = y;
    let z_rot_i = x * p.i.sin() + z * p.i.cos();
    // Rotate orbit based on eccentricity vector with argument of periapsis (ω)
    let arg_of_peri = libm::atan2(p.e.y, p.e.x);
    let x_rot_w = x_rot_i * arg_of_peri.cos() - y_rot_i * arg_of_peri.sin();
    let y_rot_w = x_rot_i * arg_of_peri.sin() + y_rot_i * arg_of_peri.cos();
    let z_rot_w = z_rot_i;
    // Secondary position
    let x2 = - x_rot_w / (1. + p.q);
    let y2 = - y_rot_w / (1. + p.q);
    let z2 = z_rot_w / (1. + p.q);
    // Primary position
    let x1 = - x2 * p.q;
    let y1 = - y2 * p.q;
    let z1 = - z2 * p.q;

    (Vec3{ x: x1, y: y1, z: z1} , Vec3{ x: x2, y: y2, z: z2 })
}


struct Dust {
    physics: DustPhysics,
    initial: DustInitial,
}

impl Dust {
    /// Compute gravitational acceleration. 
    fn acceleration(&self, r: Vec3, state: &State) -> Vec3 {
        match self.physics.central_object {
            CentralObject::Single { mass } => {
                let eps = self.physics.softening;
                let r2 = r.dot(r) + eps * eps;
                let acc = -mass / (r2.sqrt() * r2);
                acc * r
            }
            CentralObject::Binary { mass, q, a, e, i } => {
                let p = AnomalyParams{ 
                    mass: mass, 
                    q: q, 
                    a: a, 
                    e: e, 
                    i: i,
                    mean_anom: 0.0, 
                };
                let eps = self.physics.softening;
                let m1 = mass / (1. + p.q);
                let m2 = p.q * m1;
                let time_since_periapse = state.time;
                let (r1, r2) = orbital_state(p, time_since_periapse);
                let r1_sep = r - r1;
                let r2_sep = r - r2;
                let r1_sq = r1_sep.dot(r1_sep) + eps * eps;
                let r2_sq = r2_sep.dot(r2_sep) + eps * eps;

                let acc1 = - m1 / r1_sq / r1_sq.sqrt();
                let acc2 = - m2 / r2_sq / r2_sq.sqrt();
                let a1 = acc1 * r1_sep;
                let a2 = acc2 * r2_sep;

                a1 + a2
            }
        }
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
        let mut rs: Vec<Vec3> = Vec::with_capacity(n);
        let mut vs: Vec<Vec3> = Vec::with_capacity(n);
        
        let r1: Vec3;
        let r2: Vec3;
        let rshift: Vec3;
        let vshift: Vec3;
        let mshift: f64;
        match self.physics.central_object {
            CentralObject::Binary { mass, q, a, e, i } => {
                let p = AnomalyParams { mass: mass, q: q, a: a, e: e, mean_anom: 0.0, i: i };
                (r1, r2) = orbital_state(p, self.physics.tstart);
                let r12 = r2 - r1;
                let theta1 = atan2(r1.y, r1.x);
                let theta2 = atan2(r2.y, r2.x);
                match self.initial.disk_center {
                    DiskCenter::Primary => {
                        mshift = mass / (1. + q);
                        let v1 = (mshift * mshift * q * q / mass / r12.mag()).sqrt();
                        rshift = r1;
                        vshift = Vec3::new(- v1 * theta1.sin(), v1 * theta1.cos(),0.);
                    }
                    DiskCenter::Secondary => {
                        mshift = mass * q / (1. + q);
                        let v2 = (mshift * mshift / q / q / mass / r12.mag()).sqrt();
                        rshift = r2;
                        vshift = Vec3::new(- v2 * theta2.sin(), v2 * theta2.cos(), 0.)
                    }
                    DiskCenter::Arbitrary { x, y , z} => {
                        rshift = Vec3{ x:x , y:y, z:z };
                        vshift = Vec3::default();
                        mshift = mass;
                    }
                }
            }
            CentralObject::Single { mass } => {
                rshift = Vec3::default();
                vshift = Vec3::default();
                mshift = mass;
            }
        }
        
        match self.initial.setup {
            DustSetup::Ring { radius } => {
                for i in 0..n {
                    let theta = 2.0 * PI * i as f64 / n as f64;
                    let r = radius;
                    let px = r * theta.cos();
                    let py = r * theta.sin();
                    let pz = 0.0;
                    rs.push(Vec3::new(px, py, pz) + rshift);
                    // Circular orbital velocity: v = sqrt(GM/r)
                    let v = (mshift / r).sqrt();
                    let vx = -v * theta.sin();
                    let vy = v * theta.cos();
                    let vz = 0.0;
                    vs.push(Vec3::new(vx, vy, vz) + vshift);
                }
            }
            DustSetup::RandomDisk{ inner_radius, outer_radius} => {
                //  Really random disk
                let h = 0.; // Constant 'aspect ratio' of disk
                let mut rng = rand::thread_rng();
                for _i in 0..n {
                    let r: f64 = (rng.gen_range(inner_radius*inner_radius .. outer_radius*outer_radius)).sqrt();
                    let theta = rng.gen_range(0. .. 2.*PI);
                    let px = r * theta.cos();
                    let py = r * theta.sin();
                    let pz = h * r;
                    rs.push(Vec3::new(px, py, pz) + rshift);
                    // Circular orbital velocity: v = sqrt(GM/r)
                    let v = (mshift / r).sqrt();
                    let vx = -v * theta.sin();
                    let vy = v * theta.cos();
                    let vz = 0.0;
                    vs.push(Vec3::new(vx, vy, vz) + vshift);
                }
            }
            DustSetup::UniformDisk{ inner_radius, outer_radius } => {
                // Simple deterministic pseudo-random using golden ratio
                let h = 0.; // Constant 'aspect ratio' of disk
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                for i in 0..n {
                    let r = inner_radius + (outer_radius - inner_radius) * (i as f64 / n as f64).sqrt();
                    let theta = 2.0 * PI * (i as f64 * phi);
                    let px = r * theta.cos();
                    let py = r * theta.sin();
                    let pz = h * r;
                    rs.push(Vec3::new(px, py, pz) + rshift);
                    // Circular orbital velocity: v = sqrt(GM/r)
                    let v = (mshift / r).sqrt();
                    let vx = -v * theta.sin();
                    let vy = v * theta.cos();
                    let vz = 0.0;
                    vs.push(Vec3::new(vx, vy, vz) + vshift);
                }
            }
        }

        State {
            time: self.physics.tstart,
            positions: rs,
            velocities: vs,
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
        let n = state.positions.len();

        // Leapfrog (kick-drift-kick)
        for i in 0..n {
            let a = self.acceleration(state.positions[i], &state);
            state.velocities[i] += 0.5 * dt * a;
        }
        for i in 0..n {
            state.positions[i] += dt * state.velocities[i];
        }
        for i in 0..n {
            let a = self.acceleration(state.positions[i], &state);
            state.velocities[i] += 0.5 * dt * a;
        }

        state.time += dt;
        state
    }

    fn products(&self, state: &State) -> DustProducts {
        DustProducts {
            x: state.positions.iter().map(|v| v.x).collect(),
            y: state.positions.iter().map(|v| v.y).collect(),
            z: state.positions.iter().map(|v| v.z).collect(),
        }
    }

    fn status(&self, state: &State) -> DustStatus {
        DustStatus {
            time: state.time,
            num_particles: state.positions.len(),
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

fn rgb_tuple(r: u8, g: u8, b: u8) -> gpui::Rgba {
    gpui::rgb((r as u32) << 16 | (g as u32) << 8 | (b as u32))
}
struct DustApp {
    handle: DriverHandle,
    form: Entity<SchemaForm>,
    plot: Entity<Plot>,
    editor: Entity<InputState>,
    focus_handle: FocusHandle,
    schema: schemars::Schema,
    snapshot_reader: SnapshotReader,
    log: DriverLog,
    show_log: Rc<Cell<bool>>,
    left_tab: LeftTab,
    config_source: ConfigSource,
    binary_params: Option<AnomalyParams>, 
}

impl Focusable for DustApp {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl DustApp {
    /// Subscribe to form change events to push config updates to the driver.
    fn subscribe_form(&self, _window: &mut Window, cx: &mut Context<Self>) {
        cx.subscribe(&self.form, |this, form, _event: &SchemaFormEvent, cx| {
            let value = form.read(cx).to_value();
            this.config_source = ConfigSource::Form;
            this.send(Command::UpdateConfig(value));
        })
        .detach();
    }

    /// Read the latest snapshot, diff it, drain events, and perform
    /// app-specific updates (plot data, config filter, editor/form sync).
    fn read_snapshot(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let diff = self.snapshot_reader.update(&self.handle.snapshot);

        // Drain events — log handles logging, we get back app-relevant ones
        let app_events = self.log.drain_events(&self.handle.event_rx);

        // Process app events (config sync, etc.)
        for event in app_events {
            self.handle_app_event(event, window, cx);
        }

        // Log iteration messages when iteration advances
        if diff.iteration_advanced && !self.snapshot_reader.status_text().is_empty() {
            self.log
                .log_iteration(self.snapshot_reader.status_text().to_string());
        }

        // App-specific: update plot data (preserves pan/zoom view state)
        if diff.iteration_advanced || diff.state_changed {
            let snap = self.snapshot_reader.snapshot();
            let time = snap.time;
            let style = PlotStyle::from_theme(cx.theme());
            let mut star_series = Vec::new();
            if let Some(params) = self.binary_params.clone() {
                let (r1, r2) = orbital_state(params, time);
                star_series.push(
                    Series::scatter(vec![r1.x], vec![r1.y])
                        .label("primary")
                        .marker_radius(6.0)
                        .color(rgb_tuple(255, 220, 120))
                );
                star_series.push(
                    Series::scatter(vec![r2.x], vec![r2.y])
                        .label("secondary")
                        .marker_radius(5.0)
                        .color(rgb_tuple(255, 149, 120))
                );
            }
            let first_data = diff.state_changed && snap.has_state;
            if let (Some(x), Some(y)) = (snap.linear.get("x"), snap.linear.get("y")) {
                self.plot.update(cx, |plot, cx| {
                    let mut series = vec![
                        Series::scatter(x.clone(), y.clone())
                            .label("particles")
                            .marker_radius(1.5),
                    ];
                    series.extend(star_series);
                    plot.set_series(series);
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

    fn editing(&self, window: &Window, cx: &Context<Self>) -> bool {
        self.form.read(cx).editing()
            || (self.left_tab == LeftTab::Editor
                && self.editor.read(cx).focus_handle(cx).is_focused(window))
    }

    fn editor_focused(&self, window: &Window, cx: &Context<Self>) -> bool {
        self.left_tab == LeftTab::Editor
            && self.editor.read(cx).focus_handle(cx).is_focused(window)
    }

    fn send(&self, cmd: Command) {
        let _ = self.handle.cmd_tx.send(cmd);
    }

    /// Handle an app-level event returned by `drain_events`.
    fn handle_app_event(
        &mut self,
        event: DriverEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            // Config sections (RON) — update the editor unless it was the source
            DriverEvent::ConfigSections {
                driver,
                physics,
                initial,
                compute,
            } => {
                if self.config_source != ConfigSource::Editor {
                    let ron = assemble_config_ron(&driver, &physics, &initial, &compute);
                    self.editor.update(cx, |editor, cx| {
                        editor.set_value(ron, window, cx);
                    });
                }
                self.config_source = ConfigSource::Driver;
            }
            // Full config JSON — update the form unless it was the source
            DriverEvent::Config(value) => {
                if let Ok(config) = serde_json::from_value::<SimulationConfig<Dust>>(value.clone()) {
                    if let CentralObject::Binary { mass, q, a, e , i} = config.physics.central_object {
                        self.binary_params = Some(AnomalyParams {
                            mass,
                            q,
                            a,
                            e,
                            i, 
                            mean_anom: 0.0,
                        });
                    } else {
                        self.binary_params = None;
                    }
                }
                if self.config_source != ConfigSource::Form {
                    let form = cx.new(|cx| {
                        let mut form = SchemaForm::new(&self.schema, &value, window, cx);
                        form.set_filter(
                            DustFilter {
                                has_state: self.snapshot_reader.has_state(),
                            },
                            window,
                            cx,
                        );
                        form
                    });
                    self.form = form;
                    self.subscribe_form(window, cx);
                }
                self.config_source = ConfigSource::Driver;
            }
            // Config update result — on error from editor, don't overwrite editor text
            DriverEvent::ConfigUpdated(Err(_)) => {
                // Error already logged by DriverLog. Reset source so next
                // successful update will sync both widgets.
                self.config_source = ConfigSource::Driver;
            }
            DriverEvent::ConfigUpdated(Ok(())) => {
                // Success — the ConfigSections event that follows will sync the editor
            }
            // State created/destroyed — request fresh config to sync both widgets
            DriverEvent::StateCreated | DriverEvent::StateDestroyed => {
                self.send(Command::QueryConfig);
                self.send(Command::QueryConfigRon);
            }
            // Config/checkpoint loaded — request both representations
            DriverEvent::ConfigLoaded { .. } | DriverEvent::CheckpointLoaded { .. } => {
                self.config_source = ConfigSource::Driver;
                self.send(Command::QueryConfig);
                self.send(Command::QueryConfigRon);
            }
            _ => {}
        }
    }

    /// Send the editor's RON content to the driver as a config update.
    fn apply_editor_config(&mut self, cx: &mut Context<Self>) {
        let ron = self.editor.read(cx).value().to_string();
        self.config_source = ConfigSource::Editor;
        self.send(Command::UpdateConfigRon(ron));
        // Also request the JSON form so the schema form gets updated
        self.send(Command::QueryConfig);
    }
}

impl Render for DustApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.read_snapshot(window, cx);

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

        // Left panel: tab bar + content
        let tab_idx = match self.left_tab {
            LeftTab::Config => 0,
            LeftTab::Editor => 1,
        };
        let left_content = match self.left_tab {
            LeftTab::Config => div()
                .flex_1()
                .min_h_0()
                .overflow_y_scrollbar()
                .child(self.form.clone())
                .into_any_element(),
            LeftTab::Editor => div()
                .flex_1()
                .min_h_0()
                .size_full()
                .font_family(cx.theme().mono_font_family.clone())
                .child(Input::new(&self.editor).w_full().h_full())
                .into_any_element(),
        };
        let left_panel = v_flex()
            .size_full()
            .child(
                TabBar::new("left-tabs")
                    .child("Config")
                    .child("Editor")
                    .selected_index(tab_idx)
                    .on_click(cx.listener(|this, idx: &usize, _window, _cx| {
                        this.left_tab = match idx {
                            0 => LeftTab::Config,
                            _ => LeftTab::Editor,
                        };
                    })),
            )
            .child(left_content);

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
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                // Cmd+S: apply editor config (works even when editor is focused)
                if event.keystroke.key.as_str() == "s"
                    && event.keystroke.modifiers.platform
                {
                    if this.editor_focused(window, cx) {
                        this.apply_editor_config(cx);
                    }
                    return;
                }
                if this.editing(window, cx) {
                    return;
                }
                let key = event.keystroke.key.as_str();
                match key {
                    "p" => {
                        if !this.has_state() {
                            return;
                        }
                        if this.running() {
                            this.send(Command::Pause);
                        } else {
                            this.send(Command::Run);
                        }
                    }
                    "s" => {
                        if this.has_state() && !this.running() {
                            this.send(Command::Step);
                        }
                    }
                    "n" => {
                        if !this.has_state() {
                            this.send(Command::CreateState);
                        }
                    }
                    "d" => {
                        if this.has_state() && !this.running() {
                            this.send(Command::DestroyState);
                        }
                    }
                    "c" => {
                        if this.has_state() && !this.running() {
                            this.send(Command::Checkpoint);
                        }
                    }
                    "w" => {
                        this.send(Command::WriteConfig("config.ron".into()));
                    }
                    "l" => {
                        this.show_log.set(!this.show_log.get());
                    }
                    _ => {}
                }
            }))
            .child(
                // Main content: resizable config panel + plot/log
                h_resizable("main-split")
                    .child(
                        resizable_panel()
                            .size(px(400.0))
                            .size_range(px(150.0)..px(800.0))
                            .child(left_panel),
                    )
                    .child(resizable_panel().child(
                        div()
                            .id("right-panel")
                            .size_full()
                            .on_click(move |_, window, _cx| {
                                focus_handle.focus(window);
                            })
                            .child(right_panel),
                    )),
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

/// Assemble individually pretty-printed RON sections into a single
/// `SimulationConfig(...)` string with correct indentation.
fn assemble_config_ron(driver: &str, physics: &str, initial: &str, compute: &str) -> String {
    fn indent(section: &str, prefix: &str) -> String {
        section
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line.to_string()
                } else {
                    format!("{prefix}{line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
    let pad = "    ";
    format!(
        "SimulationConfig(\n\
         {pad}driver: {},\n\
         {pad}physics: {},\n\
         {pad}initial: {},\n\
         {pad}compute: {},\n\
         )",
        indent(driver, pad),
        indent(physics, pad),
        indent(initial, pad),
        indent(compute, pad),
    )
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
        // Only register cmd-q globally; single-letter shortcuts are
        // handled via on_key_down so they don't steal from the editor.
        cx.bind_keys([gpui::KeyBinding::new("cmd-q", Quit, None)]);

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

                let initial_ron = ron::ser::to_string_pretty(
                    &config,
                    ron::ser::PrettyConfig::new()
                        .depth_limit(8)
                        .struct_names(true)
                        .enumerate_arrays(false),
                )
                .unwrap_or_default();
                let editor = cx.new(|cx| {
                    let mut state = InputState::new(window, cx)
                        .code_editor("rust")
                        .line_number(true);
                    state.set_value(initial_ron, window, cx);
                    state
                });

                let app_entity = cx.new(|cx| {
                    let focus_handle = cx.focus_handle();
                    focus_handle.focus(window);
                    let app = DustApp {
                        handle,
                        form,
                        plot,
                        editor,
                        focus_handle,
                        schema,
                        snapshot_reader: SnapshotReader::new(),
                        log: DriverLog::new(),
                        show_log: Rc::new(Cell::new(false)),
                        left_tab: LeftTab::Config,
                        config_source: ConfigSource::Driver,
                        binary_params: None, 
                    };
                    app.subscribe_form(window, cx);
                    app
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
