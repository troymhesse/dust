//! Composable GPUI widgets for simulation driver frontends.
//!
//! Provides [`DriverLog`], [`DriverFooter`], [`SnapshotReader`], and
//! [`register_keybindings`] — reusable building blocks that any GPUI-based
//! simulation app can compose into its own layout.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use gpui::{
    actions, div, px, App, InteractiveElement as _, IntoElement, KeyBinding, ParentElement,
    StatefulInteractiveElement as _, Styled,
};
use gpui_component::{h_flex, scroll::ScrollableElement, ActiveTheme};

use crate::command::{Command, DriverMode, Event};
use crate::watch::{Snapshot, Watch};

// ============================================================================
// Actions
// ============================================================================

actions!(
    driver,
    [
        Quit,
        ToggleRun,
        Step,
        CreateState,
        DestroyState,
        WriteCheckpoint,
        WriteConfig,
        ToggleLog,
    ]
);

/// Register the standard driver keybindings (no key context, so they
/// fire regardless of focus unless a text input consumes the keystroke).
pub fn register_keybindings(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("p", ToggleRun, None),
        KeyBinding::new("s", Step, None),
        KeyBinding::new("n", CreateState, None),
        KeyBinding::new("d", DestroyState, None),
        KeyBinding::new("c", WriteCheckpoint, None),
        KeyBinding::new("w", WriteConfig, None),
        KeyBinding::new("l", ToggleLog, None),
    ]);
}

// ============================================================================
// Log
// ============================================================================

/// Severity level for display color selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Error,
    Iteration,
}

/// A single log entry.
#[derive(Debug, Clone)]
pub struct LogMessage {
    pub text: String,
    pub level: LogLevel,
}

/// Log storage with event draining and rendering helpers.
pub struct DriverLog {
    pub lines: Vec<LogMessage>,
    pub last_message: Option<LogMessage>,
}

impl Default for DriverLog {
    fn default() -> Self {
        Self::new()
    }
}

impl DriverLog {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            last_message: None,
        }
    }

    fn push(&mut self, level: LogLevel, text: String) {
        let msg = LogMessage { text, level };
        self.last_message = Some(msg.clone());
        self.lines.push(msg);
    }

    pub fn log_info(&mut self, msg: impl Into<String>) {
        self.push(LogLevel::Info, msg.into());
    }
    pub fn log_error(&mut self, msg: impl Into<String>) {
        self.push(LogLevel::Error, msg.into());
    }
    pub fn log_iteration(&mut self, msg: impl Into<String>) {
        self.push(LogLevel::Iteration, msg.into());
    }

    /// Drain all pending [`Event`]s into log entries.
    pub fn drain_events(&mut self, rx: &Receiver<Event>) {
        loop {
            match rx.try_recv() {
                Ok(ev) => self.handle_event(ev),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Error(e) => self.log_error(format!("error: {e}")),
            Event::ConfigUpdated(Ok(())) => self.log_info("config updated"),
            Event::ConfigUpdated(Err(e)) => self.log_error(format!("config error: {e}")),
            Event::SimulationDone => self.log_info("simulation done"),
            Event::CheckpointWritten { path } => self.log_info(format!("wrote {path}")),
            Event::StateCreated => self.log_info("state created"),
            Event::StateDestroyed => self.log_info("state destroyed"),
            Event::ConfigWritten { path } => self.log_info(format!("wrote config to {path}")),
            Event::ConfigLoaded { path } => self.log_info(format!("loaded config from {path}")),
            Event::CheckpointLoaded { path } => {
                self.log_info(format!("loaded checkpoint from {path}"))
            }
            _ => {}
        }
    }

    /// Scrollable, color-coded log view element.
    pub fn render_log_view(&self, cx: &App) -> impl IntoElement {
        let mono = cx.theme().mono_font_family.clone();
        let lines = self.lines.clone();
        div()
            .flex_1()
            .min_h_0()
            .overflow_y_scrollbar()
            .p_2()
            .bg(cx.theme().background)
            .children(lines.into_iter().map(|msg| {
                div()
                    .text_xs()
                    .font_family(mono.clone())
                    .text_color(log_color(msg.level, cx))
                    .child(msg.text)
                    .into_any_element()
            }))
    }

    /// Element showing the most recent log message (empty div if none).
    pub fn last_message_element(&self, cx: &App) -> gpui::AnyElement {
        if let Some(ref msg) = self.last_message {
            div()
                .text_xs()
                .font_family(cx.theme().mono_font_family.clone())
                .text_color(log_color(msg.level, cx))
                .child(msg.text.clone())
                .into_any_element()
        } else {
            div().into_any_element()
        }
    }
}

fn log_color(level: LogLevel, cx: &App) -> gpui::Hsla {
    match level {
        LogLevel::Info => cx.theme().success.into(),
        LogLevel::Error => cx.theme().danger.into(),
        LogLevel::Iteration => cx.theme().muted_foreground.into(),
    }
}

// ============================================================================
// Snapshot reader
// ============================================================================

/// What changed between the previous and current snapshot.
pub struct SnapshotDiff {
    pub iteration_advanced: bool,
    pub state_changed: bool,
    pub mode_changed: bool,
}

/// Tracks the last-read snapshot and computes diffs each frame.
pub struct SnapshotReader {
    last: Snapshot,
}

impl Default for SnapshotReader {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotReader {
    pub fn new() -> Self {
        Self {
            last: Snapshot::default(),
        }
    }

    /// Read the watch channel, diff against previous, store the new snapshot.
    pub fn update(&mut self, watch: &Watch<Snapshot>) -> SnapshotDiff {
        let snap = watch.read();
        let diff = SnapshotDiff {
            iteration_advanced: snap.iteration != self.last.iteration,
            state_changed: snap.has_state != self.last.has_state,
            mode_changed: snap.mode != self.last.mode,
        };
        self.last = snap;
        diff
    }

    pub fn snapshot(&self) -> &Snapshot {
        &self.last
    }
    pub fn running(&self) -> bool {
        self.last.mode == DriverMode::Running
    }
    pub fn has_state(&self) -> bool {
        self.last.has_state
    }
    pub fn iteration(&self) -> i64 {
        self.last.iteration
    }
    pub fn time(&self) -> f64 {
        self.last.time
    }
    pub fn status_text(&self) -> &str {
        &self.last.status_text
    }
}

// ============================================================================
// Footer
// ============================================================================

/// Context-sensitive footer with clickable hint labels, iteration info,
/// and the most recent log message.
pub struct DriverFooter<'a> {
    running: bool,
    has_state: bool,
    show_log: Rc<Cell<bool>>,
    iteration: i64,
    time: f64,
    log: &'a DriverLog,
    cmd_tx: Option<Sender<Command>>,
}

impl<'a> DriverFooter<'a> {
    /// `show_log` is a shared flag — the footer toggles it on click,
    /// the app reads it each frame to decide which panel to show.
    pub fn new(reader: &SnapshotReader, log: &'a DriverLog, show_log: Rc<Cell<bool>>) -> Self {
        Self {
            running: reader.running(),
            has_state: reader.has_state(),
            show_log,
            iteration: reader.iteration(),
            time: reader.time(),
            log,
            cmd_tx: None,
        }
    }

    pub fn cmd_tx(mut self, tx: Sender<Command>) -> Self {
        self.cmd_tx = Some(tx);
        self
    }

    pub fn render(self, cx: &App) -> gpui::AnyElement {
        let mono = cx.theme().mono_font_family.clone();
        let key_color: gpui::Hsla = cx.theme().warning.into();
        let text_color: gpui::Hsla = cx.theme().muted_foreground.into();
        let bg_color = cx.theme().title_bar;

        // (key, description, element_id, command-or-toggle)
        let mut hints: Vec<(&str, &str, &str, Option<Command>)> = Vec::new();
        let r = self.running;
        let s = self.has_state;

        if r {
            hints.push(("p", "pause", "hint-p", Some(Command::Pause)));
        } else if s {
            hints.push(("p", "play", "hint-p", Some(Command::Run)));
        }
        if s && !r {
            hints.push(("s", "step", "hint-s", Some(Command::Step)));
        }
        if !s {
            hints.push(("n", "new", "hint-n", Some(Command::CreateState)));
        }
        if s && !r {
            hints.push(("d", "destroy", "hint-d", Some(Command::DestroyState)));
            hints.push(("c", "checkpoint", "hint-c", Some(Command::Checkpoint)));
            hints.push((
                "w",
                "write-cfg",
                "hint-w",
                Some(Command::WriteConfig("config.ron".into())),
            ));
        }
        let log_label = if self.show_log.get() { "plot" } else { "log" };
        hints.push(("l", log_label, "hint-l", None)); // None = toggle-log

        let footer_hints = h_flex().gap_0p5().items_center().flex_shrink_0().children(
            hints
                .into_iter()
                .enumerate()
                .map(|(i, (key, desc, id, cmd))| {
                    let cmd_tx = self.cmd_tx.clone();
                    let show_log = self.show_log.clone();
                    let sep = if i > 0 { " " } else { "" };

                    div()
                        .id(id)
                        .cursor_pointer()
                        .child(
                            h_flex()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_family(mono.clone())
                                        .text_color(text_color)
                                        .child(sep),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .font_family(mono.clone())
                                        .text_color(key_color)
                                        .child(key),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .font_family(mono.clone())
                                        .text_color(text_color)
                                        .child(format!(":{desc}")),
                                ),
                        )
                        .on_click(move |_, _, _| match &cmd {
                            None => show_log.set(!show_log.get()),
                            Some(cmd) => {
                                if let Some(tx) = &cmd_tx {
                                    let _ = tx.send(cmd.clone());
                                }
                            }
                        })
                        .into_any_element()
                }),
        );

        let iter_text = if self.has_state {
            format!("[{:06}] t={:.6e}", self.iteration, self.time)
        } else {
            String::new()
        };

        h_flex()
            .h(px(30.0))
            .px_2()
            .items_center()
            .border_t_1()
            .border_color(cx.theme().border)
            .bg(bg_color)
            .child(footer_hints)
            .child(div().flex_1())
            .child(
                h_flex()
                    .gap_3()
                    .flex_shrink_0()
                    .items_center()
                    .child(self.log.last_message_element(cx))
                    .child(
                        div()
                            .text_xs()
                            .font_family(mono)
                            .text_color(text_color)
                            .child(iter_text),
                    ),
            )
            .into_any_element()
    }
}
