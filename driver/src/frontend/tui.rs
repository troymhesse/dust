//! Interactive TUI frontend.
//!
//! Three-column layout: file browser, config editor, log/plot.

use crate::command::{Command, DriverMode, Event as DriverEvent};
use crate::watch::Snapshot;
use crate::worker::DriverHandle;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseButton,
    MouseEvent, MouseEventKind,
};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap,
    canvas::{Canvas, Line as CanvasLine},
};
use schema_tui::{EventResult, NodeFilter};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Top-level entry point
// ============================================================================

pub fn run(handle: DriverHandle, disabled_paths: Vec<String>) {
    let mut terminal = ratatui::init();
    crossterm::execute!(std::io::stdout(), EnableMouseCapture).unwrap();
    let mut state = TuiState::new(disabled_paths);

    // Request initial schema and config
    let _ = handle.cmd_tx.send(Command::QuerySchema);
    let _ = handle.cmd_tx.send(Command::QueryConfig);
    std::thread::sleep(Duration::from_millis(50));
    while let Ok(event) = handle.event_rx.try_recv() {
        state.handle_event(event, &handle);
    }
    state.refresh_files();

    loop {
        // Read snapshot (continuous data)
        state.read_snapshot(&handle.snapshot);

        // Drain events (non-blocking)
        while let Ok(event) = handle.event_rx.try_recv() {
            if matches!(event, DriverEvent::Finished) {
                state.log_info("driver finished");
                let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
                ratatui::restore();
                return;
            }
            state.handle_event(event, &handle);
        }

        // Periodic file refresh
        if state.last_file_scan.elapsed() >= Duration::from_secs_f64(0.1) {
            state.refresh_files();
            state.last_file_scan = Instant::now();
        }

        // Draw
        terminal
            .draw(|frame| {
                draw(frame, &mut state);
                // Show cursor when editing in tree
                if state.focus == Focus::Middle && state.middle_tab == MiddleTab::Config {
                    if let Some(ref tree) = state.tree_state {
                        if matches!(tree.edit_mode, schema_tui::EditMode::Editing { .. }) {
                            if let Some((cx, cy)) = tree.cursor_position {
                                frame.set_cursor_position((cx, cy));
                            }
                        }
                    }
                }
            })
            .unwrap();

        // Poll input
        if event::poll(Duration::from_millis(50)).unwrap() {
            match event::read().unwrap() {
                Event::Key(key) => {
                    if let Some(cmd) = state.handle_key(key.code, key.modifiers) {
                        let is_quit = matches!(cmd, Command::Quit);
                        let _ = handle.cmd_tx.send(cmd);
                        if is_quit {
                            break;
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    state.handle_mouse(mouse);
                }
                _ => {}
            }
        }
    }

    let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
}

// ============================================================================
// Config filter
// ============================================================================

struct ConfigFilter {
    has_state: bool,
    disabled_paths: Vec<String>,
}

impl NodeFilter for ConfigFilter {
    fn enabled(&self, path: &str) -> bool {
        if self.has_state
            && (path == "initial"
                || path.starts_with("initial.")
                || path == "compute"
                || path.starts_with("compute."))
        {
            return false;
        }
        !self.disabled_paths.iter().any(|d| path == d.as_str())
    }
}

// ============================================================================
// State types
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Focus {
    Left,
    Middle,
    Right,
}

#[derive(Clone, Copy, PartialEq)]
enum FileTab {
    Configs,
    Checkpoints,
}

#[derive(Clone, Copy, PartialEq)]
enum MiddleTab {
    Config,
    State,
}

#[derive(Clone, Copy, PartialEq)]
enum RightTab {
    Log,
    Plot1D,
    Plot2D,
}

struct TuiState {
    focus: Focus,
    sidebar_visible: bool,
    column_areas: [Rect; 3],

    // File browser
    file_tab: FileTab,
    config_files: Vec<String>,
    checkpoint_files: Vec<String>,
    file_list_state: ListState,

    // Middle column
    middle_tab: MiddleTab,

    // Config editor (schema-tui tree)
    tree_state: Option<schema_tui::TreeState>,
    schema_value: Option<serde_json::Value>,
    pending_config: Option<serde_json::Value>,
    disabled_paths: Vec<String>,
    has_state: bool,

    // State info
    state_info_text: String,

    // Driver status
    mode: DriverMode,

    // Right column
    right_tab: RightTab,

    // Log
    log_lines: Vec<Line<'static>>,

    // 1D plot
    linear_data: HashMap<String, Vec<f64>>,
    linear_keys: Vec<String>,
    x_axis_index: usize,
    y_cursor: usize,
    y_selected: Vec<bool>,
    y_legend_rect: Rect,
    x_radio_spans: Vec<(u16, u16)>, // (start_col, end_col) per radio item
    x_radio_y: u16,

    // 2D plot
    planar_data: HashMap<String, (usize, usize, Vec<f64>)>,
    planar_keys: Vec<String>,
    planar_list: ListState,

    // Footer message
    last_message: Option<(String, Color)>,

    // Output dir for file scanning
    output_dir: String,
    last_file_scan: Instant,
}

impl TuiState {
    fn new(disabled_paths: Vec<String>) -> Self {
        TuiState {
            focus: Focus::Right,
            sidebar_visible: true,
            column_areas: [Rect::default(); 3],
            file_tab: FileTab::Configs,
            config_files: Vec::new(),
            checkpoint_files: Vec::new(),
            file_list_state: ListState::default(),
            middle_tab: MiddleTab::Config,
            tree_state: None,
            schema_value: None,
            pending_config: None,
            disabled_paths,
            has_state: false,
            state_info_text: "no state".into(),
            mode: DriverMode::Idle,
            right_tab: RightTab::Log,
            log_lines: Vec::new(),
            linear_data: HashMap::new(),
            linear_keys: Vec::new(),
            x_axis_index: 0,
            y_cursor: 0,
            y_selected: Vec::new(),
            y_legend_rect: Rect::default(),
            x_radio_spans: Vec::new(),
            x_radio_y: 0,
            planar_data: HashMap::new(),
            planar_keys: Vec::new(),
            planar_list: ListState::default(),
            last_message: None,
            output_dir: ".".into(),
            last_file_scan: Instant::now(),
        }
    }

    fn log_info(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.last_message = Some((msg.clone(), Color::Green));
        self.log_lines
            .push(Line::styled(msg, Style::default().fg(Color::Green)));
    }

    fn log_error(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.last_message = Some((msg.clone(), Color::Red));
        self.log_lines
            .push(Line::styled(msg, Style::default().fg(Color::Red)));
    }

    fn refresh_files(&mut self) {
        self.config_files = scan_files(".", "ron");
        self.checkpoint_files = scan_files(&self.output_dir, "mpk");
    }

    fn active_file_list(&self) -> &[String] {
        match self.file_tab {
            FileTab::Configs => &self.config_files,
            FileTab::Checkpoints => &self.checkpoint_files,
        }
    }

    fn build_tree_state(
        &mut self,
        schema_value: &serde_json::Value,
        config_value: &serde_json::Value,
    ) {
        let schema: schemars::Schema = match schema_value.clone().try_into() {
            Ok(s) => s,
            Err(_) => {
                self.log_error("invalid schema");
                return;
            }
        };
        let mut tree = schema_tui::TreeState::new(&schema, config_value);
        tree.set_filter(ConfigFilter {
            has_state: self.has_state,
            disabled_paths: self.disabled_paths.clone(),
        });
        self.tree_state = Some(tree);
    }

    fn update_filter(&mut self) {
        if let Some(ref mut tree) = self.tree_state {
            tree.set_filter(ConfigFilter {
                has_state: self.has_state,
                disabled_paths: self.disabled_paths.clone(),
            });
        }
    }

    // ========================================================================
    // Message handling
    // ========================================================================

    /// Read the latest snapshot from the watch channel and update TUI state.
    fn read_snapshot(&mut self, snapshot: &crate::watch::Watch<Snapshot>) {
        let snap = snapshot.read();

        // Update mode
        self.mode = snap.mode;

        // Update has_state and filter
        let new_has_state = snap.has_state;
        if new_has_state != self.has_state {
            self.has_state = new_has_state;
            self.update_filter();
            if !new_has_state {
                self.linear_data.clear();
                self.linear_keys.clear();
                self.y_selected.clear();
                self.y_cursor = 0;
                self.planar_data.clear();
                self.planar_keys.clear();
            }
        }

        // Update state info text
        if snap.has_state {
            self.state_info_text =
                format!("iteration: {}\ntime: {:.6e}", snap.iteration, snap.time,);
            if !snap.status_text.is_empty() {
                self.state_info_text.push('\n');
                self.state_info_text.push_str(&snap.status_text);
            }
        } else {
            self.state_info_text = "no state".into();
        }

        // Update plot data from snapshot
        self.update_plot_data(snap.linear, snap.planar);
    }

    /// Update plot data, preserving y-axis selections where possible.
    fn update_plot_data(
        &mut self,
        linear: HashMap<String, Vec<f64>>,
        planar: HashMap<String, (usize, usize, Vec<f64>)>,
    ) {
        let mut lkeys: Vec<String> = linear.keys().cloned().collect();
        lkeys.sort();

        // Rebuild y_selected, preserving selections for keys that still exist
        let old_keys = std::mem::take(&mut self.linear_keys);
        let old_sel = std::mem::take(&mut self.y_selected);
        let mut new_sel = vec![false; lkeys.len()];
        for (oi, ok) in old_keys.iter().enumerate() {
            if oi < old_sel.len() && old_sel[oi] {
                if let Some(ni) = lkeys.iter().position(|k| k == ok) {
                    new_sel[ni] = true;
                }
            }
        }
        self.linear_keys = lkeys;
        self.y_selected = new_sel;
        self.linear_data = linear;

        let n = self.linear_keys.len();
        if n > 0 {
            let xi = self.x_axis_index.min(n - 1);
            self.x_axis_index = xi;
            self.y_cursor = self.y_cursor.min(n - 1);
        }

        let mut pkeys: Vec<String> = planar.keys().cloned().collect();
        pkeys.sort();
        self.planar_keys = pkeys;
        self.planar_data = planar;
        let n = self.planar_keys.len();
        if n > 0 {
            let pi = self.planar_list.selected().unwrap_or(0).min(n - 1);
            self.planar_list.select(Some(pi));
        }
    }

    fn handle_event(&mut self, event: DriverEvent, handle: &DriverHandle) {
        match event {
            DriverEvent::SimulationDone => {
                self.log_info("simulation done");
            }
            DriverEvent::Config(v) => {
                if let Some(ref schema) = self.schema_value {
                    let schema = schema.clone();
                    self.build_tree_state(&schema, &v);
                } else {
                    self.pending_config = Some(v);
                }
            }
            DriverEvent::ConfigSections { .. } => {}
            DriverEvent::Schema(v) => {
                self.schema_value = Some(v);
                if let Some(config) = self.pending_config.take() {
                    let schema = self.schema_value.as_ref().unwrap().clone();
                    self.build_tree_state(&schema, &config);
                }
            }
            DriverEvent::ConfigUpdated(Ok(())) => {
                self.log_info("config updated");
            }
            DriverEvent::ConfigUpdated(Err(e)) => {
                self.log_error(format!("config error: {}", e));
                // Revert tree to driver's authoritative state
                let _ = handle.cmd_tx.send(Command::QueryConfig);
            }
            DriverEvent::CheckpointWritten { path } => {
                self.log_info(format!("wrote {}", path));
                self.refresh_files();
            }
            DriverEvent::StateCreated => {
                self.log_info("state created");
            }
            DriverEvent::StateDestroyed => {
                self.log_info("state destroyed");
            }
            DriverEvent::ConfigWritten { path } => {
                self.log_info(format!("wrote config to {}", path));
            }
            DriverEvent::ConfigLoaded { path } => {
                self.log_info(format!("loaded config from {}", path));
                let _ = handle.cmd_tx.send(Command::QueryConfig);
            }
            DriverEvent::CheckpointLoaded { path } => {
                self.log_info(format!("loaded checkpoint from {}", path));
                let _ = handle.cmd_tx.send(Command::QueryConfig);
            }
            DriverEvent::Error(e) => {
                self.log_error(format!("error: {}", e));
            }
            DriverEvent::Finished => {} // handled in main loop
        }
    }

    // ========================================================================
    // Key handling
    // ========================================================================

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Option<Command> {
        // When focused on the config tree, forward to term-schema first
        if self.focus == Focus::Middle && self.middle_tab == MiddleTab::Config {
            if let Some(ref mut tree) = self.tree_state {
                let event = crossterm::event::KeyEvent::new(code, modifiers);
                let result = schema_tui::handle_key_event(tree, event);
                match result {
                    EventResult::Consumed {
                        value_changed: true,
                    } => {
                        let value = tree.to_value();
                        return Some(Command::UpdateConfig(value));
                    }
                    EventResult::Consumed {
                        value_changed: false,
                    } => {
                        return None;
                    }
                    EventResult::Ignored => {
                        // Fall through to global keys below
                    }
                }
            }
        }

        // Global keys
        match code {
            KeyCode::Char('q') => return Some(Command::Quit),
            KeyCode::Char('p') => {
                if self.mode == DriverMode::Running {
                    return Some(Command::Pause);
                } else {
                    return Some(Command::Run);
                }
            }
            KeyCode::Char('s') => return Some(Command::Step),
            KeyCode::Char('n') => return Some(Command::CreateState),
            KeyCode::Char('d') => return Some(Command::DestroyState),
            KeyCode::Char('c') => return Some(Command::Checkpoint),
            KeyCode::Char('b') => {
                self.sidebar_visible = !self.sidebar_visible;
                if !self.sidebar_visible && self.focus == Focus::Left {
                    self.focus = Focus::Middle;
                }
                return None;
            }
            KeyCode::Char('w') => {
                let path = format!("{}/config.ron", self.output_dir);
                return Some(Command::WriteConfig(path));
            }
            KeyCode::Char('1') => {
                if self.sidebar_visible {
                    self.focus = Focus::Left;
                }
                return None;
            }
            KeyCode::Char('2') => {
                self.focus = Focus::Middle;
                return None;
            }
            KeyCode::Char('3') => {
                self.focus = Focus::Right;
                return None;
            }
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Left => Focus::Middle,
                    Focus::Middle => Focus::Right,
                    Focus::Right => {
                        if self.sidebar_visible {
                            Focus::Left
                        } else {
                            Focus::Middle
                        }
                    }
                };
                return None;
            }
            KeyCode::BackTab => {
                self.focus = match self.focus {
                    Focus::Left => Focus::Right,
                    Focus::Middle => {
                        if self.sidebar_visible {
                            Focus::Left
                        } else {
                            Focus::Right
                        }
                    }
                    Focus::Right => Focus::Middle,
                };
                return None;
            }
            _ => {}
        }

        // Focus-specific keys
        match self.focus {
            Focus::Left => self.handle_key_left(code),
            Focus::Middle => self.handle_key_middle(code),
            Focus::Right => self.handle_key_right(code),
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if let MouseEventKind::Down(MouseButton::Left) = mouse.kind {
            let pos = Position {
                x: mouse.column,
                y: mouse.row,
            };

            if self.right_tab == RightTab::Plot1D {
                // Click on y-legend checkbox toggles the series
                if self.y_legend_rect.contains(pos) {
                    let row = (pos.y - self.y_legend_rect.y) as usize;
                    if row < self.y_selected.len() {
                        self.y_selected[row] = !self.y_selected[row];
                        self.y_cursor = row;
                    }
                    return;
                }
                // Click on x-axis radio button selects it
                if pos.y == self.x_radio_y {
                    for (i, &(start, end)) in self.x_radio_spans.iter().enumerate() {
                        if pos.x >= start && pos.x < end {
                            self.x_axis_index = i;
                            return;
                        }
                    }
                }
            }

            let [left, middle, right] = self.column_areas;
            if self.sidebar_visible && left.contains(pos) {
                self.focus = Focus::Left;
            } else if middle.contains(pos) {
                self.focus = Focus::Middle;
            } else if right.contains(pos) {
                self.focus = Focus::Right;
            }
        }
    }

    fn handle_key_middle(&mut self, code: KeyCode) -> Option<Command> {
        match code {
            KeyCode::Char('[') => {
                self.middle_tab = match self.middle_tab {
                    MiddleTab::Config => MiddleTab::State,
                    MiddleTab::State => MiddleTab::Config,
                };
            }
            KeyCode::Char(']') => {
                self.middle_tab = match self.middle_tab {
                    MiddleTab::Config => MiddleTab::State,
                    MiddleTab::State => MiddleTab::Config,
                };
            }
            _ => {}
        }
        None
    }

    fn handle_key_left(&mut self, code: KeyCode) -> Option<Command> {
        match code {
            KeyCode::Char('[') | KeyCode::Char(']') => {
                self.file_tab = match self.file_tab {
                    FileTab::Configs => FileTab::Checkpoints,
                    FileTab::Checkpoints => FileTab::Configs,
                };
                self.file_list_state.select(Some(0));
            }
            KeyCode::Up => {
                self.file_list_state.select_previous();
            }
            KeyCode::Down => {
                self.file_list_state.select_next();
            }
            KeyCode::Enter => {
                if let Some(idx) = self.file_list_state.selected() {
                    let files = self.active_file_list();
                    if let Some(path) = files.get(idx) {
                        let path = path.clone();
                        return match self.file_tab {
                            FileTab::Configs => Some(Command::LoadConfig(path)),
                            FileTab::Checkpoints => Some(Command::LoadCheckpoint(path)),
                        };
                    }
                }
            }
            KeyCode::Delete | KeyCode::Backspace => {
                if let Some(idx) = self.file_list_state.selected() {
                    let files = self.active_file_list();
                    if let Some(path) = files.get(idx).cloned() {
                        match std::fs::remove_file(&path) {
                            Ok(()) => self.log_info(format!("deleted {}", path)),
                            Err(e) => self.log_error(format!("delete {}: {}", path, e)),
                        }
                        self.refresh_files();
                        let len = self.active_file_list().len();
                        if len == 0 {
                            self.file_list_state.select(None);
                        } else {
                            self.file_list_state.select(Some(idx.min(len - 1)));
                        }
                    }
                }
            }
            _ => {}
        }
        None
    }

    fn handle_key_right(&mut self, code: KeyCode) -> Option<Command> {
        match code {
            KeyCode::Char('[') => {
                self.right_tab = match self.right_tab {
                    RightTab::Log => RightTab::Plot2D,
                    RightTab::Plot1D => RightTab::Log,
                    RightTab::Plot2D => RightTab::Plot1D,
                };
            }
            KeyCode::Char(']') => {
                self.right_tab = match self.right_tab {
                    RightTab::Log => RightTab::Plot1D,
                    RightTab::Plot1D => RightTab::Plot2D,
                    RightTab::Plot2D => RightTab::Log,
                };
            }
            _ => match self.right_tab {
                RightTab::Log => {}
                RightTab::Plot1D => match code {
                    KeyCode::Char('x') if !self.linear_keys.is_empty() => {
                        let n = self.linear_keys.len();
                        let i = (self.x_axis_index + 1) % n;
                        self.x_axis_index = i;
                    }
                    KeyCode::Char('X') if !self.linear_keys.is_empty() => {
                        let n = self.linear_keys.len();
                        let i = self.x_axis_index;
                        self.x_axis_index = i.checked_sub(1).unwrap_or(n - 1);
                    }
                    KeyCode::Char('y') if !self.linear_keys.is_empty() => {
                        let n = self.linear_keys.len();
                        self.y_cursor = (self.y_cursor + 1) % n;
                        self.y_selected[self.y_cursor] = !self.y_selected[self.y_cursor];
                    }
                    KeyCode::Char('Y') if !self.linear_keys.is_empty() => {
                        let n = self.linear_keys.len();
                        self.y_cursor = self.y_cursor.checked_sub(1).unwrap_or(n - 1);
                        self.y_selected[self.y_cursor] = !self.y_selected[self.y_cursor];
                    }
                    _ => {}
                },
                RightTab::Plot2D => match code {
                    KeyCode::Char('y') if !self.planar_keys.is_empty() => {
                        let n = self.planar_keys.len();
                        let i = (self.planar_list.selected().unwrap_or(0) + 1) % n;
                        self.planar_list.select(Some(i));
                    }
                    KeyCode::Char('Y') if !self.planar_keys.is_empty() => {
                        let n = self.planar_keys.len();
                        let i = self.planar_list.selected().unwrap_or(0);
                        self.planar_list
                            .select(Some(i.checked_sub(1).unwrap_or(n - 1)));
                    }
                    _ => {}
                },
            },
        }
        None
    }
}

// ============================================================================
// Drawing
// ============================================================================

fn draw(frame: &mut Frame, state: &mut TuiState) {
    let area = frame.area();

    // Main area + footer
    let [main_area, footer_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);

    if state.sidebar_visible {
        let [left_area, middle_area, right_area] = Layout::horizontal([
            Constraint::Percentage(15),
            Constraint::Percentage(25),
            Constraint::Percentage(60),
        ])
        .areas(main_area);
        state.column_areas = [left_area, middle_area, right_area];
        draw_file_browser(frame, state, left_area);
        draw_middle(frame, state, middle_area);
        draw_right(frame, state, right_area);
    } else {
        let [middle_area, right_area] =
            Layout::horizontal([Constraint::Percentage(30), Constraint::Percentage(70)])
                .areas(main_area);
        state.column_areas = [Rect::default(), middle_area, right_area];
        draw_middle(frame, state, middle_area);
        draw_right(frame, state, right_area);
    }
    draw_footer(frame, state, footer_area);
}

fn draw_file_browser(frame: &mut Frame, state: &mut TuiState, area: Rect) {
    let focused = state.focus == Focus::Left;
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let tabs = Tabs::new(vec!["Configs", "Checkpoints"])
        .select(match state.file_tab {
            FileTab::Configs => 0,
            FileTab::Checkpoints => 1,
        })
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    let [tab_area, list_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);

    frame.render_widget(tabs, tab_area);

    let files: Vec<String> = state.active_file_list().to_vec();
    let items: Vec<ListItem> = files.iter().map(|f| ListItem::new(f.as_str())).collect();
    let list = List::new(items)
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style),
        );
    frame.render_stateful_widget(list, list_area, &mut state.file_list_state);
}

fn draw_middle(frame: &mut Frame, state: &mut TuiState, area: Rect) {
    let focused = state.focus == Focus::Middle;
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let editing = state
        .tree_state
        .as_ref()
        .is_some_and(|t| matches!(t.edit_mode, schema_tui::EditMode::Editing { .. }));

    // Tab row + content
    let tab_names = if editing {
        vec!["Config [editing]", "State"]
    } else {
        vec!["Config", "State"]
    };
    let tabs = Tabs::new(tab_names)
        .select(match state.middle_tab {
            MiddleTab::Config => 0,
            MiddleTab::State => 1,
        })
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    let [tab_area, content_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);

    frame.render_widget(tabs, tab_area);

    match state.middle_tab {
        MiddleTab::Config => {
            if let Some(ref mut tree) = state.tree_state {
                let tree_border = if editing {
                    Style::default().fg(Color::Yellow)
                } else {
                    border_style
                };
                let widget = schema_tui::SchemaTree::default()
                    .border(true)
                    .border_style(tree_border)
                    .show_help(false)
                    .highlight_style(if focused {
                        Style::default().bg(Color::DarkGray)
                    } else {
                        Style::default()
                    })
                    .key_style(Style::default().fg(Color::Cyan))
                    .value_style(Style::default().fg(Color::White))
                    .edit_style(Style::default().fg(Color::Yellow).bg(Color::DarkGray))
                    .disabled_style(Style::default().fg(Color::DarkGray));
                frame.render_stateful_widget(widget, content_area, tree);
            } else {
                let block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style);
                let p = Paragraph::new("loading...").block(block);
                frame.render_widget(p, content_area);
            }
        }
        MiddleTab::State => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(border_style);
            let info_lines = crate::format::highlight_ron_lines(&state.state_info_text);
            let info = Paragraph::new(info_lines)
                .wrap(Wrap { trim: false })
                .block(block);
            frame.render_widget(info, content_area);
        }
    }
}

fn draw_right(frame: &mut Frame, state: &mut TuiState, area: Rect) {
    let focused = state.focus == Focus::Right;
    let border_style = if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let tabs = Tabs::new(vec!["Log", "1D", "2D"])
        .select(match state.right_tab {
            RightTab::Log => 0,
            RightTab::Plot1D => 1,
            RightTab::Plot2D => 2,
        })
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

    let [tab_area, content_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);

    frame.render_widget(tabs, tab_area);

    match state.right_tab {
        RightTab::Log => draw_log(frame, state, content_area, border_style),
        RightTab::Plot1D => draw_plot_1d(frame, state, content_area, border_style),
        RightTab::Plot2D => draw_plot_2d(frame, state, content_area, border_style),
    }
}

fn draw_log(frame: &mut Frame, state: &TuiState, area: Rect, border_style: Style) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style);

    let inner_height = block.inner(area).height as usize;
    let skip = state.log_lines.len().saturating_sub(inner_height);

    let items: Vec<ListItem> = state
        .log_lines
        .iter()
        .skip(skip)
        .cloned()
        .map(ListItem::new)
        .collect();

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

const SERIES_COLORS: [Color; 6] = [
    Color::Cyan,
    Color::Yellow,
    Color::Green,
    Color::Magenta,
    Color::Red,
    Color::Blue,
];

fn draw_plot_1d(frame: &mut Frame, state: &mut TuiState, area: Rect, border_style: Style) {
    if state.linear_keys.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style);
        let p = Paragraph::new("no data").block(block);
        frame.render_widget(p, area);
        return;
    }

    // Layout: plot area above, centered x-tabs below
    let [plot_area, x_tab_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);

    // x-axis radio buttons (centered)
    let xi = state.x_axis_index;
    let mut x_spans: Vec<Span> = Vec::new();
    // Build radio items: (o) name  ( ) name ...
    let mut item_texts: Vec<String> = Vec::new();
    for (i, k) in state.linear_keys.iter().enumerate() {
        let dot = if i == xi { "o" } else { " " };
        item_texts.push(format!("({}) {}", dot, k));
    }
    let full_text = item_texts.join("  ");
    let prefix = "x: ";
    let total_len = prefix.len() + full_text.len();
    // Center: compute left padding
    let pad = (x_tab_area.width as usize).saturating_sub(total_len) / 2;

    // Record span positions for click hit-testing
    state.x_radio_spans.clear();
    state.x_radio_y = x_tab_area.y;
    let mut col = x_tab_area.x + pad as u16 + prefix.len() as u16;
    for text in &item_texts {
        let start = col;
        let end = col + text.len() as u16;
        state.x_radio_spans.push((start, end));
        col = end + 2; // "  " separator
    }

    x_spans.push(Span::raw(" ".repeat(pad)));
    x_spans.push(Span::styled(prefix, Style::default().fg(Color::DarkGray)));
    for (i, text) in item_texts.iter().enumerate() {
        if i > 0 {
            x_spans.push(Span::raw("  "));
        }
        let style = if i == xi {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        x_spans.push(Span::styled(text.clone(), style));
    }
    frame.render_widget(Paragraph::new(Line::from(x_spans)), x_tab_area);

    // Collect selected y-series with their colors
    let mut selected_series: Vec<(usize, &str, Color)> = Vec::new();
    for (i, key) in state.linear_keys.iter().enumerate() {
        if i < state.y_selected.len() && state.y_selected[i] {
            let color = SERIES_COLORS[selected_series.len() % SERIES_COLORS.len()];
            selected_series.push((i, key.as_str(), color));
        }
    }

    // Get x data
    let x_name = &state.linear_keys[xi];
    let x_data = match state.linear_data.get(x_name) {
        Some(d) if !d.is_empty() => d,
        _ => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(border_style);
            let p = Paragraph::new("missing x data").block(block);
            frame.render_widget(p, plot_area);
            return;
        }
    };

    // Compute bounds from x data and all selected y-series
    let x_min = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, name, _) in &selected_series {
        if let Some(yd) = state.linear_data.get(name) {
            for &v in yd {
                if v < y_min {
                    y_min = v;
                }
                if v > y_max {
                    y_max = v;
                }
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }

    let x_pad = (x_max - x_min).abs() * 0.05 + 1e-15;
    let y_pad = (y_max - y_min).abs() * 0.05 + 1e-15;

    // Build title from selected series names
    let title = if selected_series.is_empty() {
        format!(" 1d (x={}) ", x_name)
    } else {
        let names: Vec<&str> = selected_series.iter().map(|&(_, n, _)| n).collect();
        format!(" {} vs {} ", names.join(", "), x_name)
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);
    let inner = block.inner(plot_area);

    // Draw canvas with all selected y-series
    let canvas = Canvas::default()
        .block(block)
        .x_bounds([x_min - x_pad, x_max + x_pad])
        .y_bounds([y_min - y_pad, y_max + y_pad])
        .paint(|ctx| {
            for &(_, name, color) in &selected_series {
                if let Some(yd) = state.linear_data.get(name) {
                    let n = x_data.len().min(yd.len());
                    for i in 0..n.saturating_sub(1) {
                        ctx.draw(&CanvasLine {
                            x1: x_data[i],
                            y1: yd[i],
                            x2: x_data[i + 1],
                            y2: yd[i + 1],
                            color,
                        });
                    }
                }
            }
        });
    frame.render_widget(canvas, plot_area);

    // Inset y-legend overlaid on the plot (top-right corner)
    let legend_w = state.linear_keys.iter().map(|k| k.len()).max().unwrap_or(0) + 5; // "[x] " + name + padding
    let legend_h = state.linear_keys.len() as u16;
    if inner.width as usize >= legend_w + 2 && inner.height >= legend_h {
        state.y_legend_rect = Rect::new(
            inner.x + inner.width - legend_w as u16,
            inner.y,
            legend_w as u16,
            legend_h,
        );
        let lr = state.y_legend_rect;
        let buf = frame.buffer_mut();

        // Assign colors: only selected series get cycling palette colors
        let mut color_idx = 0usize;
        for (i, key) in state.linear_keys.iter().enumerate() {
            let selected = i < state.y_selected.len() && state.y_selected[i];
            let color = if selected {
                let c = SERIES_COLORS[color_idx % SERIES_COLORS.len()];
                color_idx += 1;
                c
            } else {
                Color::DarkGray
            };
            let check = if selected { "x" } else { " " };
            let label = format!("[{}] {}", check, key);
            let y = lr.y + i as u16;
            if y >= lr.y + lr.height {
                break;
            }
            let is_cursor = i == state.y_cursor;
            for (ci, ch) in label.chars().enumerate() {
                let x = lr.x + ci as u16;
                if x >= lr.x + lr.width {
                    break;
                }
                let cell = &mut buf[(x, y)];
                cell.set_symbol(&ch.to_string());
                if is_cursor {
                    cell.set_style(Style::default().fg(color).add_modifier(Modifier::BOLD));
                } else {
                    cell.set_style(Style::default().fg(color));
                }
            }
        }
    } else {
        state.y_legend_rect = Rect::default();
    }
}

fn draw_plot_2d(frame: &mut Frame, state: &mut TuiState, area: Rect, border_style: Style) {
    if state.planar_keys.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style);
        let p = Paragraph::new("no data").block(block);
        frame.render_widget(p, area);
        return;
    }

    // Layout: plot above, field selector row below
    let [plot_area, selector_area] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);

    let pi = state.planar_list.selected().unwrap_or(0);
    let name = &state.planar_keys[pi];

    // Field selector radio buttons (centered)
    let mut item_texts: Vec<String> = Vec::new();
    for (i, k) in state.planar_keys.iter().enumerate() {
        let dot = if i == pi { "o" } else { " " };
        item_texts.push(format!("({}) {}", dot, k));
    }
    let full_text = item_texts.join("  ");
    let total_len = full_text.len();
    let pad = (selector_area.width as usize).saturating_sub(total_len) / 2;

    let mut spans: Vec<Span> = Vec::new();
    spans.push(Span::raw(" ".repeat(pad)));
    for (i, text) in item_texts.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("  "));
        }
        let style = if i == pi {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::styled(text.clone(), style));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), selector_area);

    let (rows, cols, data) = match state.planar_data.get(name) {
        Some(d) => d,
        None => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(border_style);
            let p = Paragraph::new("missing data").block(block);
            frame.render_widget(p, plot_area);
            return;
        }
    };

    let (rows, cols) = (*rows, *cols);
    if data.is_empty() || rows == 0 || cols == 0 {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style);
        let p = Paragraph::new("empty data").block(block);
        frame.render_widget(p, plot_area);
        return;
    }

    // Find min/max for colormap, skipping NaN/Inf
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in data {
        if v.is_finite() {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
        }
    }
    if !vmin.is_finite() || !vmax.is_finite() {
        vmin = 0.0;
        vmax = 1.0;
    }
    let range = (vmax - vmin).max(1e-15);

    let title = format!(
        " {} ({}x{}) [{:.3e}, {:.3e}] ",
        name, rows, cols, vmin, vmax
    );
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(border_style);
    let inner = block.inner(plot_area);
    frame.render_widget(block, plot_area);

    let w = inner.width as usize;
    let h = inner.height as usize;
    if w == 0 || h == 0 {
        return;
    }

    let heatmap = Heatmap {
        data,
        rows,
        cols,
        vmin,
        range,
    };
    frame.render_widget(heatmap, inner);
}

struct Heatmap<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
    vmin: f64,
    range: f64,
}

impl Widget for Heatmap<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let w = area.width as usize;
        let h = area.height as usize;
        if w == 0 || h == 0 {
            return;
        }
        for ty in 0..h {
            for tx in 0..w {
                let dr = ty as f64 / h as f64;
                let dc = tx as f64 / w as f64;
                let r = (dr * self.rows as f64).min(self.rows as f64 - 1.0).max(0.0) as usize;
                let c = (dc * self.cols as f64).min(self.cols as f64 - 1.0).max(0.0) as usize;
                let v = self.data[r * self.cols + c];
                let color = colormap((v - self.vmin) / self.range);
                let cell = &mut buf[(area.x + tx as u16, area.y + ty as u16)];
                cell.set_symbol("\u{2588}");
                cell.set_fg(color);
            }
        }
    }
}

/// Map a normalized value [0, 1] to a color via a 5-stop colormap:
/// blue -> cyan -> green -> yellow -> red
fn colormap(t: f64) -> Color {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    Color::Rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

fn build_footer_hints(state: &TuiState) -> Vec<(&'static str, &'static str)> {
    let mut hints: Vec<(&str, &str)> = Vec::new();

    // Check if tree is in edit mode
    if state.focus == Focus::Middle && state.middle_tab == MiddleTab::Config {
        if let Some(ref tree) = state.tree_state {
            if matches!(tree.edit_mode, schema_tui::EditMode::Editing { .. }) {
                hints.push(("Enter", "confirm"));
                hints.push(("Esc", "cancel"));
                return hints;
            }
        }
    }

    // Focus-specific hints
    match state.focus {
        Focus::Left => {
            hints.push(("[/]", "tab"));
            hints.push(("↑↓", "navigate"));
            hints.push(("Enter", "load"));
            hints.push(("Del", "delete"));
        }
        Focus::Middle => {
            hints.push(("[/]", "tab"));
            match state.middle_tab {
                MiddleTab::Config => {
                    hints.push(("↑↓", "navigate"));
                    hints.push(("Enter", "edit"));
                    hints.push(("Space", "toggle"));
                }
                MiddleTab::State => {}
            }
        }
        Focus::Right => {
            hints.push(("[/]", "tab"));
            match state.right_tab {
                RightTab::Log => {}
                RightTab::Plot1D => {
                    hints.push(("x/X", "x-series"));
                    hints.push(("y/Y", "y-series"));
                }
                RightTab::Plot2D => {
                    hints.push(("y/Y", "field"));
                }
            }
        }
    }

    // Global hints
    if state.mode == DriverMode::Running {
        hints.push(("p", "pause"));
    } else {
        hints.push(("p", "play"));
    }
    hints.push(("s", "step"));
    hints.push(("n", "new-state"));
    hints.push(("d", "del-state"));
    hints.push(("c", "chkpt"));
    hints.push(("b", "sidebar"));
    hints.push(("w", "dump-cfg"));
    hints.push(("q", "quit"));

    hints
}

fn draw_footer(frame: &mut Frame, state: &TuiState, area: Rect) {
    let mode_str = match state.mode {
        DriverMode::Idle => "idle",
        DriverMode::Running => "running",
    };

    let bg = Color::Rgb(30, 30, 30);
    let key_style = Style::default().fg(Color::Yellow).bg(bg);
    let text_style = Style::default().fg(Color::Gray).bg(bg);

    let mut spans = vec![Span::styled(format!(" {} ", mode_str), text_style)];

    for (i, (key, desc)) in build_footer_hints(state).iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" ", text_style));
        }
        spans.push(Span::styled(*key, key_style));
        spans.push(Span::styled(format!(":{}", desc), text_style));
    }

    let [hints_area, msg_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Percentage(40)]).areas(area);

    let hints = Paragraph::new(Line::from(spans)).style(Style::default().bg(bg));
    frame.render_widget(hints, hints_area);

    if let Some((ref msg, color)) = state.last_message {
        let msg_paragraph = Paragraph::new(Span::styled(
            format!("{} ", msg),
            Style::default().fg(color).bg(bg),
        ))
        .alignment(Alignment::Right);
        frame.render_widget(msg_paragraph, msg_area);
    } else {
        frame.render_widget(Paragraph::new("").style(Style::default().bg(bg)), msg_area);
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn scan_files(dir: &str, ext: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                let s = path.display().to_string();
                Some(s.strip_prefix("./").map(String::from).unwrap_or(s))
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}
