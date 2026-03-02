//! Display formatting: RON syntax highlighting and JSON Schema rendering.

use ratatui::prelude::*;
use schemars::Schema;
use serde_json::Value;

// ============================================================================
// RON syntax highlighting
// ============================================================================

/// Token categories produced by the RON tokenizer.
#[derive(Clone, Copy, PartialEq)]
enum RonToken {
    String,
    Number,
    Keyword,
    TypeName,
    Punctuation,
    Comment,
    Plain,
}

/// Tokenize a single line of RON into `(token_kind, text)` pairs.
fn tokenize_ron_line(line: &str) -> Vec<(RonToken, String)> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") {
        return vec![(RonToken::Comment, line.to_string())];
    }

    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '"' => {
                let mut word = String::from("\"");
                chars.next();
                loop {
                    match chars.next() {
                        Some('\\') => {
                            word.push('\\');
                            if let Some(esc) = chars.next() {
                                word.push(esc);
                            }
                        }
                        Some('"') => {
                            word.push('"');
                            break;
                        }
                        Some(c) => word.push(c),
                        None => break,
                    }
                }
                tokens.push((RonToken::String, word));
            }
            '0'..='9' | '-'
                if ch == '-' && chars.clone().nth(1).is_some_and(|c| c.is_ascii_digit())
                    || ch.is_ascii_digit() =>
            {
                let mut word = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || c == '+'
                        || c == '-'
                    {
                        word.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push((RonToken::Number, word));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut word = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        word.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let kind = match word.as_str() {
                    "true" | "false" | "None" | "Some" => RonToken::Keyword,
                    _ => {
                        let next = chars.peek().copied();
                        if next == Some('(') || next == Some(':') {
                            RonToken::TypeName
                        } else {
                            RonToken::Plain
                        }
                    }
                };
                tokens.push((kind, word));
            }
            '(' | ')' | '{' | '}' | '[' | ']' | ':' | ',' => {
                tokens.push((RonToken::Punctuation, String::from(ch)));
                chars.next();
            }
            _ => {
                tokens.push((RonToken::Plain, String::from(ch)));
                chars.next();
            }
        }
    }
    tokens
}

/// Apply ANSI syntax highlighting to a RON string for terminal display.
pub fn highlight_ron(s: &str) -> String {
    const RESET: &str = "\x1b[0m";
    const CYAN: &str = "\x1b[1;36m";
    const YELLOW: &str = "\x1b[33m";
    const GREEN: &str = "\x1b[32m";
    const MAGENTA: &str = "\x1b[35m";
    const DIM: &str = "\x1b[2m";

    let mut out = String::with_capacity(s.len() * 2);

    for line in s.lines() {
        for (kind, text) in tokenize_ron_line(line) {
            let color = match kind {
                RonToken::String => GREEN,
                RonToken::Number => YELLOW,
                RonToken::Keyword => MAGENTA,
                RonToken::TypeName => CYAN,
                RonToken::Punctuation | RonToken::Comment => DIM,
                RonToken::Plain => {
                    out.push_str(&text);
                    continue;
                }
            };
            out.push_str(color);
            out.push_str(&text);
            out.push_str(RESET);
        }
        out.push('\n');
    }
    out
}

/// Syntax-highlight a RON string into ratatui `Line`/`Span` objects for TUI display.
pub fn highlight_ron_lines(s: &str) -> Vec<Line<'static>> {
    let cyan = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let yellow = Style::default().fg(Color::Yellow);
    let green = Style::default().fg(Color::Green);
    let magenta = Style::default().fg(Color::Magenta);
    let dim = Style::default().fg(Color::DarkGray);

    s.lines()
        .map(|line| {
            let spans: Vec<Span<'static>> = tokenize_ron_line(line)
                .into_iter()
                .map(|(kind, text)| {
                    let style = match kind {
                        RonToken::String => green,
                        RonToken::Number => yellow,
                        RonToken::Keyword => magenta,
                        RonToken::TypeName => cyan,
                        RonToken::Punctuation | RonToken::Comment => dim,
                        RonToken::Plain => Style::default(),
                    };
                    Span::styled(text, style)
                })
                .collect();
            Line::from(spans)
        })
        .collect()
}

// ============================================================================
// JSON Schema rendering
// ============================================================================

const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[1;36m";
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";

/// Print a schemars schema as a human-readable tree to stdout.
pub fn print_schema(schema: &Schema) {
    let val = schema.as_value();
    let defs = val.get("$defs").and_then(|d| d.as_object());
    let color = std::io::IsTerminal::is_terminal(&std::io::stdout());

    // Print top-level title if present
    if let Some(title) = val.get("title").and_then(|t| t.as_str()) {
        if color {
            println!("{}{}{}", CYAN, title, RESET);
        } else {
            println!("{}", title);
        }
    }
    print_node(val, defs, "  ", color);
}

fn print_node(
    schema: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    color: bool,
) {
    let schema = resolve(schema, defs);

    // oneOf — enum with variants
    if let Some(variants) = schema.get("oneOf").and_then(|v| v.as_array()) {
        for variant in variants {
            print_variant(variant, defs, indent, color);
        }
        return;
    }

    // All-unit enum: {"type": "string", "enum": ["Error", "Warn", ...]}
    if let Some(values) = schema.get("enum").and_then(|v| v.as_array()) {
        for val in values {
            if let Some(name) = val.as_str() {
                if color {
                    println!("{}{}| {}{}{}", indent, DIM, GREEN, name, RESET);
                } else {
                    println!("{}| {}", indent, name);
                }
            }
        }
        return;
    }

    // Object with properties — struct
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (name, prop_schema) in props {
            let type_name = describe_type(prop_schema, defs);
            let compound = is_compound_type(prop_schema, defs);

            if color {
                println!(
                    "{}{}{}{}: {}{}{}",
                    indent, name, DIM, RESET, CYAN, type_name, RESET
                );
            } else {
                println!("{}{}: {}", indent, name, type_name);
            }

            if compound {
                let resolved = resolve(prop_schema, defs);
                let inner = unwrap_option(resolved, defs).unwrap_or(resolved);
                let inner = if inner.get("type").and_then(|t| t.as_str()) == Some("array") {
                    inner
                        .get("items")
                        .map(|i| resolve(i, defs))
                        .unwrap_or(inner)
                } else {
                    inner
                };
                print_node(inner, defs, &format!("{}  ", indent), color);
            }
        }
        return;
    }
}

fn print_variant(
    variant: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    color: bool,
) {
    let variant = resolve(variant, defs);

    // Unit variant: {"type": "string", "enum": ["Sod"]}
    if let Some(values) = variant.get("enum").and_then(|v| v.as_array()) {
        if let Some(name) = values.first().and_then(|v| v.as_str()) {
            if color {
                println!("{}{}| {}{}{}", indent, DIM, GREEN, name, RESET);
            } else {
                println!("{}| {}", indent, name);
            }
            return;
        }
    }

    // Also handle {"const": "Sod"} form
    if let Some(name) = variant.get("const").and_then(|v| v.as_str()) {
        if color {
            println!("{}{}| {}{}{}", indent, DIM, GREEN, name, RESET);
        } else {
            println!("{}| {}", indent, name);
        }
        return;
    }

    // Struct/newtype variant: {"type": "object", "properties": {"VariantName": {...}}}
    if let Some(props) = variant.get("properties").and_then(|p| p.as_object()) {
        if let Some((name, inner_schema)) = props.iter().next() {
            let inner = resolve(inner_schema, defs);
            if let Some(inner_props) = inner.get("properties").and_then(|p| p.as_object()) {
                let fields: Vec<String> = inner_props
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, describe_type(v, defs)))
                    .collect();
                if color {
                    println!(
                        "{}{}| {}{}{} {}{{ {} }}{}",
                        indent,
                        DIM,
                        GREEN,
                        name,
                        RESET,
                        DIM,
                        fields.join(", "),
                        RESET
                    );
                } else {
                    println!("{}| {} {{ {} }}", indent, name, fields.join(", "));
                }
            } else {
                let type_name = describe_type(inner_schema, defs);
                if color {
                    println!(
                        "{}{}| {}{}{}{}({}){}",
                        indent, DIM, GREEN, name, RESET, DIM, type_name, RESET
                    );
                } else {
                    println!("{}| {}({})", indent, name, type_name);
                }
            }
            return;
        }
    }

    println!("{}| ???", indent);
}

/// Render a section's schema as a compact multi-line string.
///
/// `full_schema` is the JSON Schema for the whole `SimulationConfig`.
/// `section` is one of "driver", "physics", "initial", "compute".
pub fn section_schema_text(full_schema: &Value, section: &str) -> String {
    let defs = full_schema.get("$defs").and_then(|d| d.as_object());
    let prop = full_schema
        .get("properties")
        .and_then(|p| p.get(section));
    let prop = match prop {
        Some(p) => resolve(p, defs),
        None => return format!("no schema for {}", section),
    };
    let mut out = String::new();
    render_node(prop, defs, "", &mut out);
    out
}

/// Render a section's schema as styled ratatui Lines (matching CLI colors).
pub fn section_schema_lines(full_schema: &Value, section: &str) -> Vec<Line<'static>> {
    let defs = full_schema.get("$defs").and_then(|d| d.as_object());
    let prop = full_schema
        .get("properties")
        .and_then(|p| p.get(section));
    let prop = match prop {
        Some(p) => resolve(p, defs),
        None => return vec![Line::raw(format!("no schema for {}", section))],
    };
    let mut lines = Vec::new();
    styled_node(prop, defs, "", &mut lines);
    lines
}

fn styled_node(
    schema: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    lines: &mut Vec<Line<'static>>,
) {
    let schema = resolve(schema, defs);

    if let Some(variants) = schema.get("oneOf").and_then(|v| v.as_array()) {
        for variant in variants {
            styled_variant(variant, defs, indent, lines);
        }
        return;
    }

    if let Some(values) = schema.get("enum").and_then(|v| v.as_array()) {
        for val in values {
            if let Some(name) = val.as_str() {
                lines.push(Line::from(vec![
                    Span::raw(format!("{}", indent)),
                    Span::styled("| ", Style::default().dim()),
                    Span::styled(name.to_string(), Style::default().fg(Color::Green)),
                ]));
            }
        }
        return;
    }

    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (name, prop_schema) in props {
            let type_name = describe_type(prop_schema, defs);
            lines.push(Line::from(vec![
                Span::raw(format!("{}", indent)),
                Span::raw(name.clone()),
                Span::styled(": ", Style::default().dim()),
                Span::styled(type_name, Style::default().fg(Color::Cyan).bold()),
            ]));

            if is_compound_type(prop_schema, defs) {
                let resolved = resolve(prop_schema, defs);
                let inner = unwrap_option(resolved, defs).unwrap_or(resolved);
                let inner = if inner.get("type").and_then(|t| t.as_str()) == Some("array") {
                    inner
                        .get("items")
                        .map(|i| resolve(i, defs))
                        .unwrap_or(inner)
                } else {
                    inner
                };
                styled_node(inner, defs, &format!("{}  ", indent), lines);
            }
        }
    }
}

fn styled_variant(
    variant: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    lines: &mut Vec<Line<'static>>,
) {
    let variant = resolve(variant, defs);

    if let Some(values) = variant.get("enum").and_then(|v| v.as_array()) {
        if let Some(name) = values.first().and_then(|v| v.as_str()) {
            lines.push(Line::from(vec![
                Span::raw(format!("{}", indent)),
                Span::styled("| ", Style::default().dim()),
                Span::styled(name.to_string(), Style::default().fg(Color::Green)),
            ]));
            return;
        }
    }

    if let Some(name) = variant.get("const").and_then(|v| v.as_str()) {
        lines.push(Line::from(vec![
            Span::raw(format!("{}", indent)),
            Span::styled("| ", Style::default().dim()),
            Span::styled(name.to_string(), Style::default().fg(Color::Green)),
        ]));
        return;
    }

    if let Some(props) = variant.get("properties").and_then(|p| p.as_object()) {
        if let Some((name, inner_schema)) = props.iter().next() {
            let inner = resolve(inner_schema, defs);
            if let Some(inner_props) = inner.get("properties").and_then(|p| p.as_object()) {
                let fields: Vec<String> = inner_props
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, describe_type(v, defs)))
                    .collect();
                lines.push(Line::from(vec![
                    Span::raw(format!("{}", indent)),
                    Span::styled("| ", Style::default().dim()),
                    Span::styled(name.clone(), Style::default().fg(Color::Green)),
                    Span::styled(format!(" {{ {} }}", fields.join(", ")), Style::default().dim()),
                ]));
            } else {
                let type_name = describe_type(inner_schema, defs);
                lines.push(Line::from(vec![
                    Span::raw(format!("{}", indent)),
                    Span::styled("| ", Style::default().dim()),
                    Span::styled(name.clone(), Style::default().fg(Color::Green)),
                    Span::styled(format!("({})", type_name), Style::default().dim()),
                ]));
            }
            return;
        }
    }

    lines.push(Line::from(vec![
        Span::raw(format!("{}", indent)),
        Span::styled("| ???", Style::default().dim()),
    ]));
}

fn render_node(
    schema: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    out: &mut String,
) {
    let schema = resolve(schema, defs);

    if let Some(variants) = schema.get("oneOf").and_then(|v| v.as_array()) {
        for variant in variants {
            render_variant(variant, defs, indent, out);
        }
        return;
    }

    if let Some(values) = schema.get("enum").and_then(|v| v.as_array()) {
        for val in values {
            if let Some(name) = val.as_str() {
                out.push_str(&format!("{}| {}\n", indent, name));
            }
        }
        return;
    }

    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (name, prop_schema) in props {
            let type_name = describe_type(prop_schema, defs);
            out.push_str(&format!("{}{}: {}\n", indent, name, type_name));

            if is_compound_type(prop_schema, defs) {
                let resolved = resolve(prop_schema, defs);
                let inner = unwrap_option(resolved, defs).unwrap_or(resolved);
                let inner = if inner.get("type").and_then(|t| t.as_str()) == Some("array") {
                    inner
                        .get("items")
                        .map(|i| resolve(i, defs))
                        .unwrap_or(inner)
                } else {
                    inner
                };
                render_node(inner, defs, &format!("{}  ", indent), out);
            }
        }
    }
}

fn render_variant(
    variant: &Value,
    defs: Option<&serde_json::Map<String, Value>>,
    indent: &str,
    out: &mut String,
) {
    let variant = resolve(variant, defs);

    if let Some(values) = variant.get("enum").and_then(|v| v.as_array()) {
        if let Some(name) = values.first().and_then(|v| v.as_str()) {
            out.push_str(&format!("{}| {}\n", indent, name));
            return;
        }
    }

    if let Some(name) = variant.get("const").and_then(|v| v.as_str()) {
        out.push_str(&format!("{}| {}\n", indent, name));
        return;
    }

    if let Some(props) = variant.get("properties").and_then(|p| p.as_object()) {
        if let Some((name, inner_schema)) = props.iter().next() {
            let inner = resolve(inner_schema, defs);
            if let Some(inner_props) = inner.get("properties").and_then(|p| p.as_object()) {
                let fields: Vec<String> = inner_props
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, describe_type(v, defs)))
                    .collect();
                out.push_str(&format!("{}| {} {{ {} }}\n", indent, name, fields.join(", ")));
            } else {
                let type_name = describe_type(inner_schema, defs);
                out.push_str(&format!("{}| {}({})\n", indent, name, type_name));
            }
            return;
        }
    }

    out.push_str(&format!("{}| ???\n", indent));
}

// ============================================================================
// Schema helpers
// ============================================================================

fn describe_type(schema: &Value, defs: Option<&serde_json::Map<String, Value>>) -> String {
    // Extract name from $ref before resolving
    let ref_name = schema
        .get("$ref")
        .and_then(|v| v.as_str())
        .and_then(|r| r.strip_prefix("#/$defs/"));

    let schema = resolve(schema, defs);

    // anyOf with null — Option<T>
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        let non_null: Vec<&Value> = any_of
            .iter()
            .filter(|s| resolve(s, defs).get("type").and_then(|t| t.as_str()) != Some("null"))
            .collect();
        if non_null.len() == 1 && any_of.len() == 2 {
            return format!("Option<{}>", describe_type(non_null[0], defs));
        }
    }

    // All-unit enum: {"type": "string", "enum": [...]}
    if schema.get("enum").is_some() {
        return type_name(schema, ref_name, "enum");
    }

    // oneOf — enum
    if schema.get("oneOf").is_some() {
        return type_name(schema, ref_name, "enum");
    }

    // Object with properties — struct
    if schema.get("properties").is_some() {
        return type_name(schema, ref_name, "struct");
    }

    // Type can be a string or an array (e.g. ["number", "null"] for Option<f64>)
    match schema.get("type") {
        Some(Value::String(ty)) => {
            return match ty.as_str() {
                "string" => "String".to_string(),
                "boolean" => "bool".to_string(),
                "integer" => format_int(schema),
                "number" => format_float(schema),
                "array" => {
                    if let Some(items) = schema.get("items") {
                        format!("Vec<{}>", describe_type(items, defs))
                    } else {
                        "Vec<?>".to_string()
                    }
                }
                "null" => "null".to_string(),
                other => other.to_string(),
            };
        }
        Some(Value::Array(types)) => {
            // ["number", "null"] → Option<f64>
            let non_null: Vec<&str> = types
                .iter()
                .filter_map(|t| t.as_str())
                .filter(|t| *t != "null")
                .collect();
            if non_null.len() == 1 && types.len() == 2 {
                let inner = primitive_type(non_null[0], schema);
                return format!("Option<{}>", inner);
            }
        }
        _ => {}
    }

    "?".to_string()
}

fn type_name(schema: &Value, ref_name: Option<&str>, fallback: &str) -> String {
    schema
        .get("title")
        .and_then(|t| t.as_str())
        .or(ref_name)
        .unwrap_or(fallback)
        .to_string()
}

fn primitive_type(ty: &str, schema: &Value) -> String {
    match ty {
        "string" => "String".into(),
        "boolean" => "bool".into(),
        "integer" => format_int(schema),
        "number" => format_float(schema),
        other => other.into(),
    }
}

fn format_int(schema: &Value) -> String {
    match schema.get("format").and_then(|f| f.as_str()) {
        Some("uint8") => "u8".into(),
        Some("uint16") => "u16".into(),
        Some("uint32") => "u32".into(),
        Some("uint64") => "u64".into(),
        Some("uint") => "usize".into(),
        Some("int8") => "i8".into(),
        Some("int16") => "i16".into(),
        Some("int32") => "i32".into(),
        Some("int64") => "i64".into(),
        Some("int") => "isize".into(),
        _ => "integer".into(),
    }
}

fn format_float(schema: &Value) -> String {
    match schema.get("format").and_then(|f| f.as_str()) {
        Some("float") => "f32".into(),
        Some("double") => "f64".into(),
        _ => "number".into(),
    }
}

fn is_compound_type(schema: &Value, defs: Option<&serde_json::Map<String, Value>>) -> bool {
    let schema = resolve(schema, defs);

    // Unwrap Option
    if let Some(any_of) = schema.get("anyOf").and_then(|v| v.as_array()) {
        let non_null: Vec<&Value> = any_of
            .iter()
            .filter(|s| resolve(s, defs).get("type").and_then(|t| t.as_str()) != Some("null"))
            .collect();
        if non_null.len() == 1 && any_of.len() == 2 {
            return is_compound_type(non_null[0], defs);
        }
    }

    // Look through array items
    if schema.get("type").and_then(|t| t.as_str()) == Some("array") {
        if let Some(items) = schema.get("items") {
            return is_compound_type(items, defs);
        }
    }

    schema.get("properties").is_some()
        || schema.get("oneOf").is_some()
        || schema.get("enum").is_some()
}

fn resolve<'a>(schema: &'a Value, defs: Option<&'a serde_json::Map<String, Value>>) -> &'a Value {
    if let Some(r) = schema.get("$ref").and_then(|v| v.as_str()) {
        if let Some(name) = r.strip_prefix("#/$defs/") {
            if let Some(resolved) = defs.and_then(|d| d.get(name)) {
                return resolved;
            }
        }
    }
    schema
}

fn unwrap_option<'a>(
    schema: &'a Value,
    defs: Option<&'a serde_json::Map<String, Value>>,
) -> Option<&'a Value> {
    let any_of = schema.get("anyOf")?.as_array()?;
    let non_null: Vec<&Value> = any_of
        .iter()
        .filter(|s| resolve(s, defs).get("type").and_then(|t| t.as_str()) != Some("null"))
        .collect();
    if non_null.len() == 1 && any_of.len() == 2 {
        Some(resolve(non_null[0], defs))
    } else {
        None
    }
}
