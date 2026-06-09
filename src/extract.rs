//! Parsing agent session JSONL files into `ChangeEvent`s, plus helpers for
//! session-file discovery and full-change re-extraction.
//!
//! Three harness dialects are supported, auto-detected per line by
//! [`extract_change_events`]: **Claude Code** (`type: "assistant"`, `tool_use`
//! blocks with `input`), **pi** (`type: "message"` + assistant role, `toolCall`
//! blocks with `arguments`), and **Codex** (OpenAI Codex CLI;
//! `type: "response_item"`, an `apply_patch` payload). Claude and pi both nest a
//! `message.content[]` array of typed blocks, so they share the block loop and
//! the event-building helpers; Codex has no content array, so it dispatches at
//! the line level and parses the apply-patch envelope.
//!
//! Each harness has its own session-directory layout and discovery model,
//! exposed as separate discovery functions ([`claude_session_files`],
//! [`pi_session_files`], [`codex_session_files`]) plus a merged
//! [`session_files`]. Claude and pi encode the worktree in the directory path;
//! Codex records it in each file's first line, so its discovery matches by
//! content (see [`codex_session_files`]).
//!
//! Inlines `encode_cwd` and `parse_iso8601_ms` from the host project's
//! `activity/events.rs` so this crate has no back-dependency.

use crate::event::{ChangeDetail, ChangeEvent, ChangeSource, ChangeTool};
use std::path::{Path, PathBuf};

pub(crate) const SUMMARY_MAX_CHARS: usize = 80;

/// Bounded number of characters retained per side of a diff peek.
pub const DETAIL_MAX_CHARS: usize = 600;

fn clip(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn tool_from_name(name: &str) -> Option<ChangeTool> {
    match name {
        "Edit" => Some(ChangeTool::Edit),
        "MultiEdit" => Some(ChangeTool::MultiEdit),
        "Write" => Some(ChangeTool::Write),
        "NotebookEdit" => Some(ChangeTool::NotebookEdit),
        _ => None,
    }
}

/// True if a line looks like a declaration worth surfacing.
fn looks_like_decl(line: &str) -> bool {
    let t = line.trim_start();
    const KW: [&str; 11] = [
        "fn ", "pub ", "def ", "class ", "struct ", "impl ", "enum ", "trait ", "func ", "type ",
        "const ",
    ];
    KW.iter().any(|k| t.starts_with(k))
}

fn truncate_summary(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= SUMMARY_MAX_CHARS {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(SUMMARY_MAX_CHARS - 1).collect();
    out.push('…');
    out
}

/// Summarize an Edit/MultiEdit: prefer a declaration among lines present in
/// `new` but not `old`; else the first non-blank line of `new` not in `old`;
/// else the first non-blank line of `new`.
pub(crate) fn summarize_edit(old: &str, new: &str) -> String {
    let old_lines: std::collections::HashSet<&str> = old.lines().collect();
    let changed: Vec<&str> = new
        .lines()
        .filter(|l| !old_lines.contains(*l) && !l.trim().is_empty())
        .collect();
    if let Some(decl) = changed.iter().find(|l| looks_like_decl(l)) {
        return truncate_summary(decl);
    }
    if let Some(first) = changed.first() {
        return truncate_summary(first);
    }
    match new.lines().find(|l| !l.trim().is_empty()) {
        Some(l) => truncate_summary(l),
        None => "edit".to_string(),
    }
}

/// Summarize a Write: the first declaration in the content, else "new file".
pub(crate) fn summarize_write(content: &str) -> String {
    match content.lines().find(|l| looks_like_decl(l)) {
        Some(decl) => truncate_summary(decl),
        None => "new file".to_string(),
    }
}

/// Which harness wrote a session line. Claude and pi nest a `message.content[]`
/// array of typed blocks (they differ only in marker fields), so they share the
/// block loop. Codex has no content array — its mutating calls live directly in
/// a per-line `payload` — so it dispatches at the line level instead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dialect {
    /// Claude Code: `type == "assistant"`, `tool_use` blocks with `input`.
    Claude,
    /// pi: `type == "message"` + `message.role == "assistant"`, `toolCall`
    /// blocks with `arguments`.
    Pi,
    /// Codex: `type == "response_item"`, a `payload` carrying a `function_call`
    /// or `custom_tool_call` (no `message.content[]`).
    Codex,
}

/// Identify the harness dialect of a session line, or `None` for lines that
/// can't carry mutating tool calls (user turns, tool results, metadata).
fn detect_dialect(v: &serde_json::Value) -> Option<Dialect> {
    match v.get("type").and_then(|t| t.as_str())? {
        "assistant" => Some(Dialect::Claude),
        "message" => match v
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
        {
            Some("assistant") => Some(Dialect::Pi),
            _ => None,
        },
        "response_item" => Some(Dialect::Codex),
        _ => None,
    }
}

/// Append a `Write` event. `index_in_line` is the current output length so the
/// source back-reference survives full re-extraction.
fn push_write(
    out: &mut Vec<ChangeEvent>,
    ts: i64,
    file_path: PathBuf,
    content: &str,
    detail_max: usize,
) {
    out.push(ChangeEvent {
        timestamp_ms: ts,
        tool: ChangeTool::Write,
        file_path,
        summary: summarize_write(content),
        detail: ChangeDetail::Write {
            head: clip(content, detail_max),
        },
        source: ChangeSource {
            session_file: PathBuf::new(),
            line_index: 0,
            index_in_line: out.len(),
        },
    });
}

/// Append an edit event with the given tool variant and old/new text.
fn push_edit(
    out: &mut Vec<ChangeEvent>,
    ts: i64,
    tool: ChangeTool,
    file_path: PathBuf,
    old: &str,
    new: &str,
    detail_max: usize,
) {
    out.push(ChangeEvent {
        timestamp_ms: ts,
        tool,
        file_path,
        summary: summarize_edit(old, new),
        detail: ChangeDetail::Edit {
            old: clip(old, detail_max),
            new: clip(new, detail_max),
        },
        source: ChangeSource {
            session_file: PathBuf::new(),
            line_index: 0,
            index_in_line: out.len(),
        },
    });
}

/// Handle one Claude `tool_use` block, appending any events it produces.
fn extract_claude_block(
    block: &serde_json::Value,
    ts: i64,
    detail_max: usize,
    out: &mut Vec<ChangeEvent>,
) {
    if block.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
        return;
    }
    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let Some(tool) = tool_from_name(name) else {
        return;
    };
    let input = block.get("input").unwrap_or(&serde_json::Value::Null);
    let file = input
        .get("file_path")
        .or_else(|| input.get("notebook_path"))
        .and_then(|p| p.as_str());
    let Some(file) = file else { return };
    let file_path = PathBuf::from(file);

    match tool {
        ChangeTool::Write => {
            let content = input.get("content").and_then(|c| c.as_str()).unwrap_or("");
            push_write(out, ts, file_path, content, detail_max);
        }
        ChangeTool::MultiEdit => {
            if let Some(edits) = input.get("edits").and_then(|e| e.as_array()) {
                for e in edits {
                    let old = e.get("old_string").and_then(|s| s.as_str()).unwrap_or("");
                    let new = e.get("new_string").and_then(|s| s.as_str()).unwrap_or("");
                    push_edit(out, ts, tool, file_path.clone(), old, new, detail_max);
                }
            }
        }
        ChangeTool::Edit | ChangeTool::NotebookEdit => {
            let old = input
                .get("old_string")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            let new = input
                .get("new_string")
                .or_else(|| input.get("new_source"))
                .and_then(|s| s.as_str())
                .unwrap_or("");
            push_edit(out, ts, tool, file_path, old, new, detail_max);
        }
    }
}

/// Handle one pi `toolCall` block, appending any events it produces. pi's
/// `edit` always carries an `edits[]` array (mapped to `MultiEdit`, one event
/// per element); `write` carries `content`.
fn extract_pi_block(
    block: &serde_json::Value,
    ts: i64,
    detail_max: usize,
    out: &mut Vec<ChangeEvent>,
) {
    if block.get("type").and_then(|t| t.as_str()) != Some("toolCall") {
        return;
    }
    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let args = block.get("arguments").unwrap_or(&serde_json::Value::Null);
    let Some(file) = args.get("path").and_then(|p| p.as_str()) else {
        return;
    };
    let file_path = PathBuf::from(file);

    match name {
        "edit" => {
            if let Some(edits) = args.get("edits").and_then(|e| e.as_array()) {
                for e in edits {
                    let old = e.get("oldText").and_then(|s| s.as_str()).unwrap_or("");
                    let new = e.get("newText").and_then(|s| s.as_str()).unwrap_or("");
                    push_edit(
                        out,
                        ts,
                        ChangeTool::MultiEdit,
                        file_path.clone(),
                        old,
                        new,
                        detail_max,
                    );
                }
            }
        }
        "write" => {
            let content = args.get("content").and_then(|c| c.as_str()).unwrap_or("");
            push_write(out, ts, file_path, content, detail_max);
        }
        _ => {}
    }
}

/// Pull the raw apply-patch envelope text out of one Codex `payload`, or `None`
/// for any other tool. Codex carries `apply_patch` two ways depending on how the
/// tool was registered, and we accept both:
/// - `custom_tool_call` (freeform tool): the patch is the `input` string verbatim.
/// - `function_call` (JSON tool): `arguments` is a JSON-encoded *string* — decode
///   it, then read the patch from the object's `input` field. (If `arguments`
///   itself decodes straight to a string, treat that as the patch.)
///
/// `exec_command`/shell calls carry arbitrary shell (`sed`, heredocs, …) and are
/// not reliably extractable as structured changes — they return `None` here and
/// are documented as a known gap.
fn codex_apply_patch_text(payload: &serde_json::Value) -> Option<String> {
    let name = payload.get("name").and_then(|n| n.as_str()).unwrap_or("");
    if name != "apply_patch" {
        return None;
    }
    match payload.get("type").and_then(|t| t.as_str()) {
        Some("custom_tool_call") => payload
            .get("input")
            .and_then(|i| i.as_str())
            .map(str::to_string),
        Some("function_call") => {
            let raw = payload.get("arguments").and_then(|a| a.as_str())?;
            match serde_json::from_str::<serde_json::Value>(raw) {
                Ok(serde_json::Value::Object(map)) => map
                    .get("input")
                    .and_then(|i| i.as_str())
                    .map(str::to_string),
                Ok(serde_json::Value::String(s)) => Some(s),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Parse one Codex `response_item` line: if its payload is an `apply_patch`
/// invocation, parse the patch envelope into events. Other payloads
/// (`exec_command`, `function_call_output`, `reasoning`, assistant `message`, …)
/// produce nothing.
fn extract_codex_line(
    v: &serde_json::Value,
    ts: i64,
    detail_max: usize,
    out: &mut Vec<ChangeEvent>,
) {
    let Some(payload) = v.get("payload") else {
        return;
    };
    if let Some(patch) = codex_apply_patch_text(payload) {
        parse_apply_patch(&patch, ts, detail_max, out);
    }
}

/// Parse a Codex apply-patch envelope into `ChangeEvent`s. The envelope is a
/// sequence of file sections framed by `*** Begin Patch` / `*** End Patch`:
/// - `*** Add File: <path>` followed by `+`-prefixed lines → one `Write`.
/// - `*** Update File: <path>` followed by one or more `@@` hunks of
///   ` `/`-`/`+`-prefixed lines → one `MultiEdit` per hunk (mirrors how a pi
///   `edit` emits one event per element, so `index_in_line` stays stable for
///   `load_full_change`). A trailing `*** Move to: <dest>` retargets the section
///   to `<dest>` (the path the edits live at after the patch applies).
/// - `*** Delete File: <path>` → skipped (no change text to show).
///
/// Lenient: leading text before `*** Begin Patch` is ignored, and the parser
/// does not require the framing markers to be present.
fn parse_apply_patch(patch: &str, ts: i64, detail_max: usize, out: &mut Vec<ChangeEvent>) {
    // State for the section currently being accumulated.
    enum Section {
        None,
        Add { path: PathBuf, lines: Vec<String> },
        Update { path: PathBuf, hunk: Option<Hunk> },
    }
    #[derive(Default)]
    struct Hunk {
        old: Vec<String>,
        new: Vec<String>,
    }

    fn flush_hunk(path: &Path, hunk: Hunk, ts: i64, detail_max: usize, out: &mut Vec<ChangeEvent>) {
        push_edit(
            out,
            ts,
            ChangeTool::MultiEdit,
            path.to_path_buf(),
            &hunk.old.join("\n"),
            &hunk.new.join("\n"),
            detail_max,
        );
    }

    fn flush_section(section: Section, ts: i64, detail_max: usize, out: &mut Vec<ChangeEvent>) {
        match section {
            Section::None => {}
            Section::Add { path, lines } => {
                push_write(out, ts, path, &lines.join("\n"), detail_max);
            }
            Section::Update { path, hunk } => {
                if let Some(hunk) = hunk {
                    flush_hunk(&path, hunk, ts, detail_max, out);
                }
            }
        }
    }

    let mut section = Section::None;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("*** ") {
            // `Move to:` renames the file being updated: retarget the current
            // Update section to the destination so events attribute to the path
            // that exists after the patch applies (not the now-gone original).
            if let Some(dest) = rest.strip_prefix("Move to: ") {
                if let Section::Update { path, .. } = &mut section {
                    *path = PathBuf::from(dest.trim());
                }
                continue;
            }
            // `End of File` is a hunk annotation, not a section boundary.
            if rest.starts_with("End of File") {
                continue;
            }
            if let Some(p) = rest.strip_prefix("Add File: ") {
                flush_section(
                    std::mem::replace(&mut section, Section::None),
                    ts,
                    detail_max,
                    out,
                );
                section = Section::Add {
                    path: PathBuf::from(p.trim()),
                    lines: Vec::new(),
                };
            } else if let Some(p) = rest.strip_prefix("Update File: ") {
                flush_section(
                    std::mem::replace(&mut section, Section::None),
                    ts,
                    detail_max,
                    out,
                );
                section = Section::Update {
                    path: PathBuf::from(p.trim()),
                    hunk: None,
                };
            } else {
                // `Delete File:`, `Begin Patch`, `End Patch`, `Environment ID:`,
                // or anything else: end the current section, accumulate nothing.
                flush_section(
                    std::mem::replace(&mut section, Section::None),
                    ts,
                    detail_max,
                    out,
                );
            }
            continue;
        }
        match &mut section {
            Section::Add { lines, .. } => {
                // Added lines are `+`-prefixed; tolerate a stray bare line.
                lines.push(line.strip_prefix('+').unwrap_or(line).to_string());
            }
            Section::Update { path, hunk } => {
                if line.starts_with("@@") {
                    // Hunk boundary: flush the previous hunk, start fresh.
                    if let Some(h) = hunk.take() {
                        flush_hunk(path, h, ts, detail_max, out);
                    }
                    continue;
                }
                let h = hunk.get_or_insert_with(Hunk::default);
                match line.chars().next() {
                    Some('-') => h.old.push(line[1..].to_string()),
                    Some('+') => h.new.push(line[1..].to_string()),
                    Some(' ') => {
                        // Context line: present in both sides.
                        h.old.push(line[1..].to_string());
                        h.new.push(line[1..].to_string());
                    }
                    // A blank line is blank context; anything else is treated as
                    // context too (defensive — valid hunks only use ` `/`+`/`-`).
                    _ => {
                        h.old.push(line.to_string());
                        h.new.push(line.to_string());
                    }
                }
            }
            Section::None => {}
        }
    }
    flush_section(section, ts, detail_max, out);
}

/// Extract zero or more `ChangeEvent`s from one parsed session JSONL line.
/// Auto-detects the harness dialect (Claude, pi, or Codex). Claude/pi loop the
/// `message.content[]` blocks; Codex parses the line's `payload` directly. Only
/// mutating tool calls produce events; a multi-edit or multi-hunk call produces
/// one event per edit/hunk. `detail_max` caps the chars retained per diff side;
/// pass `DETAIL_MAX_CHARS` for normal use or `usize::MAX` for full unclipped
/// re-extraction.
pub fn extract_change_events(v: &serde_json::Value, detail_max: usize) -> Vec<ChangeEvent> {
    let mut out = Vec::new();
    let Some(dialect) = detect_dialect(v) else {
        return out;
    };
    let ts = v
        .get("timestamp")
        .and_then(|t| t.as_str())
        .and_then(parse_iso8601_ms)
        .unwrap_or(0);
    match dialect {
        Dialect::Claude | Dialect::Pi => {
            let Some(blocks) = v
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_array())
            else {
                return out;
            };
            for block in blocks {
                match dialect {
                    Dialect::Claude => extract_claude_block(block, ts, detail_max, &mut out),
                    Dialect::Pi => extract_pi_block(block, ts, detail_max, &mut out),
                    Dialect::Codex => unreachable!("Codex handled at the line level"),
                }
            }
        }
        Dialect::Codex => extract_codex_line(v, ts, detail_max, &mut out),
    }
    out
}

/// 1-based line to open the editor at, given the file's current contents and
/// the change detail. The chronology records changes that were already applied,
/// so the file holds the NEW text — locate the first non-blank line of `new` in
/// `contents`; for a Write (or anything not found), line 1.
pub fn resolve_line(contents: &str, detail: &ChangeDetail) -> u32 {
    let needle = match detail {
        ChangeDetail::Edit { new, .. } => new.lines().find(|l| !l.trim().is_empty()),
        _ => None,
    };
    let Some(needle) = needle else { return 1 };
    for (i, line) in contents.lines().enumerate() {
        if line.contains(needle) {
            return (i + 1) as u32;
        }
    }
    1
}

/// Read the file at `path` and resolve the line for `detail`. Returns 1 when
/// the file can't be read (deleted/renamed since the edit).
pub fn resolve_line_in_file(path: &Path, detail: &ChangeDetail) -> u32 {
    match std::fs::read_to_string(path) {
        Ok(contents) => resolve_line(&contents, detail),
        Err(_) => 1,
    }
}

/// Parse every line of a session file into `ChangeEvent`s. Malformed lines are
/// skipped silently (matches the existing tail-loop tolerance).
pub fn parse_file(path: &Path) -> Vec<ChangeEvent> {
    use std::io::{BufRead, BufReader};
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (line_index, line) in BufReader::new(file)
        .lines()
        .map_while(|l| l.ok())
        .enumerate()
    {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
            for mut ev in extract_change_events(&v, DETAIL_MAX_CHARS) {
                ev.source.session_file = path.to_path_buf();
                ev.source.line_index = line_index;
                out.push(ev);
            }
        }
    }
    out
}

/// Re-read the un-clipped change for `ev` from its session log. `None` when the
/// source is empty/unreadable or the line/event is gone (callers fall back to
/// the clipped `detail`).
pub fn load_full_change(ev: &ChangeEvent) -> Option<ChangeDetail> {
    use std::io::{BufRead, BufReader};
    if ev.source.session_file.as_os_str().is_empty() {
        return None;
    }
    let file = std::fs::File::open(&ev.source.session_file).ok()?;
    let line = BufReader::new(file)
        .lines()
        .map_while(|l| l.ok())
        .nth(ev.source.line_index)?;
    let v: serde_json::Value = serde_json::from_str(&line).ok()?;
    let evs = extract_change_events(&v, usize::MAX);
    evs.into_iter()
        .nth(ev.source.index_in_line)
        .map(|e| e.detail)
}

/// All `.jsonl` files directly in `dir` (non-recursive). Returns empty if the
/// directory is missing or unreadable.
fn collect_jsonl(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return files;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    files
}

/// All `.jsonl` session files under `<home>/.claude/projects/<encoded-cwd>/`.
/// Testable variant taking an explicit home dir and canonical worktree path.
pub(crate) fn session_files_in(home: &Path, abs_worktree: &Path) -> Vec<PathBuf> {
    collect_jsonl(&home.join(".claude/projects").join(encode_cwd(abs_worktree)))
}

/// All `.jsonl` session files under
/// `<home>/.pi/agent/sessions/<pi-encoded-cwd>/`. Testable variant taking an
/// explicit home dir and canonical worktree path.
pub(crate) fn pi_session_files_in(home: &Path, abs_worktree: &Path) -> Vec<PathBuf> {
    collect_jsonl(
        &home
            .join(".pi/agent/sessions")
            .join(pi_encode_cwd(abs_worktree)),
    )
}

/// Cap how many Codex rollout files we content-scan per discovery, newest-first,
/// so a long session history can't make a refresh loop pathological.
const CODEX_SCAN_CAP: usize = 500;

/// True if `path` looks like a Codex rollout file (`rollout-*.jsonl`).
fn is_rollout_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.starts_with("rollout-") && name.ends_with(".jsonl")
}

/// Collect up to `cap` `rollout-*.jsonl` files under `dir`, newest-first, then
/// stop. Codex names both its partition dirs (`YYYY`/`MM`/`DD`) and its
/// `rollout-<ISO>-…` files so that a descending sort is chronological at every
/// level, so a depth-first walk that visits each directory's entries in
/// descending order yields rollouts globally newest-first — and bailing out once
/// `cap` are collected means a long history's older partitions are never read.
/// Depth-agnostic (doesn't hard-code the `YYYY/MM/DD` layout) so it tolerates a
/// future partitioning change.
fn collect_rollouts(dir: &Path, cap: usize, out: &mut Vec<PathBuf>) {
    if out.len() >= cap {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<std::fs::DirEntry> = rd.flatten().collect();
    // Descending by name → newest first. Uses the dirent name only (no stat).
    entries.sort_by_key(|b| std::cmp::Reverse(b.file_name()));
    for entry in entries {
        if out.len() >= cap {
            return;
        }
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => collect_rollouts(&path, cap, out),
            Ok(_) if is_rollout_file(&path) => out.push(path),
            _ => {}
        }
    }
}

/// Read only the first line of a rollout file and return its recorded `cwd`
/// (`session_meta.payload.cwd`), or `None` if the first line isn't a
/// `session_meta` record.
fn rollout_cwd(path: &Path) -> Option<PathBuf> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).ok()?;
    let mut first = String::new();
    BufReader::new(file).read_line(&mut first).ok()?;
    let v: serde_json::Value = serde_json::from_str(first.trim_end()).ok()?;
    if v.get("type").and_then(|t| t.as_str()) != Some("session_meta") {
        return None;
    }
    v.get("payload")
        .and_then(|p| p.get("cwd"))
        .and_then(|c| c.as_str())
        .map(PathBuf::from)
}

/// True if a rollout's recorded `cwd` matches `abs` (the canonical worktree),
/// comparing on the canonicalized cwd when it still exists and falling back to a
/// raw path compare.
fn rollout_cwd_matches(path: &Path, abs: &Path) -> bool {
    let Some(cwd) = rollout_cwd(path) else {
        return false;
    };
    std::fs::canonicalize(&cwd).ok().as_deref() == Some(abs) || cwd == abs
}

/// All Codex rollout session files for a worktree, under `sessions_root`
/// (`<home>/.codex/sessions`). Codex does NOT encode the worktree in the path —
/// it stores the originating directory in each file's first `session_meta` line
/// — so discovery is by content: scan rollout files newest-first (capped at
/// [`CODEX_SCAN_CAP`]) and keep those whose recorded `cwd` matches
/// `abs_worktree`. Testable variant taking an explicit sessions root + canonical
/// worktree path.
pub(crate) fn codex_session_files_in(sessions_root: &Path, abs_worktree: &Path) -> Vec<PathBuf> {
    // `collect_rollouts` already returns the newest `CODEX_SCAN_CAP` rollouts,
    // newest-first, without walking older history; keep those whose cwd matches.
    let mut rollouts = Vec::new();
    collect_rollouts(sessions_root, CODEX_SCAN_CAP, &mut rollouts);
    rollouts.retain(|p| rollout_cwd_matches(p, abs_worktree));
    rollouts
}

/// Codex session files for a worktree. Resolves the real home dir and canonical
/// worktree path.
pub fn codex_session_files(worktree: &Path) -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let Ok(abs) = std::fs::canonicalize(worktree) else {
        return Vec::new();
    };
    codex_session_files_in(&home.join(".codex/sessions"), &abs)
}

/// Claude Code session files for a worktree. Resolves the real home dir and
/// canonical worktree path.
pub fn claude_session_files(worktree: &Path) -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let Ok(abs) = std::fs::canonicalize(worktree) else {
        return Vec::new();
    };
    session_files_in(&home, &abs)
}

/// pi session files for a worktree. Resolves the real home dir and canonical
/// worktree path.
pub fn pi_session_files(worktree: &Path) -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let Ok(abs) = std::fs::canonicalize(worktree) else {
        return Vec::new();
    };
    pi_session_files_in(&home, &abs)
}

/// Session files for a worktree across every supported harness (Claude + pi +
/// Codex), concatenated. `extract_change_events` auto-detects each line's
/// dialect, and `Timeline` merges events across all files newest-first, so a
/// worktree that was driven by more than one harness yields a single unified
/// chronology.
///
/// Resolves the home dir and canonical worktree once, then fans out to the
/// per-harness `_in` helpers — cheap to call repeatedly in a refresh loop.
pub fn session_files(worktree: &Path) -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let Ok(abs) = std::fs::canonicalize(worktree) else {
        return Vec::new();
    };
    let mut files = session_files_in(&home, &abs);
    files.extend(pi_session_files_in(&home, &abs));
    files.extend(codex_session_files_in(&home.join(".codex/sessions"), &abs));
    files
}

// ── inlined helpers from activity/events.rs ──────────────────────────────────

/// Encode an absolute path the way Claude Code does for `~/.claude/projects/`.
///
/// Claude Code maps every character that is not ASCII-alphanumeric to `-` (not
/// just `/` and `.`), so repo names containing spaces or other punctuation are
/// flattened. Mirror that exactly, otherwise the encoded directory won't match
/// the one Claude actually writes.
pub fn encode_cwd(path: &Path) -> String {
    path.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// Encode an absolute path the way pi does for `~/.pi/agent/sessions/`: strip
/// the leading `/`, replace remaining `/` with `-`, and wrap with `--`. Unlike
/// Claude, pi does NOT replace `.`, so `/home/eben/.local` becomes
/// `--home-eben-.local--`.
pub fn pi_encode_cwd(path: &Path) -> String {
    let inner = path
        .to_string_lossy()
        .trim_start_matches('/')
        .replace('/', "-");
    format!("--{inner}--")
}

/// Minimal ISO 8601 parser for the format Claude Code emits:
/// `YYYY-MM-DDTHH:MM:SS.fffZ` (always UTC, always millisecond precision).
pub fn parse_iso8601_ms(s: &str) -> Option<i64> {
    // Strip trailing Z; we treat the timestamp as UTC.
    let s = s.strip_suffix('Z').unwrap_or(s);
    // Split date and time at 'T'.
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let y: i32 = date_parts.next()?.parse().ok()?;
    let mo: u32 = date_parts.next()?.parse().ok()?;
    let d: u32 = date_parts.next()?.parse().ok()?;

    let (hms, frac) = match time.split_once('.') {
        Some((hms, frac)) => (hms, frac),
        None => (time, "0"),
    };
    let mut tp = hms.split(':');
    let h: u32 = tp.next()?.parse().ok()?;
    let mi: u32 = tp.next()?.parse().ok()?;
    let se: u32 = tp.next()?.parse().ok()?;
    // Treat fractional seconds as milliseconds (truncate/pad to 3 digits).
    let mut frac_ms_str = String::new();
    for c in frac.chars().take(3) {
        frac_ms_str.push(c);
    }
    while frac_ms_str.len() < 3 {
        frac_ms_str.push('0');
    }
    let ms: i64 = frac_ms_str.parse().ok()?;

    let days = days_from_civil(y, mo, d);
    let secs_of_day = h as i64 * 3600 + mi as i64 * 60 + se as i64;
    Some(days * 86_400_000 + secs_of_day * 1000 + ms)
}

/// Howard Hinnant's `days_from_civil` algorithm — days since 1970-01-01 for a
/// proleptic Gregorian calendar date. Avoids pulling in chrono just for this.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era as i64 * 146_097 + doe as i64 - 719_468
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod extract_tests {
    use super::*;

    fn line(json: &str) -> serde_json::Value {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn extracts_edit_event() {
        let v = line(
            r#"{"type":"assistant","timestamp":"2026-05-14T17:32:02.744Z","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/wt/a.rs","old_string":"let x=1;","new_string":"pub fn foo() {}"}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Edit);
        assert_eq!(evs[0].file_path, std::path::PathBuf::from("/wt/a.rs"));
        assert_eq!(evs[0].summary, "pub fn foo() {}");
        assert!(matches!(evs[0].detail, ChangeDetail::Edit { .. }));
        assert_eq!(
            evs[0].timestamp_ms,
            parse_iso8601_ms("2026-05-14T17:32:02.744Z").unwrap()
        );
    }

    #[test]
    fn extracts_write_event() {
        let v = line(
            r#"{"type":"assistant","timestamp":"2026-05-14T17:32:02.744Z","message":{"content":[{"type":"tool_use","id":"t2","name":"Write","input":{"file_path":"/wt/new.rs","content":"pub struct Z;"}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Write);
        assert_eq!(evs[0].summary, "pub struct Z;");
        assert!(
            matches!(&evs[0].detail, ChangeDetail::Write { head } if head.contains("struct Z"))
        );
    }

    #[test]
    fn multiedit_emits_one_event_per_edit() {
        let v = line(
            r#"{"type":"assistant","timestamp":"2026-05-14T17:32:02.744Z","message":{"content":[{"type":"tool_use","id":"t3","name":"MultiEdit","input":{"file_path":"/wt/a.rs","edits":[{"old_string":"a","new_string":"pub fn one(){}"},{"old_string":"b","new_string":"pub fn two(){}"}]}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].tool, ChangeTool::MultiEdit);
        assert_eq!(evs[1].summary, "pub fn two(){}");
    }

    #[test]
    fn ignores_read_and_bash() {
        let v = line(
            r#"{"type":"assistant","timestamp":"2026-05-14T17:32:02.744Z","message":{"content":[{"type":"tool_use","id":"t4","name":"Read","input":{"file_path":"/wt/a.rs"}},{"type":"tool_use","id":"t5","name":"Bash","input":{"command":"ls"}}]}}"#,
        );
        assert!(extract_change_events(&v, DETAIL_MAX_CHARS).is_empty());
    }

    #[test]
    fn ignores_non_assistant_lines() {
        let v = line(
            r#"{"type":"user","timestamp":"2026-05-14T17:32:02.744Z","message":{"role":"user","content":"hi"}}"#,
        );
        assert!(extract_change_events(&v, DETAIL_MAX_CHARS).is_empty());
    }
}

#[cfg(test)]
mod pi_tests {
    use super::*;

    fn line(json: &str) -> serde_json::Value {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn extracts_pi_edit_event() {
        let v = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_1","name":"edit","arguments":{"path":"/wt/a.rs","edits":[{"oldText":"let x=1;","newText":"pub fn foo() {}"}]}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::MultiEdit);
        assert_eq!(evs[0].file_path, std::path::PathBuf::from("/wt/a.rs"));
        assert_eq!(evs[0].summary, "pub fn foo() {}");
        assert!(
            matches!(&evs[0].detail, ChangeDetail::Edit { new, .. } if new == "pub fn foo() {}")
        );
        assert_eq!(
            evs[0].timestamp_ms,
            parse_iso8601_ms("2026-05-24T02:29:38.132Z").unwrap()
        );
    }

    #[test]
    fn pi_edit_emits_one_event_per_edit() {
        let v = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_1","name":"edit","arguments":{"path":"/wt/a.rs","edits":[{"oldText":"a","newText":"pub fn one(){}"},{"oldText":"b","newText":"pub fn two(){}"}]}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].summary, "pub fn one(){}");
        assert_eq!(evs[1].summary, "pub fn two(){}");
        assert_eq!(evs[1].source.index_in_line, 1);
    }

    #[test]
    fn extracts_pi_write_event() {
        let v = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_2","name":"write","arguments":{"path":"/wt/new.rs","content":"pub struct Z;"}}]}}"#,
        );
        let evs = extract_change_events(&v, DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Write);
        assert_eq!(evs[0].file_path, std::path::PathBuf::from("/wt/new.rs"));
        assert_eq!(evs[0].summary, "pub struct Z;");
        assert!(
            matches!(&evs[0].detail, ChangeDetail::Write { head } if head.contains("struct Z"))
        );
    }

    #[test]
    fn ignores_pi_read_and_bash() {
        let v = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"/wt/a.rs"}},{"type":"toolCall","id":"c2","name":"bash","arguments":{"command":"ls"}}]}}"#,
        );
        assert!(extract_change_events(&v, DETAIL_MAX_CHARS).is_empty());
    }

    #[test]
    fn ignores_pi_user_and_toolresult_lines() {
        let user = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
        );
        assert!(extract_change_events(&user, DETAIL_MAX_CHARS).is_empty());
        let result = line(
            r#"{"type":"message","timestamp":"2026-05-24T02:29:38.132Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"edit","content":[{"type":"text","text":"ok"}]}}"#,
        );
        assert!(extract_change_events(&result, DETAIL_MAX_CHARS).is_empty());
    }

    #[test]
    fn encode_cwd_replaces_all_non_alphanumerics() {
        // Claude Code maps EVERY non-alphanumeric char to '-', not just '/' and
        // '.'. A repo whose name contains a space (e.g. "meals backend") yields a
        // worktree path with a space, which must be encoded to '-' to match the
        // real `~/.claude/projects/` directory Claude writes.
        assert_eq!(
            encode_cwd(Path::new(
                "/home/eben/.local/state/wsx/worktrees/meals backend/miniature-lupin"
            )),
            "-home-eben--local-state-wsx-worktrees-meals-backend-miniature-lupin"
        );
        // Existing hyphens are preserved (hyphen maps to hyphen).
        assert_eq!(encode_cwd(Path::new("/a-b/c.d")), "-a-b-c-d");
    }

    #[test]
    fn pi_encode_cwd_strips_leading_slash_keeps_dots() {
        assert_eq!(pi_encode_cwd(Path::new("/home/eben")), "--home-eben--");
        // Unlike Claude, pi does NOT replace '.' — only '/'.
        assert_eq!(
            pi_encode_cwd(Path::new("/home/eben/.local/state")),
            "--home-eben-.local-state--"
        );
    }

    #[test]
    fn pi_session_files_lists_jsonl_in_encoded_dir() {
        let home = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let abs = std::fs::canonicalize(work.path()).unwrap();
        let dir = home
            .path()
            .join(".pi/agent/sessions")
            .join(pi_encode_cwd(&abs));
        std::fs::create_dir_all(&dir).unwrap();
        use std::io::Write as _;
        for name in ["a.jsonl", "b.jsonl", "notes.txt"] {
            let mut f = std::fs::File::create(dir.join(name)).unwrap();
            writeln!(f, "{{}}").unwrap();
        }
        let files = pi_session_files_in(home.path(), &abs);
        assert_eq!(files.len(), 2, "only .jsonl files counted");
    }
}

#[cfg(test)]
mod codex_tests {
    use super::*;
    use std::io::Write;

    /// Build a Codex `response_item` line whose payload is a `custom_tool_call`
    /// (freeform) `apply_patch` carrying `patch` verbatim in `input`.
    fn custom_apply_patch(patch: &str) -> serde_json::Value {
        serde_json::json!({
            "timestamp": "2026-06-06T12:56:06.504Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": patch,
                "call_id": "call_1",
            }
        })
    }

    /// Build a Codex `response_item` line whose payload is a JSON `function_call`
    /// `apply_patch` — `arguments` is a JSON-encoded *string* `{"input": patch}`.
    fn function_apply_patch(patch: &str) -> serde_json::Value {
        let args = serde_json::to_string(&serde_json::json!({ "input": patch })).unwrap();
        serde_json::json!({
            "timestamp": "2026-06-06T12:56:06.504Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "apply_patch",
                "arguments": args,
                "call_id": "call_1",
            }
        })
    }

    #[test]
    fn add_file_becomes_write_event() {
        let patch = "*** Begin Patch\n\
                     *** Add File: /wt/new.rs\n\
                     +pub struct Z;\n\
                     +impl Z {}\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Write);
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/new.rs"));
        assert_eq!(evs[0].summary, "pub struct Z;");
        assert!(matches!(
            &evs[0].detail,
            ChangeDetail::Write { head } if head == "pub struct Z;\nimpl Z {}"
        ));
        assert_eq!(
            evs[0].timestamp_ms,
            parse_iso8601_ms("2026-06-06T12:56:06.504Z").unwrap()
        );
    }

    #[test]
    fn update_file_single_hunk_becomes_edit_event() {
        let patch = "*** Begin Patch\n\
                     *** Update File: /wt/a.rs\n\
                     @@\n\
                     -let x = 1;\n\
                     +pub fn foo() {}\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::MultiEdit);
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/a.rs"));
        assert_eq!(evs[0].summary, "pub fn foo() {}");
        assert!(matches!(
            &evs[0].detail,
            ChangeDetail::Edit { old, new } if old == "let x = 1;" && new == "pub fn foo() {}"
        ));
    }

    #[test]
    fn update_context_lines_appear_on_both_sides() {
        // Built by join (not a `\`-continuation literal) so the leading space on
        // real context lines survives.
        let patch = [
            "*** Begin Patch",
            "*** Update File: /wt/a.rs",
            "@@",
            " ctx before",
            "-old line",
            "+new line",
            " ctx after",
            "*** End Patch",
        ]
        .join("\n");
        let evs = extract_change_events(&custom_apply_patch(&patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        // The ` ` prefix denotes context: stripped, then kept on both sides.
        match &evs[0].detail {
            ChangeDetail::Edit { old, new } => {
                assert_eq!(old, "ctx before\nold line\nctx after");
                assert_eq!(new, "ctx before\nnew line\nctx after");
            }
            _ => panic!("expected Edit"),
        }
    }

    #[test]
    fn update_file_emits_one_event_per_hunk() {
        let patch = "*** Begin Patch\n\
                     *** Update File: /wt/a.rs\n\
                     @@\n\
                     -a\n\
                     +pub fn one() {}\n\
                     @@\n\
                     -b\n\
                     +pub fn two() {}\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].summary, "pub fn one() {}");
        assert_eq!(evs[1].summary, "pub fn two() {}");
        assert_eq!(evs[0].source.index_in_line, 0);
        assert_eq!(evs[1].source.index_in_line, 1);
    }

    #[test]
    fn multiple_files_in_one_patch() {
        let patch = "*** Begin Patch\n\
                     *** Add File: /wt/new.rs\n\
                     +pub fn added() {}\n\
                     *** Update File: /wt/a.rs\n\
                     @@\n\
                     -x\n\
                     +pub fn updated() {}\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].tool, ChangeTool::Write);
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/new.rs"));
        assert_eq!(evs[1].tool, ChangeTool::MultiEdit);
        assert_eq!(evs[1].file_path, PathBuf::from("/wt/a.rs"));
        assert_eq!(evs[1].source.index_in_line, 1);
    }

    #[test]
    fn update_with_move_to_retargets_to_destination() {
        let patch = "*** Begin Patch\n\
                     *** Update File: /wt/a.rs\n\
                     *** Move to: /wt/b.rs\n\
                     @@\n\
                     -old\n\
                     +pub fn moved() {}\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1, "Move to is metadata, not a section break");
        assert_eq!(evs[0].summary, "pub fn moved() {}");
        // The edit attributes to the post-rename path (the one that exists after
        // the patch applies), not the original `Update File:` path.
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/b.rs"));
    }

    #[test]
    fn delete_file_is_skipped() {
        let patch = "*** Begin Patch\n\
                     *** Delete File: /wt/gone.rs\n\
                     *** End Patch";
        let evs = extract_change_events(&custom_apply_patch(patch), DETAIL_MAX_CHARS);
        assert!(evs.is_empty(), "deletes carry no change text to show");
    }

    #[test]
    fn function_call_json_arguments_variant() {
        let patch = "*** Begin Patch\n\
                     *** Add File: /wt/new.rs\n\
                     +pub struct Z;\n\
                     *** End Patch";
        let evs = extract_change_events(&function_apply_patch(patch), DETAIL_MAX_CHARS);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Write);
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/new.rs"));
    }

    #[test]
    fn ignores_exec_command_and_outputs_and_reasoning() {
        for v in [
            // exec_command shell call — out of scope (arbitrary shell).
            serde_json::json!({
                "timestamp": "2026-06-06T12:56:06.504Z",
                "type": "response_item",
                "payload": {"type": "function_call", "name": "exec_command",
                    "arguments": "{\"cmd\":\"sed -i s/a/b/ f\"}", "call_id": "c1"}
            }),
            // tool result.
            serde_json::json!({
                "timestamp": "2026-06-06T12:56:06.504Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "c1", "output": "ok"}
            }),
            // assistant narration / reasoning.
            serde_json::json!({
                "timestamp": "2026-06-06T12:56:06.504Z",
                "type": "response_item",
                "payload": {"type": "reasoning", "summary": []}
            }),
            // non-response_item lines.
            serde_json::json!({
                "timestamp": "2026-06-06T12:56:06.504Z",
                "type": "session_meta",
                "payload": {"cwd": "/wt"}
            }),
            serde_json::json!({
                "timestamp": "2026-06-06T12:56:06.504Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hi"}
            }),
        ] {
            assert!(
                extract_change_events(&v, DETAIL_MAX_CHARS).is_empty(),
                "must ignore: {v}"
            );
        }
    }

    #[test]
    fn detail_max_clips_codex_edits() {
        let big = "z".repeat(50);
        let patch =
            format!("*** Begin Patch\n*** Update File: /wt/a.rs\n@@\n-x\n+{big}\n*** End Patch");
        let clipped = extract_change_events(&custom_apply_patch(&patch), 4);
        match &clipped[0].detail {
            ChangeDetail::Edit { new, .. } => assert_eq!(new, "zzzz"),
            _ => panic!("expected Edit"),
        }
    }

    #[test]
    fn parse_file_then_load_full_change_round_trips_codex() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("rollout-x.jsonl");
        let big = "q".repeat(DETAIL_MAX_CHARS + 25);
        let patch = format!("*** Begin Patch\n*** Add File: /wt/a.rs\n+{big}\n*** End Patch");
        let line = custom_apply_patch(&patch).to_string();
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{line}").unwrap();

        let evs = parse_file(&path);
        assert_eq!(evs.len(), 1);
        match &evs[0].detail {
            ChangeDetail::Write { head } => {
                assert_eq!(head.chars().count(), DETAIL_MAX_CHARS, "in-memory clipped");
            }
            _ => panic!("expected Write"),
        }
        match load_full_change(&evs[0]).expect("full re-extract") {
            ChangeDetail::Write { head } => {
                assert_eq!(
                    head.chars().count(),
                    DETAIL_MAX_CHARS + 25,
                    "full content recovered"
                );
            }
            _ => panic!("expected Write"),
        }
    }

    fn write_rollout(dir: &Path, name: &str, cwd: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let meta = serde_json::json!({
            "timestamp": "2026-06-06T08:53:03.019Z",
            "type": "session_meta",
            "payload": {"id": "abc", "cwd": cwd, "originator": "codex-tui"}
        });
        std::fs::write(dir.join(name), format!("{meta}\n")).unwrap();
    }

    #[test]
    fn discovery_matches_embedded_cwd_newest_first() {
        let root = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let abs = std::fs::canonicalize(work.path()).unwrap();
        let cwd = abs.to_string_lossy();
        let day = root.path().join("2026/06/06");
        write_rollout(&day, "rollout-2026-06-06T08-00-00-aaaa.jsonl", &cwd);
        write_rollout(&day, "rollout-2026-06-06T09-00-00-bbbb.jsonl", &cwd);
        // A rollout for a different worktree must be excluded.
        write_rollout(
            &day,
            "rollout-2026-06-06T10-00-00-cccc.jsonl",
            "/somewhere/else",
        );

        let files = codex_session_files_in(root.path(), &abs);
        assert_eq!(files.len(), 2, "only matching-cwd rollouts");
        assert!(
            files[0].ends_with("rollout-2026-06-06T09-00-00-bbbb.jsonl"),
            "newest first: {files:?}"
        );
        assert!(files[1].ends_with("rollout-2026-06-06T08-00-00-aaaa.jsonl"));
    }

    #[test]
    fn discovery_only_counts_rollout_files() {
        let root = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let abs = std::fs::canonicalize(work.path()).unwrap();
        let cwd = abs.to_string_lossy();
        let day = root.path().join("2026/06/06");
        write_rollout(&day, "rollout-2026-06-06T08-00-00-aaaa.jsonl", &cwd);
        // Not a rollout file name → ignored even though its cwd matches.
        write_rollout(&day, "notes.jsonl", &cwd);
        let files = codex_session_files_in(root.path(), &abs);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn discovery_missing_root_returns_empty() {
        let abs = PathBuf::from("/nonexistent/worktree");
        assert!(codex_session_files_in(Path::new("/nonexistent/sessions"), &abs).is_empty());
    }

    #[test]
    fn collect_rollouts_caps_and_orders_newest_first_across_partitions() {
        let root = tempfile::TempDir::new().unwrap();
        // Spread rollouts across multiple day partitions; names sort
        // chronologically so the newest overall live under the latest dir.
        for (day, names) in [
            (
                "2026/06/05",
                ["rollout-2026-06-05T01.jsonl", "rollout-2026-06-05T02.jsonl"],
            ),
            (
                "2026/06/06",
                ["rollout-2026-06-06T01.jsonl", "rollout-2026-06-06T02.jsonl"],
            ),
        ] {
            let dir = root.path().join(day);
            std::fs::create_dir_all(&dir).unwrap();
            for name in names {
                std::fs::write(dir.join(name), "{}\n").unwrap();
            }
        }
        let mut out = Vec::new();
        collect_rollouts(root.path(), 3, &mut out);
        // Capped at 3, and the 3 collected are the newest 3 in descending order
        // (older 06-05T01 is never reached).
        assert_eq!(out.len(), 3);
        assert!(out[0].ends_with("rollout-2026-06-06T02.jsonl"));
        assert!(out[1].ends_with("rollout-2026-06-06T01.jsonl"));
        assert!(out[2].ends_with("rollout-2026-06-05T02.jsonl"));
    }
}

#[cfg(test)]
mod line_tests {
    use super::*;

    #[test]
    fn finds_line_of_new_string_first_line() {
        let file = "fn a() {}\nfn b2() {}\nfn c() {}\n";
        let detail = ChangeDetail::Edit {
            old: "fn b() {}".into(),
            new: "fn b2() {}".into(),
        };
        assert_eq!(resolve_line(file, &detail), 2);
    }

    #[test]
    fn write_resolves_to_line_one() {
        let detail = ChangeDetail::Write {
            head: "anything".into(),
        };
        assert_eq!(resolve_line("whatever\n", &detail), 1);
    }

    #[test]
    fn missing_new_string_falls_back_to_line_one() {
        let detail = ChangeDetail::Edit {
            old: "x".into(),
            new: "nonexistent".into(),
        };
        assert_eq!(resolve_line("fn a() {}\n", &detail), 1);
    }

    #[test]
    fn none_detail_falls_back_to_line_one() {
        assert_eq!(resolve_line("fn a() {}\n", &ChangeDetail::None), 1);
    }
}

#[cfg(test)]
mod locate_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn lists_all_jsonl_files_in_session_dir() {
        let home = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let abs = std::fs::canonicalize(work.path()).unwrap();
        let dir = home.path().join(".claude/projects").join(encode_cwd(&abs));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["a.jsonl", "b.jsonl", "notes.txt"] {
            let mut f = std::fs::File::create(dir.join(name)).unwrap();
            writeln!(f, "{{}}").unwrap();
        }
        let files = session_files_in(home.path(), &abs);
        assert_eq!(files.len(), 2, "only .jsonl files counted");
    }

    #[test]
    fn missing_dir_returns_empty() {
        let home = tempfile::TempDir::new().unwrap();
        let abs = std::path::PathBuf::from("/nonexistent/worktree");
        assert!(session_files_in(home.path(), &abs).is_empty());
    }
}

#[cfg(test)]
mod parse_file_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_events_from_a_jsonl_file_skipping_garbage() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"type":"assistant","timestamp":"2026-05-14T17:00:00.000Z","message":{{"content":[{{"type":"tool_use","name":"Write","input":{{"file_path":"/wt/x.rs","content":"pub fn x(){{}}"}}}}]}}}}"#).unwrap();
        writeln!(f, "not json at all").unwrap();
        writeln!(f, r#"{{"type":"user","message":{{"content":"hi"}}}}"#).unwrap();
        let evs = parse_file(&path);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tool, ChangeTool::Write);
    }
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    #[test]
    fn prefers_declaration_line() {
        let s = summarize_edit("let x = 1;\n", "let x = 1;\npub fn foo() {}\n");
        assert_eq!(s, "pub fn foo() {}");
    }

    #[test]
    fn falls_back_to_first_nonblank_changed_line() {
        let s = summarize_edit("a = 1\n", "a = 2\n");
        assert_eq!(s, "a = 2");
    }

    #[test]
    fn write_new_file_when_no_decl() {
        let s = summarize_write("plain text\nmore text\n");
        assert_eq!(s, "new file");
    }

    #[test]
    fn write_uses_first_declaration_when_present() {
        let s = summarize_write("# header\nclass Thing:\n    pass\n");
        assert_eq!(s, "class Thing:");
    }

    #[test]
    fn truncates_long_summaries() {
        let long = "x".repeat(200);
        let s = summarize_edit("", &format!("{long}\n"));
        assert!(s.chars().count() <= SUMMARY_MAX_CHARS);
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn extract_assigns_index_in_line_and_respects_detail_max() {
        let v: serde_json::Value = serde_json::from_str(r#"{"type":"assistant","timestamp":"2026-05-14T17:00:00.000Z","message":{"content":[{"type":"tool_use","name":"MultiEdit","input":{"file_path":"/wt/a.rs","edits":[{"old_string":"aaaa","new_string":"bbbb"},{"old_string":"cccc","new_string":"dddd"}]}}]}}"#).unwrap();
        let clipped = extract_change_events(&v, 2);
        assert_eq!(clipped.len(), 2);
        assert_eq!(clipped[0].source.index_in_line, 0);
        assert_eq!(clipped[1].source.index_in_line, 1);
        if let ChangeDetail::Edit { new, .. } = &clipped[0].detail {
            assert_eq!(new, "bb", "detail_max=2 clips new_string");
        } else {
            panic!("expected Edit");
        }
        let full = extract_change_events(&v, usize::MAX);
        if let ChangeDetail::Edit { new, .. } = &full[1].detail {
            assert_eq!(new, "dddd", "usize::MAX keeps full text");
        } else {
            panic!("expected Edit");
        }
    }

    #[test]
    fn load_full_change_round_trips_uncliped() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{}}").unwrap();
        writeln!(f, r#"{{"type":"assistant","timestamp":"2026-05-14T17:00:00.000Z","message":{{"content":[{{"type":"tool_use","name":"Edit","input":{{"file_path":"/wt/a.rs","old_string":"OLD","new_string":"A_VERY_LONG_NEW_STRING_BEYOND_ANY_CLIP"}}}}]}}}}"#).unwrap();
        let ev = ChangeEvent {
            timestamp_ms: 0,
            tool: ChangeTool::Edit,
            file_path: PathBuf::from("/wt/a.rs"),
            summary: String::new(),
            detail: ChangeDetail::Edit {
                old: "OLD".into(),
                new: "A_VERY".into(),
            },
            source: ChangeSource {
                session_file: path.clone(),
                line_index: 1,
                index_in_line: 0,
            },
        };
        let full = load_full_change(&ev).expect("re-extract");
        if let ChangeDetail::Edit { new, .. } = full {
            assert_eq!(new, "A_VERY_LONG_NEW_STRING_BEYOND_ANY_CLIP");
        } else {
            panic!("expected Edit");
        }
    }

    #[test]
    fn load_full_change_none_when_source_empty() {
        let ev = ChangeEvent {
            timestamp_ms: 0,
            tool: ChangeTool::Write,
            file_path: PathBuf::from("/wt/a.rs"),
            summary: String::new(),
            detail: ChangeDetail::Write { head: "x".into() },
            source: ChangeSource::default(),
        };
        assert!(load_full_change(&ev).is_none());
    }

    #[test]
    fn parse_file_then_load_full_change_returns_untruncated() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let big = "x".repeat(DETAIL_MAX_CHARS + 50);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","timestamp":"2026-05-14T17:00:00.000Z","message":{{"content":[{{"type":"tool_use","name":"Write","input":{{"file_path":"/wt/a.rs","content":"{big}"}}}}]}}}}"#
        )
        .unwrap();
        let evs = parse_file(&path);
        assert_eq!(evs.len(), 1);
        match &evs[0].detail {
            ChangeDetail::Write { head } => {
                assert_eq!(
                    head.chars().count(),
                    DETAIL_MAX_CHARS,
                    "in-memory detail is clipped"
                );
            }
            _ => panic!("expected Write"),
        }
        match load_full_change(&evs[0]).expect("full re-extract") {
            ChangeDetail::Write { head } => {
                assert_eq!(
                    head.chars().count(),
                    DETAIL_MAX_CHARS + 50,
                    "full content recovered"
                );
            }
            _ => panic!("expected Write"),
        }
    }
}

#[cfg(test)]
mod iso8601_tests {
    use super::*;

    #[test]
    fn iso8601_parser_roundtrips_known_value() {
        let ms = parse_iso8601_ms("2026-05-14T17:32:02.744Z").unwrap();
        let days = days_from_civil(2026, 5, 14);
        let expected = days * 86_400_000 + (17 * 3600 + 32 * 60 + 2) * 1000 + 744;
        assert_eq!(ms, expected);
    }
}
