//! Parsing a single Claude Code JSONL line into a [`ParsedLine`] — the display
//! event plus the tool-tracking and classification signals the tailer
//! aggregates.

use super::format::{MAX_DISPLAY_CHARS, collapse_ws, truncate_display};
use super::parse_iso8601_ms;
use super::{EventKind, EventSnapshot, StopReason};

/// Result of parsing a single JSONL line: at most one display event, plus
/// any tool-tracking signals derived from its content blocks.
#[derive(Debug, Default)]
pub struct ParsedLine {
    pub event: Option<EventSnapshot>,
    pub tool_use_starts: Vec<(String, String, i64)>,
    pub tool_use_resolves: Vec<String>,
    /// The stop_reason on an assistant line, if present. None for any other
    /// line type (user, tool_result, unknown).
    pub stop_reason: Option<StopReason>,
    /// True if this line is a plain-text user message (real human input).
    /// Tool_result lines wrapped as `user` do not set this.
    pub is_user_text: bool,
    /// The text of the last `text` content block in this assistant message.
    /// Forwarded to `WorkspaceEvents.last_assistant_text`; consumed by
    /// `WorkspaceEvents::last_text_ends_with_question`. None for any
    /// non-assistant line, or for assistant messages with no text blocks.
    pub last_assistant_text: Option<String>,
    /// The longest `text` content block (by char count) in this assistant
    /// message. Distinct from `last_assistant_text`: when a message has
    /// multiple text blocks (e.g. a long recap followed by a terse
    /// "Done."), this surfaces the substantive one. Forwarded to the
    /// recap pipeline via `TailUpdate.longest_assistant_text_in_batch`.
    pub longest_text_in_message: Option<String>,
    /// True if this user line is the Claude Code "[Request interrupted by
    /// user for tool use]" sentinel. Doesn't set `is_user_text`: the
    /// sentinel is system-generated, not a real human reply, so the
    /// `human_replied_after_last_stop` machinery should ignore it.
    pub user_interrupt_sentinel: bool,
    /// Plain user text content for the first real user message in this
    /// line (None for tool_result or non-user lines). Aggregated into
    /// `TailUpdate.first_user_text` upstream.
    pub first_user_text: Option<String>,
    /// File paths extracted from Read/Edit/MultiEdit/Write/NotebookEdit
    /// tool_use blocks on this line, in source order. Empty for any
    /// other tool / non-assistant line.
    pub edited_file_paths: Vec<String>,
}

/// Parse a single JSONL line into a [`ParsedLine`]. Malformed lines and
/// uninteresting top-level types yield an empty result.
///
/// User `tool_result` content blocks DO NOT produce an `EventSnapshot` (they
/// stay skipped from the display log) but DO populate `tool_use_resolves`.
pub fn parse_jsonl_line(line: &str) -> ParsedLine {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return ParsedLine::default();
    };
    let Some(kind) = v.get("type").and_then(|t| t.as_str()) else {
        return ParsedLine::default();
    };
    let timestamp_ms = parse_timestamp(v.get("timestamp"));
    match kind {
        "user" => parse_user(&v, timestamp_ms),
        "assistant" => parse_assistant(&v, timestamp_ms),
        _ => ParsedLine::default(),
    }
}

fn parse_user(v: &serde_json::Value, timestamp_ms: i64) -> ParsedLine {
    let mut out = ParsedLine::default();
    let Some(content) = v.get("message").and_then(|m| m.get("content")) else {
        return out;
    };
    // User content is either:
    //   (a) a plain string (the user's prompt) — emit a display event;
    //   (b) an array containing tool_result blocks — emit resolves but no
    //       display event (tool outputs aren't user prompts).
    if let Some(text) = content.as_str() {
        if text.trim().is_empty() {
            return out;
        }
        let display = truncate_display(&format!("user: {}", collapse_ws(text)), MAX_DISPLAY_CHARS);
        out.event = Some(EventSnapshot {
            kind: EventKind::UserMessage,
            display,
            timestamp_ms,
        });
        out.is_user_text = true;
        out.first_user_text = Some(text.to_string());
        return out;
    }
    if let Some(blocks) = content.as_array() {
        for block in blocks {
            let Some(bt) = block.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            match bt {
                "tool_result" => {
                    if let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) {
                        out.tool_use_resolves.push(id.to_string());
                    }
                }
                "text" => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str())
                        && t == INTERRUPT_SENTINEL
                    {
                        out.user_interrupt_sentinel = true;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Exact text Claude Code writes as a synthetic user message when the
/// human cancels an in-flight tool call. Used to distinguish "agent was
/// interrupted" from "agent is stalled."
const INTERRUPT_SENTINEL: &str = "[Request interrupted by user for tool use]";

/// Render a subagent dispatch (Claude Code's `Agent` tool) for the activity
/// display. The bare tool name "Agent" is uninformative — the useful signal is
/// the subagent type (e.g. "Explore") and how many ran in parallel. A single
/// type collapses to "using Explore agent"; N of the same type to "using 2
/// Explore agents"; mixed or unknown types degrade to a bare count or the old
/// "using Agent" fallback.
fn format_agent_dispatch(subtypes: &[&str]) -> String {
    let first = subtypes.first().copied().unwrap_or("");
    let all_same = !first.is_empty() && subtypes.iter().all(|s| *s == first);
    match (subtypes.len(), all_same) {
        (1, true) => format!("using {first} agent"),
        (n, true) => format!("using {n} {first} agents"),
        (n, false) if n > 1 => format!("using {n} agents"),
        // 0, or a single dispatch with no subagent_type: nothing better to say.
        _ => "using Agent".to_string(),
    }
}

fn parse_assistant(v: &serde_json::Value, timestamp_ms: i64) -> ParsedLine {
    let mut out = ParsedLine::default();
    // stop_reason lives at message.stop_reason. Some lines (e.g. partial
    // streaming snapshots) may omit it; in that case we leave the previous
    // sticky value in place upstream.
    if let Some(sr) = v
        .get("message")
        .and_then(|m| m.get("stop_reason"))
        .and_then(|s| s.as_str())
    {
        out.stop_reason = Some(StopReason::from_json_str(sr));
    }
    let Some(blocks) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
    else {
        return out;
    };
    // Prefer tool_use over text — tool use is the most concrete signal of
    // "what's happening right now". Fall back to assistant text.
    let mut last_text: Option<&str> = None;
    let mut longest_text: Option<&str> = None;
    let mut last_tool: Option<(&str, &serde_json::Value)> = None;
    // Subagent dispatches (the `Agent` tool) — collected across the whole
    // message so parallel dispatches can be counted and summarized by type.
    let mut agent_subtypes: Vec<&str> = Vec::new();
    for block in blocks {
        let Some(bt) = block.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        match bt {
            "text" => {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    last_text = Some(t);
                    let new_len = t.chars().count();
                    let beats = longest_text
                        .map(|cur| cur.chars().count() < new_len)
                        .unwrap_or(true);
                    if beats {
                        longest_text = Some(t);
                    }
                }
            }
            "tool_use" => {
                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let input = block.get("input").unwrap_or(&serde_json::Value::Null);
                last_tool = Some((name, input));
                if name == "Agent" {
                    agent_subtypes.push(
                        input
                            .get("subagent_type")
                            .and_then(|s| s.as_str())
                            .unwrap_or(""),
                    );
                }
                // Track every tool_use we see — multiple in one message is rare
                // but possible. The id is required for matching tool_results.
                if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                    out.tool_use_starts
                        .push((id.to_string(), name.to_string(), timestamp_ms));
                }
                // Only mutating tools count as a "recent edit" — Read
                // never modifies the worktree, so files the agent just
                // read shouldn't show up in the detail bar's
                // RECENT FILES list (they'd render without a diff count
                // and confuse the user).
                if matches!(name, "Edit" | "MultiEdit" | "Write" | "NotebookEdit")
                    && let Some(p) = input.get("file_path").and_then(|p| p.as_str())
                {
                    out.edited_file_paths.push(p.to_string());
                }
            }
            _ => {}
        }
    }
    // Capture the final text block for the classifier BEFORE returning down
    // the tool-use display path. The display preference (tool > text) is
    // unchanged; we just also remember the text for downstream classification.
    if let Some(t) = last_text {
        out.last_assistant_text = Some(t.to_string());
    }
    if let Some(t) = longest_text {
        out.longest_text_in_message = Some(t.to_string());
    }
    if let Some((name, input)) = last_tool {
        let body = if name == "Bash" {
            let cmd = input
                .get("command")
                .and_then(|c| c.as_str())
                .unwrap_or("(no command)");
            format!("ran `{}`", collapse_ws(cmd))
        } else if name == "Agent" {
            format_agent_dispatch(&agent_subtypes)
        } else if name.is_empty() {
            "using a tool".to_string()
        } else {
            format!("using {}", name)
        };
        out.event = Some(EventSnapshot {
            kind: EventKind::AssistantToolUse,
            display: truncate_display(&body, MAX_DISPLAY_CHARS),
            timestamp_ms,
        });
        return out;
    }
    if let Some(t) = last_text {
        let trimmed = t.trim();
        if trimmed.is_empty() {
            return out;
        }
        out.event = Some(EventSnapshot {
            kind: EventKind::AssistantText,
            display: truncate_display(&collapse_ws(trimmed), MAX_DISPLAY_CHARS),
            timestamp_ms,
        });
    }
    out
}

/// Parse an ISO 8601 timestamp (e.g. `2026-05-14T17:32:02.744Z`) to epoch
/// milliseconds. Returns the current time on failure.
fn parse_timestamp(v: Option<&serde_json::Value>) -> i64 {
    let now_ms = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };
    let Some(v) = v else { return now_ms() };
    // Could also be an epoch number — handle both.
    if let Some(n) = v.as_i64() {
        // Heuristic: > 10^12 means already ms; else seconds.
        return if n > 1_000_000_000_000 { n } else { n * 1000 };
    }
    let Some(s) = v.as_str() else { return now_ms() };
    parse_iso8601_ms(s).unwrap_or_else(now_ms)
}
