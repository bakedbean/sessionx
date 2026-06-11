//! Tail Claude Code session JSONL files for activity events.
//!
//! Claude Code writes one JSONL file per session at
//! `~/.claude/projects/<encoded-cwd>/<uuid>.jsonl`, where the cwd encoding
//! replaces `/` and `.` with `-` (see [`crate::extract::encode_cwd`]).
//!
//! ## JSONL schema (as of Claude Code v2.x)
//!
//! Each line is one JSON object. Lines we care about look roughly like:
//!
//! ```jsonc
//! // User text message:
//! {
//!   "type": "user",
//!   "message": { "role": "user", "content": "<text>" },
//!   "uuid": "...", "timestamp": "2026-05-14T17:32:02.744Z",
//!   "sessionId": "...", "cwd": "...", "gitBranch": "...", ...
//! }
//!
//! // Assistant text message (content is an array of content blocks):
//! {
//!   "type": "assistant",
//!   "message": {
//!     "role": "assistant",
//!     "content": [
//!       { "type": "thinking", "thinking": "...", "signature": "..." },
//!       { "type": "text", "text": "<text>" }
//!     ], ...
//!   },
//!   "uuid": "...", "timestamp": "2026-05-14T...", ...
//! }
//!
//! // Assistant tool use (also in content array):
//! {
//!   "type": "assistant",
//!   "message": {
//!     "content": [
//!       { "type": "tool_use", "id": "...", "name": "Bash",
//!         "input": { "command": "git status", "description": "..." } }
//!     ], ...
//!   }, ...
//! }
//!
//! // Tool result (back as "user" with structured content array — skipped):
//! { "type": "user", "message": { "role": "user", "content": [
//!     { "tool_use_id": "...", "type": "tool_result", "content": "...", "is_error": false }
//!   ] }, ... }
//! ```
//!
//! Other top-level `type` values seen: `attachment`, `last-prompt`,
//! `permission-mode`, `ai-title`, `file-history-snapshot`. We skip those.
//!
//! Timestamps are ISO 8601 with millisecond precision and a trailing `Z`.
//! We parse them ourselves to avoid pulling in chrono.

use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

mod discovery;
mod format;
mod parse;

// Re-export the submodules' public surface so the long-standing
// `activity::events::*` paths (used by lib.rs, the Codex/Pi/Hermes tailers, and
// wsx) keep resolving unchanged after the split.
pub use discovery::{encode_cwd, locate_session_file};
pub use format::{MAX_DISPLAY_CHARS, clean_recap, collapse_ws, truncate_display};
pub use parse::{ParsedLine, parse_jsonl_line};

// The ISO-8601 parser is shared with the change-extraction path; keep a single
// implementation in `extract` and re-export it here so the long-standing
// `activity::events::parse_iso8601_ms` path (used by the Codex/Pi tailers and
// `parse`'s `parse_timestamp`) keeps resolving.
pub use crate::extract::parse_iso8601_ms;

const MAX_LOG: usize = 50;

/// Why the assistant's most recent message stopped. Mirrors the Anthropic
/// API's `stop_reason` field. `EndTurn`, `MaxTokens`, and `StopSequence` all
/// mean "the agent is no longer running and is awaiting user input";
/// `ToolUse` means it stopped to call a tool and will resume after the
/// `tool_result` is delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

impl StopReason {
    /// True iff the agent has stopped and is waiting on the human (as opposed
    /// to waiting on its own tool-call result).
    pub fn is_awaiting_user(&self) -> bool {
        matches!(
            self,
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::StopSequence
        )
    }

    pub fn from_json_str(s: &str) -> Self {
        match s {
            "end_turn" => StopReason::EndTurn,
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            "stop_sequence" => StopReason::StopSequence,
            other => StopReason::Other(other.to_string()),
        }
    }
}

/// Running tallies of tool_use blocks by category. Populated by the
/// tail loop as JSONL lines parse. Used by the dashboard detail bar
/// to synthesize a one-line action trace like "read 14 files, edited
/// 3 files, ran 2 commands".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolUseCounts {
    pub read: u32,
    pub edit: u32,
    pub write: u32,
    pub bash: u32,
    pub other: u32,
}

impl ToolUseCounts {
    /// Increment the appropriate field based on the Claude Code tool name.
    /// Edit/MultiEdit count as `edit`; Write/NotebookEdit count as `write`;
    /// Bash counts as `bash`; Read counts as `read`; everything else
    /// (Task, Glob, Grep, WebFetch, …) counts as `other`.
    pub fn increment(&mut self, tool_name: &str) {
        match tool_name {
            "Read" => self.read += 1,
            "Edit" | "MultiEdit" => self.edit += 1,
            "Write" | "NotebookEdit" => self.write += 1,
            "Bash" => self.bash += 1,
            _ => self.other += 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceEvents {
    pub latest: Option<EventSnapshot>,
    /// Recent events, oldest first; bounded to MAX_LOG.
    pub log: VecDeque<EventSnapshot>,
    pub file_path: Option<PathBuf>,
    pub byte_offset: u64,
    /// Tool_use ids the assistant has emitted but for which we haven't yet
    /// seen a matching tool_result. Used to detect permission prompts —
    /// a tool_use pending for ≥3s is almost certainly waiting on user
    /// approval. Map: id → (tool name, first-seen epoch ms).
    pub pending_tool_uses: HashMap<String, (String, i64)>,
    /// The most recently observed assistant `stop_reason`. None until the
    /// first assistant message arrives, or after a session file reset.
    pub last_stop_reason: Option<StopReason>,
    /// Set when a real user text message arrives after the latest
    /// awaiting-user stop_reason. Used to decide whether the agent is still
    /// idle (waiting on the human) or has resumed (received new input but
    /// hasn't produced its next assistant message yet).
    pub user_replied_since_stop: bool,
    /// Epoch-ms of the last time the JSONL log was observed to have grown.
    /// Updated by the tail loop whenever a new event is appended. Used by
    /// `is_stalled` to detect sessions where claude has gone quiet
    /// mid-tool-chain without writing a terminal stop_reason.
    pub last_log_activity_ms: i64,
    /// The text of the most recent assistant text content block, if any.
    /// Used by the question-vs-complete classifier to decide whether a
    /// stopped turn ended on a trailing `?`. Cleared on session reset.
    pub last_assistant_text: Option<String>,
    /// True when the most recent user-side signal in the session was an
    /// interrupt sentinel — Claude Code writes
    /// `[Request interrupted by user for tool use]` as a user text block
    /// when the human cancels the agent mid-tool-call. The agent never
    /// emits a follow-up `end_turn` for these, so without this flag the
    /// session drifts into `Stalled` after 60s. Cleared by any subsequent
    /// real assistant message or real user text.
    pub last_user_interrupted: bool,
    /// First plain-text user content block observed since the most
    /// recent session reset. Set once per session; preserved across
    /// log rotation past MAX_LOG. Used by the detail bar's SESSION
    /// SUMMARY column.
    pub first_user_text: Option<String>,
    /// Running tallies of tool_use blocks by category. Populated by
    /// the tail loop. Used by the detail bar to synthesize a
    /// one-line action trace.
    pub tool_use_counts: ToolUseCounts,
    /// Most-recent-first ring of edited file paths, bounded to 7.
    /// A path already in the ring is moved to the front rather than
    /// duplicated, so repeated edits to the same file don't appear
    /// multiple times in the dashboard's RECENT FILES list.
    pub recent_edited_files: VecDeque<String>,
    /// Per-turn accumulator: the longest assistant text block seen
    /// since the last is-awaiting-user stop_reason. Updated by the
    /// tail loop via `record_batch_longest_text`. Snapshotted into
    /// `last_completed_turn_text` (after `clean_recap` filtering) at
    /// the end of each turn. Cleared on session reset.
    pub longest_text_this_turn: Option<String>,
    /// Cleaned recap of the most-recently-completed agent turn. Pinned
    /// (i.e. does not mutate mid-turn or wipe on filter rejection) so
    /// the SESSION SUMMARY column has stable text to render between
    /// turns. Cleared on session reset.
    pub last_completed_turn_text: Option<String>,
}

impl Default for WorkspaceEvents {
    fn default() -> Self {
        Self {
            latest: None,
            log: VecDeque::with_capacity(MAX_LOG),
            file_path: None,
            byte_offset: 0,
            pending_tool_uses: HashMap::new(),
            last_stop_reason: None,
            user_replied_since_stop: false,
            last_log_activity_ms: 0,
            last_assistant_text: None,
            last_user_interrupted: false,
            first_user_text: None,
            tool_use_counts: ToolUseCounts::default(),
            recent_edited_files: VecDeque::with_capacity(7),
            longest_text_this_turn: None,
            last_completed_turn_text: None,
        }
    }
}

impl WorkspaceEvents {
    /// Push a path onto `recent_edited_files`, moving any existing
    /// entry for that path to the front instead of duplicating it.
    /// Bounds the ring to 7 entries.
    pub fn push_recent_edited_file(&mut self, path: String) {
        self.recent_edited_files.retain(|p| p != &path);
        self.recent_edited_files.push_front(path);
        while self.recent_edited_files.len() > 7 {
            self.recent_edited_files.pop_back();
        }
    }

    /// Clear all session-derived state. Used when the underlying jsonl file
    /// is replaced or truncated — stale tool_uses and stop_reasons from the
    /// prior session must not leak into the new one.
    pub fn reset_session_state(&mut self) {
        self.pending_tool_uses.clear();
        self.last_stop_reason = None;
        self.user_replied_since_stop = false;
        self.last_log_activity_ms = 0;
        self.last_assistant_text = None;
        self.last_user_interrupted = false;
        self.first_user_text = None;
        self.tool_use_counts = ToolUseCounts::default();
        self.recent_edited_files.clear();
        self.longest_text_this_turn = None;
        self.last_completed_turn_text = None;
    }

    /// Merge a batch's longest assistant text into the per-turn
    /// accumulator. Keeps whichever candidate is longer (by character
    /// count). Called once per tail batch by the background poller.
    pub fn record_batch_longest_text(&mut self, batch_longest: &str) {
        let new_len = batch_longest.chars().count();
        let cur_len = self
            .longest_text_this_turn
            .as_ref()
            .map(|s| s.chars().count())
            .unwrap_or(0);
        if new_len > cur_len {
            self.longest_text_this_turn = Some(batch_longest.to_string());
        }
    }

    /// At the end of a turn (an "awaiting user" stop_reason), run
    /// `clean_recap` over the accumulated longest text and pin the
    /// result into `last_completed_turn_text`. Always clears the
    /// accumulator. If the candidate fails the filter the prior
    /// recap survives — the user keeps seeing the most recent valid
    /// recap rather than blanking on a noisy turn.
    pub fn snapshot_recap_at_turn_end(&mut self) {
        if let Some(text) = self.longest_text_this_turn.take()
            && let Some(cleaned) = clean_recap(&text)
        {
            self.last_completed_turn_text = Some(cleaned);
        }
    }

    /// The agent is stopped and the human hasn't replied yet.
    pub fn is_awaiting_user(&self) -> bool {
        !self.user_replied_since_stop
            && self
                .last_stop_reason
                .as_ref()
                .is_some_and(StopReason::is_awaiting_user)
    }

    /// If any pending `tool_use` is `AskUserQuestion` or `ExitPlanMode`,
    /// return the tool name. These tools mean "the agent has explicitly
    /// asked the human for input" — distinct from a generic permission
    /// prompt. Returns the first match (order across HashMap iteration is
    /// unspecified, but in practice at most one such tool is pending).
    pub fn pending_question_tool(&self) -> Option<&str> {
        for (name, _) in self.pending_tool_uses.values() {
            if name == "AskUserQuestion" || name == "ExitPlanMode" {
                return Some(name.as_str());
            }
        }
        None
    }

    /// Return the oldest pending `tool_use` that looks like a permission
    /// prompt: pending for ≥ `stale_ms` and not one of the tools we know
    /// are not prompts. Excludes `AskUserQuestion` / `ExitPlanMode`
    /// (surfaced separately as `AwaitingAnswer`) and `Agent` (subagent
    /// dispatches that legitimately run for minutes and would otherwise
    /// pin the workspace to the false-positive `?` glyph).
    pub fn pending_permission_tool(&self, now_ms: i64, stale_ms: i64) -> Option<(String, i64)> {
        let mut oldest: Option<(&str, i64)> = None;
        for (name, ts) in self.pending_tool_uses.values() {
            if matches!(name.as_str(), "AskUserQuestion" | "ExitPlanMode" | "Agent") {
                continue;
            }
            if now_ms.saturating_sub(*ts) >= stale_ms {
                match oldest {
                    None => oldest = Some((name.as_str(), *ts)),
                    Some((_, t)) if *ts < t => oldest = Some((name.as_str(), *ts)),
                    _ => {}
                }
            }
        }
        oldest.map(|(n, ts)| (n.to_string(), ts))
    }

    /// True iff the most recent assistant text block ends with `?` (after
    /// stripping trailing whitespace and markdown noise — `*`, `_`, `` ` ``).
    /// Fallback signal used by the question-vs-complete classifier when
    /// neither `AskUserQuestion` nor `ExitPlanMode` was invoked.
    pub fn last_text_ends_with_question(&self) -> bool {
        let Some(text) = self.last_assistant_text.as_deref() else {
            return false;
        };
        let trimmed =
            text.trim_end_matches(|c: char| c.is_whitespace() || matches!(c, '*' | '_' | '`'));
        trimmed.ends_with('?')
    }

    /// True iff claude appears to have stalled mid-tool-chain: the JSONL
    /// log was last appended >`stall_threshold_ms` ago, there's no
    /// pending tool_use (so it's not just a slow tool), and we've seen
    /// at least one stop_reason (so we know claude has been active in
    /// this session — fresh sessions with no events yet don't flag).
    pub fn is_stalled(&self, now_ms: i64, stall_threshold_ms: i64) -> bool {
        self.last_stop_reason.is_some()
            && self.pending_tool_uses.is_empty()
            && self.last_log_activity_ms > 0
            && now_ms.saturating_sub(self.last_log_activity_ms) > stall_threshold_ms
    }
}

/// Output of a single `tail_session` call.
///
/// Carries both display-bound events and tool-tracking signals that the caller
/// uses to maintain a per-workspace pending-tool map.
#[derive(Debug, Clone, Default)]
pub struct TailUpdate {
    pub new_offset: u64,
    pub events: Vec<EventSnapshot>,
    /// (tool_use_id, tool_name, first-seen epoch ms) for each tool_use block
    /// observed in this batch.
    pub tool_use_starts: Vec<(String, String, i64)>,
    /// tool_use_ids resolved by a `tool_result` block in this batch.
    pub tool_use_resolves: Vec<String>,
    /// The stop_reason on the last assistant message in this batch, if any.
    /// Later batches with a fresh assistant message override this; batches
    /// containing only user/tool_result lines leave it None.
    pub last_stop_reason: Option<StopReason>,
    /// True iff at least one plain-text user message appears in this batch
    /// AFTER the latest assistant `stop_reason` in this batch (or anywhere in
    /// the batch if there is no new stop_reason). The caller uses this to
    /// decide whether to flip `user_replied_since_stop` on. Within-batch
    /// ordering matters: `end_turn` then user-text means "user replied";
    /// user-text then `end_turn` means "agent stopped again, no reply yet".
    pub human_replied_after_last_stop: bool,
    /// True if `tail_session` had to rewind to offset 0 because the file
    /// shrank since the previous call (truncation or replacement). The caller
    /// should treat all prior session-derived state as stale.
    pub reset_from_zero: bool,
    /// The most recent assistant text block observed in this batch, if
    /// any. The caller stores this on WorkspaceEvents for the classifier.
    /// None means "no new text in this batch" — keep the prior value.
    pub last_assistant_text: Option<String>,
    /// The longest assistant text block observed in this batch, by
    /// character count. The caller merges this into a per-turn
    /// accumulator and snapshots it at end-of-turn for the SESSION
    /// SUMMARY recap line. Tracking the *longest* (not the latest)
    /// block is a heuristic that filters out short pre-tool narration
    /// in favor of substantive end-of-turn recaps.
    pub longest_assistant_text_in_batch: Option<String>,
    /// Final interrupt-sentinel state at the end of the batch.
    /// `Some(true)`  — batch ended with an interrupt sentinel,
    /// `Some(false)` — batch saw something that overrides the sentinel
    ///                 (a real assistant message or real user text),
    /// `None`        — batch was silent on this signal; caller keeps prior.
    pub last_user_interrupted: Option<bool>,
    /// First user-text content block observed in this batch (in line
    /// order). The caller assigns this to `WorkspaceEvents.first_user_text`
    /// only when the destination is currently `None` — once the first
    /// prompt is captured, subsequent user messages don't overwrite it.
    pub first_user_text: Option<String>,
    /// Tool-use category increments observed in this batch. The caller
    /// adds these into `WorkspaceEvents.tool_use_counts` (saturating).
    pub tool_use_counts: ToolUseCounts,
    /// File paths the agent touched in this batch, in source order
    /// (most-recent last). The caller push-fronts each entry into
    /// `WorkspaceEvents.recent_edited_files`, deduping consecutive
    /// same-path entries and bounding to 7.
    pub edited_file_paths: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EventSnapshot {
    pub kind: EventKind,
    /// Pre-formatted line ready to render. Already truncated.
    pub display: String,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    UserMessage,
    AssistantText,
    AssistantToolUse,
    Other,
}

/// Read new lines from `path` starting at `offset` and parse them.
/// Returns the new committed offset (only fully terminated lines count) plus
/// the parsed events and tool-tracking signals.
pub fn tail_session(path: &Path, offset: u64) -> Result<TailUpdate> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    // Handle truncation/replacement: if the file is now smaller than our
    // offset, reset to 0 — likely a new session in the same path (rare).
    let reset_from_zero = offset > file_size;
    let start = if reset_from_zero { 0 } else { offset };
    file.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file);
    let mut update = TailUpdate {
        reset_from_zero,
        ..TailUpdate::default()
    };
    let mut buf = String::new();
    let mut consumed = start;
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        // Only fully-terminated lines (ending in '\n') are committed. A
        // partial trailing line may still be in flight; the next poll picks
        // it up after it completes.
        if !buf.ends_with('\n') {
            break;
        }
        consumed += n as u64;
        let parsed = parse_jsonl_line(buf.trim_end());
        if let Some(snap) = parsed.event {
            update.events.push(snap);
        }
        // Borrow parsed.tool_use_starts to increment counts BEFORE the
        // extend (which moves it).
        for (_id, name, _ts) in &parsed.tool_use_starts {
            update.tool_use_counts.increment(name);
        }
        update.tool_use_starts.extend(parsed.tool_use_starts);
        update.tool_use_resolves.extend(parsed.tool_use_resolves);
        update.edited_file_paths.extend(parsed.edited_file_paths);
        if update.first_user_text.is_none()
            && let Some(t) = parsed.first_user_text
        {
            update.first_user_text = Some(t);
        }
        // Order-aware: a fresh stop_reason restarts the "has the user
        // replied since this stop?" count. A user_text after it sets it.
        if let Some(sr) = parsed.stop_reason {
            update.last_stop_reason = Some(sr);
            update.human_replied_after_last_stop = false;
            // A new assistant message means the agent is past any prior
            // interrupt.
            update.last_user_interrupted = Some(false);
        }
        if parsed.is_user_text {
            update.human_replied_after_last_stop = true;
            // Real user text overrides any prior interrupt sentinel.
            update.last_user_interrupted = Some(false);
        }
        if parsed.user_interrupt_sentinel {
            update.last_user_interrupted = Some(true);
        }
        // Track the longest text block seen in this batch from each
        // message's longest-in-message (NOT its last text). This avoids
        // missing a substantive recap when the agent emitted a terse
        // closing block ("Done.") last in the same message.
        if let Some(longest) = parsed.longest_text_in_message {
            let len = longest.chars().count();
            let beats = update
                .longest_assistant_text_in_batch
                .as_ref()
                .map(|cur| cur.chars().count() < len)
                .unwrap_or(true);
            if beats {
                update.longest_assistant_text_in_batch = Some(longest);
            }
        }
        if let Some(text) = parsed.last_assistant_text {
            update.last_assistant_text = Some(text);
        }
    }
    update.new_offset = consumed;
    Ok(update)
}

/// Append `event` into a [`WorkspaceEvents`] log, evicting the oldest entry
/// once the cap is hit. Updates `latest` to the appended event.
pub fn push_event(store: &mut WorkspaceEvents, event: EventSnapshot) {
    if store.log.len() >= MAX_LOG {
        store.log.pop_front();
    }
    store.latest = Some(event.clone());
    store.log.push_back(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::EnvGuard;

    #[test]
    fn encode_cwd_maps_every_non_alphanumeric_to_dash() {
        // Worktree paths for repos with spaces (e.g. "meals backend") must
        // encode the space to '-' to match the real ~/.claude/projects dir.
        assert_eq!(
            encode_cwd(Path::new(
                "/home/eben/.local/state/wsx/worktrees/meals backend/miniature-lupin"
            )),
            "-home-eben--local-state-wsx-worktrees-meals-backend-miniature-lupin"
        );
    }

    #[test]
    fn parses_user_text_message() {
        let line = r#"{"type":"user","message":{"role":"user","content":"how do I add a new migration?"},"uuid":"u1","timestamp":"2026-05-14T17:32:02.744Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::UserMessage);
        assert!(
            ev.display.starts_with("user: how do I add"),
            "{}",
            ev.display
        );
        // 2026-05-14T17:32:02.744Z is a real, finite epoch — sanity check.
        assert!(ev.timestamp_ms > 1_700_000_000_000);
    }

    #[test]
    fn parses_assistant_text_message() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I'll rename the branch."}]},"timestamp":"2026-05-14T17:32:13.536Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantText);
        assert!(ev.display.contains("I'll rename"), "{}", ev.display);
    }

    #[test]
    fn parses_assistant_bash_tool_use() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cargo test --workspace","description":"run all tests"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert!(
            ev.display.contains("ran `cargo test --workspace`"),
            "{}",
            ev.display
        );
    }

    #[test]
    fn parses_assistant_other_tool_use() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/x"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(ev.display, "using Read");
    }

    #[test]
    fn agent_dispatch_single_shows_subagent_type() {
        // A subagent dispatch is the `Agent` tool with the real "what" living
        // in input.subagent_type. The bare tool name "Agent" is uninformative,
        // so surface the subagent type instead.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Agent","input":{"subagent_type":"Explore","description":"find dashboard logic"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(ev.display, "using Explore agent");
    }

    #[test]
    fn agent_dispatch_parallel_same_type_shows_count() {
        // Parallel dispatch = multiple Agent tool_use blocks in one message.
        // All the same type → "using 2 Explore agents".
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Agent","input":{"subagent_type":"Explore","description":"a"}},{"type":"tool_use","id":"t2","name":"Agent","input":{"subagent_type":"Explore","description":"b"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(ev.display, "using 2 Explore agents");
    }

    #[test]
    fn agent_dispatch_parallel_mixed_types_shows_count_only() {
        // Mixed subagent types can't be summarized by one type name, so fall
        // back to a bare count.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Agent","input":{"subagent_type":"Explore","description":"a"}},{"type":"tool_use","id":"t2","name":"Agent","input":{"subagent_type":"general-purpose","description":"b"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(ev.display, "using 2 agents");
    }

    #[test]
    fn agent_dispatch_without_subagent_type_falls_back() {
        // If subagent_type is missing we can't do better than the old behavior.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Agent","input":{"description":"a"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(ev.display, "using Agent");
    }

    #[test]
    fn tool_use_wins_over_text_in_same_message() {
        // When an assistant message has both a thinking block, a text block,
        // and a tool_use block, we surface the tool_use (most concrete).
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"running the tests"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cargo test"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert!(ev.display.contains("cargo test"));
    }

    #[test]
    fn skips_tool_result_user_messages() {
        // A "user" line whose content is an array (tool results, not a real
        // user prompt) should be skipped from the display log entirely. It
        // STILL emits a resolve so the caller can clear the pending entry.
        let line = r#"{"type":"user","message":{"role":"user","content":[{"tool_use_id":"t1","type":"tool_result","content":"ok","is_error":false}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.event.is_none());
        assert_eq!(parsed.tool_use_resolves, vec!["t1".to_string()]);
    }

    #[test]
    fn skips_unknown_line_types() {
        let line = r#"{"type":"attachment","content":"x","timestamp":"2026-05-14T17:32:14.000Z"}"#;
        assert!(parse_jsonl_line(line).event.is_none());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_jsonl_line("{ not json").event.is_none());
        assert!(parse_jsonl_line("").event.is_none());
    }

    #[test]
    fn truncates_long_messages() {
        let long = "x".repeat(600);
        let line = format!(
            r#"{{"type":"user","message":{{"role":"user","content":"{long}"}},"timestamp":"2026-05-14T17:32:02.744Z"}}"#
        );
        let ev = parse_jsonl_line(&line).event.expect("should parse");
        assert!(ev.display.chars().count() <= MAX_DISPLAY_CHARS);
        assert!(ev.display.ends_with('\u{2026}'));
    }

    #[test]
    fn collapses_whitespace_in_display() {
        let line = r#"{"type":"user","message":{"role":"user","content":"hello\n\n  world\t!"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        let ev = parse_jsonl_line(line).event.expect("should parse");
        assert_eq!(ev.display, "user: hello world !");
    }

    #[test]
    fn parser_emits_tool_use_start_on_assistant_tool_use() {
        let line = r#"{"type":"assistant","timestamp":"2026-05-14T20:00:00.000Z","message":{"content":[{"type":"tool_use","id":"toolu_abc","name":"Bash","input":{"command":"ls"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        // Existing behavior: an AssistantToolUse display event.
        let ev = parsed.event.as_ref().expect("display event");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        // New: tracking emission for the tool_use block.
        assert_eq!(parsed.tool_use_starts.len(), 1);
        assert_eq!(parsed.tool_use_starts[0].0, "toolu_abc");
        assert_eq!(parsed.tool_use_starts[0].1, "Bash");
        assert!(parsed.tool_use_resolves.is_empty());
    }

    #[test]
    fn parser_emits_tool_use_resolve_on_user_tool_result() {
        let line = r#"{"type":"user","timestamp":"2026-05-14T20:00:05.000Z","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_abc","content":"ok"}]}}"#;
        let parsed = parse_jsonl_line(line);
        // User tool_result rows stay skipped from the display log.
        assert!(parsed.event.is_none());
        assert_eq!(parsed.tool_use_resolves, vec!["toolu_abc".to_string()]);
        assert!(parsed.tool_use_starts.is_empty());
    }

    #[test]
    fn parser_handles_assistant_text_and_tool_use_in_same_message() {
        // For mixed messages we still surface the tool_use as the display
        // event AND emit a tool_use_start for it.
        let line = r#"{"type":"assistant","timestamp":"2026-05-14T20:00:00.000Z","message":{"content":[{"type":"text","text":"I'll run this"},{"type":"tool_use","id":"toolu_xyz","name":"Bash","input":{"command":"ls"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        let ev = parsed.event.as_ref().expect("display event");
        assert_eq!(ev.kind, EventKind::AssistantToolUse);
        assert_eq!(parsed.tool_use_starts.len(), 1);
        assert_eq!(parsed.tool_use_starts[0].0, "toolu_xyz");
        assert_eq!(parsed.tool_use_starts[0].1, "Bash");
    }

    #[test]
    fn tail_session_emits_pairs_across_lines() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let line_a = r#"{"type":"assistant","timestamp":"2026-05-14T20:00:00.000Z","message":{"content":[{"type":"tool_use","id":"a1","name":"Bash","input":{"command":"x"}}]}}"#;
        let line_b = r#"{"type":"user","timestamp":"2026-05-14T20:00:01.000Z","message":{"content":[{"type":"tool_result","tool_use_id":"a1","content":"ok"}]}}"#;
        std::fs::write(&path, format!("{line_a}\n{line_b}\n")).unwrap();
        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.tool_use_starts.len(), 1);
        assert_eq!(update.tool_use_starts[0].0, "a1");
        assert_eq!(update.tool_use_starts[0].1, "Bash");
        assert_eq!(update.tool_use_resolves, vec!["a1".to_string()]);
    }

    #[test]
    fn tail_session_reads_all_then_nothing_then_appended() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let line1 = r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        let line2 = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]},"timestamp":"2026-05-14T17:32:03.000Z"}"#;
        std::fs::write(&path, format!("{line1}\n{line2}\n")).unwrap();

        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.events.len(), 2);
        assert_eq!(update.events[0].kind, EventKind::UserMessage);
        assert_eq!(update.events[1].kind, EventKind::AssistantText);

        // Re-tailing from the same offset returns nothing.
        let update2 = tail_session(&path, update.new_offset).unwrap();
        assert!(update2.events.is_empty());
        assert_eq!(update2.new_offset, update.new_offset);

        // Append a new complete line and verify only it comes back.
        let line3 = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-05-14T17:32:04.000Z"}"#;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        writeln!(f, "{line3}").unwrap();
        let update3 = tail_session(&path, update2.new_offset).unwrap();
        assert_eq!(update3.events.len(), 1);
        assert_eq!(update3.events[0].kind, EventKind::AssistantToolUse);
    }

    #[test]
    fn tail_session_ignores_unterminated_trailing_line() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let line1 = r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        // Note: no trailing newline on the second line.
        let partial = r#"{"type":"user","message":{"role":"user","content":"oops"}"#;
        std::fs::write(&path, format!("{line1}\n{partial}")).unwrap();

        let update = tail_session(&path, 0).unwrap();
        // Only the first, terminated line should be committed.
        assert_eq!(update.events.len(), 1);
        // Offset advanced only past the completed line.
        assert_eq!(update.new_offset as usize, line1.len() + 1);

        // Now complete the second line and verify it's picked up.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        writeln!(f, r#","timestamp":"2026-05-14T17:32:03.000Z"}}"#).unwrap();
        let update2 = tail_session(&path, update.new_offset).unwrap();
        assert_eq!(update2.events.len(), 1);
    }

    #[test]
    fn tail_session_resets_when_offset_exceeds_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let line = r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        std::fs::write(&path, format!("{line}\n")).unwrap();
        // Offset way past EOF — should reset to 0 and re-read.
        let update = tail_session(&path, 9_999_999).unwrap();
        assert_eq!(update.events.len(), 1);
        assert!(update.reset_from_zero);
    }

    #[test]
    fn parses_assistant_stop_reason_end_turn() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"done"}]},"timestamp":"2026-05-14T17:32:13.536Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.stop_reason, Some(StopReason::EndTurn));
        assert!(parsed.stop_reason.unwrap().is_awaiting_user());
    }

    #[test]
    fn parses_assistant_stop_reason_tool_use() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"tool_use","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.stop_reason, Some(StopReason::ToolUse));
        assert!(!parsed.stop_reason.unwrap().is_awaiting_user());
    }

    #[test]
    fn parses_assistant_stop_reason_max_tokens_and_stop_sequence() {
        for (sr, expected) in [
            ("max_tokens", StopReason::MaxTokens),
            ("stop_sequence", StopReason::StopSequence),
        ] {
            let line = format!(
                r#"{{"type":"assistant","message":{{"role":"assistant","stop_reason":"{sr}","content":[{{"type":"text","text":"x"}}]}},"timestamp":"2026-05-14T17:32:13.536Z"}}"#
            );
            let parsed = parse_jsonl_line(&line);
            assert_eq!(parsed.stop_reason, Some(expected.clone()));
            assert!(expected.is_awaiting_user());
        }
    }

    #[test]
    fn assistant_without_stop_reason_yields_none() {
        // Some streaming-snapshot lines may omit stop_reason.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"thinking"}]},"timestamp":"2026-05-14T17:32:13.536Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.stop_reason, None);
    }

    #[test]
    fn assistant_unknown_stop_reason_is_other() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"refusal","content":[{"type":"text","text":"x"}]},"timestamp":"2026-05-14T17:32:13.536Z"}"#;
        let parsed = parse_jsonl_line(line);
        match parsed.stop_reason {
            Some(StopReason::Other(s)) => assert_eq!(s, "refusal"),
            other => panic!("expected Other(\"refusal\"), got {other:?}"),
        }
    }

    #[test]
    fn user_text_message_sets_is_user_text() {
        let line = r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.is_user_text);
    }

    #[test]
    fn user_tool_result_does_not_set_is_user_text() {
        let line = r#"{"type":"user","message":{"role":"user","content":[{"tool_use_id":"t1","type":"tool_result","content":"ok"}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(!parsed.is_user_text);
    }

    #[test]
    fn tail_session_aggregates_last_stop_reason_and_no_user_text_between() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        // tool_use then end_turn, with only tool_result in between — the
        // last assistant stop_reason wins, and no real user text appears.
        let l1 = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"tool_use","content":[{"type":"tool_use","id":"t","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-05-14T17:32:13.536Z"}"#;
        let l2 = r#"{"type":"user","message":{"role":"user","content":[{"tool_use_id":"t","type":"tool_result","content":"ok"}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let l3 = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"done"}]},"timestamp":"2026-05-14T17:32:15.000Z"}"#;
        std::fs::write(&path, format!("{l1}\n{l2}\n{l3}\n")).unwrap();
        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.last_stop_reason, Some(StopReason::EndTurn));
        assert!(!update.human_replied_after_last_stop);
        assert!(!update.reset_from_zero);
    }

    #[test]
    fn tail_session_flags_user_text_after_stop() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let l1 = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"done"}]},"timestamp":"2026-05-14T17:32:15.000Z"}"#;
        let l2 = r#"{"type":"user","message":{"role":"user","content":"more please"},"timestamp":"2026-05-14T17:32:20.000Z"}"#;
        std::fs::write(&path, format!("{l1}\n{l2}\n")).unwrap();
        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.last_stop_reason, Some(StopReason::EndTurn));
        assert!(update.human_replied_after_last_stop);
    }

    #[test]
    fn tail_session_user_text_before_a_later_stop_does_not_count() {
        // user_text comes first, then assistant ends turn — the agent is
        // awaiting input AGAIN, the prior user_text does not count.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let l1 = r#"{"type":"user","message":{"role":"user","content":"go"},"timestamp":"2026-05-14T17:32:00.000Z"}"#;
        let l2 = r#"{"type":"assistant","message":{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"done"}]},"timestamp":"2026-05-14T17:32:15.000Z"}"#;
        std::fs::write(&path, format!("{l1}\n{l2}\n")).unwrap();
        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.last_stop_reason, Some(StopReason::EndTurn));
        assert!(!update.human_replied_after_last_stop);
    }

    #[test]
    fn tail_session_user_text_with_no_stop_in_batch_still_flags() {
        // No stop_reason in this batch, only a user_text. The caller will
        // keep its prior last_stop_reason; user_replied_since_stop should
        // be flipped on.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("s.jsonl");
        let l1 = r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-05-14T17:32:00.000Z"}"#;
        std::fs::write(&path, format!("{l1}\n")).unwrap();
        let update = tail_session(&path, 0).unwrap();
        assert_eq!(update.last_stop_reason, None);
        assert!(update.human_replied_after_last_stop);
    }

    #[test]
    fn workspace_events_is_awaiting_user() {
        let mut ws = WorkspaceEvents::default();
        assert!(!ws.is_awaiting_user()); // no stop_reason yet
        ws.last_stop_reason = Some(StopReason::EndTurn);
        assert!(ws.is_awaiting_user());
        ws.user_replied_since_stop = true;
        assert!(!ws.is_awaiting_user()); // human spoke after end_turn
        ws.user_replied_since_stop = false;
        ws.last_stop_reason = Some(StopReason::ToolUse);
        assert!(!ws.is_awaiting_user()); // tool_use stops aren't on the user
    }

    #[test]
    fn workspace_events_reset_session_state_clears_everything() {
        let mut ws = WorkspaceEvents {
            last_stop_reason: Some(StopReason::EndTurn),
            user_replied_since_stop: true,
            last_log_activity_ms: 12_345,
            ..Default::default()
        };
        ws.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 1000));
        ws.reset_session_state();
        assert!(ws.pending_tool_uses.is_empty());
        assert_eq!(ws.last_stop_reason, None);
        assert!(!ws.user_replied_since_stop);
        assert_eq!(ws.last_log_activity_ms, 0);
    }

    #[test]
    fn workspace_events_is_stalled_requires_prior_stop_reason() {
        // Fresh sessions with no stop_reason yet must not flag — we'd
        // misclassify normal startup quiet as a stall.
        let ws = WorkspaceEvents {
            last_log_activity_ms: 1_000,
            ..Default::default()
        };
        assert!(!ws.is_stalled(100_000, 60_000));
    }

    #[test]
    fn workspace_events_is_stalled_false_when_tool_use_pending() {
        // Pending tool_use means claude is mid-call — not a stall, just
        // a slow tool.
        let mut ws = WorkspaceEvents {
            last_stop_reason: Some(StopReason::ToolUse),
            last_log_activity_ms: 1_000,
            ..Default::default()
        };
        ws.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 500));
        assert!(!ws.is_stalled(100_000, 60_000));
    }

    #[test]
    fn workspace_events_is_stalled_false_within_threshold() {
        let ws = WorkspaceEvents {
            last_stop_reason: Some(StopReason::ToolUse),
            last_log_activity_ms: 50_000,
            ..Default::default()
        };
        // delta = 60_000 - 50_000 = 10s, well under the 60s threshold.
        assert!(!ws.is_stalled(60_000, 60_000));
    }

    #[test]
    fn workspace_events_is_stalled_true_when_all_conditions_met() {
        let ws = WorkspaceEvents {
            last_stop_reason: Some(StopReason::ToolUse),
            last_log_activity_ms: 1_000,
            ..Default::default()
        };
        // delta = 100_000 - 1_000 = 99s, above the 60s threshold.
        assert!(ws.is_stalled(100_000, 60_000));
    }

    #[test]
    fn workspace_events_is_stalled_false_when_log_activity_never_set() {
        // last_log_activity_ms = 0 means we've never observed the log
        // grow — guard against false positives before the tailer runs.
        let ws = WorkspaceEvents {
            last_stop_reason: Some(StopReason::ToolUse),
            ..Default::default()
        };
        assert!(!ws.is_stalled(100_000, 60_000));
    }

    #[test]
    fn locate_session_file_finds_newest() {
        let home = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let abs = std::fs::canonicalize(work.path()).unwrap();
        let encoded = encode_cwd(&abs);
        let session_dir = home.path().join(".claude/projects").join(&encoded);
        std::fs::create_dir_all(&session_dir).unwrap();
        let older = session_dir.join("older.jsonl");
        let newer = session_dir.join("newer.jsonl");
        std::fs::write(&older, "{}").unwrap();
        // Sleep a hair to guarantee a different mtime even on coarse fs.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&newer, "{}").unwrap();

        let mut env = EnvGuard::new();
        env.set("HOME", home.path());
        let result = locate_session_file(work.path());
        assert_eq!(result, Some(newer));
    }

    #[test]
    fn locate_session_file_returns_none_when_dir_missing() {
        let home = tempfile::TempDir::new().unwrap();
        let work = tempfile::TempDir::new().unwrap();
        let mut env = EnvGuard::new();
        env.set("HOME", home.path());
        let result = locate_session_file(work.path());
        assert!(result.is_none());
    }

    #[test]
    fn pending_question_tool_matches_ask_user_question() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("AskUserQuestion".into(), 1));
        assert_eq!(evt.pending_question_tool(), Some("AskUserQuestion"));
    }

    #[test]
    fn pending_question_tool_matches_exit_plan_mode() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("ExitPlanMode".into(), 1));
        assert_eq!(evt.pending_question_tool(), Some("ExitPlanMode"));
    }

    #[test]
    fn pending_question_tool_ignores_other_tools() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 1));
        evt.pending_tool_uses
            .insert("t2".into(), ("Read".into(), 2));
        assert_eq!(evt.pending_question_tool(), None);
    }

    #[test]
    fn pending_permission_tool_returns_stale_bash() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 0));
        let hit = evt.pending_permission_tool(5_000, 3_000);
        assert_eq!(hit, Some(("Bash".into(), 0)));
    }

    #[test]
    fn pending_permission_tool_ignores_fresh_tools() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 4_000));
        // Only 1s old — below the 3s stale threshold.
        assert_eq!(evt.pending_permission_tool(5_000, 3_000), None);
    }

    #[test]
    fn pending_permission_tool_excludes_question_tools() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("AskUserQuestion".into(), 0));
        evt.pending_tool_uses
            .insert("t2".into(), ("ExitPlanMode".into(), 0));
        assert_eq!(evt.pending_permission_tool(10_000, 3_000), None);
    }

    #[test]
    fn pending_permission_tool_excludes_agent_dispatch() {
        // Agent subagent dispatches routinely run for minutes by design;
        // they should never be misread as a permission prompt.
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("Agent".into(), 0));
        assert_eq!(evt.pending_permission_tool(60_000, 3_000), None);
    }

    #[test]
    fn pending_permission_tool_picks_oldest_among_eligible() {
        let mut evt = WorkspaceEvents::default();
        evt.pending_tool_uses
            .insert("t1".into(), ("Bash".into(), 100));
        evt.pending_tool_uses
            .insert("t2".into(), ("Read".into(), 50));
        evt.pending_tool_uses
            .insert("t3".into(), ("Agent".into(), 0));
        // Agent (oldest overall) is excluded; Read at ts=50 wins.
        let hit = evt.pending_permission_tool(10_000, 3_000);
        assert_eq!(hit, Some(("Read".into(), 50)));
    }

    #[test]
    fn last_text_ends_with_question_true_for_simple_question() {
        let evt = WorkspaceEvents {
            last_assistant_text: Some("Want me to also handle X?".into()),
            ..Default::default()
        };
        assert!(evt.last_text_ends_with_question());
    }

    #[test]
    fn last_text_ends_with_question_strips_trailing_markdown() {
        // Claude often writes `Want me to refactor `foo`?*` where the literal
        // final char is `*` — we still want this classified as a question.
        let evt = WorkspaceEvents {
            last_assistant_text: Some("Want me to refactor `foo`?*".into()),
            ..Default::default()
        };
        assert!(evt.last_text_ends_with_question());
    }

    #[test]
    fn last_text_ends_with_question_strips_trailing_whitespace() {
        let evt = WorkspaceEvents {
            last_assistant_text: Some("Should I proceed?\n   ".into()),
            ..Default::default()
        };
        assert!(evt.last_text_ends_with_question());
    }

    #[test]
    fn last_text_ends_with_question_false_for_period_ending() {
        let evt = WorkspaceEvents {
            last_assistant_text: Some("Done. Let me know if you'd like changes.".into()),
            ..Default::default()
        };
        assert!(!evt.last_text_ends_with_question());
    }

    #[test]
    fn last_text_ends_with_question_false_when_question_in_middle() {
        // A `?` in the middle followed by a declarative final sentence should
        // not trip the heuristic. Only the trailing char (after markdown trim)
        // matters.
        let evt = WorkspaceEvents {
            last_assistant_text: Some("I considered: does this work? Yes, it works.".into()),
            ..Default::default()
        };
        assert!(!evt.last_text_ends_with_question());
    }

    #[test]
    fn parse_assistant_captures_last_text_for_classifier() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Want me to also run tests?"}],"stop_reason":"end_turn"},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(
            parsed.last_assistant_text.as_deref(),
            Some("Want me to also run tests?")
        );
    }

    #[test]
    fn parse_assistant_skips_capturing_text_when_only_tool_use() {
        // When the assistant message has only tool_use blocks, there is no
        // trailing text to feed the classifier. `last_assistant_text` stays None.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.last_assistant_text, None);
    }

    #[test]
    fn last_text_ends_with_question_false_for_empty_or_missing() {
        let evt = WorkspaceEvents::default();
        assert!(!evt.last_text_ends_with_question());
        let mut evt = evt;
        evt.last_assistant_text = Some(String::new());
        assert!(!evt.last_text_ends_with_question());
        evt.last_assistant_text = Some("   \n  ".into());
        assert!(!evt.last_text_ends_with_question());
    }

    #[test]
    fn parse_user_detects_interrupt_sentinel() {
        // Claude Code writes this exact text as a synthetic user message
        // when the human cancels mid-tool-call.
        let line = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.478Z","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user for tool use]"}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.user_interrupt_sentinel);
        // The sentinel is system-generated, not a real reply — must not
        // be counted as the user replying.
        assert!(!parsed.is_user_text);
        // And it shouldn't emit a display event either.
        assert!(parsed.event.is_none());
    }

    #[test]
    fn parse_user_does_not_flag_other_text_blocks() {
        // A real user text block looking similar but not exact must NOT
        // be treated as the sentinel.
        let line = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.478Z","message":{"role":"user","content":[{"type":"text","text":"Request interrupted by user (typed manually)"}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert!(!parsed.user_interrupt_sentinel);
    }

    #[test]
    fn tail_session_flags_interrupt_as_last_signal() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let assistant = r#"{"type":"assistant","timestamp":"2026-05-18T13:51:01.236Z","message":{"stop_reason":"tool_use","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]}}"#;
        let tool_result = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.470Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#;
        let interrupt = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.478Z","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user for tool use]"}]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{assistant}").unwrap();
        writeln!(f, "{tool_result}").unwrap();
        writeln!(f, "{interrupt}").unwrap();
        let u = tail_session(&path, 0).unwrap();
        assert_eq!(u.last_user_interrupted, Some(true));
        // And the interrupt sentinel must not count as a human reply.
        assert!(!u.human_replied_after_last_stop);
    }

    #[test]
    fn tail_session_clears_interrupt_when_assistant_replies_later() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let interrupt = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.478Z","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user for tool use]"}]}}"#;
        let assistant_after = r#"{"type":"assistant","timestamp":"2026-05-18T13:55:00.000Z","message":{"stop_reason":"end_turn","content":[{"type":"text","text":"resumed"}]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{interrupt}").unwrap();
        writeln!(f, "{assistant_after}").unwrap();
        let u = tail_session(&path, 0).unwrap();
        assert_eq!(u.last_user_interrupted, Some(false));
    }

    #[test]
    fn tail_session_clears_interrupt_when_real_user_text_follows() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let interrupt = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.478Z","message":{"role":"user","content":[{"type":"text","text":"[Request interrupted by user for tool use]"}]}}"#;
        let real_user = r#"{"type":"user","timestamp":"2026-05-18T13:55:00.000Z","message":{"role":"user","content":"actually try again"}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{interrupt}").unwrap();
        writeln!(f, "{real_user}").unwrap();
        let u = tail_session(&path, 0).unwrap();
        assert_eq!(u.last_user_interrupted, Some(false));
    }

    #[test]
    fn tail_session_silent_batch_keeps_interrupt_signal_none() {
        // A batch with no stop_reason, no user text, no interrupt
        // sentinel must leave the field as None so the caller doesn't
        // overwrite the sticky workspace state.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        // Just a tool_result, no other signals.
        let tool_result = r#"{"type":"user","timestamp":"2026-05-18T13:52:22.470Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{tool_result}").unwrap();
        let u = tail_session(&path, 0).unwrap();
        assert_eq!(u.last_user_interrupted, None);
    }

    #[test]
    fn push_event_bounds_log() {
        let mut ws = WorkspaceEvents::default();
        for i in 0..(MAX_LOG + 10) {
            push_event(
                &mut ws,
                EventSnapshot {
                    kind: EventKind::Other,
                    display: format!("e{i}"),
                    timestamp_ms: i as i64,
                },
            );
        }
        assert_eq!(ws.log.len(), MAX_LOG);
        assert_eq!(
            ws.latest.as_ref().unwrap().display,
            format!("e{}", MAX_LOG + 9)
        );
        // Oldest entry should have been evicted.
        assert_eq!(ws.log.front().unwrap().display, format!("e{}", 10));
    }

    #[test]
    fn workspace_events_new_fields_default_to_empty() {
        let evt = WorkspaceEvents::default();
        assert!(evt.first_user_text.is_none());
        assert_eq!(evt.tool_use_counts.read, 0);
        assert_eq!(evt.tool_use_counts.edit, 0);
        assert_eq!(evt.tool_use_counts.write, 0);
        assert_eq!(evt.tool_use_counts.bash, 0);
        assert_eq!(evt.tool_use_counts.other, 0);
        assert!(evt.recent_edited_files.is_empty());
    }

    #[test]
    fn push_recent_edited_file_moves_repeats_to_front_no_duplicates() {
        // A re-edit of an already-tracked file moves it to the front
        // rather than appearing twice in the ring.
        let mut evt = WorkspaceEvents::default();
        evt.push_recent_edited_file("a.rs".into());
        evt.push_recent_edited_file("b.rs".into());
        evt.push_recent_edited_file("a.rs".into());
        let entries: Vec<&str> = evt.recent_edited_files.iter().map(String::as_str).collect();
        assert_eq!(entries, vec!["a.rs", "b.rs"], "no duplicate a.rs");
    }

    #[test]
    fn push_recent_edited_file_bounds_to_seven() {
        let mut evt = WorkspaceEvents::default();
        for i in 0..10 {
            evt.push_recent_edited_file(format!("f{i}.rs"));
        }
        assert_eq!(evt.recent_edited_files.len(), 7);
        // Newest at front, oldest dropped.
        assert_eq!(
            evt.recent_edited_files.front().map(String::as_str),
            Some("f9.rs")
        );
        assert!(
            !evt.recent_edited_files.iter().any(|p| p == "f0.rs"),
            "oldest evicted"
        );
    }

    #[test]
    fn reset_session_state_clears_new_fields() {
        let mut evt = WorkspaceEvents {
            first_user_text: Some("hello".to_string()),
            tool_use_counts: ToolUseCounts {
                read: 3,
                bash: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        evt.recent_edited_files
            .push_front("src/main.rs".to_string());

        evt.reset_session_state();

        assert!(evt.first_user_text.is_none());
        assert_eq!(evt.tool_use_counts.read, 0);
        assert_eq!(evt.tool_use_counts.bash, 0);
        assert!(evt.recent_edited_files.is_empty());
    }

    #[test]
    fn parse_user_surfaces_first_user_text() {
        let line = r#"{"type":"user","message":{"role":"user","content":"summarize this repo"},"timestamp":"2026-05-14T17:32:02.744Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(
            parsed.first_user_text.as_deref(),
            Some("summarize this repo")
        );
    }

    #[test]
    fn parse_user_omits_first_user_text_for_tool_results() {
        // A "user" line whose content is a tool_result array is not a real
        // user prompt — first_user_text must stay None.
        let line = r#"{"type":"user","message":{"role":"user","content":[{"tool_use_id":"t1","type":"tool_result","content":"ok","is_error":false}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.first_user_text.is_none());
    }

    #[test]
    fn parse_assistant_surfaces_edited_file_paths() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/tmp/x/src/main.rs","old_string":"a","new_string":"b"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(
            parsed.edited_file_paths,
            vec!["/tmp/x/src/main.rs".to_string()]
        );
    }

    #[test]
    fn parse_assistant_skips_read_paths() {
        // Read never modifies the worktree — its file_path should not
        // be captured as a recent edit (otherwise the dashboard detail
        // bar lists files with no diff count next to them).
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/tmp/x/Cargo.toml"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.edited_file_paths.is_empty());
    }

    #[test]
    fn parse_assistant_skips_paths_for_non_file_tools() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}]},"timestamp":"2026-05-14T17:32:14.000Z"}"#;
        let parsed = parse_jsonl_line(line);
        assert!(parsed.edited_file_paths.is_empty());
    }

    #[test]
    fn tail_session_aggregates_first_user_text_and_counts() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"type":"user","message":{{"role":"user","content":"do the thing"}},"timestamp":"2026-05-14T17:32:02.744Z"}}"#).unwrap();
        writeln!(f, r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","id":"t1","name":"Read","input":{{"file_path":"/a.rs"}}}}]}},"timestamp":"2026-05-14T17:32:03.744Z"}}"#).unwrap();
        writeln!(f, r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","id":"t2","name":"Edit","input":{{"file_path":"/b.rs","old_string":"x","new_string":"y"}}}}]}},"timestamp":"2026-05-14T17:32:04.744Z"}}"#).unwrap();
        writeln!(f, r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","id":"t3","name":"Bash","input":{{"command":"ls"}}}}]}},"timestamp":"2026-05-14T17:32:05.744Z"}}"#).unwrap();
        drop(f);

        let upd = tail_session(&path, 0).unwrap();
        assert_eq!(upd.first_user_text.as_deref(), Some("do the thing"));
        assert_eq!(upd.tool_use_counts.read, 1);
        assert_eq!(upd.tool_use_counts.edit, 1);
        assert_eq!(upd.tool_use_counts.bash, 1);
        // Only Edit contributes to edited_file_paths — Read does not.
        assert_eq!(upd.edited_file_paths, vec!["/b.rs".to_string()]);
    }

    // clean_recap: cleans a raw assistant text block into a recap candidate
    // suitable for the SESSION SUMMARY column. See module-level helper.

    #[test]
    fn clean_recap_keeps_simple_prose() {
        let s = "Here's the sketch. Three small pieces — no plumbing changes.";
        assert_eq!(clean_recap(s).as_deref(), Some(s));
    }

    #[test]
    fn clean_recap_rejects_let_me_narration() {
        assert!(clean_recap("Let me check the events module.").is_none());
        assert!(clean_recap("let me grep for that next.").is_none());
    }

    #[test]
    fn clean_recap_rejects_ill_narration() {
        assert!(clean_recap("I'll grep for that next.").is_none());
        assert!(clean_recap("I'm going to read the parser first.").is_none());
        assert!(clean_recap("Let's check the test fixtures.").is_none());
    }

    #[test]
    fn clean_recap_rejects_empty_or_whitespace() {
        assert!(clean_recap("").is_none());
        assert!(clean_recap("   \n\n  ").is_none());
    }

    #[test]
    fn clean_recap_rejects_pure_decoration() {
        assert!(clean_recap("`─────────────`").is_none());
        assert!(clean_recap("```rust").is_none());
        assert!(clean_recap("───\n\n```").is_none());
    }

    #[test]
    fn clean_recap_strips_leading_blank_and_rule_lines() {
        let s = "\n\n`───`\nHere is the recap.";
        assert_eq!(clean_recap(s).as_deref(), Some("Here is the recap."));
    }

    #[test]
    fn clean_recap_skips_entire_insight_block() {
        // Opening `★ Insight ─` banner, bullets inside, closing `─` rule,
        // then the real recap prose. Filter should land on the prose.
        let s =
            "`★ Insight ─────`\n- meta point 1\n- meta point 2\n`─────`\n\nHere's what I did: X.";
        assert_eq!(clean_recap(s).as_deref(), Some("Here's what I did: X."));
    }

    #[test]
    fn clean_recap_insight_block_without_closing_banner_falls_back() {
        // Unbalanced banner: skip just the opener, accept whatever's next.
        let s = "`★ Insight ─────`\nThis is the body.";
        let got = clean_recap(s);
        assert!(got.is_some(), "should not reject unbalanced banner");
        assert!(
            got.as_deref().unwrap().contains("This is the body."),
            "got: {got:?}"
        );
    }

    #[test]
    fn clean_recap_skips_leading_code_fence_to_post_code_prose() {
        let s = "```rust\nlet x = 1;\nlet y = 2;\n```\n\nThis adds two constants.";
        assert_eq!(clean_recap(s).as_deref(), Some("This adds two constants."));
    }

    #[test]
    fn clean_recap_unbalanced_code_fence_falls_back() {
        // Missing closing fence: skip just the opener.
        let s = "```\nstill content here";
        let got = clean_recap(s);
        assert!(got.is_some(), "should not reject unbalanced fence");
    }

    #[test]
    fn clean_recap_rejects_pure_short_trailing_question() {
        assert!(clean_recap("Want me to update the tests?").is_none());
        assert!(clean_recap("What should I do next?").is_none());
    }

    #[test]
    fn clean_recap_keeps_statement_then_question() {
        let s = "I made the changes. Want me to update the tests?";
        assert_eq!(clean_recap(s).as_deref(), Some(s));
    }

    #[test]
    fn clean_recap_rejects_short_non_ascii_trailing_question() {
        // 21 chars, ~61 bytes — short by character count but long by
        // byte count. The 50-char cap must compare characters, not
        // bytes, so this short non-ASCII closing question gets
        // rejected like its ASCII counterparts.
        let s = "漢字漢字漢字漢字漢字漢字漢字漢字漢字漢字?";
        assert!(
            clean_recap(s).is_none(),
            "expected short non-ASCII question to be rejected; got {:?}",
            clean_recap(s)
        );
    }

    #[test]
    fn clean_recap_keeps_long_trailing_question() {
        // Above the 50-char short-question cap, accept even though it
        // ends with `?` — long questions carry substantive content.
        let s = "Want me to do a thorough audit of every callsite that touches that field too?";
        assert!(clean_recap(s).is_some());
    }

    #[test]
    fn clean_recap_preserves_paragraph_structure() {
        // Body with multi-line paragraphs survives intact (wrap happens
        // downstream in the renderer, not here).
        let s = "Done.\n\nNext: address the lint warning in foo.rs.";
        assert_eq!(clean_recap(s).as_deref(), Some(s));
    }

    #[test]
    fn clean_recap_real_hyacinth_turn_4() {
        // Regression: real recap from the dusty-hyacinth session that
        // motivated this feature. Should pass through clean.
        let s = "Here's the sketch. Three small pieces — no plumbing changes, no new dependencies.";
        assert_eq!(clean_recap(s).as_deref(), Some(s));
    }

    #[test]
    fn record_batch_longest_text_replaces_when_longer() {
        let mut evt = WorkspaceEvents::default();
        evt.record_batch_longest_text("short");
        evt.record_batch_longest_text("a much longer recap candidate");
        assert_eq!(
            evt.longest_text_this_turn.as_deref(),
            Some("a much longer recap candidate")
        );
    }

    #[test]
    fn record_batch_longest_text_keeps_when_shorter() {
        let mut evt = WorkspaceEvents::default();
        evt.record_batch_longest_text("a much longer recap candidate");
        evt.record_batch_longest_text("short");
        assert_eq!(
            evt.longest_text_this_turn.as_deref(),
            Some("a much longer recap candidate")
        );
    }

    #[test]
    fn record_batch_longest_text_initializes_from_empty() {
        let mut evt = WorkspaceEvents::default();
        assert!(evt.longest_text_this_turn.is_none());
        evt.record_batch_longest_text("first text");
        assert_eq!(evt.longest_text_this_turn.as_deref(), Some("first text"));
    }

    #[test]
    fn snapshot_recap_at_turn_end_writes_cleaned_text_and_clears_accumulator() {
        let mut evt = WorkspaceEvents::default();
        evt.record_batch_longest_text("Here's the recap. Done.");
        evt.snapshot_recap_at_turn_end();
        assert_eq!(
            evt.last_completed_turn_text.as_deref(),
            Some("Here's the recap. Done.")
        );
        assert!(evt.longest_text_this_turn.is_none());
    }

    #[test]
    fn snapshot_recap_at_turn_end_strips_insight_banner() {
        let mut evt = WorkspaceEvents::default();
        evt.record_batch_longest_text("`★ Insight ─────`\n- meta\n`─────`\n\nHere is the recap.");
        evt.snapshot_recap_at_turn_end();
        assert_eq!(
            evt.last_completed_turn_text.as_deref(),
            Some("Here is the recap.")
        );
    }

    #[test]
    fn snapshot_recap_at_turn_end_preserves_prior_when_candidate_is_narration() {
        let mut evt = WorkspaceEvents {
            last_completed_turn_text: Some("Prior good recap.".to_string()),
            ..WorkspaceEvents::default()
        };
        evt.record_batch_longest_text("Let me check the parser.");
        evt.snapshot_recap_at_turn_end();
        // Prior recap survives because the new candidate was narration.
        assert_eq!(
            evt.last_completed_turn_text.as_deref(),
            Some("Prior good recap.")
        );
        // Accumulator still cleared — next turn starts fresh.
        assert!(evt.longest_text_this_turn.is_none());
    }

    #[test]
    fn snapshot_recap_at_turn_end_with_empty_accumulator_does_nothing() {
        let mut evt = WorkspaceEvents {
            last_completed_turn_text: Some("Prior recap.".to_string()),
            ..WorkspaceEvents::default()
        };
        evt.snapshot_recap_at_turn_end();
        assert_eq!(
            evt.last_completed_turn_text.as_deref(),
            Some("Prior recap.")
        );
    }

    #[test]
    fn reset_session_state_clears_recap_fields() {
        let mut evt = WorkspaceEvents {
            longest_text_this_turn: Some("foo".to_string()),
            last_completed_turn_text: Some("bar".to_string()),
            ..WorkspaceEvents::default()
        };
        evt.reset_session_state();
        assert!(evt.longest_text_this_turn.is_none());
        assert!(evt.last_completed_turn_text.is_none());
    }

    #[test]
    fn tail_session_captures_longest_text_in_batch() {
        // The recap pipeline cares about the *longest* assistant text
        // in a turn, not the last — narration is short, recaps are
        // long. tail_session should expose the batch's longest text
        // alongside the existing last-text field.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let short = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Let me look."}]},"timestamp":"2026-05-14T17:32:02.000Z"}"#;
        let long = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Here is the substantial recap of what got done in this turn including details."}]},"timestamp":"2026-05-14T17:32:03.000Z"}"#;
        let final_short = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Done."}]},"timestamp":"2026-05-14T17:32:04.000Z"}"#;
        std::fs::write(&path, format!("{short}\n{long}\n{final_short}\n")).unwrap();
        let upd = tail_session(&path, 0).unwrap();
        // last_assistant_text remains the final one (existing behavior).
        assert_eq!(upd.last_assistant_text.as_deref(), Some("Done."));
        // longest captures the middle, long block.
        assert_eq!(
            upd.longest_assistant_text_in_batch.as_deref(),
            Some("Here is the substantial recap of what got done in this turn including details.")
        );
    }

    #[test]
    fn tail_session_longest_picks_longest_block_within_one_message() {
        // A single assistant message containing multiple text blocks —
        // long recap first, terse "Done." last. The batch-longest must
        // be the long block. Catches the bug where the parser only
        // surfaces the last text block to the tail loop.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Here is a substantial recap that summarises the turn at length."},{"type":"text","text":"Done."}]},"timestamp":"2026-05-14T17:32:02.000Z"}"#;
        std::fs::write(&path, format!("{line}\n")).unwrap();
        let upd = tail_session(&path, 0).unwrap();
        assert_eq!(
            upd.longest_assistant_text_in_batch.as_deref(),
            Some("Here is a substantial recap that summarises the turn at length.")
        );
    }

    #[test]
    fn tail_session_longest_text_is_none_when_no_text() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        // Only a tool_use line — no assistant text.
        let only_tool = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Bash","input":{"command":"x"}}]},"timestamp":"2026-05-14T17:32:02.000Z"}"#;
        std::fs::write(&path, format!("{only_tool}\n")).unwrap();
        let upd = tail_session(&path, 0).unwrap();
        assert!(upd.longest_assistant_text_in_batch.is_none());
    }

    #[test]
    fn parse_assistant_captures_context_tokens_and_model() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"model":"claude-opus-4-8","stop_reason":"end_turn","usage":{"input_tokens":2,"cache_creation_input_tokens":4874,"cache_read_input_tokens":72081,"output_tokens":277},"content":[{"type":"text","text":"hi"}]}}"#;
        let parsed = parse_jsonl_line(line);
        // context = input + cache_creation + cache_read = 2 + 4874 + 72081
        assert_eq!(parsed.context_tokens, Some(76_957));
        assert_eq!(parsed.model_id.as_deref(), Some("claude-opus-4-8"));
    }

    #[test]
    fn parse_assistant_current_action_is_bash_command() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cargo test --lib"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.current_action.as_deref(), Some("cargo test --lib"));
    }

    #[test]
    fn parse_assistant_current_action_is_now_basename_for_edit() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"/abs/src/ui/dashboard/column_content.rs"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(
            parsed.current_action.as_deref(),
            Some("now column_content.rs")
        );
    }

    #[test]
    fn parse_assistant_no_current_action_for_read() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/abs/x.rs"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.current_action, None);
    }

    #[test]
    fn parse_assistant_captures_ask_user_question_header() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"AskUserQuestion","input":{"questions":[{"header":"Auth method","question":"Which auth approach?"}]}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.pending_question_text.as_deref(), Some("Auth method"));
    }

    #[test]
    fn parse_assistant_ask_user_question_falls_back_to_question() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"AskUserQuestion","input":{"questions":[{"question":"Which auth approach?"}]}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(
            parsed.pending_question_text.as_deref(),
            Some("Which auth approach?")
        );
    }

    #[test]
    fn parse_assistant_current_action_for_notebook_edit_uses_notebook_path() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"NotebookEdit","input":{"notebook_path":"/abs/notes/analysis.ipynb"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.current_action.as_deref(), Some("now analysis.ipynb"));
    }

    #[test]
    fn parse_assistant_current_action_for_write_uses_file_path() {
        let line = r#"{"type":"assistant","timestamp":"2026-06-11T00:00:00.000Z","message":{"content":[{"type":"tool_use","id":"t1","name":"Write","input":{"file_path":"/abs/src/new_mod.rs"}}]}}"#;
        let parsed = parse_jsonl_line(line);
        assert_eq!(parsed.current_action.as_deref(), Some("now new_mod.rs"));
    }

    #[test]
    fn clean_recap_real_hibiscus_turn_2() {
        // Regression: insight banner + bullets + closing banner + prose.
        let s = "`★ Insight ─────`\n- DetailContext is a borrowed snapshot — zero allocations per draw.\n- The four current modules each tap a different layer.\n`─────`\n\nHere are ideas grouped by layer.";
        let got = clean_recap(s).expect("should keep content");
        assert!(
            got.starts_with("Here are ideas grouped by layer."),
            "expected post-banner prose, got: {got:?}"
        );
    }
}
