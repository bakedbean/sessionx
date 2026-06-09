//! Core change event types: the data model for a single file change.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The mutating tool that produced a change. Read and non-mutating tools are
/// never recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeTool {
    Edit,
    MultiEdit,
    Write,
    NotebookEdit,
}

impl ChangeTool {
    /// Compact label for display (`edit` / `write` / …).
    pub fn label(self) -> &'static str {
        match self {
            ChangeTool::Edit => "edit",
            ChangeTool::MultiEdit => "edit",
            ChangeTool::Write => "write",
            ChangeTool::NotebookEdit => "edit",
        }
    }
}

/// Bounded change text retained for the expandable diff peek (C fidelity).
/// `None` when the agent did not expose the changed text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeDetail {
    Edit { old: String, new: String },
    Write { head: String },
    None,
}

/// Where a `ChangeEvent` was extracted from, so the full (un-clipped) change can
/// be re-read on demand for the detail modal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeSource {
    pub session_file: PathBuf,
    pub line_index: usize,
    pub index_in_line: usize,
}

/// One change the agent made at one moment in time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeEvent {
    /// Epoch milliseconds, parsed from the JSONL line's `timestamp`.
    pub timestamp_ms: i64,
    pub tool: ChangeTool,
    /// Absolute path as the agent reported it (display layer makes it relative).
    pub file_path: PathBuf,
    /// One-line "what" summary (B fidelity).
    pub summary: String,
    /// Change text for the C-expand peek.
    pub detail: ChangeDetail,
    /// Back-reference to the session log line for full re-extraction.
    pub source: ChangeSource,
}
