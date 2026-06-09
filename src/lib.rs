//! Parse agent session logs into a navigable model and a live event stream.
//!
//! `sessionx` is the framework-agnostic core shared by `wsx` and
//! `chronox-tui`. It has no UI dependency; rendering lives in the consumers.
//! Two complementary views over the same Claude Code (and Codex/Pi) session
//! logs:
//!
//! - **Change timeline** ([`Timeline`], [`ChangeEvent`]): what files an agent
//!   changed and when, with syntax-tokenized diffs ([`syntax`]).
//! - **Event stream** ([`activity::events`]): the agent's live session state —
//!   user/assistant turns, tool calls, stop reasons ([`StopReason`],
//!   [`EventSnapshot`], [`WorkspaceEvents`]).

pub mod activity;
pub mod config;
pub mod error;
pub mod event;
pub mod extract;
pub mod nav;
pub mod syntax;
pub mod timeline;

#[cfg(test)]
#[allow(dead_code)]
mod test_support;

// --- Change-timeline model ---
pub use config::{ChronologyConfig, ChronologyOverride, ConfigSource, Side, WidthSpec};
pub use config::{resolve, resolve_global_only};
pub use event::{ChangeDetail, ChangeEvent, ChangeSource, ChangeTool};
pub use extract::DETAIL_MAX_CHARS;
pub use nav::{NavAction, NavKey};
pub use syntax::{
    CellKind, DiffCell, DiffLine, DiffMarker, LangSpec, SideRow, Token, TokenKind,
    change_detail_side_by_side, lang_for_path,
};
pub use timeline::Timeline;

// --- Session event stream (shared types; per-agent tailers live in `activity`) ---
pub use activity::events::{
    EventKind, EventSnapshot, StopReason, TailUpdate, ToolUseCounts, WorkspaceEvents, clean_recap,
    push_event,
};

pub use error::{Error, Result};
