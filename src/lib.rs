//! Parse agent session logs into a navigable model and a live event stream.
//!
//! `sessionx` is the framework-agnostic core shared by `wsx` and
//! `chronox-tui`. It has no UI dependency; rendering lives in the consumers.
//! Two complementary views over the same Claude Code (and Codex/Pi) session
//! logs:
//!
//! - **Change timeline** ([`Timeline`], [`ChangeEvent`]): what files an agent
//!   changed and when, with syntax-tokenized diffs ([`syntax`]).
//! - **Event stream** (`activity::events`): the agent's live session state —
//!   user/assistant turns, tool calls, stop reasons.

pub mod config;
pub mod error;
pub mod event;
pub mod extract;
pub mod nav;
pub mod syntax;
pub mod timeline;

// --- Change-timeline model ---
pub use config::{ChronologyConfig, ChronologyOverride, ConfigSource, Side, WidthSpec};
pub use config::{resolve, resolve_global_only};
pub use event::{ChangeDetail, ChangeEvent, ChangeSource, ChangeTool};
pub use extract::DETAIL_MAX_CHARS;
pub use nav::{NavAction, NavKey};
pub use syntax::{DiffLine, DiffMarker, LangSpec, Token, TokenKind, lang_for_path};
pub use timeline::Timeline;

pub use error::{Error, Result};
