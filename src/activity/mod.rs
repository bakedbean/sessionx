//! Tailing of live agent session logs into a display-ready event stream.
//!
//! `events` tails Claude Code JSONL session logs; `codex_events` and
//! `pi_events` are the Codex and Pi variants built on top of its shared
//! types (`EventSnapshot`, `EventKind`, `StopReason`, `TailUpdate`).

pub mod codex_events;
pub mod events;
pub mod pi_events;
