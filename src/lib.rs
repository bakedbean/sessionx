//! Parse agent session logs into a navigable model and a live event stream.
//!
//! `sessionx` is the framework-agnostic core shared by `wsx` and
//! `chronox-tui`. It has no UI dependency; rendering lives in the consumers.

pub mod error;

pub use error::{Error, Result};
