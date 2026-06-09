//! Error type for sessionx.
//!
//! The parsing entry points (`tail_session` and friends) only ever surface
//! filesystem and JSON-decoding failures, so the surface is intentionally
//! tiny. Consumers can map these into their own richer error enums.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}
