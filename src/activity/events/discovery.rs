//! Locating Claude Code session files on disk.

use std::path::{Path, PathBuf};

/// Encode an absolute path the way Claude Code does for `~/.claude/projects/`.
///
/// Delegates to [`crate::extract::encode_cwd`], the single source of truth for
/// this encoding, so the change-extraction and event-stream paths can never
/// drift. (They previously each replaced only `/` and `.`, missing spaces and
/// other punctuation — repos with a space in the name then resolved to a
/// non-existent directory.)
pub fn encode_cwd(path: &Path) -> String {
    crate::extract::encode_cwd(path)
}

/// Locate the active session file for a worktree.
///
/// Returns the latest-modified `.jsonl` in
/// `~/.claude/projects/<encoded-cwd>/`, if any.
pub fn locate_session_file(worktree: &Path) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let abs = std::fs::canonicalize(worktree).ok()?;
    let encoded = encode_cwd(&abs);
    let session_dir = home.join(".claude/projects").join(encoded);
    if !session_dir.is_dir() {
        return None;
    }
    let mut newest: Option<(PathBuf, std::time::SystemTime)> = None;
    for entry in std::fs::read_dir(&session_dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        match &newest {
            None => newest = Some((path, mtime)),
            Some((_, prev)) if mtime > *prev => newest = Some((path, mtime)),
            _ => {}
        }
    }
    newest.map(|(p, _)| p)
}
