//! Per-workspace `Timeline` cache: merges events across session files,
//! reparsing only when a file's `(size, mtime)` changes.

use crate::event::ChangeEvent;
use crate::extract::parse_file;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A per-file cache key. Reparse only when size or mtime changes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    mtime: SystemTime,
}

fn stamp(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(path).ok()?;
    Some(FileStamp {
        size: meta.len(),
        mtime: meta.modified().ok()?,
    })
}

/// Merged, newest-first chronology of `ChangeEvent`s across a workspace's
/// session files. Caches parsed events per file by `(size, mtime)`.
#[derive(Debug, Default)]
pub struct Timeline {
    /// Per-file parsed events + the stamp they were parsed at.
    per_file: HashMap<PathBuf, (FileStamp, Vec<ChangeEvent>)>,
    /// Flattened, sorted view rebuilt on each refresh.
    merged: Vec<ChangeEvent>,
    /// Test/diagnostic counter of how many file parses have occurred.
    parses: usize,
}

impl Timeline {
    /// Re-scan `files`, reparsing only those whose `(size, mtime)` changed,
    /// dropping cache entries for files no longer present, then rebuild the
    /// merged newest-first view.
    pub fn refresh(&mut self, files: &[PathBuf]) {
        let present: std::collections::HashSet<&PathBuf> = files.iter().collect();
        self.per_file.retain(|p, _| present.contains(p));

        for path in files {
            let Some(st) = stamp(path) else { continue };
            let needs = match self.per_file.get(path) {
                Some((prev, _)) => *prev != st,
                None => true,
            };
            if needs {
                let evs = parse_file(path);
                self.parses += 1;
                self.per_file.insert(path.clone(), (st, evs));
            }
        }

        let mut merged: Vec<ChangeEvent> = self
            .per_file
            .values()
            .flat_map(|(_, evs)| evs.iter().cloned())
            .collect();
        // Newest first. Tie-break on file_path so equal-timestamp events have a
        // deterministic order across refreshes (the per_file source is a HashMap
        // whose iteration order is not stable).
        merged.sort_by(|a, b| {
            b.timestamp_ms
                .cmp(&a.timestamp_ms)
                .then_with(|| a.file_path.cmp(&b.file_path))
        });
        self.merged = merged;
    }

    /// The merged newest-first events.
    pub fn events(&self) -> &[ChangeEvent] {
        &self.merged
    }

    #[cfg(test)]
    pub fn parse_count(&self) -> usize {
        self.parses
    }
}

#[cfg(test)]
mod timeline_tests {
    use super::*;
    use std::io::Write;

    fn write_event(path: &Path, ts: &str, file: &str) {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","timestamp":"{ts}","message":{{"content":[{{"type":"tool_use","name":"Write","input":{{"file_path":"{file}","content":"x"}}}}]}}}}"#
        )
        .unwrap();
    }

    #[test]
    fn merges_files_newest_first() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        write_event(&a, "2026-05-14T17:00:00.000Z", "/wt/old.rs");
        write_event(&b, "2026-05-14T18:00:00.000Z", "/wt/new.rs");
        let mut tl = Timeline::default();
        tl.refresh(&[a.clone(), b.clone()]);
        let evs = tl.events();
        assert_eq!(evs.len(), 2);
        assert_eq!(
            evs[0].file_path,
            PathBuf::from("/wt/new.rs"),
            "newest first"
        );
        assert_eq!(evs[1].file_path, PathBuf::from("/wt/old.rs"));
    }

    #[test]
    fn unchanged_file_is_not_reparsed() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.jsonl");
        write_event(&a, "2026-05-14T17:00:00.000Z", "/wt/old.rs");
        let mut tl = Timeline::default();
        tl.refresh(std::slice::from_ref(&a));
        assert_eq!(tl.parse_count(), 1);
        tl.refresh(std::slice::from_ref(&a)); // same size+mtime → cache hit
        assert_eq!(tl.parse_count(), 1, "should not reparse unchanged file");
    }

    #[test]
    fn grown_file_is_reparsed() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.jsonl");
        write_event(&a, "2026-05-14T17:00:00.000Z", "/wt/old.rs");
        let mut tl = Timeline::default();
        tl.refresh(std::slice::from_ref(&a));
        write_event(&a, "2026-05-14T19:00:00.000Z", "/wt/newer.rs");
        tl.refresh(std::slice::from_ref(&a));
        assert_eq!(tl.parse_count(), 2, "size changed → reparse");
        assert_eq!(tl.events().len(), 2);
    }

    #[test]
    fn removed_file_events_drop_from_merged() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        write_event(&a, "2026-05-14T17:00:00.000Z", "/wt/a.rs");
        write_event(&b, "2026-05-14T18:00:00.000Z", "/wt/b.rs");
        let mut tl = Timeline::default();
        tl.refresh(&[a.clone(), b.clone()]);
        assert_eq!(tl.events().len(), 2);
        // b.jsonl no longer in the file list → its events must disappear.
        tl.refresh(std::slice::from_ref(&a));
        let evs = tl.events();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].file_path, PathBuf::from("/wt/a.rs"));
    }
}
