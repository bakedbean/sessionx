//! Basic, dependency-free syntax highlighting core. Emits neutral `TokenKind`
//! tokens — no ratatui dependency. The ratatui mapping lives in `render.rs`.

use crate::event::ChangeDetail;
use std::path::Path;

/// Minimal language description driving the generic tokenizer.
pub struct LangSpec {
    pub keywords: &'static [&'static str],
    pub line_comment: &'static [&'static str],
    pub string_delims: &'static [char],
}

static RUST: LangSpec = LangSpec {
    keywords: &[
        "fn", "let", "mut", "pub", "use", "struct", "enum", "impl", "trait", "for", "in", "if",
        "else", "match", "while", "loop", "return", "self", "Self", "mod", "const", "static",
        "move", "ref", "as", "where", "async", "await", "dyn", "crate", "super", "type", "unsafe",
        "break", "continue", "true", "false",
    ],
    line_comment: &["//"],
    string_delims: &['"'],
};

static CLIKE: LangSpec = LangSpec {
    keywords: &[
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "break",
        "continue",
        "return",
        "struct",
        "class",
        "const",
        "static",
        "void",
        "int",
        "char",
        "bool",
        "new",
        "delete",
        "public",
        "private",
        "protected",
        "function",
        "var",
        "let",
        "import",
        "export",
        "from",
        "default",
        "true",
        "false",
        "null",
    ],
    line_comment: &["//"],
    string_delims: &['"', '\''],
};

static PYTHON: LangSpec = LangSpec {
    keywords: &[
        "def", "class", "return", "if", "elif", "else", "for", "while", "import", "from", "as",
        "with", "try", "except", "finally", "raise", "lambda", "None", "True", "False", "and",
        "or", "not", "in", "is", "pass", "yield", "global", "nonlocal",
    ],
    line_comment: &["#"],
    string_delims: &['"', '\''],
};

static SHELL: LangSpec = LangSpec {
    keywords: &[
        "if", "then", "else", "elif", "fi", "for", "in", "do", "done", "while", "case", "esac",
        "function", "return", "export", "local",
    ],
    line_comment: &["#"],
    string_delims: &['"', '\''],
};

/// Pick a `LangSpec` from a path's extension; `None` → no highlighting.
pub fn lang_for_path(path: &Path) -> Option<&'static LangSpec> {
    let ext = path.extension().and_then(|e| e.to_str())?;
    match ext {
        "rs" => Some(&RUST),
        "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" | "js" | "jsx" | "ts" | "tsx" | "go"
        | "java" | "cs" | "json" => Some(&CLIKE),
        "py" => Some(&PYTHON),
        "sh" | "bash" | "zsh" => Some(&SHELL),
        _ => None,
    }
}

fn take_while(rest: &str, pred: impl Fn(char) -> bool) -> (String, usize) {
    let mut tok = String::new();
    let mut bytes = 0;
    for c in rest.chars() {
        if pred(c) {
            tok.push(c);
            bytes += c.len_utf8();
        } else {
            break;
        }
    }
    (tok, bytes)
}

fn take_string(rest: &str, delim: char) -> (String, usize) {
    let mut tok = String::new();
    let mut bytes = 0;
    let mut chars = rest.chars();
    let open = chars.next().unwrap();
    tok.push(open);
    bytes += open.len_utf8();
    let mut escaped = false;
    for c in chars {
        tok.push(c);
        bytes += c.len_utf8();
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == delim {
            break;
        }
    }
    (tok, bytes)
}

/// Highlight kind for a token run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Default,
    Keyword,
    Str,
    Number,
    Comment,
}

/// A run of source text and its highlight kind.
pub type Token = (String, TokenKind);

/// Add/remove marker for a diff line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMarker {
    Added,
    Removed,
}

/// One display line of a change's diff: a fixed-width line-number gutter, an
/// add/remove marker, and the (optionally tokenized) code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
    pub gutter: String,
    pub marker: DiffMarker,
    pub code: Vec<Token>,
}

/// Which side/kind a side-by-side cell represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    Context,
    Added,
    Removed,
}

/// One side (left=old or right=new) of a side-by-side diff row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffCell {
    pub gutter: String,
    pub kind: CellKind,
    pub code: Vec<Token>,
}

/// One row of a side-by-side diff. Either side may be absent (a blank column).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideRow {
    pub left: Option<DiffCell>,
    pub right: Option<DiffCell>,
}

/// Internal line-alignment op produced by `lcs_ops`.
enum DiffOp {
    Equal(usize, usize),
    Removed(usize),
    Added(usize),
}

/// Cap on the LCS DP table size (`old_len * new_len`). Details normally come
/// from clipped change text, but `load_full_change` can hand us an unbounded
/// `old`/`new`; without a cap the O(n*m) table could allocate gigabytes and
/// OOM. Above the cap we skip context detection and fall back to a naive
/// all-removed / all-added alignment (still O(n+m)).
const LCS_MAX_CELLS: usize = 1_000_000;

/// Longest-common-subsequence alignment of two line sequences. O(n*m) DP —
/// the blocks here are normally a single edit's old/new text, so this stays
/// cheap and keeps `syntax.rs` dependency-free. For pathologically large
/// inputs (see `LCS_MAX_CELLS`) it degrades to a naive alignment.
fn lcs_ops(old: &[&str], new: &[&str]) -> Vec<DiffOp> {
    let (n, m) = (old.len(), new.len());
    if n.saturating_mul(m) > LCS_MAX_CELLS {
        // Naive fallback: every old line is Removed, every new line Added. The
        // caller zips these into side-by-side rows just like a fully-changed
        // block — no context lines, but bounded memory.
        let mut ops = Vec::with_capacity(n + m);
        ops.extend((0..n).map(DiffOp::Removed));
        ops.extend((0..m).map(DiffOp::Added));
        return ops;
    }
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if old[i] == new[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut ops = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if old[i] == new[j] {
            ops.push(DiffOp::Equal(i, j));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            ops.push(DiffOp::Removed(i));
            i += 1;
        } else {
            ops.push(DiffOp::Added(j));
            j += 1;
        }
    }
    while i < n {
        ops.push(DiffOp::Removed(i));
        i += 1;
    }
    while j < m {
        ops.push(DiffOp::Added(j));
        j += 1;
    }
    ops
}

/// Flush a pending change block (consecutive removed and/or added lines) as
/// zipped side-by-side rows: row k pairs the k-th removed line (left, red) with
/// the k-th added line (right, green); the shorter side gets a blank cell.
fn push_change_block(
    rows: &mut Vec<SideRow>,
    rem: &mut Vec<usize>,
    add: &mut Vec<usize>,
    old_lines: &[&str],
    new_lines: &[&str],
    base_line: u32,
    lang: Option<&LangSpec>,
) {
    let k = rem.len().max(add.len());
    for idx in 0..k {
        let left = rem.get(idx).map(|&i| DiffCell {
            gutter: "     ".to_string(),
            kind: CellKind::Removed,
            code: code_tokens(old_lines[i], lang),
        });
        let right = add.get(idx).map(|&j| DiffCell {
            gutter: format!("{:>4} ", base_line.saturating_add(j as u32)),
            kind: CellKind::Added,
            code: code_tokens(new_lines[j], lang),
        });
        rows.push(SideRow { left, right });
    }
    rem.clear();
    add.clear();
}

/// Build a side-by-side diff: old on the left, new on the right, aligned by an
/// LCS so only genuinely-changed lines are marked. Right-side (new-file) lines
/// are numbered from `base_line`; the left gutter is blank (no reliable
/// old-file numbers — matches the removed-line convention of `change_detail_diff`).
pub fn change_detail_side_by_side(
    detail: &ChangeDetail,
    base_line: u32,
    lang: Option<&LangSpec>,
) -> Vec<SideRow> {
    match detail {
        ChangeDetail::Edit { old, new } => {
            let old_lines: Vec<&str> = old.lines().collect();
            let new_lines: Vec<&str> = new.lines().collect();
            let mut rows: Vec<SideRow> = Vec::new();
            let mut rem: Vec<usize> = Vec::new();
            let mut add: Vec<usize> = Vec::new();
            for op in lcs_ops(&old_lines, &new_lines) {
                match op {
                    DiffOp::Equal(i, j) => {
                        push_change_block(
                            &mut rows, &mut rem, &mut add, &old_lines, &new_lines, base_line, lang,
                        );
                        rows.push(SideRow {
                            left: Some(DiffCell {
                                gutter: "     ".to_string(),
                                kind: CellKind::Context,
                                code: code_tokens(old_lines[i], lang),
                            }),
                            right: Some(DiffCell {
                                gutter: format!("{:>4} ", base_line.saturating_add(j as u32)),
                                kind: CellKind::Context,
                                code: code_tokens(new_lines[j], lang),
                            }),
                        });
                    }
                    DiffOp::Removed(i) => rem.push(i),
                    DiffOp::Added(j) => add.push(j),
                }
            }
            push_change_block(
                &mut rows, &mut rem, &mut add, &old_lines, &new_lines, base_line, lang,
            );
            rows
        }
        ChangeDetail::Write { head } => head
            .lines()
            .enumerate()
            .map(|(j, l)| SideRow {
                left: None,
                right: Some(DiffCell {
                    gutter: format!("{:>4} ", base_line.saturating_add(j as u32)),
                    kind: CellKind::Added,
                    code: code_tokens(l, lang),
                }),
            })
            .collect(),
        ChangeDetail::None => Vec::new(),
    }
}

fn push_default(out: &mut Vec<Token>, buf: &mut String) {
    if !buf.is_empty() {
        out.push((std::mem::take(buf), TokenKind::Default));
    }
}

/// Tokenize ONE line of code into (text, kind) runs by `spec`. Priority: line
/// comment (rest of line) > string > number > keyword/identifier > default.
pub fn tokenize_line(text: &str, spec: &LangSpec) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if spec.line_comment.iter().any(|c| rest.starts_with(c)) {
            push_default(&mut out, &mut buf);
            out.push((rest.to_string(), TokenKind::Comment));
            return out;
        }
        let ch = rest.chars().next().unwrap();
        if spec.string_delims.contains(&ch) {
            push_default(&mut out, &mut buf);
            let (tok, consumed) = take_string(rest, ch);
            out.push((tok, TokenKind::Str));
            i += consumed;
        } else if ch.is_ascii_digit() {
            push_default(&mut out, &mut buf);
            let (tok, consumed) = take_while(rest, |c| c.is_ascii_digit() || c == '.' || c == '_');
            out.push((tok, TokenKind::Number));
            i += consumed;
        } else if ch.is_alphabetic() || ch == '_' {
            let (tok, consumed) = take_while(rest, |c| c.is_alphanumeric() || c == '_');
            if spec.keywords.contains(&tok.as_str()) {
                push_default(&mut out, &mut buf);
                out.push((tok, TokenKind::Keyword));
            } else {
                buf.push_str(&tok);
            }
            i += consumed;
        } else {
            buf.push(ch);
            i += ch.len_utf8();
        }
    }
    push_default(&mut out, &mut buf);
    out
}

fn code_tokens(code: &str, lang: Option<&LangSpec>) -> Vec<Token> {
    match lang {
        Some(spec) => tokenize_line(code, spec),
        None => vec![(code.to_string(), TokenKind::Default)],
    }
}

/// Build the neutral diff model: removed (`old`) lines with a blank gutter and
/// `Removed` marker, then added (`new`/`head`) lines numbered from `base_line`
/// with an `Added` marker. No line cap — the modal scrolls.
pub fn change_detail_diff(
    detail: &ChangeDetail,
    base_line: u32,
    lang: Option<&LangSpec>,
) -> Vec<DiffLine> {
    let mut out = Vec::new();
    match detail {
        ChangeDetail::Edit { old, new } => {
            for l in old.lines() {
                out.push(DiffLine {
                    gutter: "     ".to_string(),
                    marker: DiffMarker::Removed,
                    code: code_tokens(l, lang),
                });
            }
            for (k, l) in new.lines().enumerate() {
                let n = base_line.saturating_add(k as u32);
                out.push(DiffLine {
                    gutter: format!("{n:>4} "),
                    marker: DiffMarker::Added,
                    code: code_tokens(l, lang),
                });
            }
        }
        ChangeDetail::Write { head } => {
            for (k, l) in head.lines().enumerate() {
                let n = base_line.saturating_add(k as u32);
                out.push(DiffLine {
                    gutter: format!("{n:>4} "),
                    marker: DiffMarker::Added,
                    code: code_tokens(l, lang),
                });
            }
        }
        ChangeDetail::None => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tokenize_priority_comment_string_number_keyword() {
        let spec = lang_for_path(Path::new("a.rs")).unwrap();
        // keyword + identifier + number + string
        let toks = tokenize_line("let x = 42 + \"hi\"", spec);
        // first run is the "let" keyword
        assert_eq!(toks[0], ("let".to_string(), TokenKind::Keyword));
        // a number run somewhere
        assert!(
            toks.iter()
                .any(|(t, k)| t == "42" && *k == TokenKind::Number)
        );
        // a string run including quotes
        assert!(
            toks.iter()
                .any(|(t, k)| t == "\"hi\"" && *k == TokenKind::Str)
        );
        // line comment swallows the rest of the line
        let c = tokenize_line("x // tail", spec);
        assert_eq!(
            c.last().unwrap(),
            &("// tail".to_string(), TokenKind::Comment)
        );
    }

    #[test]
    fn change_detail_diff_gutter_and_marker() {
        let detail = ChangeDetail::Edit {
            old: "old".into(),
            new: "let y = 1".into(),
        };
        let lines = change_detail_diff(&detail, 7, lang_for_path(Path::new("a.rs")));
        // removed line: blank gutter, Removed marker
        assert_eq!(lines[0].gutter, "     ");
        assert_eq!(lines[0].marker, DiffMarker::Removed);
        // added line: gutter "   7 ", Added marker, "let" tokenized as keyword
        assert_eq!(lines[1].gutter, "   7 ");
        assert_eq!(lines[1].marker, DiffMarker::Added);
        assert!(
            lines[1]
                .code
                .iter()
                .any(|(t, k)| t == "let" && *k == TokenKind::Keyword)
        );
    }

    #[test]
    fn no_lang_is_single_default_run() {
        let detail = ChangeDetail::Write {
            head: "let y = 1".into(),
        };
        let lines = change_detail_diff(&detail, 1, None);
        assert_eq!(
            lines[0].code,
            vec![("let y = 1".to_string(), TokenKind::Default)]
        );
    }

    #[test]
    fn lang_for_path_maps_extensions() {
        assert!(lang_for_path(Path::new("a.rs")).is_some());
        assert!(lang_for_path(Path::new("a.py")).is_some());
        assert!(lang_for_path(Path::new("a.c")).is_some());
        assert!(lang_for_path(Path::new("a.js")).is_some());
        assert!(lang_for_path(Path::new("a.sh")).is_some());
        assert!(lang_for_path(Path::new("a.txt")).is_none());
        assert!(lang_for_path(Path::new("noext")).is_none());
    }

    #[test]
    fn tokenize_string_keeps_escaped_quote_in_one_run() {
        let spec = lang_for_path(Path::new("a.rs")).unwrap();
        let toks = tokenize_line(r#""a\"b""#, spec);
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].0, r#""a\"b""#);
        assert_eq!(toks[0].1, TokenKind::Str);
    }

    #[test]
    fn non_keyword_identifier_is_default() {
        let spec = lang_for_path(Path::new("a.rs")).unwrap();
        let toks = tokenize_line("foobar", spec);
        assert_eq!(toks, vec![("foobar".to_string(), TokenKind::Default)]);
    }

    // Helper: join a cell's code tokens back into a plain string.
    fn cell_code(cell: &Option<DiffCell>) -> Option<String> {
        cell.as_ref()
            .map(|c| c.code.iter().map(|(t, _)| t.clone()).collect())
    }

    #[test]
    fn side_by_side_pure_replace_zips_old_and_new() {
        let detail = ChangeDetail::Edit {
            old: "a\nb".into(),
            new: "x\ny".into(),
        };
        let rows = change_detail_side_by_side(&detail, 5, None);
        assert_eq!(rows.len(), 2);
        // row 0: removed "a" (blank gutter) | added "x" (numbered from base 5)
        let l0 = rows[0].left.as_ref().unwrap();
        let r0 = rows[0].right.as_ref().unwrap();
        assert_eq!(l0.kind, CellKind::Removed);
        assert_eq!(l0.gutter, "     ");
        assert_eq!(cell_code(&rows[0].left).unwrap(), "a");
        assert_eq!(r0.kind, CellKind::Added);
        assert_eq!(r0.gutter, "   5 ");
        assert_eq!(cell_code(&rows[0].right).unwrap(), "x");
        // row 1: new line numbered 6
        assert_eq!(rows[1].right.as_ref().unwrap().gutter, "   6 ");
    }

    #[test]
    fn side_by_side_keeps_context_and_blanks_short_side() {
        let detail = ChangeDetail::Edit {
            old: "ctx\nlet x = 1;".into(),
            new: "ctx\nlet x = 2;\nlet y = 3;".into(),
        };
        let rows = change_detail_side_by_side(&detail, 10, None);
        // row 0 is the shared context line, present on both sides
        assert_eq!(rows[0].left.as_ref().unwrap().kind, CellKind::Context);
        assert_eq!(rows[0].right.as_ref().unwrap().kind, CellKind::Context);
        assert_eq!(cell_code(&rows[0].left).unwrap(), "ctx");
        // a replace row (let x = 1; -> let x = 2;) then an add-only row (let y = 3;)
        assert_eq!(cell_code(&rows[1].left).unwrap(), "let x = 1;");
        assert_eq!(cell_code(&rows[1].right).unwrap(), "let x = 2;");
        assert!(
            rows[2].left.is_none(),
            "the extra added line has no left side"
        );
        assert_eq!(cell_code(&rows[2].right).unwrap(), "let y = 3;");
        assert_eq!(rows[2].right.as_ref().unwrap().kind, CellKind::Added);
    }

    #[test]
    fn side_by_side_write_is_added_only_on_the_right() {
        let detail = ChangeDetail::Write {
            head: "one\ntwo".into(),
        };
        let rows = change_detail_side_by_side(&detail, 1, None);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].left.is_none());
        assert_eq!(rows[0].right.as_ref().unwrap().gutter, "   1 ");
        assert_eq!(rows[0].right.as_ref().unwrap().kind, CellKind::Added);
        assert_eq!(rows[1].right.as_ref().unwrap().gutter, "   2 ");
    }

    #[test]
    fn side_by_side_none_is_empty() {
        assert!(change_detail_side_by_side(&ChangeDetail::None, 1, None).is_empty());
    }

    #[test]
    fn side_by_side_pure_add() {
        let detail = ChangeDetail::Edit {
            old: "".into(),
            new: "x".into(),
        };
        let rows = change_detail_side_by_side(&detail, 1, None);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].left.is_none());
        assert_eq!(rows[0].right.as_ref().unwrap().kind, CellKind::Added);
        assert_eq!(cell_code(&rows[0].right).unwrap(), "x");
    }

    #[test]
    fn side_by_side_pure_remove() {
        let detail = ChangeDetail::Edit {
            old: "y".into(),
            new: "".into(),
        };
        let rows = change_detail_side_by_side(&detail, 1, None);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].right.is_none());
        assert_eq!(rows[0].left.as_ref().unwrap().kind, CellKind::Removed);
        assert_eq!(cell_code(&rows[0].left).unwrap(), "y");
    }

    #[test]
    fn side_by_side_falls_back_above_cell_cap() {
        // n*m exceeds LCS_MAX_CELLS (1100*1100 ≈ 1.21M), so context detection
        // is skipped: even identical old/new degrade to a naive all-removed /
        // all-added zip. This both verifies the fallback fires and guards
        // against the O(n*m) table OOMing on unbounded `load_full_change` text.
        let block: String = (0..1100).map(|i| format!("line {i}\n")).collect();
        let detail = ChangeDetail::Edit {
            old: block.clone(),
            new: block,
        };
        let rows = change_detail_side_by_side(&detail, 1, None);
        assert_eq!(rows.len(), 1100);
        assert!(
            rows.iter().all(|r| {
                r.left.as_ref().map(|c| c.kind) == Some(CellKind::Removed)
                    && r.right.as_ref().map(|c| c.kind) == Some(CellKind::Added)
            }),
            "above the cap, identical lines are not detected as context"
        );
    }
}
