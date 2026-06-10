//! Display-string formatting: whitespace collapsing, truncation, and turning a
//! raw assistant text block into a SESSION SUMMARY recap candidate.

/// Cap on a rendered display line's length (characters, not bytes).
pub const MAX_DISPLAY_CHARS: usize = 512;

pub fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

pub fn truncate_display(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('\u{2026}'); // ellipsis
        out
    }
}

/// Clean a raw assistant text block into a recap candidate suitable
/// for the SESSION SUMMARY column. Returns `None` if the text looks
/// like pre-tool narration, a pure short closing question, or pure
/// decoration with no prose content.
///
/// Conservative — when in doubt, return `Some`. Steps:
/// 1. Skip leading blank lines.
/// 2. If the first content line opens a `★ Insight` block, skip past
///    its closing horizontal-rule banner.
/// 3. If the first content line opens a fenced code block (```), skip
///    past its closing fence.
/// 4. Strip residual decoration lines (horizontal rules, orphan
///    fences, blank lines).
/// 5. Reject when the first remaining content line starts with a
///    narration prefix (`Let me `, `I'll `, `I'm going to `, `Let's `).
/// 6. Reject when the entire cleaned content is a single short
///    trailing question (≤50 chars, ends with `?`, no internal `.` or
///    `!`). 50-char cap excludes substantive questions like "Want me
///    to audit every callsite that touches that field too?" while
///    still catching closers like "Want me to update the tests?".
pub fn clean_recap(raw: &str) -> Option<String> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut idx = 0;

    let skip_blanks = |lines: &[&str], mut i: usize| {
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        i
    };

    idx = skip_blanks(&lines, idx);
    idx = skip_insight_block(&lines, idx);
    idx = skip_blanks(&lines, idx);
    idx = skip_code_fence(&lines, idx);
    idx = skip_blanks(&lines, idx);
    while idx < lines.len() && is_decoration_line(lines[idx]) {
        idx += 1;
    }

    if idx >= lines.len() {
        return None;
    }

    let first = lines[idx].trim_start();
    const NARRATION_PREFIXES: &[&str] = &[
        "Let me ",
        "let me ",
        "I'll ",
        "i'll ",
        "I'm going to ",
        "i'm going to ",
        "Let's ",
        "let's ",
    ];
    if NARRATION_PREFIXES.iter().any(|p| first.starts_with(p)) {
        return None;
    }

    let cleaned = lines[idx..].join("\n").trim_end().to_string();
    if cleaned.trim().is_empty() {
        return None;
    }
    if is_pure_short_question(&cleaned) {
        return None;
    }
    Some(cleaned)
}

fn is_decoration_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return true;
    }
    // Lone code-fence marker (``` or ```rust). `skip_code_fence`
    // already paired-stripped leading fences; this catches orphans.
    if t.starts_with("```") {
        return true;
    }
    let inner = t.trim_matches('`').trim();
    !inner.is_empty()
        && inner
            .chars()
            .all(|c| c == '─' || c == '-' || c.is_whitespace())
}

fn skip_insight_block(lines: &[&str], start: usize) -> usize {
    let Some(line) = lines.get(start) else {
        return start;
    };
    let inner = line.trim().trim_matches('`').trim();
    let is_opener = inner.starts_with('★') && inner.contains('─');
    if !is_opener {
        return start;
    }
    let window_end = (start + 1 + 20).min(lines.len());
    for (i, line) in lines.iter().enumerate().take(window_end).skip(start + 1) {
        let candidate = line.trim().trim_matches('`').trim();
        if !candidate.is_empty()
            && candidate
                .chars()
                .all(|c| c == '─' || c == '-' || c.is_whitespace())
        {
            return i + 1;
        }
    }
    // Unbalanced opener — fall back to skipping just the opener line.
    start + 1
}

fn skip_code_fence(lines: &[&str], start: usize) -> usize {
    if !lines
        .get(start)
        .map(|l| l.trim().starts_with("```"))
        .unwrap_or(false)
    {
        return start;
    }
    for (i, line) in lines.iter().enumerate().skip(start + 1) {
        if line.trim().starts_with("```") {
            return i + 1;
        }
    }
    // Unbalanced fence — fall back to skipping just the opener.
    start + 1
}

fn is_pure_short_question(s: &str) -> bool {
    let t = s.trim();
    // Character count, not byte count — non-ASCII trailing questions
    // would otherwise sneak past the threshold.
    if t.chars().count() > 50 || !t.ends_with('?') {
        return false;
    }
    // "Statement. Question?" carries signal in the statement — keep.
    // `?` is single-byte ASCII so `t.len() - 1` is a safe boundary.
    let body = &t[..t.len() - 1];
    !body.contains(['.', '!'])
}
