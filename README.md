# sessionx

The framework-agnostic core for parsing agent **session logs** into two
complementary views:

- **Change timeline** — what files an agent changed and when, with
  syntax-tokenized diffs. (`Timeline`, `ChangeEvent`, `syntax`)
- **Event stream** — the agent's live session state: user/assistant turns,
  tool calls, stop reasons, stall/permission detection. (`WorkspaceEvents`,
  `EventSnapshot`, `StopReason`, `activity::events`)

Both read the same Claude Code session JSONL files
(`~/.claude/projects/<encoded-cwd>/<uuid>.jsonl`); the Codex and Pi variants
parse their respective formats on top of the shared event types.

`sessionx` has **no UI dependency** — rendering lives in the consumers. It is
the shared core behind [`wsx`](https://github.com/bakedbean/workspacex) and
[`chronox`](https://github.com/bakedbean/chronox).

## Layout

| Module | Concern |
|--------|---------|
| `extract`, `event`, `timeline` | Parse JSONL → `ChangeEvent`s → newest-first `Timeline` (Claude/Pi/Codex dialects) |
| `syntax` | Diff tokenizing + side-by-side layout (no ratatui) |
| `nav`, `config` | Navigation transitions and display-config resolution |
| `activity::events` | Tail Claude Code sessions → live `EventSnapshot` stream |
| `activity::codex_events`, `activity::pi_events` | Codex / Pi event-stream variants |
| `error` | Minimal `Error`/`Result` for the parsing entry points |

## License

MIT OR Apache-2.0.
