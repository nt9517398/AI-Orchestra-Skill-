# Agent compatibility (Phase 2)

Built by `docs/diagnostics/agent_compat.py report` from the raw logs in `docs/agent-compat-logs/2026-11-24`. 0 model requests logged; last entry 2026-11-24T08:50:16.

## What Claude Code sends

Measured against a local recording server with Claude Code `2.1.331 (Claude Code)` and the model id `meta/muse-spark-1.3`.

| CLAUDE_CODE_MAX_OUTPUT_TOKENS | max_tokens sent | thinking sent | tools |
|---|---|---|---|
| (unset) | 32000 | {"type": "adaptive", "display": "omitted"} | 3 |
| 120000 | 120000 | {"type": "adaptive", "display": "omitted"} | 3 |
| 640000 | 128000 | {"type": "adaptive", "display": "omitted"} | 3 |

## Feature matrix

No model has been run through the live matrix yet.

## Rules

- Eligible means both gates pass: `tool_loop` and `claude_code_worker`.
- `caching`, `parallel_tools` and `thinking` are informational; they are not gates.
- One run per model unless the logs show more. A pass is not a reliability figure.
- `pin` checks the request is accepted with the registry's host pin; `partial` means the response did not say which host served it.
