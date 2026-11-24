# Orchestra v21 upgrade plan: fleet-powered local workers

Status (2026-11-24): **the plan is replaced by the bare-minimum rebuild in *Revised direction* below; M0–M4 are done offline.** Earlier status: **Phases 0 and 1 are done; v20.1 (2026-11-23) repaired the baseline and did the offline part of Phase 1.5; v20.2 (2026-11-24) removed the routers and applied the fleet seats in code; v20.3 added three floating "latest" aliases (GPT Sol, Grok, Gemini Pro) as unvetted candidates. Phase 2 is started: the harness is built and tested and the Claude Code output-cap check is measured; the live model matrix has not been run. Phases 3 to 7 are not started.** Evidence: `FINDINGS-2026-11-23.md`.

## Revised direction (2026-11-24): bare-minimum Orchestra (v21-min)

**Status (2026-11-24): approved by the owner; M0–M4 are done offline (server pruned, `ask`, swarm, docs, 142 tests); M5 (live run) and M6 (release) wait for an OpenRouter key and the owner's local install.** Owner's answers: measurement off; git worktree isolation, public and test projects only; no Bash for fleet agents; swarm launched by a script. **Workflows, adversarial mode and memory are kept**, with their briefs and contracts. **Ratio:** 60–70% fleet agents, and 80–90% when a whole number cannot reach 60–70%.

**Owner's decision.** Orchestra is over-engineered. Rebuild it to the bare minimum around three uses, and nothing else:

1. **Ask.** Claude sends a **JSON brief**, rendered to an **XML envelope**, with an **output contract** checked on the reply (owner's decision, 2026-11-24: the structured request stays; free text is refused).
2. **Decide.** Claude runs closed questions that matter (dispatch or not, which answer is better, does a synthesis pass its checklist) through the decision bench: `~typesafe/jev-latest` + `~perplexity/pplx-decider-latest`.
3. **Swarm.** When Claude Code deploys subagents, **60–70% of them are driven by fleet models** and the rest by default Claude models.

This section supersedes Phases 3–7 above and the seat, promotion-rule and routing material under *Fleet redesign*. Those stay in the document as history. The full v20.5 is commit `7a95b10` (a git tag `v20.5-full` exists locally; pushing a tag was refused by the proxy, so the commit hash is the rollback point).

### Target fleet (unchanged from v20.5, plus the Jev alias and the Kimi pin of 2026-11-24)

| Model (wire id) | Class | Host | Typical use |
|---|---|---|---|
| `~openai/gpt-terra-latest` | US-CLOSED | unpinned | Review of code and logic; general reasoning |
| `~google/gemini-flash-latest` | GOOGLE-CLOSED | unpinned | Second reviewer; fast generation |
| `~x-ai/grok-latest` | XAI-CLOSED | unpinned | Worker and generator |
| `qwen/qwen4-max` | CN-OW | unpinned | Lead worker: reasoning and coding |
| `tencent/hy4` | CN-OW | unpinned | Drafts and bulk |
| `moonshotai/kimi-k3.1` | CN-OW | `wafer/us` | Math and code where a check grades it |
| `meta/muse-spark-1.3` | META-CLOSED | Meta | Long-context work |
| `meta/muse-ember` | META-CLOSED | Meta | Cheap subtasks and drafts |
| `glm-5.3` (direct API) | CN-OW | Zhipu | Software engineering |
| `deepseek-v4-pro` (direct API) | CN-OW | DeepSeek | Algorithmic code |
| `deepseek-flash` (direct API, fleet id `deepseek-v4.1-flash`) | CN-OW | DeepSeek | Cheapest drafts; never checks facts |
| Bench: `~typesafe/jev-latest` | US-CLOSED | OpenRouter decisions | Gatekeeper, ≤ 32K state |
| Bench: `~perplexity/pplx-decider-latest` | CN-OW (inference) | OpenRouter decisions | Screener, long states up to 262K |

The Typical use column is advice in the docs, not code: no seats, no routing prior and no role enforcement. Class labels stay, as a single field shown by `list_fleet`, because the one rule worth keeping is "a check should come from a different class than the work it checks".

### What stays, what goes

Line numbers refer to `orchestra_mcp_server.py` at v20.5 (4,353 lines). Result: 2,950 lines (the owner's keeps account for the gap to the first estimate of about 1,000).

| Part (v20.5) | Decision | Why |
|---|---|---|
| Key loading, `_CONFIGS`, generator registry (`_MODELS`: provider, class, max_out, pin, api_id, served_as, auto_ceiling) | **Keep**, minus `_SEATS`, `_UNVETTED`, `_NO_VERIFY` | Without them no call can be made safely |
| Streaming `_call` with the silence rule, provenance (`[SUBSTITUTED]`), host pinning, failure tags (`[ERROR]`, `[TIMEOUT]`, `[EMPTY]`, `[TRUNCATED]`) | **Keep** | They stop a wrong model, a wrong host or a cut-off reply from passing as a real answer |
| Live output ceilings (`_live_ceiling`) for `~` aliases and unpinned models | **Keep** | A typed-in ceiling above the host's makes every call fail |
| Decision transport (`_decide_raw`), `decide`, `decide_panel` (two classes), polarity canary in `fleet_check` | **Keep** | This is the bench |
| `decide_compare` | **Dropped (done)** | One `decide` with a `choice` question asked in both orders does the same job |
| Error-body capture, key redaction, save-on-return (`results/<task_id>/`), `recover_results` | **Keep** | They fixed real losses (D11: a finished answer lost to a killed run) |
| Async jobs (`orchestra_start` → `check_job`) | **Keep, simplified** to one start tool | Claude Desktop cuts tool calls at roughly 60–240 s, and deep calls take longer |
| `fleet_check` and `python orchestra_mcp_server.py --check` | **Keep** | The only way to verify slugs, ceilings and hosts against the live service |
| Call log (`dispatch_log.jsonl`: model, served model, host, tokens, latency, error) | **Keep, automatic only** | Cheap; it is the evidence when something goes wrong |
| Structured briefs, contracts, roles (`work_type` is now only a log tag) | **Keep**, for workflows, adversarial mode and memory | Owner's decision. Plain `ask` does not use them |
| Workflow engine (`workflow_start`) | **Keep** | Owner's decision |
| Adversarial mode (`orchestra_start(mode="adversarial")`, `orchestra_adversarial`, `orchestra_parallel`) | **Keep** | Owner's decision |
| Dynamic fleet (`fleet_add`, `fleet_remove`, guests, `fleet_probe`; ~535–710, 2817–2969) | **Drop** | Switched off since v18.1; a model joins by a registry line |
| Memory bank (`memory_update`, `memory_read`, `memory_review`) | **Keep** | Owner's decision |
| Routing (`route_evidence`, `route_decide`, router record; ~3619–4004) | **Drop** | Claude picks the model; the docs give advice |
| Outcome measurement (`log_outcome`, `log_override`, `fleet_stats`) | **Drop** (see open decision 1) | It only paid off through routing, which goes |
| Seats, `_UNVETTED`, promotion rule, `_ROUTE_BLURB` | **Drop** | No roles to enforce |

### Tool surface (as built)

| Tool | Does |
|---|---|
| `ask_start(models, brief)` | The same validated brief, XML-rendered, to 1–6 models in the background; each reply contract-checked. (`call_model` is the quick single-model call.) |
| `check_job(job_id)` | Poll any background job |
| `decide`, `decide_panel` | The bench: Jev (`~typesafe/jev-latest`) and the Decider (`~perplexity/pplx-decider-latest`) |
| `call_model`, `orchestra_start`, `orchestra_parallel`, `orchestra_adversarial`, `workflow_start` | The structured path, kept |
| `memory_update`, `memory_read`, `memory_review` | Kept |
| `swarm_plan(n_agents, tasks, repo)` | The fleet/Claude split, a model per fleet agent, the launch command |
| `list_fleet`, `fleet_check`, `recover_results` | Availability, live verification, what a killed run left (no task id: unfinished calls) |

Removed: `fleet_probe`, `fleet_add`, `fleet_remove`, `decide_compare`, `route_evidence`, `route_decide`, `log_outcome`, `log_override`, `fleet_stats`.

### The swarm

**How a fleet agent runs.** A Claude Code subagent started with the Agent tool always uses the session's own Claude model, so it cannot be driven by a fleet model. A fleet agent is therefore a separate headless Claude Code process, `claude -p "<task>"`, with these variables, which send its API calls to OpenRouter's Anthropic-compatible endpoint:

```
ANTHROPIC_BASE_URL=https://openrouter.ai/api
ANTHROPIC_AUTH_TOKEN=<OpenRouter key>
ANTHROPIC_API_KEY=""
ANTHROPIC_MODEL=<wire id, e.g. qwen/qwen4-max>
CLAUDE_CODE_MAX_OUTPUT_TOKENS=<registry max_out, at most 120000>
```

Claude agents are ordinary subagents from the Agent tool, on default Claude models.

**The ratio rule (owner, 2026-11-24).** For N agents, the fleet count F is a whole number giving 60–70% fleet agents; if none exists, 80–90%; if neither, the share nearest 65%. At least one agent is always Claude. A single agent is not a swarm: Claude decides.

| N | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fleet | 1 | 2 | 3 | 3 | 4 | 6 | 5 | 6 | 7 | 7 | 8 |
| Share | 50% | 67% | 75% | 60% | 67% | 86% | 63% | 67% | 70% | 64% | 67% |

N = 2 and 4 reach neither band; N = 7 uses 80–90%. A task marked `claude_only` always goes to a Claude agent, even if that lowers the share.

**Which task goes where.** Fleet agents take bounded tasks with a check: tests to pass, a schema to meet, a file to produce. Claude agents take the tasks that need judgment, repository-wide context or anything sensitive, and they check the fleet agents' work. A fleet agent's output counts only after its check passes (L1); review by a Claude agent or a fleet model of another class comes second.

**Which fleet model.** `swarm_plan` assigns by task type from the table above, and spreads agents across classes when tasks are similar, so that one model's blind spot does not repeat across the swarm.

**Launcher.** `orchestra_swarm.py` (stdlib only) starts the fleet agents named by `swarm_plan`. Each agent gets its own git worktree of `HEAD` (tracked files only), the variables above, file tools only, a turn cap and a time limit. Claude Code has no spend flag, so the spend limit is an OpenRouter key limit. The launcher collects each agent's diff, exit status and cost into `results/<task_id>/`, so a killed swarm keeps what finished. Claude reviews each diff and applies it; fleet agents never write to the real checkout.

**Eligibility.** Only models that pass the Phase 2 harness (`docs/diagnostics/agent_compat.py`: tool loop and a real Claude Code worker) should drive a swarm agent. `report --eligible-json swarm_eligible.json` exports them and `swarm_plan` reads that file; without it every OpenRouter model is offered with a warning.

**Data rule (owner's decision 5, unchanged).** Fleet agents run on test and public projects only until isolation is decided (open decision 2).

### Skill documents

`SKILL.md` is now 73 lines (was about 270): Gate 0, Ask, Decide, Swarm, a short verification ladder, failure tags and an index. The package has six references: `fleet-card.md` (rewritten), `decision-bench.md` (trimmed), `swarm.md` (new), `dispatch-protocol.md`, `verification.md` and `memory.md` (patched for removed tools). `governance.md`, `evidence-base.md`, `research-alignment.md` and `dynamic-fleet.md` are removed from the package and stay in commit `7a95b10`.

### Phases

| # | Do | Exit check |
|---|---|---|
| M0 | Record v20.5 as the rollback point | **Done.** Commit `7a95b10` (165 tests); the tag could not be pushed |
| M1 | Owner approves this plan and answers the open decisions | **Done 2026-11-24**, see decision 15 |
| M2 | Prune the server to the tool surface above (**done**: 4,353 → 2,950 lines, 142 tests) | All new tests pass offline. Kept behaviour has a test each: provenance, host pin, silence timeout, failure tags, live ceiling, save-on-return, error body, redaction, panel class rule, polarity canary, `--check` exit codes 0/1/2. No network in tests |
| M3 | `swarm_plan` and `orchestra_swarm.py` (**done**, 7 tests, mutation-checked) | Offline tests with a fake `claude` binary: the ratio table, worktree per agent, variables set and the key never logged, caps enforced, results kept after a kill, ineligible models skipped |
| M4 | Rewrite `SKILL.md` and the references; rebuild `orchestra.skill` (**done**, 7 files) | Packaging tests (description ≤ 1,004 characters, every file indexed, every registry id in the fleet card); a fresh session asked to "fan this out" plans a 60–70% swarm from the skill alone |
| M5 | Live run, with an OpenRouter key in the `ai-orchestra-skill` environment | `fleet_check` gives every model OK or a named problem; the Phase 2 matrix decides swarm eligibility; one real swarm on a public repo finishes with diffs Claude applied after tests passed; real spend within the cap |
| M6 | Release: merge to `main`, update the owner's local copy, test rollback | `orchestra.skill` and the server installed locally; switching back to `v20.5-full` works |

M0–M4 needed no key. M5 cannot be done without one; M6 needs the owner's machine.

### Risks

| Risk | Control |
|---|---|
| Rules that moved from code to docs (class-matching in workflow plans, "a model grading its own output") | Stated in the docs; the server still refuses the barred roles, same-class adversarial pairs and same-class panels |
| A fleet agent reads or leaks something it should not | Worktree only, scrubbed environment, test and public projects only; container isolation is open decision 2 |
| OpenRouter's Anthropic endpoint or a model mishandles Claude Code's tool use | Phase 2 harness is the gate; ineligible models never drive an agent |
| A floating alias changes model under a running swarm | Served model logged per call; `fleet_check` before a large swarm |
| Unverified slugs (Terra, Gemini Flash, Kimi K3.1, the `~` decision aliases) | `fleet_check` in M5 names any that do not exist |
| Runaway spend | Per-agent spend cap in the launcher; an OpenRouter key limit as a backstop |

### Decisions (answered 2026-11-24)

1. Track whether each model's answers were right: **no** for now; the automatic call log is enough.
2. Isolation for fleet agents: **a git worktree** with a scrubbed environment, public and test projects only. Docker later if needed.
3. Bash for fleet agents: **off** until isolation is decided.
4. Swarm launch: **a script** that Claude runs with Bash (`orchestra_swarm.py`), not the MCP server.
5. Keep **workflows, adversarial mode and memory**.
6. Ratio: **60–70%**, else **80–90%**.

### Still open

- Run the live matrix and a real swarm (M5) with an OpenRouter key and a spend limit on it.
- Confirm the unchecked slugs (Terra, Gemini Flash, Kimi K3.1, Qwen 4 Max, Tencent HY4) and whether the decisions endpoint accepts the `~` aliases.
- Install on the owner's machine: copy `orchestra_mcp_server.py` **and** `orchestra_swarm.py`, `requirements.txt`, `.env`; upload the new `orchestra.skill`; merge the branch to `main` (M6).

## Objective (original v21 plan; see *Revised direction* above)

Let Orchestra run isolated Claude Code workers whose brain is a fleet model, while Claude stays conductor, so that file-level work can move off the Claude usage limit without losing Orchestra's verification, cost control or data safety.

## Goals and measures

| # | Goal | Measure |
|---|---|---|
| G1 | Trustworthy baseline | 124 of 124 tests pass; v20 committed; `.env` ignored |
| G2 | A worker is one Orchestra call | `agent_start` → `check_job` returns diff, test result, tokens, real cost, served model and host |
| G3 | Workers cannot reach secrets or the real repo | All escape tests in Phase 3 pass |
| G4 | Spend is capped by real cost | No test run exceeds its cap by more than one request |
| G5 | Routing rests on measured results | At least 20 graded tasks per candidate model in `route_evidence` |
| G6 | Worth using | On the benchmark, a fleet worker with a Claude or cross-class review matches the agreed pass rate at lower Claude usage |

**Out of scope for v21:** replacing the main Claude session, a gateway for all traffic, workers spawning workers, fleet workers on search, quotes, citations or gate words, and enabling the dynamic fleet.

## Source of truth

Build from this folder, `C:/Users/nguyen/orchestra-mcp` (v20). The v15 copy in `W:/Claude Skills/02-Skills-Library/Orchestration-MultiAgent/Orchestra-files` is not a source.

## Phases

Each phase must pass its exit check before the next starts.

### Phase 0: Secure and baseline (done 2026-11-23, see findings)

**Do:** restore `.gitignore`; review and commit v20 on a branch; make the test runner report every failure; diagnose the 3 decision-panel failures first (a possible safety-guard regression); then update the 6 stale Muse ceiling tests and resolve the missing `fleet_extra.json`. A test changes only after confirming the server behaviour is the intended one.

**Verify:**
- `git check-ignore .env` prints `.env`; `.env` appears in no commit.
- `python test_orchestra.py` reports 124 of 124, exit 0.
- A deliberately broken test makes the runner exit non-zero and still run the rest.

### Phase 1: Repair the fleet (done 2026-11-23, see findings)

**Do:** (Decider swap to `perplexity/pplx-decider-v1.2` done 2026-11-23.) (Default panel changed to Jev + Decider v1.2 on 2026-11-23; finding D10.) Decide whether Solar stays in the router chain; add a polarity canary to `fleet_check`; make `fleet_check` make a real call for decision models; decide Muse's output ceiling; pin Muse's host (owner chooses Meta, Google, or an allowed set).

**Verify:**
- `fleet_check` reports every model OK with one real call each.
- `route_decide` works with the first router disabled.
- 10 Muse calls all report the pinned host.

### Phase 1.5: Reliable multi-model runs

Why: on 2026-11-23 a two-model review lost a finished GPT Astra answer when a 590 s limit killed the process while GLM was still working (finding D11). Workers in Phase 4 would inherit this failure.

**Do:**
1. Save each model's result as soon as it finishes, in `orchestra_parallel` and `fleet.py`. A kill or slow model loses only its own half.
2. Write a `started` log row before each call, so a killed or hung call is visible. `fleet_stats` must not count it as a success or lose it.
3. Keep the provider's status and error text in the log and in `fleet_check` rows, with secrets removed.
4. Base caller time limits on measured model speed, or rely on the silence rule alone. Document that a wrapper must not hide the exit code behind a pipe.
5. Reproduce the GLM HTTP 400 with the saved v5.7 review brief and diff, find the cause, and fix or document it.

**Verify:**
- Kill a two-model run after the fast model finishes: its answer is on disk, intact, and the other model is marked incomplete.
- A killed call appears in the log as started and unfinished.
- A forced provider 400 shows its status and error text, and a planted key never appears in the log.
- `fleet.py` exits non-zero when any model fails or is missing.
- The GLM 400 has a confirmed cause, and the same review either succeeds or fails with a clear message.
- All tests pass.

**Status (2026-11-23, v20.1).** Done offline and tested: items 1 to 3 (every generator call logs a `started` row and saves its reply at once; `recover_results` reads them back; provider error text is kept, redacted), and the measured-latency half of item 4 (`fleet_stats` per-model `elapsed_s`; the wrapper rule is in `dispatch-protocol.md`). **Not done:** item 5 needs live access and the saved v5.7 brief and diff, and `fleet.py` is not in this repository, so its non-zero exit cannot be fixed or tested here. One cause of the unexplained 400 is found: the error body was read after the stream had closed, so every provider error arrived with no text. That explains the missing text, not the 400 itself.

### Phase 2: Compatibility matrix

**Do:** run the same small script for each fleet model through the Messages endpoint: forced tool call, multi-turn tool loop, parallel tools, streaming, caching, thinking blocks, a real Claude Code worker.

**Verify:** a committed `docs/agent-compatibility.md` with pass or fail per feature and model, from raw logs. Only models passing the tool loop and the worker test become eligible.

**Status (2026-11-24).** Built: `docs/diagnostics/agent_compat.py` runs the matrix (basic, host pin, forced tool, tool loop, parallel tools, streaming, caching, thinking, and a real Claude Code worker) through the Messages endpoint, logs every exchange to JSONL without the key, and `report` rebuilds `docs/agent-compatibility.md` from the logs alone. It is tested offline against a fake Messages server, and every guard in it was checked by breaking it. Measured now, with no key: what Claude Code sends (`docs/agent-compat-logs/2026-11-24/`). **Not run:** the live matrix. It needs OpenRouter reachable from the session and a key; this session had neither, so no model has a verdict yet and none is eligible.

To run it in a session on the `ai-orchestra-skill` environment (created 2026-11-24 so the key stays off the shared Default): approve the OpenRouter credential when asked, then `python docs/diagnostics/agent_compat.py run --out docs/agent-compat-logs/<date> --worker`, then `report`. A per-model request cap (default 60) bounds the spend; set an OpenRouter key limit of about $2 as well.

### Phase 3: Isolation

**Do:** run workers in a disposable Docker container holding only a copy of the task's files; a scrubbed environment; network limited to the model endpoint; a short-lived OpenRouter key with its own spend limit if available; results returned as a diff that Claude applies.

**Verify, with a hostile test worker that tries to:**
- read `.env`, `~/.claude`, or a planted canary file outside the task;
- print environment variables and find anything but its own limited credential;
- reach another network host;
- change the host repo directly;
- follow a prompt injection planted in a task file.

Every attempt must fail and the canary must appear in no output or provider request log. **Bash stays disabled until this phase passes.**

### Phase 4: Worker tool

**Do:** add `agent_start(model, brief, files, allowed_tools, max_turns, max_usd, test_cmd)`, built on the existing job layer. Reuse `check_job`, `[TAG]` failures, provenance and `log_outcome`. Add a `worker` role and cost, host, turns and requests to the dispatch log. Take real cost from OpenRouter's response or generation record.

**Verify (mocked and live):**
- Replays the `median()` proof of concept with a passing test.
- Timeout, crash, turn cap and budget cap each return a tagged failure and leave no process or container.
- Reported cost is within 5% of the OpenRouter log.
- A wrong served model returns `[SUBSTITUTED]`.
- An unchanged or failing result is never reported as success: Orchestra runs `test_cmd` itself.

### Phase 5: Worker and verifier loop

**Do:** worker → deterministic test → cross-class read-only reviewer → accept or revise, at most 3 worker turns. Log outcomes on the L1 basis. Allow parallel workers, starting at 2, only on non-overlapping files.

**Verify:**
- Seeded-bug tasks: the loop rejects a solution that passes tests but breaks the spec.
- Two parallel workers produce no file conflicts and stay under a combined cap.
- `route_evidence` shows the new outcomes.

**Gap to close in this phase.** Workflow plans do not check classes: `_validate_workflow` has no class comparison, so a verifier step can read a same-class worker's output. The class rule is enforced for adversarial pairs and decision panels only. Add the check for the verifier role here.

### Phase 6: Benchmark and decision

**Do:** a frozen set of about 20 tasks from easy to hard, with hidden tests, written before results are seen. Compare Claude subagent, each eligible fleet worker alone, and fleet worker plus reviewer.

**Report:** pass rate, cost, time, tool errors, human fixes needed, Claude usage.

**Verify:** thresholds are fixed before the run. The result decides per work type: adopt, adopt with review, or reject. A negative result is reported as one.

### Phase 7: Skill and release

**Do:** update `SKILL.md`, fleet card, dispatch and governance references; add the worker role, data rules and failure handling; package v21.

**Verify:** full tests and `fleet_check` pass; the packaged skill matches the source; a fresh session follows the skill to run a worker correctly from a plain request; rollback to the v20 tag is tested.

## Fleet redesign (added 2026-11-23)

Roles applied in code and docs on 2026-11-24 (v20.2); routers removed the same day. Live checks on 2026-11-23 were one small call per route, so they show availability, not quality.

### Live check

| Model | $/M in / out | Context | Messages API + tool call | Observed |
|---|---|---|---|---|
| `meta/muse-spark-1.3` | 1.25 / 4.25 | 1.05M | Pass | Hosts: Google, Meta |
| `meta/muse-ember` | 0.10 / 0.50 | 1.05M | Pass | Hosts: Google, Meta |
| `ibm-granite/granite-4.2-8b` | 0.06 / 0.15 (CoreWeave) | 131K | Pass | Hosts: CoreWeave, DeepInfra |
| GPT Astra | 8 / 40 listed | 1.05M | Not retested | Removed from the fleet 2026-11-24 |

### Decision (2026-11-24): no routers

The owner removed the routers. `openrouter/auto` is out of the fleet and `call_router` is deleted (v20.2); `openrouter/auto-code` and `openrouter/fusion` were never wired in and are dropped from the plan. This replaces the 2026-11-23 decision that routers would assist the then chief reviewer, GPT Astra, who has since been removed too. The reasons recorded then still describe what the routers would have cost: Orchestra's verification depends on knowing **which model answered and that it differs in class from the worker**, and a router chooses that at run time, so:

- it can pick the worker's own family, silently voiding a cross-class check;
- the `[SUBSTITUTED]` provenance guard cannot tell a legitimate route from a wrong model unless it checks against an allowlist;
- `route_evidence` learns nothing about a moving target unless each result is logged under the served model;
- Fusion can answer with Claude, so it is not independent of a Claude-authored synthesis, and it uses web search on the prompt.

### Decision (2026-11-24): Astra removed; roles from the owner's table

The owner removed **GPT Astra, `openrouter/auto` and `openrouter/auto-code`**, and approved the roles below. Applied in v20.4: registry, seats, routing prior, default pair, tests, the Phase 2 harness and the packaged docs. The three floating aliases from v20.3 stay as the models for Sol, Grok and Gemini (the owner chose `~…-latest` over fixed version numbers).

### Target roles (applied 2026-11-24: `_SEATS` in code, `list_fleet`, and the seat tables in `SKILL.md` and `governance.md`)

| Seat | Model | Class | Rule |
|---|---|---|---|
| Conductor | Claude | US-CLOSED | Unchanged |
| **Chief reviewer (code and logic)** | `~openai/gpt-sol-latest` | US-CLOSED | Reviews as a `critic`. Candidate: no routing prior, no factual role |
| Second reviewer and generator | `~google/gemini-pro-latest` | GOOGLE-CLOSED | Its own class, so Sol and Gemini can review each other. Candidate |
| Worker and generator | `~x-ai/grok-latest` | XAI-CLOSED | Its own class. Candidate |
| Lead worker (reasoning and coding) | `qwen/qwen4-max` | CN-OW | Same class as DeepSeek and GLM. Candidate. Slug not yet checked against OpenRouter |
| Worker (drafts and bulk) | `tencent/hy4` | CN-OW | Same class as DeepSeek and GLM. Candidate. Slug not yet checked against OpenRouter |
| Lead long-context worker | Muse Spark 1.3 | META-CLOSED | Long-context agentic and coding work. Pinned host, worker output cap |
| Efficient worker | Muse Ember | META-CLOSED | Repeated tool calls, subtasks, extraction, first drafts |
| Math and code, only where a check grades it | Granite 4.2 8B | US-OW | Never without a deterministic check |
| GLM-5.3, DeepSeek V4 Pro, DeepSeek V4.1 Flash | unchanged | CN-OW | Lead engineer and memory keeper; Algorithmist; Drafter |

**Reviews.** A worker is reviewed by a model from a different class. Sol and Gemini can review each other because Google has its own class. Qwen and Tencent share a class with GLM and DeepSeek, so none of those can review another. Spark and Ember share a class and never check each other.

**Starting status.** Sol, Gemini, Grok, Qwen and Tencent start as candidates: no routing prior, no factual role. The prior is now GLM-5.3 and DeepSeek only, with no cross-class partner named, and the default pair is GLM-5.3 + Sol.

**Consequences the owner should see.**

1. **No cross-class factual verifier.** Astra was the only generator outside CN-OW with a factual role. A factual flag on DeepSeek or GLM output can now be confirmed only by a fetched source or a check (L1); otherwise it ships labelled unconfirmed. The owner can lift this by moving a candidate out of `_UNVETTED` after a measured fabrication record, which needs the Phase 6 benchmark.
2. **Sol reviews as a `critic`, not a `verifier`.** The server refuses a verifier brief to an unvetted model. Phase 5's worker and verifier loop therefore gates on the L1 check, with Sol's defect list as a flag, until Sol is vetted.
3. **Research authorship moved to Claude** (reviewed by Sol), because the "lowest hallucination rate" reason for Astra authoring it no longer has a holder.
4. **Qwen and Tencent facts are unverified.** The slugs come from the owner's table. Nothing was read from OpenRouter (no key in the build environment): `fleet_check` or `fleet_probe` will show whether the slugs exist, their ceilings, prices and hosts.
5. **Whole non-CN-OW fleet sits behind one aggregator**, as before, but now there is no direct-API generator outside CN-OW at all.

### Promotion rule: how a model can take the chief reviewer seat

Another model may take Sol's reviewer seat for a work type only if, on the Phase 6 frozen benchmark:

1. it catches at least as many seeded defects as Sol, within a margin fixed before the run;
2. its false-accept rate is no higher than Sol's;
3. it has at least 20 graded verdicts in that work type;
4. a measured fabrication check clears the `_UNVETTED` restriction;
5. the reviewer is still a different class from the worker.

Sol stays as the fallback until all five hold. (This rule used to name Muse Spark and Astra; the structure is unchanged.)

### Added work

- **Phase 1:** register Ember and Granite as unvetted. (Done. The router work that was listed here was removed.)
- **Phase 2:** add Ember and Granite to the worker compatibility matrix.
- **Phase 2:** add Qwen and Tencent to the worker compatibility matrix (done in the harness; not yet run).
- **Phase 6:** add arms for Spark, Ember, Granite, Qwen, Tencent, Grok, and Spark + Sol review, and a fabrication test for each candidate that might hold a factual role.

## Release gates

v21 ships only when all are true:

- [ ] All tests pass and v20 is recoverable from git
- [ ] No canary leak in the Phase 3 tests
- [ ] Budget caps hold against OpenRouter's billing record
- [ ] Every failure mode returns a tag, never false success
- [ ] Benchmark report written, including negative results
- [ ] A killed multi-model run keeps every finished result
- [ ] Lab rules intact: no fleet worker on search, quotes, citations or gate words

## Main risks

| Risk | Control |
|---|---|
| Code or secrets sent to a provider | Container with task files only; pinned hosts; no `.env` |
| Runaway agent loop | Turn cap, real-cost cap, provider key limit |
| Weak tool use in long runs | Phase 2 matrix, Phase 6 benchmark |
| A Claude Code update breaks the route | Pin the worker's version; run the compatibility check on upgrade |
| Fleet quality oversold | Deterministic tests decide; agreement is not verification |
| Provider terms on automated or proxied use | Review each provider's terms before Phase 4 |

## Decisions from the owner

All answered 2026-11-23:

1. Decider replaced by `perplexity/pplx-decider-v1.2`. Done.
2. Muse host: **Meta**.
3. Muse worker output cap: **at least 120,000 tokens**. Set 120,000. Claude Code reported a 32,000 output limit in the proof of concept, so Phase 2 must confirm a worker really reaches 120,000 (for example through `CLAUDE_CODE_MAX_OUTPUT_TOKENS`). One full response costs about $0.51.
5. Code leaving the machine: **test and public projects only** at first.
6. Commit v20 as baseline once `.gitignore` is restored: **yes**.
7. Astra stays reviewer until Muse earns promotion: **yes**. *Superseded 2026-11-24: Astra was removed; Sol is chief reviewer under the same promotion rule.*
8. `openrouter/auto`: **allowlist only**. *Superseded 2026-11-24: the routers were removed.*
9. Fusion: **opt-in, public content only**. *Superseded 2026-11-24: Fusion is out of the plan.*

4. Spend limits during testing: **$2 per worker, $10 per day**. Enforced from OpenRouter's reported cost, and replaced with measured limits after the Phase 6 benchmark.

10. Reviewer of Muse workers: **Astra** (answered 2026-11-23; *superseded 2026-11-24: now Sol*). Because the class rule would have refused a Muse and Astra pair, Meta models became their own class, `META-CLOSED` (v20.1). Muse Spark and Muse Ember share it and still cannot check each other.

Phase 2 note on decision 3 (measured 2026-11-24, Claude Code 2.1.331): a worker sends `max_tokens` 32,000 for an unrecognised model id; with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000` it sends 120,000, so the 120,000 cap is reachable. A request for 640,000 is clamped to 128,000, so Muse Spark's 640,000 registry ceiling can never apply to a worker.

11. Floating "latest" aliases (answered 2026-11-24): use `~openai/gpt-sol-latest`, `~x-ai/grok-latest` and `~google/gemini-pro-latest` instead of fixed version numbers. Added as unvetted candidates (v20.3). Open: pin each to a host once `fleet_check` shows who serves it; whether any of them should be its own class like Meta.

12. Roles table (answered 2026-11-24): Astra, `openrouter/auto` and `openrouter/auto-code` removed; Sol chief reviewer, Gemini second reviewer and generator, Grok worker and generator, Qwen lead worker, Tencent worker, Spark lead long-context worker, Ember efficient worker, Granite math and code where a check grades it; Google and xAI are their own classes; the new five start as candidates. Applied in v20.4. Open: confirm the Qwen and Tencent slugs, pin hosts, and decide whether Sol should be allowed any factual role.

13. **Simpler fleet (answered 2026-11-24, v20.5).** Solar Decide removed; the bench is Jev + `~perplexity/pplx-decider-latest`. GPT Sol replaced by `~openai/gpt-terra-latest`; Gemini Pro by `~google/gemini-flash-latest`; IBM Granite by `moonshotai/kimi-k3.1` (CN-OW, same seat). Grok, Muse Spark, Muse Ember, Tencent HY4 and Qwen 4 Max unchanged. The swarm pattern (Claude Code subagents on fleet models through OpenRouter) is in `SKILL.md`, marked unverified. Open: the Kimi K3.1, Terra and Flash slugs are not checked against OpenRouter, and whether the decisions endpoint accepts the Decider's `~` alias is unknown; the live run answers both.

14. Decision bench aliases (answered 2026-11-24): `~typesafe/jev-latest` replaces `typesafe/jev-1.13`; with `~perplexity/pplx-decider-latest` it forms the bench. Kimi K3.1 host: chosen by Claude from the owner's three options as `wafer/us`, because the other two are 4-bit quantized by name and the seat is math and code. Bare-minimum rebuild approved in principle, plan first: see *Revised direction*.

15. Rebuild go-ahead (answered 2026-11-24): proceed with the bare-minimum rebuild on the recommendations above, keeping workflows, adversarial mode and memory, with the 60–70% / 80–90% ratio. Done offline as M0–M4 (commits after `7a95b10`). Jev is `~typesafe/jev-latest`; Kimi K3.1 is pinned to `wafer/us`.

16. Requests stay structured (answered 2026-11-24): JSON brief + XML envelope + output contract is how Claude sends every request. The plain `ask` tool was removed; `ask_start` takes a brief.
