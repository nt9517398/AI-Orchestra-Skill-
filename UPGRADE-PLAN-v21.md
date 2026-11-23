# Orchestra v21 upgrade plan: fleet-powered local workers

Status: **draft for the owner's review. Nothing is implemented.** Evidence: `FINDINGS-2026-11-23.md`.

## Objective

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

### Phase 2: Compatibility matrix

**Do:** run the same small script for each fleet model through the Messages endpoint: forced tool call, multi-turn tool loop, parallel tools, streaming, caching, thinking blocks, a real Claude Code worker.

**Verify:** a committed `docs/agent-compatibility.md` with pass or fail per feature and model, from raw logs. Only models passing the tool loop and the worker test become eligible.

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

### Phase 6: Benchmark and decision

**Do:** a frozen set of about 20 tasks from easy to hard, with hidden tests, written before results are seen. Compare Claude subagent, each eligible fleet worker alone, and fleet worker plus reviewer.

**Report:** pass rate, cost, time, tool errors, human fixes needed, Claude usage.

**Verify:** thresholds are fixed before the run. The result decides per work type: adopt, adopt with review, or reject. A negative result is reported as one.

### Phase 7: Skill and release

**Do:** update `SKILL.md`, fleet card, dispatch and governance references; add the worker role, data rules and failure handling; package v21.

**Verify:** full tests and `fleet_check` pass; the packaged skill matches the source; a fresh session follows the skill to run a worker correctly from a plain request; rollback to the v20 tag is tested.

## Fleet redesign (added 2026-11-23)

Proposed, not implemented. Live checks on 2026-11-23 were one small call per route, so they show availability, not quality.

### Live check

| Model | $/M in / out | Context | Messages API + tool call | Observed |
|---|---|---|---|---|
| `meta/muse-spark-1.3` | 1.25 / 4.25 | 1.05M | Pass | Hosts: Google, Meta |
| `meta/muse-ember` | 0.10 / 0.50 | 1.05M | Pass | Hosts: Google, Meta |
| `ibm-granite/granite-4.2-8b` | 0.06 / 0.15 (CoreWeave) | 131K | Pass | Hosts: CoreWeave, DeepInfra |
| `openrouter/auto` | Served model's price | 2M | Pass | Honoured a Muse/Granite allowlist 3 of 3 times |
| `openrouter/auto-code` | Served model's price | 2M | Pass unrestricted | **503 with the Muse/Granite allowlist**; unrestricted it chose `openai/gpt-6.2-sol` |
| `openrouter/fusion` | Varies | 1M | **Fail (400)**; listed without tools | Chat call was answered by `anthropic/claude-opus-5.5` |
| GPT Astra (current) | 8 / 40 listed | 1.05M | Not retested | The registry comment still says $10 / $50 |

### Decision (accepted 2026-11-23): routers assist Astra; they do not replace it yet

Orchestra's verification depends on knowing **which model answered and that it differs in class from the worker**. A router chooses that at run time, so:

- it can pick the worker's own family, silently voiding a cross-class check;
- the `[SUBSTITUTED]` provenance guard cannot tell a legitimate route from a wrong model unless it checks against an allowlist;
- `route_evidence` learns nothing about a moving target unless each result is logged under the served model;
- Fusion can answer with Claude, so it is not independent of a Claude-authored synthesis, and it uses web search on the prompt.

### Target roles

| Seat | Model | Rule |
|---|---|---|
| Conductor | Claude | Unchanged |
| **Lead worker candidate** | Muse Spark 1.3 | Long-context agentic and coding work, deep drafting. Pinned host, worker output cap |
| Efficient worker | Muse Ember | Repeated tool calls, subtasks, extraction, first drafts |
| Math / code specialist candidate | Granite 4.2 8B | Only where a deterministic check grades it. Possible new correlation class, to be confirmed from its lineage |
| Chief reviewer | **GPT Astra, for now** | Verifier and high-stakes review |
| Everyday router | `openrouter/auto` **with an allowlist** | Low-stakes generator work only, never verification |
| Coding router | `openrouter/auto-code` | Disabled until it works with an approved allowlist, or for public code only |
| Optional second opinion | `openrouter/fusion` | Explicit request, public content only. Never a verifier or worker |

Muse Spark and Ember share a family, so neither may verify the other. Neither takes a factual or verifier role until measured.

### Promotion rule: how Muse can replace Astra

Muse Spark may take Astra's reviewer seat for a work type only if, on the Phase 6 frozen benchmark:

1. it catches at least as many seeded defects as Astra, within a margin fixed before the run;
2. its false-accept rate is no higher than Astra's;
3. it has at least 20 graded verdicts in that work type;
4. a measured fabrication check clears the `_UNVETTED` restriction;
5. the reviewer is still a different class from the worker. If Muse is reviewer, the worker cannot be Muse.

Astra stays as the fallback until all five hold.

### Added work

- **Phase 1:** register Ember and Granite as unvetted; add a `router` entry type with an allowlist; log the served model and host for every routed call; refuse any router in a verifier, ratification or factual role.
- **Phase 2:** add Ember, Granite and both routers to the worker compatibility matrix.
- **Phase 4:** a worker may use `openrouter/auto` only with an allowlist.
- **Phase 6:** add arms for Spark, Ember, Granite, allowlisted Auto, and Spark + Astra review.

**Verify:**
- A router result from outside the allowlist returns `[SUBSTITUTED]`.
- A request for a router as verifier is refused by the server, in tests.
- 20 routed calls all log a served model inside the allowlist.
- `route_evidence` credits the served model, never the router's name.

## Release gates

v21 ships only when all are true:

- [ ] All tests pass and v20 is recoverable from git
- [ ] No canary leak in the Phase 3 tests
- [ ] Budget caps hold against OpenRouter's billing record
- [ ] Every failure mode returns a tag, never false success
- [ ] Benchmark report written, including negative results
- [ ] No router can occupy a verifier, ratification or factual seat
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
7. Astra stays reviewer until Muse earns promotion: **yes**.
8. `openrouter/auto`: **allowlist only**. *Owner's reply was numbered 7 ("Yes"); read as this question, to be confirmed.*
9. Fusion: **opt-in, public content only**. *Owner's reply was numbered 8 ("Opt-in"); read as this question, to be confirmed.*

4. Spend limits during testing: **$2 per worker, $10 per day**. Enforced from OpenRouter's reported cost, and replaced with measured limits after the Phase 6 benchmark.
