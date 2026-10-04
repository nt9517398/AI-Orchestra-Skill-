"""
orchestra_mcp_server.py — Local MCP server for Claude Desktop

v15 — STREAMING  (2026-08-31)
-----------------------------
Every per-model timeout, every per-model output guess, and the whole job wall-budget
tower are deleted. _call streams (stream=True), so requests' read timeout is the gap
BETWEEN chunks (_SILENCE_SECONDS) rather than a bound on the entire generation:

  • a model that thinks for 40 minutes while emitting tokens never times out
  • a genuinely hung provider dies in 120s instead of ~20 minutes
  • a stream that dies mid-flight RETURNS its partial text under [TIMEOUT] instead of
    having it discarded by an orphaned-thread guard

Removed with it: default_max and both doubling retries that existed to undo it;
timeout/job_timeout (14 hand-tuned numbers); _job_wall_budget and wall_deadline;
_no_reasoning and the 400-strip chain, by deleting its CAUSE (reasoning.enabled=false,
now remapped to effort "low" — see _effort); the "deepseek"/"glm" legacy mode aliases;
and fleet_Registry.xml, which no code ever read.

The failure-tag vocabulary is UNCHANGED. [TIMEOUT] simply stops being empty.

stealth/ox-alpha removed from _MODELS (2026-08-31): delisted from OpenRouter — a live
GET /api/v1/models returned 396 models without the slug, and direct calls now 404.
Stealth listings get withdrawn when a model graduates to a public name or is pulled;
this is an external change, not a code defect. It was the fleet's only UNKNOWN-class
member and the default adversarial model_b; both roles move to x-ai/grok-4.6. Fleet is
now 6 models / 2 classes (CN-OW, US-CLOSED).

CRITICAL (stdio transport): never print() / write to stdout — it corrupts JSON-RPC.
Every function RETURNS its result.

Setup:
    pip install "mcp[cli]" python-dotenv requests
    # API keys in a .env next to this script (see .env.example). Never hardcode.
    # OPENROUTER_API_KEY is required; missing it removes 3 of 6 models and the only
    # non-CN-OW correlation class, US-CLOSED.

Tests:  python test_orchestra.py     (offline; no keys needed)

v14.6 — GLM-5.3 + STEALTH/OX-ALPHA SWAP  (2026-08-25)
------------------------------------------------------
  1. GLM-5.2 -> GLM-5.3 (docs.bigmodel.cn, verified 2026-08-25): same "glm-5.3" id,
     same 131072 ceiling — but thinking is now MANDATORY; the old
     thinking.type="disabled" path (sent whenever reasoning_effort="none") now
     hard-FAILS on GLM-5.3. Fixed at the source: always send enabled, remap "none"
     -> "low" instead of erroring. Caught from the datasheet before deploy, not
     from a live 400 — the Gemini lesson from v14.5, applied proactively this time.
  2. thinkingmachines/inkling -> stealth/ox-alpha (OpenRouter /endpoints, verified
     2026-08-25): provider_name "Stealth", 131072 output / 1.05M ctx, free, single
     host, 24 tok/s (slower than fleet median — reachability math is in the
     registry comment). See fleet-card.md.

v14.4 — DISPATCH OUTCOME LOG + CITATION FIX  (2026-08-15)
--------------------------------------------------------
Two changes, both from a /karpathy first-principles review of this fleet:
  1. OUTCOME LOG (the data flywheel). Every routing prior in fleet-card.md — including
     which correlation CLASSES decorrelate — was guessed from lab lineage and never
     measured. This adds an append-only JSONL log (_log_event / _logged_call), two tools
     (log_outcome, fleet_stats), and one shared task_id per operation so cross-class
     dispatches can be paired and a real co-failure rate computed. Schema was hardened by
     a cross-class adversarial review (Kimi K3 + Grok both returned REVISE): per-model
     verdicts, correctness split from disposition, coverage reported, adversarial mode
     excluded, sub-threshold rates refused. See the OUTCOME LOG block below and SKILL.md.
  2. CITATION FIX. The docstrings/comments cited SKILL.md sections by NUMBER (a
     "section 5.4" / "section 12.6" scheme) but SKILL.md was refactored to PROSE headers
     and has no numbers — every one was a dangling pointer. All are now section TITLES
     (e.g. 'Verification ladder'), which do not re-orphan on renumbering. If you re-add a
     citation, cite the TITLE, not a number.

v14 — 5-MODEL / 4-LAB FLEET + PROVENANCE GUARD  (2026-07-30)
------------------------------------------------------------
Adds Kimi K3, Grok 4.5 and Inkling via OpenRouter alongside the direct DeepSeek and
GLM endpoints. Three structural changes:

  1. _MODELS registry replaces the 2-entry provider dict as the source of truth.
     _call() is now keyed by MODEL, not provider — provider is derived. This is what
     lets three fleet members share one endpoint (openrouter) without colliding.
  2. OpenRouter requests are provider-PINNED (only + allow_fallbacks=False). Default
     OpenRouter routing is price-weighted load balancing across hosts serving the same
     weights at different quantizations, which would make results non-reproducible.
  3. [SUBSTITUTED] tag — the response's own `model`/`provider` fields are compared
     against what was requested BEFORE the content is treated as a claim. Every other
     tag detects an INCOMPLETE response; this one detects a COMPLETE response from the
     wrong source (HTTP 200, non-empty, finish_reason="stop", inside timeout).

v14.3 — GEMINI 3.7 FLASH ADDED, GROK 4.5→4.6  (2026-08-14)
----------------------------------------------------------
Fleet now 7 registry entries / 6 labs, and TWO US-CLOSED members for the first time.
  • Grok 4.5 -> 4.6: pin "xAI" and the null output ceiling BOTH re-verified via a live
    /endpoints call today (500k ctx, max_completion_tokens still null). Pure version bump.
  • Gemini 3.7 Flash added: first-party Google, order=["Google AI Studio","Google"].
    /endpoints (2026-08-14) shows a DECLARED ceiling of 65536 (unlike Grok) and 1.05M ctx.
    Vertex's provider_name is "Google", NOT the "Google Vertex" the model page displays —
    display-name != provider_name, the slug trap named above; `order` uses the API names.
Two US-CLOSED members means an all-US-CLOSED pair (Grok+Gemini) is now SAME-class and
fails the SKILL.md 'Verification ladder' (L2 cross-class) cross-class check — pair each against a CN-OW member, per the fleet card.

v11 — ASYNC JOB LAYER (atomic task decomposition)
--------------------------------------------------
Claude Desktop enforces its own client-side ceiling on MCP tool calls (observed
~60s–240s depending on OS/version/transport; not configurable, not documented).
A long call_deepseek(model="deepseek-v4-pro") or GLM max-reasoning run can therefore
be silently dropped by the CLIENT even when this server's own HTTP call succeeds.

v11 fix: no tool call ever blocks on the long work. Long tasks go through:
    orchestra_start(...)  -> returns {"job_id": ..., "status": "running"} in <1s;
                             real work runs in a daemon thread
    check_job(job_id)     -> long-polls up to 45s (safely under any observed client
                             ceiling), returns running/complete/failed/not_found
This makes the exact client timeout value irrelevant — no call approaches it.

The four original synchronous tools are kept for quick calls, but their docstrings
now steer anything potentially >30s to orchestra_start.

State is an in-memory dict: fine while the server process lives (stdio servers live
for the whole Desktop session). Jobs do NOT survive a Claude Desktop restart —
acceptable for session-scoped orchestration; use SQLite if that ever changes.
"""

import json
import os
import time
import threading
import uuid
import itertools
from collections import defaultdict
from typing import Optional

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# v14: load the .env sitting NEXT TO THIS FILE, not relative to the process CWD.
# Claude Desktop launches this server with an unpredictable working directory, so a
# bare load_dotenv() can silently find nothing — and a silently-missing key now costs
# three models instead of one.
from pathlib import Path
load_dotenv(Path(__file__).with_name(".env"))

mcp = FastMCP("orchestra")

# ---------------------------------------------------------------------------
# Core calling logic — kept in sync with orchestra SKILL.md SKILL.md 'Regime selection'.
# ---------------------------------------------------------------------------

_CONFIGS = {
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "glm": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "key_env": "ZHIPU_API_KEY",
    },
    "openrouter": {                                              # v14
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# v15 MODEL REGISTRY — four fields, no tuning knobs.
#
# klass = correlation class (skill SKILL.md 'Verification ladder'). A Verifier must not
# share the Worker's class. This is a PRIOR from lab lineage, not a measurement;
# fleet_stats() measures it.
#
# max_out = the output budget actually sent, ALWAYS set, never None. Per fleet-card
# 'Ceiling discipline': the budget is a ceiling not a target (billing is on actual
# output tokens, so a high budget is nearly free on short answers), and an undeclared
# ceiling is a HAZARD — set an explicit number rather than omitting the parameter and
# inheriting a provider's undocumented default.
#
# There is deliberately no per-model timeout. The silence rule (_SILENCE_SECONDS) is
# the only stop condition, and it is global. The old reachability arithmetic
# ("usable output ~= throughput x timeout") described a constraint that no longer
# exists: with streaming there is no total-time budget for throughput to run out of.
# ---------------------------------------------------------------------------
_MODELS = {
    "deepseek-v4-pro": dict(
        provider="deepseek", klass="CN-OW", max_out=384000, pin=None),
    "deepseek-v4-flash": dict(
        provider="deepseek", klass="CN-OW", max_out=384000, pin=None),
    "glm-5.3": dict(
        provider="glm", klass="CN-OW", max_out=131072, pin=None),
        # thinking is MANDATORY on 5.3 (docs.bigmodel.cn, verified 2026-08-25) — see
        # the glm branch in _call. If a SECOND glm model is ever added, that branch
        # must become version-aware rather than keyed on provider=="glm".
    "moonshotai/kimi-k3": dict(
        provider="openrouter", klass="CN-OW", max_out=128000, pin=None,
        order=["Fireworks", "Modal", "Moonshot AI"]),
        # ordered preference with allow_fallbacks=False: OpenRouter tries these in
        # sequence and ERRORS if all are down, rather than silently rerouting to an
        # unknown host at unknown precision.
        # ⚠ "Wafer" was REMOVED from this list 2026-09-01: a live GET /models/{id}/endpoints
        # returned 17 hosts and Wafer was not among them — same delisting as ox-alpha. It
        # was first preference, so every call was silently falling through to Moonshot AI.
        # ⚠ ORDER IS MEASURED, NOT GUESSED (2026-09-01). OpenRouter reports
        # throughput_last_30m = 0 for EVERY host on this model, so declared stats are
        # useless here and the order had to be benchmarked directly. Sustained rate on an
        # identical 3000-token reasoning prompt, one host forced per run via provider.only:
        #     Fireworks    60.9 deltas/s  (2,998 deltas / 50s)
        #     Modal        38.7 deltas/s  (796 / 21s)
        #     Moonshot AI  14.5 deltas/s  (1,276 / 90s)  <- was first; 4.2x slower
        # Moonshot AI's 14.5/s independently reproduces a separate 15/s measurement, so
        # the gap is real and not sampling noise. Re-benchmark before trusting this order
        # again: hosts change, and OpenRouter publishes nothing to warn you.
        # ⚠ Fireworks reports quantization "unknown" while Moonshot AI (the model's own
        # lab) is mxfp4. Speed was chosen over first-party provenance deliberately, since
        # long-horizon reasoning is what this fleet member is FOR. Moonshot AI is kept
        # last as the first-party fallback. If output quality ever looks off on Kimi,
        # suspect this line first and try pinning Moonshot AI to compare.
        # ⚠ max_out is an OPERATOR BUDGET. Three numbers matter here and they are all
        # different — confusing them is what broke this entry twice:
        #   1,048,576 = CONTEXT window (input + output combined). Sending it as max_tokens
        #               leaves zero room for input; a live probe 2026-08-31 returned HTTP
        #               400 "you requested about 1048578 tokens (2 of text input, 1048576
        #               in the output)". Every prompt failed.
        #     943,718 = real max_completion_tokens, verified 2026-09-01 via /endpoints for
        #               all three hosts above. This is the true ceiling.
        #     128,000 = what we actually send. Deliberately well under the ceiling: the
        #               rework removed the job wall-clock, so the ONLY thing bounding a
        #               runaway generation is provider silence.
        # ⚠ THROUGHPUT, measured 2026-09-01 — do NOT reuse the old ~159 t/s figure from
        # fleet-card. That was Wafer, which is delisted (see above). Every call now lands
        # on Moonshot AI, measured at ~15 deltas/s sustained (829 deltas in 55s; 1,050 in
        # 75s). That is ~10x slower, and it changes what every budget here MEANS:
        #        24,000 -> ~27 min      128,000 -> ~2.4 hrs      943,718 -> ~17.5 hrs
        # 128,000 is kept because max_out is a CEILING, not a target: billing is on actual
        # output tokens, ordinary tasks finish far below it, and truncating a long
        # reasoning run is the exact failure this rework exists to prevent. The 2.4 hrs is
        # a worst case that only a genuine runaway reaches — but nothing except provider
        # silence will stop one, so treat a raise toward 943,718 as a real decision about
        # money and patience, not a free ceiling bump.
    "x-ai/grok-4.6": dict(
        provider="openrouter", klass="US-CLOSED", max_out=35000, pin="xAI"),
        # ⚠ max_out is an OPERATOR BUDGET. xAI now DECLARES a ceiling of 450,000
        # (/endpoints, 2026-09-01) — the older comment here claimed it was "genuinely
        # null, do not guess", which was true on 2026-08-14 and is not any more. That
        # drift is exactly what test_registry_matches_live_endpoints() now catches.
        # 35,000 is KEPT: measured 16.8 deltas/s on 2026-09-01, so 35,000 is ~35 min of
        # wall time — the same envelope Kimi's 128,000 buys at its own measured rate.
        # The full 450,000 would be ~7.4 hours. Budget by wall time, not by ceiling.
    "google/gemini-3.7-flash": dict(
        provider="openrouter", klass="US-CLOSED", max_out=65536, pin=None,
        order=["Google AI Studio", "Google"]),
        # ⚠ Vertex's provider_name is literally "Google", NOT the "Google Vertex" the
        # model page displays — `order` must use API names or the pin silently never
        # matches. max_completion_tokens 65536, DECLARED (verified 2026-08-14).
}


def _resolve(model: str) -> dict:
    """Single lookup point. Fails with a specific, actionable message — never a bare
    KeyError from a dict literal buried inside a calling function (the v13 defect)."""
    if model not in _MODELS:
        raise KeyError(f"{model!r} is not in _MODELS — add a registry entry before "
                       f"dispatch. Known models: {sorted(_MODELS)}")
    m = dict(_MODELS[model])
    m.update(_CONFIGS[m["provider"]])
    m["key"] = os.environ.get(m["key_env"], "")
    return m


def _provider_block(cfg: dict) -> dict:
    """OpenRouter provider pinning.

    NOTE `quantizations` is deliberately NOT sent. It is a DISCOVERY-time filter used
    in the /endpoints call to choose a host; once `only` names one host it discriminates
    nothing. allow_fallbacks=False is the field that turns a silent model swap into a
    hard failure. require_parameters=True stops routing to a host that would ignore a
    parameter we sent.

    2026-08-06: added `order` — an ordered preference across SEVERAL hosts (Kimi K3),
    vs `pin` which names ONE. Both keep allow_fallbacks=False: outside the named set we
    want a visible [ERROR], not a silent reroute to an unknown host/precision. Verified
    live that OpenRouter accepts provider DISPLAY names in `order` and returns the same
    string in the response body, so provenance matching needs no slug translation.
    """
    if cfg["provider"] != "openrouter":
        return {}
    if cfg.get("order"):
        return {"provider": {"order": cfg["order"],
                             "allow_fallbacks": False,
                             "require_parameters": True}}
    if cfg.get("pin"):
        return {"provider": {"only": [cfg["pin"]],
                             "allow_fallbacks": False,
                             "require_parameters": True}}
    return {}


def _check_provenance(body: dict, model: str, cfg: dict):
    """v14 — return a [SUBSTITUTED] string, or None when provenance is clean.

    Why this exists: [ERROR]/[TIMEOUT]/[EMPTY]/[TRUNCATED] all detect an INCOMPLETE
    response. An aggregator serving a different host or precision returns a COMPLETE
    one — 200, non-empty, finish_reason="stop", inside timeout. All four structural
    guards pass. Only comparing the response's own provenance fields catches it.
    Absent fields are not treated as failures (direct APIs don't send them).
    """
    served_model = (body.get("model") or "").strip()
    served_prov = (body.get("provider") or "").strip()
    short = model.split("/")[-1]
    if served_model and short not in served_model:
        return (f"[SUBSTITUTED] requested {model!r} but the endpoint served "
                f"{served_model!r} — do not audit this as content; re-verify the pin")
    # 2026-08-06: allow EITHER a single pin OR an ordered host set. Because
    # allow_fallbacks=False, the served host must be inside the named set; anything
    # else is a substitution. Membership test (case-insensitive), not equality.
    allowed = cfg.get("order") or ([cfg["pin"]] if cfg.get("pin") else [])
    if allowed and served_prov:
        if served_prov.lower() not in {a.lower() for a in allowed}:
            return (f"[SUBSTITUTED] expected one of {allowed!r} but response came from "
                    f"{served_prov!r} — host set not honoured; re-check /endpoints")
    return None


# ---------------------------------------------------------------------------
# STREAMING (v15) — the silence rule replaces every total-generation timeout.
#
# With stream=True, requests' READ timeout is the gap BETWEEN chunks, not the whole
# generation. A model that thinks for 40 minutes while emitting tokens never trips it;
# a genuinely hung provider dies in _SILENCE_SECONDS instead of ~20 minutes. Partial
# output survives because it has already been accumulated locally, which is the whole
# reason the per-model timeout/job_timeout/wall-budget tower could be deleted.
# ---------------------------------------------------------------------------
_CONNECT_SECONDS = 10        # TCP connect + TLS; a dead host should fail fast
_SILENCE_SECONDS = 120       # max gap BETWEEN chunks. The only stop condition.

# Streams that end without a socket error but also without content or a finish_reason
# are treated as complete; the [EMPTY] branch below catches the no-content case.
_STREAM_DEATH = (requests.exceptions.Timeout,
                 requests.exceptions.ConnectionError,
                 requests.exceptions.ChunkedEncodingError)


def _consume_stream(lines, model: str, cfg: dict, progress=None) -> str:
    """Accumulate an OpenAI-compatible SSE stream into text, or a tagged failure.

    PURE over `lines`: any iterable of bytes/str. Production passes
    resp.iter_lines(); the tests pass a list or a generator that raises. Keeping the
    network out of this function is what makes the four checks in test_orchestra.py
    run offline.

    Provenance is checked on every chunk, BEFORE any content from it is accumulated,
    until a chunk actually carries provenance fields — then the check stops. This is
    deliberate and load-bearing: [SUBSTITUTED] is the only guard against a complete,
    well-formed 200 from the wrong host, and a substituted stream must not contribute
    a single character to the returned text.
    """
    parts, reasoning, finish, n = [], [], None, 0
    try:
        for raw in lines:
            if not raw:
                continue
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            line = line.strip()
            # ": OPENROUTER PROCESSING" keepalives are SSE COMMENTS, not data. Skipping
            # them is required (they are not JSON) — but they arrive on the wire, so
            # they reset the read timeout. That is precisely why a deep-thinking model
            # stopped tripping the silence rule.
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue                      # a malformed frame is not fatal
            # Judge EVERY chunk. There is deliberately no "already checked" flag here.
            # _check_provenance correctly treats absent model/provider fields as clean,
            # because direct APIs never send them — so any flag that disarms the guard
            # on a frame lacking those fields lets a LATER frame declare a foreign
            # model/provider and sail through untested. Two such flags have already
            # shipped and both leaked: `checked = True` (disarmed on the bare role-delta
            # frame that OpenAI-compatible gateways send FIRST) and `checked = bool(model
            # or provider)` (disarmed on whichever field arrived first, so the other was
            # never judged). Checking every chunk costs a few string comparisons and
            # cannot regress. ponytail: do not reintroduce a flag to "optimize" this —
            # the value is identical on every chunk, so there is nothing to win and a
            # fail-open provenance guard to lose.
            substituted = _check_provenance(chunk, model, cfg)
            if substituted:
                return substituted            # discard everything; wrong source
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            think = None
            if piece:
                parts.append(piece)
                n += 1
            else:
                # Reasoning models stream their thinking in a separate field, NOT in
                # `content`. Measured 2026-09-01: glm-5.3 emitted 1,281 reasoning deltas
                # in 25s and produced no content token until 1066s in; deepseek-v4-pro
                # took 158s. Counting only `content` meant a job that was working
                # perfectly reported "~0 tokens" for eighteen minutes, and an answer that
                # never arrived threw away every reasoning token already paid for.
                #
                # ⚠ TWO SPELLINGS, and missing either one silently reverts this whole fix
                # for half the fleet. The direct APIs (DeepSeek, Zhipu) send
                # `reasoning_content`; OpenRouter normalises to `reasoning` and adds a
                # structured `reasoning_details`. Shipping only `reasoning_content` left
                # kimi-k3, grok-4.6 and gemini-3.7-flash exactly as broken as before —
                # caught 2026-09-01 by a live probe showing
                # {'reasoning': 741, 'reasoning_details': 741, 'content': 88} on Kimi.
                # `reasoning_details` is deliberately NOT read: it carries the same text
                # wrapped in typed objects, so consuming both would double-count.
                think = delta.get("reasoning_content") or delta.get("reasoning")
                if think:
                    reasoning.append(think)
                    n += 1
            # ponytail: n counts DELTAS, not tokenizer tokens. Close enough for a
            # progress line; swap in usage.completion_tokens if it ever must be exact.
            if piece or think:
                # Fire on the FIRST delta as well as every 25th. Delta size varies wildly
                # by model: Kimi streams thousands of token-sized deltas, while
                # gemini-3.7-flash batches a whole run into 3-7 large ones (measured
                # 2026-09-01) and would never reach a multiple of 25 — so a Gemini job
                # showed no progress at all, looking hung while working fine.
                if progress and (n == 1 or n % 25 == 0):
                    progress(f"{model}: ~{n:,} tokens")
            finish = choices[0].get("finish_reason") or finish
    except _STREAM_DEATH as e:
        return _finish_stream(model, cfg, parts, reasoning, n,
                              f"[TIMEOUT] {model} went silent after ~{n:,} tokens "
                              f"({type(e).__name__})")

    if finish == "length":
        return _finish_stream(model, cfg, parts, reasoning, n,
                              f"[TRUNCATED] {model} hit its {cfg['max_out']}-token budget")

    content = "".join(parts).strip()
    if not content:
        return _finish_stream(model, cfg, parts, reasoning, n,
                              f"[EMPTY] {model} streamed no answer — the reasoning trace "
                              f"consumed the {cfg['max_out']}-token budget. Raise max_out "
                              f"or lower reasoning_effort")
    return content


def _finish_stream(model, cfg, parts, reasoning, n, tag_line: str) -> str:
    """Build a tagged failure string that RETURNS the work already paid for.

    A stream can end early three ways — the provider went silent, the budget ran out,
    or the whole budget went to thinking and no answer followed. In all three the user
    has already been billed for every token that arrived. Discarding them is the exact
    waste this server was reworked to stop, so whatever exists is attached: the partial
    answer if there is one, otherwise the reasoning trace, which is frequently useful
    on its own and is the only artifact a long think produces before it concludes.
    """
    content = "".join(parts).strip()
    if content:
        return f"{tag_line} — content below is INCOMPLETE:\n\n{content}"
    think = "".join(reasoning).strip()
    if think:
        return (f"{tag_line}. No answer was produced, but ~{n:,} tokens of REASONING "
                f"were streamed and are preserved below. This is a thinking trace, not "
                f"a conclusion — treat it as working notes, not an answer:\n\n{think}")
    return tag_line + "."


# 2026-08-06: single source of truth for the structural failure tags _call can return.
# Used by the adversarial fault barrier; any caller can test `out.startswith(_FAILURE_TAGS)`
# to tell "this is a failure marker" from "this is model content".
_FAILURE_TAGS = ("[ERROR]", "[TIMEOUT]", "[EMPTY]", "[TRUNCATED]",
                 "[SKIPPED]", "[SUBSTITUTED]")

# ---------------------------------------------------------------------------
# v14.4 — DISPATCH OUTCOME LOG (the data flywheel)
# ---------------------------------------------------------------------------
# Why this exists: every routing prior in fleet-card.md — including which correlation
# CLASSES actually decorrelate — is currently GUESSED from lab lineage, never measured
# on this fleet. list_fleet()'s own note calls that measurement "BLOCKING" and it had
# never run. This is the mechanism that lets it run: an append-only JSONL log of what
# the fleet was asked, what it returned, and (fed back by Claude after adjudication)
# whether it was right. Read it with fleet_stats(). See SKILL.md 'Outcome log'.
#
# Minimal on purpose (JSONL, not SQLite): greppable by hand ("sort the log, read the
# worst ten"), survives a Desktop restart (the in-memory job dict does not), zero new
# dependency. Move to SQLite only when grep stops scaling — not yet.
#
# Schema shaped by a cross-class adversarial review (Kimi K3 + Grok, 2026-08-14) that
# both returned REVISE. Confirmed-critical fixes applied here: per-MODEL verdicts (not
# per-task), ONE task_id minted at the operation entry and threaded to every dispatch
# under it, task_id surfaced back to Claude, correctness split from disposition, and
# fleet_stats reports COVERAGE + refuses a co-failure point estimate below _MIN_PAIRS +
# excludes adversarial mode. DEFERRED (named, not built — ponytail): random paired
# probes for an unbiased sample, multi-PROCESS-safe advisory locking, size-based
# rotation, per-round indices. These are real; they are iteration-2, not iteration-1.

_LOG_PATH = Path(__file__).with_name("dispatch_log.jsonl")
_LOG_LOCK = threading.Lock()      # ONE lock for both append and read — a tail read racing
                                  # an append can otherwise see a torn last line.
_LOG_FAILURES = 0                 # surfaced by fleet_stats: a silently-failing logger is a
                                  # dead flywheel, the worst failure mode the reviewers named.
_MIN_PAIRS = 20                   # ponytail: judgment call, not a measured optimum. Below this
                                  # many paired observations a co-failure rate is noise, so
                                  # fleet_stats refuses to print one. Raise once data exists.

# Two orthogonal axes, kept separate on reviewer insistence. correctness is the ERROR
# axis (what co-failure is computed on); disposition is what Claude DID with the answer.
# A DISCARDED-but-correct redundant answer must never count as an error.
_CORRECTNESS = ("CORRECT", "WRONG", "UNVERIFIED", "OVERTURNED_BY_L1")
_DISPOSITION = ("", "USED", "DISCARDED", "NA")
_ERROR_SET = frozenset({"WRONG", "OVERTURNED_BY_L1"})   # counts as a co-failure event


def _now() -> str:
    # local wall-clock, second precision — greppable by date. Local, not UTC, on purpose:
    # a single-operator log reads more naturally in the operator's own time.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _outcome_tag(out: str) -> str:
    """The INFRA outcome of a dispatch: a leading failure tag, or OK. Distinct from the
    CORRECTNESS axis that log_outcome records — Grok's point that 'the call succeeded'
    and 'the answer was right' must not be conflated."""
    s = (out or "").lstrip()
    for tag in _FAILURE_TAGS:
        if s.startswith(tag):
            return tag.strip("[]")
    return "OK"


def _log_event(rec: dict) -> None:
    """Append one JSON line under the shared lock, flushing so a Desktop kill can't
    strand the tail in a buffer. NEVER raises — the log is instrumentation, not the
    product, so a write failure must not break a dispatch. But it is COUNTED: a logger
    that fails forever in silence is the worst outcome the reviewers named, so
    _LOG_FAILURES is surfaced by fleet_stats rather than vanishing."""
    global _LOG_FAILURES
    rec.setdefault("pid", os.getpid())    # cheap multi-process breadcrumb (full locking deferred)
    try:
        line = json.dumps(rec, ensure_ascii=False)
    except Exception:
        with _LOG_LOCK:
            _LOG_FAILURES += 1
        return
    try:
        with _LOG_LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
    except Exception:
        with _LOG_LOCK:
            _LOG_FAILURES += 1


def _new_task_id() -> str:
    return str(uuid.uuid4())[:8]


def _banner(task_id: str) -> str:
    """First line of every SYNC tool result: surfaces the task_id so Claude can pair a
    verdict to it. Job mode does not need this — the job_id IS the task_id and Claude
    already has it from the job envelope."""
    return (f"[orchestra task_id={task_id}] after adjudicating, call "
            f"log_outcome(task_id, model, correctness) once per model.\n\n")


# A model may name a single host (pin) OR an ordered set (order), never both —
# ambiguous routing otherwise.
for _name, _m in _MODELS.items():
    assert not (_m.get("pin") and _m.get("order")), \
        f"{_name}: set pin OR order, not both"


def _effort(name: str) -> str:
    """One effort scale for every provider. Returns "low" | "medium" | "high".

    "none" maps to "low" and NOT to reasoning.enabled=false. That payload was the sole
    producer of Gemini 3.7 Flash's HTTP 400, which was the sole reason the
    _no_reasoning strip-and-retry path existed. GLM-5.3 already remapped it (thinking
    cannot be disabled there either); doing the same for OpenRouter deletes the whole
    strip/retry chain and the 2026-08-25 four-branch composition bug with it.
    Unknown values fall back to the default depth rather than guessing a payload.
    """
    return {"none": "low", "low": "low", "medium": "medium",
            "high": "high", "max": "high"}.get(name, "high")


def _retry_after(response) -> int:
    """Seconds to wait on a 429, honouring Retry-After, clamped to [1, 120].

    The OverflowError guard is load-bearing: a header like "1e400" parses via float()
    to inf without error, and int(inf) raises OverflowError — NOT ValueError — which
    would otherwise escape and break every caller's assumption that _call always
    returns a tagged string.
    """
    if response is None:
        return 20
    try:
        return min(120, max(1, int(float(response.headers.get("Retry-After", 20)))))
    except (TypeError, ValueError, OverflowError):
        return 20


def _call(model: str, messages: list, glm_reasoning_effort: str = "max",
          progress=None) -> str:
    """Stream one chat completion. Always returns a string: content, or a tag from
    _FAILURE_TAGS.

    v15: no `timeout`, `max_tokens`, `_retry` or `_no_reasoning` parameters. The
    silence rule is global, the budget is the registry's max_out, and there are no
    recursive call sites left — the 429/5xx retries are a two-pass loop.
    """
    try:
        cfg = _resolve(model)
    except KeyError as e:
        return f"[ERROR] {e}"
    provider = cfg["provider"]
    if not cfg["key"]:
        return f"[SKIPPED] {cfg['key_env']} not set — add it to .env next to this script"

    payload = {"model": model, "messages": messages,
               "max_tokens": cfg["max_out"], "stream": True}
    payload.update(_provider_block(cfg))     # OpenRouter pin; no-op for direct APIs
    if provider == "glm":
        # GLM-5.3 cannot disable thinking; thinking.type="disabled" hard-FAILS.
        payload["thinking"] = {"type": "enabled"}
        # GLM's reasoning_effort scale is its own: passthrough "max", "high", "medium",
        # "low", with only "none" remapped to "low" (since thinking cannot be disabled).
        # _effort() collapses "max" to "high" because OPENROUTER has no deeper tier;
        # applying that collapse here would silently downgrade every GLM call from the
        # server's default effort="max" to effort="high". Restore passthrough to match
        # the original GLM behavior.
        payload["reasoning_effort"] = "low" if glm_reasoning_effort == "none" else glm_reasoning_effort
    elif provider == "openrouter":
        # Bounds CoT depth. Without it Kimi/Grok default to max depth and can spend the
        # entire budget on an invisible trace, returning [EMPTY].
        payload["reasoning"] = {"effort": _effort(glm_reasoning_effort)}

    for attempt in (0, 1):
        try:
            # `with` guarantees resp.close() on every exit path — including
            # _consume_stream's early `return` on [SUBSTITUTED] and its `break` on
            # [DONE] — so a long-lived stdio server doesn't leak sockets across a
            # whole Claude Desktop session.
            with requests.post(
                cfg["url"],
                headers={"Authorization": f"Bearer {cfg['key']}",
                         "Content-Type": "application/json"},
                json=payload, stream=True,
                timeout=(_CONNECT_SECONDS, _SILENCE_SECONDS),
            ) as resp:
                resp.raise_for_status()
                return _consume_stream(resp.iter_lines(), model, cfg, progress)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            # 429 is the most common disruption in multi-model OpenRouter
            # orchestration; an instant fail defeats the point of a background job.
            if attempt == 0 and status == 429:
                time.sleep(_retry_after(e.response))
                continue
            if attempt == 0 and 500 <= status < 600:
                time.sleep(5)
                continue
            body_text = ""
            try:
                if e.response is not None:
                    body_text = e.response.text[:300] if e.response.text else ""
            except Exception:
                pass  # Never raise from error handling
            return f"[ERROR] {provider}/{model}: HTTP {status} — {e}" + \
                   (f" {body_text}" if body_text else "")
        except _STREAM_DEATH as e:
            # Raised before the first chunk (connect timeout / refused). A death DURING
            # the stream is caught inside _consume_stream, which keeps the partial text.
            return f"[TIMEOUT] {provider}/{model}: no stream started — {type(e).__name__}"
        except Exception as e:
            return f"[ERROR] {provider}/{model}: {e}"
    # Unreachable by construction: attempt 0 always either returns or `continue`s, and
    # on attempt 1 every `attempt == 0` retry guard above is False, so every branch
    # returns. Kept deliberately, not dead code to "clean up" — _call must ALWAYS
    # return a string, and this is the guard against a future edit to the retry loop
    # letting the function fall through and implicitly return None.
    return f"[ERROR] {provider}/{model}: retries exhausted"


def _logged_call(model, messages, *, task_id, role="", mode="sync", **call_kwargs):
    """Wrap the pure _call() so every user-facing dispatch appends EXACTLY ONE log row.
    Placed only at LEAF call sites — never around a path that itself calls _logged_call
    — so _call's internal timeout/429/5xx retries stay invisible and one logical
    dispatch is one row, not one-per-retry.

    task_id is REQUIRED and passed IN from the operation's entry point, never minted
    here. The reviewers' sharpest shared point: if each _logged_call minted its own id,
    the two models of a parallel run would never share a task_id and could never be
    paired — so the one traffic you most want to correlate would yield zero pairs. One
    operation, one task_id, threaded to every call beneath it.

    try/finally guarantees a row even when _call raises something outside its own tag
    vocabulary (a client bug, an interrupt). Without it the buggiest provider's
    dispatches would vanish from the log and it would look the most reliable."""
    t0 = time.time()
    out = None
    try:
        out = _call(model, messages, **call_kwargs)
        return out
    finally:
        tag = _outcome_tag(out) if isinstance(out, str) else "UNCAUGHT"
        _log_event({"type": "dispatch", "ts": _now(), "task_id": task_id,
                    "model": model, "klass": _MODELS.get(model, {}).get("klass", "?"),
                    "provider": _MODELS.get(model, {}).get("provider", "?"),
                    "mode": mode, "role": role, "outcome": tag,
                    "out_chars": len(out) if isinstance(out, str) else 0,
                    "elapsed_s": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------
# Shared mode logic — used by BOTH the sync tools and the job workers, so the
# two paths can't drift apart (the v9 "hand-synced copies" complaint, fixed).
# ---------------------------------------------------------------------------

def _run_parallel(models=("deepseek-v4-flash", "glm-5.3"), prompt: str = "",
                  reasoning_effort: str = "max", progress=None,
                  task_id: str = "") -> str:
    """v14: any two registry models, results keyed by MODEL ID.

    Capped at 2 candidates deliberately (skill SKILL.md 'Effort scaling' (cap 2)). `models` is a list so the
    CHOICE is free, not so the count can grow — wider fan-out on this fleet produces
    correlated agreement that reads as consensus, which is the failure SKILL.md 'Core principle' (no consensus) names.

    v14.4: both calls share ONE task_id so fleet_stats can pair them cross-class. The
    caller passes it in; we mint one only if it didn't (a bare internal call).
    """
    if len(models) != 2:
        return f"[ERROR] parallel takes exactly 2 models (skill SKILL.md 'Effort scaling' (cap 2)), got {len(models)}"
    for m in models:
        if m not in _MODELS:
            return f"[ERROR] {m!r} not in _MODELS — known: {sorted(_MODELS)}"
    task_id = task_id or _new_task_id()
    results = {}
    messages = [{"role": "user", "content": prompt}]

    def _run(model):
        # 2026-08-06: was `{"glm_reasoning_effort": "max"} if model == "glm-5.2" else {}`
        # — which hardcoded GLM to max and gave every other model nothing. Since _call
        # now routes this same knob to OpenRouter reasoning too, pass it always; the
        # deepseek branch ignores it harmlessly.
        results[model] = _logged_call(model, messages,
                                      task_id=task_id, role="candidate", mode="parallel",
                                      glm_reasoning_effort=reasoning_effort,
                                      progress=progress)

    threads = [threading.Thread(target=_run, args=(m,)) for m in models]
    if progress:
        progress(f"dispatching {models[0]} + {models[1]} in parallel")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return "\n\n".join(
        f"=== {m} [{_MODELS[m]['klass']}] ===\n{results.get(m, '[no result]')}"
        for m in models
    )


def _run_adversarial(task: str, model_a: str = "deepseek-v4-flash",
                     model_b: str = "x-ai/grok-4.6",
                     reasoning_effort: str = "max", progress=None,
                     task_id: str = "") -> str:
    """v14: cross-class ENFORCED (skill SKILL.md 'Verification ladder' (L2 cross-class)).

    A cross-critique between two models of the same correlation class is two members
    of one bloc agreeing with each other — it looks like adversarial verification and
    isn't. The v11 default pair (flash + glm-5.2) was exactly that: both CN-OW. The
    default is now deepseek-v4-flash [CN-OW] vs x-ai/grok-4.6 [US-CLOSED] — cross-class
    by construction. Refusing here rather than warning is deliberate: a silently weak
    verification is worse than none.

    v14.4: all four dispatches share ONE task_id. But fleet_stats EXCLUDES mode
    'adversarial' from its default co-failure stat — this mode drives disagreement by
    design, so its pairs would inflate measured correlation (reviewer-confirmed).
    """
    for m in (model_a, model_b):
        if m not in _MODELS:
            return f"[ERROR] {m!r} not in _MODELS — known: {sorted(_MODELS)}"
    task_id = task_id or _new_task_id()
    ka, kb = _MODELS[model_a]["klass"], _MODELS[model_b]["klass"]
    if ka == kb:
        return (f"[ERROR] ADVERSARIAL requires cross-class models (skill SKILL.md 'Verification ladder' (L2 cross-class)): "
                f"{model_a} and {model_b} are both {ka}. Pick one from a different "
                f"class — {sorted({v['klass'] for v in _MODELS.values()})}")

    messages = [{"role": "user", "content": task}]

    # Round 1 — Positioning
    if progress:
        progress("round 1/2: both models solving independently")
    round1 = {}
    threads = [
        threading.Thread(target=lambda m=model_a: round1.update(
            {m: _logged_call(m, messages, task_id=task_id, role="candidate",
                             mode="adversarial",
                             glm_reasoning_effort=reasoning_effort,
                             progress=progress)})),
        threading.Thread(target=lambda m=model_b: round1.update(
            {m: _logged_call(m, messages, task_id=task_id, role="candidate",
                             mode="adversarial",
                             glm_reasoning_effort=reasoning_effort,
                             progress=progress)})),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 2026-08-06 FAULT BARRIER. Round 2 asks each model to find weaknesses in the
    # OTHER's Round-1 output. If Round 1 returned a failure tag, the "solution" being
    # critiqued is an error string — 2 more dispatches spent producing a critique of
    # "[ERROR] HTTP 502", presented in the final report as if it were adversarial
    # verification. Abort instead. NOTE: test explicit tags, not startswith("[") —
    # a legitimate response may open with "[" (JSON array output contracts do).
    _failed = {m: out for m, out in round1.items()
               if out.strip().startswith(_FAILURE_TAGS)}
    if _failed or len(round1) < 2:
        _missing = [m for m in (model_a, model_b) if m not in round1]
        return ("[ERROR] adversarial aborted — Round 1 did not produce two auditable "
                "solutions, so a Round 2 cross-critique would be critiquing a failure "
                "string, not a solution.\n"
                + "".join(f"  {m}: {out.strip()[:200]}\n" for m, out in _failed.items())
                + "".join(f"  {m}: [no result returned]\n" for m in _missing)
                + "\nRe-run once the failing dispatch succeeds; do not audit the above "
                  "as model output.")

    # Round 2 — Assault: exactly 2 concrete weaknesses each; constraint lives in
    # the prompt (an LLM can't be forced past the API to comply).
    def assault_prompt(opponent_output: str) -> str:
        return (
            f"Original task: {task}\n\nA different model produced this solution:\n\n"
            f"{opponent_output}\n\nGive EXACTLY 2 concrete weaknesses — specific logical "
            f"flaws, unproven assumptions, or failure modes. No hedging, no \"both are good.\""
        )

    if progress:
        progress("round 2/2: cross-critique in flight")
    round2 = {}
    threads = [
        threading.Thread(target=lambda m=model_a, o=model_b: round2.update(
            {m: _logged_call(m, [{"role": "user", "content": assault_prompt(round1.get(o, ""))}],
                             task_id=task_id, role="critic", mode="adversarial",
                             glm_reasoning_effort=reasoning_effort,
                             progress=progress)})),
        threading.Thread(target=lambda m=model_b, o=model_a: round2.update(
            {m: _logged_call(m, [{"role": "user", "content": assault_prompt(round1.get(o, ""))}],
                             task_id=task_id, role="critic", mode="adversarial",
                             glm_reasoning_effort=reasoning_effort,
                             progress=progress)})),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return (
        f"=== ROUND 1: POSITIONING ===  ({model_a} [{ka}] vs {model_b} [{kb}])\n"
        f"--- {model_a} SOLUTION ---\n{round1.get(model_a, '[no result]')}\n\n"
        f"--- {model_b} SOLUTION ---\n{round1.get(model_b, '[no result]')}\n\n"
        f"=== ROUND 2: ASSAULT ===\n"
        f"--- {model_a} ATTACKS {model_b} ---\n{round2.get(model_a, '[no result]')}\n\n"
        f"--- {model_b} ATTACKS {model_a} ---\n{round2.get(model_b, '[no result]')}\n\n"
        f"[Round 3 — Battle_Test synthesis happens in Claude's own reasoning, not in this tool]"
    )


# ---------------------------------------------------------------------------
# v11 job layer — atomic task decomposition (start fast, poll fast)
# ---------------------------------------------------------------------------

_JOBS: dict = {}                 # job_id -> {"status", "mode", "started", "finished",
_JOBS_LOCK = threading.Lock()    #            "progress", "result"/"error"}
_JOB_RETENTION_SECONDS = 3600    # evict finished jobs after 1h (r3: cleanup)
_POLL_BLOCK_CAP = 45             # max long-poll block — safely under any observed client ceiling


def _prune_jobs() -> None:
    now = time.time()
    with _JOBS_LOCK:
        stale = [jid for jid, j in _JOBS.items()
                 if j["status"] in ("complete", "failed")
                 and now - j.get("finished", now) > _JOB_RETENTION_SECONDS]
        for jid in stale:
            del _JOBS[jid]


def _job_progress(job_id: str):
    def set_progress(text: str) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["progress"] = text
    return set_progress


@mcp.tool()
def orchestra_start(mode: str, prompt: str, model: str = "deepseek-v4-pro",
                    reasoning_effort: str = "max",
                    model_a: Optional[str] = None,
                    model_b: Optional[str] = None) -> str:
    """START a long-running orchestra task as a background job and return IMMEDIATELY
    with a job_id — the required path for anything that may take more than ~30 seconds
    (deep reasoning, research analysis, adversarial cross-critique, maximum-yield
    outputs). This exists because Claude Desktop silently drops MCP tool results that
    take too long; jobs make that impossible by never blocking on the work.

    mode: "model"  — single call to ANY fleet model; set `model` to a registry id.
                     This is the v14 general form; call list_fleet() to see ids.
          "parallel"    — same prompt to two models side by side (`model_a`/`model_b`,
                     default flash + glm-5.3). Capped at 2 by design (skill SKILL.md 'Effort scaling' (cap 2)).
          "adversarial" — 2-round cross-critique, 4 dispatches. `model_a`/`model_b`
                     MUST be from different correlation classes (skill SKILL.md 'Verification ladder' (L2 cross-class)); the
                     default pair is deepseek-v4-flash [CN-OW] vs x-ai/grok-4.6 [US-CLOSED].
                     Same-class pairs are REJECTED, not warned about — two models from one
                     bloc critiquing each other looks like verification and is not.

    A job never times out on its own: _call streams the response and only stops when
    the provider goes silent for _SILENCE_SECONDS between chunks. If that happens,
    whatever text had already arrived is returned under a [TIMEOUT] tag rather than
    discarded — partial work reaches the user instead of being thrown away.

    Returns JSON: {"job_id": "...", "status": "running"}.
    NEXT STEP (mandatory): call check_job(job_id) and keep polling until status is
    "complete" or "failed". Do not end your turn reporting "job started" without
    polling at least once. Each start creates a NEW independent job (no dedup).
    """
    _prune_jobs()
    mode = mode.strip().lower()
    if mode not in ("model", "parallel", "adversarial"):
        return json.dumps({"status": "failed",
                           "error": f"unknown mode '{mode}' — use model | parallel | "
                                    f"adversarial (the 'deepseek' and 'glm' aliases "
                                    f"were removed in v15; use mode='model' with a "
                                    f"registry id)"})
    if mode == "model" and model not in _MODELS:
        return json.dumps({"status": "failed",
                           "error": f"unknown model '{model}' — known: {sorted(_MODELS)}"})

    job_id = str(uuid.uuid4())[:8]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "mode": mode,
                         "started": time.time(), "progress": "queued"}
    progress = _job_progress(job_id)

    def worker():
        try:
            # The job_id IS the task_id for the outcome log, so a completed job's
            # dispatches are pairable via log_outcome(job_id, model, correctness).
            if mode == "model":
                progress(f"calling {model}")
                result = _logged_call(model, [{"role": "user", "content": prompt}],
                                      task_id=job_id, role="candidate", mode="job",
                                      glm_reasoning_effort=reasoning_effort,
                                      progress=progress)
            elif mode == "parallel":
                result = _run_parallel(models=(model_a or "deepseek-v4-flash",
                                               model_b or "glm-5.3"),
                                       prompt=prompt, reasoning_effort=reasoning_effort,
                                       progress=progress, task_id=job_id)
            else:
                result = _run_adversarial(prompt,
                                          model_a=model_a or "deepseek-v4-flash",
                                          model_b=model_b or "x-ai/grok-4.6",
                                          reasoning_effort=reasoning_effort,
                                          progress=progress, task_id=job_id)
            with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j is not None and j["status"] == "running":
                    j.update(status="complete", result=result,
                             finished=time.time(), progress="done")
        except Exception as e:   # a crashed worker must never leave "running" forever
            with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j is not None and j["status"] == "running":
                    j.update(status="failed", error=f"{type(e).__name__}: {e}",
                             finished=time.time())

    threading.Thread(target=worker, daemon=True).start()
    return json.dumps({"job_id": job_id, "status": "running",
                       "next": "call check_job with this job_id; keep polling until complete/failed"})


@mcp.tool()
def check_job(job_id: str, wait_seconds: int = 25) -> str:
    """Poll a job started by orchestra_start. LONG-POLLS: blocks up to wait_seconds
    (default 25, hard cap 45 — always safe against the client timeout) waiting for
    the job to finish before answering, which keeps the number of polls small.

    Returns JSON, one of:
      {"status": "running", "elapsed_seconds": N, "progress": "..."}
          -> CALL check_job AGAIN. Do not stop and tell the user "still running" —
             keep polling until complete or failed. Deep-reasoning jobs typically
             take 1–10 minutes (several polls).
      {"status": "complete", "result": "..."}
          -> Final payload. The result may still contain the orchestra skill's
             [ERROR]/[TIMEOUT]/[EMPTY]/[TRUNCATED]/[SKIPPED]/[SUBSTITUTED] tags from the
             underlying API call — apply the skill's SKILL.md 'Failure handling' (tag check) tag check before
             treating it as content.
      {"status": "failed", "error": "..."}  -> explicit failure, report it.
      {"status": "not_found"}  -> unknown/expired job_id (finished jobs are evicted
          after 1 hour; jobs do not survive a Claude Desktop restart).
    """
    wait_seconds = max(0, min(int(wait_seconds), _POLL_BLOCK_CAP))
    poll_deadline = time.time() + wait_seconds     # this long-poll's own window (≤45s)
    while True:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            snapshot = dict(job) if job else None
        if snapshot is None:
            return json.dumps({"status": "not_found"})
        if snapshot["status"] == "complete":
            return json.dumps({"status": "complete", "mode": snapshot["mode"],
                               "elapsed_seconds": int(snapshot["finished"] - snapshot["started"]),
                               "result": snapshot["result"]})
        if snapshot["status"] == "failed":
            return json.dumps({"status": "failed", "mode": snapshot["mode"],
                               "error": snapshot.get("error", "unknown")})
        if time.time() >= poll_deadline:
            return json.dumps({"status": "running", "mode": snapshot["mode"],
                               "elapsed_seconds": int(time.time() - snapshot["started"]),
                               "progress": snapshot.get("progress", "")})
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Legacy synchronous tools — quick calls only. Anything potentially >30s
# should go through orchestra_start instead (client-side timeout risk).
# ---------------------------------------------------------------------------

@mcp.tool()
def call_deepseek(prompt: str, model: str = "deepseek-v4-flash") -> str:
    """Call DeepSeek directly and WAIT for the answer — QUICK tasks only (short code
    snippets, short writing). For deep multi-step reasoning or model="deepseek-v4-pro",
    use orchestra_start(mode="model", model=...) instead: Claude Desktop can silently
    drop synchronous results that take too long, and the job layer is what makes that
    client-side ceiling irrelevant.

    Returns raw text, or a tagged
    [ERROR]/[TIMEOUT]/[EMPTY]/[SKIPPED]/[TRUNCATED]/[SUBSTITUTED] string — check tags
    before treating the result as content. Since v15 a [TIMEOUT] CONTAINS whatever
    streamed before the provider went silent; read it as partial content, not as a
    total loss. Result begins with a [orchestra task_id=...] banner — reuse it in
    log_outcome to record correctness.
    """
    tid = _new_task_id()
    return _banner(tid) + _logged_call(model, [{"role": "user", "content": prompt}],
                                       task_id=tid, role="candidate", mode="sync")


@mcp.tool()
def call_glm(prompt: str, reasoning_effort: str = "max") -> str:
    """Call GLM-5.3 directly and WAIT — QUICK tasks only.
    reasoning_effort: "max" (deepest, default), "high", "medium", or "low". GLM-5.3
    cannot disable thinking; "none" is accepted for back-compat and remapped to "low".
    For deep "max"-effort runs use orchestra_start(mode="model", model="glm-5.3").

    Returns raw text or a tagged string — check tags before treating it as content.
    Result begins with a [orchestra task_id=...] banner — reuse it in log_outcome.
    """
    tid = _new_task_id()
    return _banner(tid) + _logged_call("glm-5.3", [{"role": "user", "content": prompt}],
                                       task_id=tid, role="candidate", mode="sync",
                                       glm_reasoning_effort=reasoning_effort)


@mcp.tool()
def orchestra_parallel(prompt: str, reasoning_effort: str = "max") -> str:
    """Run the identical prompt on both DeepSeek (v4-flash) and GLM-5.3 concurrently
    and WAIT; returns both raw, labeled outputs side by side. QUICK
    prompts only — for anything substantial use orchestra_start(mode="parallel"),
    since the combined wait risks the client-side timeout.

    This tool does NOT judge or pick a winner. After calling it, apply the orchestra
    skill's Confidence Labeling (SKILL.md 'Confidence labelling') and Judgment Protocol (SKILL.md 'Adjudication') in Claude's own
    reasoning: audit each output's claims as HIGH / UNCERTAIN / UNTRUSTWORTHY, then
    synthesize.

    v14.4: both dispatches share the task_id shown in the [orchestra task_id=...] banner
    at the top of the result. Record each model's correctness with
    log_outcome(task_id, model, correctness) — this is the cross-class pair fleet_stats
    needs to measure real error correlation.
    """
    tid = _new_task_id()
    return _banner(tid) + _run_parallel(models=("deepseek-v4-flash", "glm-5.3"),
                                        prompt=prompt,
                                        reasoning_effort=reasoning_effort, task_id=tid)


@mcp.tool()
def orchestra_adversarial(task: str, reasoning_effort: str = "max") -> str:
    """Run ADVERSARIAL mode synchronously and WAIT: both models solve the task
    (Round 1 — Positioning), then each attacks the OTHER's solution with exactly 2
    concrete weaknesses (Round 2 — Assault). Hard-capped at 2 rounds. STRONGLY
    prefer orchestra_start(mode="adversarial") instead — two sequential rounds of
    parallel calls almost always exceed the client-side timeout window.

    v14: the default pair is now deepseek-v4-flash [CN-OW] vs x-ai/grok-4.6 [US-CLOSED].
    Cross-class is enforced (skill SKILL.md 'Verification ladder' (L2 cross-class)) — the old flash/glm-5.2
    default was two Chinese open-weight models critiquing each other.

    This tool does NOT synthesize a winner. Claude performs Round 3 (Battle_Test)
    per skill SKILL.md 'Adjudication' (synthesis): SKILL.md 'Confidence labelling' confidence labels on every claim, then a synthesis
    stating what was kept/discarded from each side and why. Explicit-trigger only —
    costs 2x a parallel call (4 dispatches).

    v14.4: dispatches are logged under the banner's task_id, but fleet_stats EXCLUDES
    adversarial mode from its default co-failure stat — this mode induces disagreement
    by design, so counting it as error correlation would be a self-inflicted bias.
    """
    tid = _new_task_id()
    return _banner(tid) + _run_adversarial(task, model_a="deepseek-v4-flash",
                                           model_b="x-ai/grok-4.6",
                                           reasoning_effort=reasoning_effort, task_id=tid)


@mcp.tool()
def call_model(model: str, prompt: str, reasoning_effort: str = "max") -> str:
    """v14 — call ANY model in the fleet by registry id and WAIT. QUICK tasks only;
    for deep work use orchestra_start(mode="model", model=...).

    Fleet ids: deepseek-v4-pro, deepseek-v4-flash, glm-5.3, moonshotai/kimi-k3,
    x-ai/grok-4.6, google/gemini-3.7-flash. Call list_fleet() for classes and budgets.

    Returns raw text, or a tagged string. [SUBSTITUTED] means the endpoint returned a
    well-formed 200 from a DIFFERENT model or host than was pinned: treat it as REJECT,
    never REFINE — re-dispatching to a host that already ignored the pin repeats the
    failure. If the substituted call was serving a Verifier role, discard the verdict;
    unknown provenance cannot satisfy skill SKILL.md 'Verification ladder'. Result
    begins with a [orchestra task_id=...] banner — reuse it in log_outcome.
    """
    tid = _new_task_id()
    return _banner(tid) + _logged_call(model, [{"role": "user", "content": prompt}],
                                       task_id=tid, role="candidate", mode="sync",
                                       glm_reasoning_effort=reasoning_effort)


@mcp.tool()
def list_fleet() -> str:
    """Return the v14 model registry: ids, correlation classes, output budgets,
    pinned host, and whether each provider's API key is actually present.

    Use this before choosing a Verifier (skill SKILL.md 'Verification ladder' (L2 cross-class) requires a class different from
    the Worker's) and whenever a call returns [SKIPPED] or [ERROR] — it shows at a
    glance which keys are missing. A missing OPENROUTER_API_KEY removes three models
    and the only non-CN-OW class, US-CLOSED, which makes the SKILL.md 'Verification ladder' (L2 cross-class) rule unsatisfiable for
    every worker; say so explicitly rather than silently same-class verifying.
    """
    rows = []
    for mid, m in _MODELS.items():
        key_present = bool(os.environ.get(_CONFIGS[m["provider"]]["key_env"], ""))
        rows.append({
            "model": mid,
            "class": m["klass"],
            "provider": m["provider"],
            # v14.5: show the ORDER host set too — a model pinned via `order` (Gemini,
            # Kimi) was previously displayed "UNPINNED", which reads as "not available"
            # and misled the caller into skipping a model that routes perfectly well. The
            # "order: " prefix keeps it from being mistaken for a single literal hostname
            # (GLM review) — it is a fallback chain, not a pin.
            "pinned_host": (m["pin"] or (("order: " + " > ".join(m["order"])) if m.get("order") else None)
                            or ("n/a" if m["provider"] != "openrouter"
                                else "UNPINNED — run /endpoints first")),
            "max_output_tokens": m["max_out"],
            "api_key_present": key_present,
        })
    classes = sorted({m["klass"] for m in _MODELS.values()})
    reachable = sorted({m["klass"] for mid, m in _MODELS.items()
                        if os.environ.get(_CONFIGS[m["provider"]]["key_env"], "")})
    return json.dumps({
        "fleet": rows,
        "correlation_classes": classes,
        "classes_reachable_now": reachable,
        "cross_class_verification_available": len(reachable) > 1,
        "note": "Verifier.class must differ from Worker.class (skill SKILL.md 'Verification ladder' (L2 cross-class)). This table "
                "is a prior based on lab lineage, not a measured error-correlation "
                "result. The mechanism to replace the guess with a measurement now "
                "exists as of v14.4 — call fleet_stats() (SKILL.md 'Outcome log'); the "
                "prior stands only until that has enough paired data.",
    }, indent=2)


@mcp.tool()
def log_outcome(task_id: str, model: str, correctness: str,
                note: str = "", disposition: str = "") -> str:
    """Record how ONE model's output on a dispatch actually turned out, AFTER Claude has
    adjudicated it. This is the step that turns the data flywheel: dispatch rows (written
    automatically by the server) say what the fleet produced; outcome rows say whether it
    was right. Without outcomes, fleet_stats can only report infra failure rates — never
    the error CORRELATION the fleet card's routing priors are currently GUESSING from lab
    lineage. See SKILL.md 'Outcome log'.

    Call it once per model you adjudicated, reusing the task_id the dispatch surfaced
    (the [orchestra task_id=...] banner in sync mode, or the job_id in job mode) and
    NAMING THE MODEL. A per-task verdict can't be attributed when 2-4 models share a
    task_id — which is exactly the paired data the flywheel needs, so the model argument
    is required (both reviewers, HIGH).

    correctness — the error axis, the only thing co-failure is computed on:
      CORRECT           — judged right (ideally after an L1 deterministic check)
      WRONG             — judged wrong
      OVERTURNED_BY_L1  — a deterministic check overrode the model
      UNVERIFIED        — no check existed; carried as unverified (L0)
    disposition (optional, orthogonal — what you DID with it): USED | DISCARDED. A
    DISCARDED-but-correct redundant answer is NOT an error and must not be logged WRONG.

    Re-adjudication is an UPDATE, not a new fact: calling again for the same
    (task_id, model) supersedes the earlier row (fleet_stats keeps last-writer-wins)."""
    c = correctness.strip().upper()
    if c not in _CORRECTNESS:
        return f"[ERROR] correctness must be one of {list(_CORRECTNESS)} — got {correctness!r}"
    d = disposition.strip().upper()
    if d and d not in _DISPOSITION:
        return f"[ERROR] disposition must be one of {[x for x in _DISPOSITION if x]} or empty — got {disposition!r}"
    if not task_id.strip() or not model.strip():
        return "[ERROR] task_id and model are both required — per-model verdicts are the whole point"
    _log_event({"type": "outcome", "ts": _now(), "task_id": task_id.strip(),
                "model": model.strip(), "correctness": c, "disposition": d,
                "note": note.strip()})
    return f"logged {c} for {model.strip()} on task {task_id.strip()}"


@mcp.tool()
def fleet_stats(last_n: int = 1000, mode: str = "") -> str:
    """Read the dispatch log and report what the fleet has actually DONE — the measured
    counterpart to fleet-card.md's ASSUMED routing priors, and the answer to list_fleet's
    'BLOCKING' correlation measurement. Reads the last `last_n` lines of the append-only
    log; optional `mode` filters dispatch rows (sync|parallel|adversarial|job).

    Reports, honestly:
      1. per_model — dispatches, infra failure-rate (SKIPPED excluded from the
         denominator — a routing skip is not a model failure), and outcome COVERAGE
         (how many dispatches got a correctness verdict fed back). Low coverage is the
         flywheel running dry, surfaced not hidden.
      2. outstanding — dispatched (task, model) pairs with NO verdict yet: the nag that
         keeps the highest-volume paths from silently contributing zero.
      3. cross_class_co_failure — over tasks where two DIFFERENT-class models each got a
         correctness verdict, how often BOTH were WRONG/OVERTURNED. This is the fleet_stats() / SKILL.md 'Outcome log'
         seed. adversarial mode is EXCLUDED (it induces disagreement by design), and a
         class-pair rate prints ONLY at n >= _MIN_PAIRS; below that it says so.
      4. worst_recent — the latest non-OK dispatches: 'sort by loss, read the ten'.
      5. log_write_failures — nonzero means the logger itself is failing and every stat
         above is stale; a silent dead flywheel is the worst case, so it is loud here."""
    lines = []
    try:
        with _LOG_LOCK:
            if _LOG_PATH.exists():
                with open(_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
    except Exception as e:
        return json.dumps({"error": f"could not read log: {e}", "log_path": str(_LOG_PATH)})
    lines = lines[-max(1, last_n):]

    dispatches, outcomes, malformed = [], [], 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)          # skip-and-count a torn tail line rather than crash
        except Exception:
            malformed += 1
            continue
        if rec.get("type") == "dispatch":
            if mode and rec.get("mode") != mode:
                continue
            dispatches.append(rec)
        elif rec.get("type") == "outcome":
            outcomes.append(rec)

    if not dispatches:
        return json.dumps({"note": "no dispatches in window — nothing measured yet",
                           "log_path": str(_LOG_PATH), "lines_scanned": len(lines),
                           "malformed_lines_skipped": malformed,
                           "log_write_failures": _LOG_FAILURES}, indent=2)

    outcome_by = {}                       # last-writer-wins per (task_id, model)
    for o in outcomes:
        outcome_by[(o.get("task_id"), o.get("model"))] = o

    per = defaultdict(lambda: {"dispatches": 0, "infra_fail": 0, "skipped": 0, "klass": "?"})
    mode_by, klass_by, dispatched_pairs = {}, {}, set()
    for d in dispatches:
        m = d.get("model"); key = (d.get("task_id"), m)
        p = per[m]; p["klass"] = d.get("klass", "?"); p["dispatches"] += 1
        tag = d.get("outcome", "OK")
        if tag == "SKIPPED":
            p["skipped"] += 1
        elif tag != "OK":
            p["infra_fail"] += 1
        mode_by[key] = d.get("mode"); klass_by[key] = d.get("klass", "?")
        dispatched_pairs.add(key)

    per_model = {}
    for m, p in per.items():
        denom = p["dispatches"] - p["skipped"]
        verdicts = sum(1 for (tid, mm) in dispatched_pairs
                       if mm == m and (tid, mm) in outcome_by)
        per_model[m] = {"class": p["klass"], "dispatches": p["dispatches"],
                        "infra_fail_rate": round(p["infra_fail"] / denom, 3) if denom else None,
                        "skipped": p["skipped"],
                        "outcome_coverage": round(verdicts / p["dispatches"], 3)}

    covered = sum(1 for pr in dispatched_pairs if pr in outcome_by)
    outstanding = len(dispatched_pairs) - covered

    # cross-class co-failure over NON-adversarial pairs with a correctness verdict
    task_models = defaultdict(dict)       # task_id -> {model: (klass, correctness)}
    for (tid, m), o in outcome_by.items():
        if mode_by.get((tid, m)) == "adversarial":
            continue
        c = o.get("correctness")
        if c in ("CORRECT", "WRONG", "OVERTURNED_BY_L1"):
            task_models[tid][m] = (klass_by.get((tid, m), "?"), c)

    pair_tot, pair_cofail = defaultdict(int), defaultdict(int)
    for tid, mm in task_models.items():
        for (m1, (k1, c1)), (m2, (k2, c2)) in itertools.combinations(mm.items(), 2):
            if k1 == k2:                  # same-class pair is not a decorrelation measurement
                continue
            pk = " x ".join(sorted((k1, k2)))
            pair_tot[pk] += 1
            if c1 in _ERROR_SET and c2 in _ERROR_SET:
                pair_cofail[pk] += 1

    co_failure = {}
    for pk, tot in pair_tot.items():
        if tot >= _MIN_PAIRS:
            co_failure[pk] = {"pairs": tot, "both_error": pair_cofail[pk],
                              "co_failure_rate": round(pair_cofail[pk] / tot, 3)}
        else:
            co_failure[pk] = f"insufficient paired data (n={tot}, need >= {_MIN_PAIRS})"

    worst = [{"ts": d.get("ts"), "model": d.get("model"), "mode": d.get("mode"),
              "outcome": d.get("outcome"), "task_id": d.get("task_id")}
             for d in dispatches if d.get("outcome") not in ("OK", "SKIPPED")][-10:]

    return json.dumps({
        "window": {"lines_scanned": len(lines), "dispatches": len(dispatches),
                   "distinct_outcomes": len(outcome_by), "mode_filter": mode or "all",
                   "malformed_lines_skipped": malformed},
        "per_model": per_model,
        "coverage": f"{covered}/{len(dispatched_pairs)} (task,model) pairs have a verdict; "
                    f"{outstanding} outstanding — low coverage means the flywheel is running dry",
        "cross_class_co_failure": co_failure or "no cross-class verdict pairs yet",
        "co_failure_caveat": "answer co-error on non-adversarial pairs only, and still "
                             "Claude-judged — a weak signal until L1-verified outcomes "
                             "dominate. The fleet_stats() / SKILL.md 'Outcome log' seed, not the finished result.",
        "worst_recent": worst,
        "log_write_failures": _LOG_FAILURES,
        "log_path": str(_LOG_PATH),
    }, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()  # stdio transport — default, correct for local Claude Desktop integration
