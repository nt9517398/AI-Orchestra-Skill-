#!/usr/bin/env python3
"""Phase 2 of the v21 plan: which fleet models can drive Claude Code's agent loop?

For each model this runs one small script through an Anthropic-compatible Messages
endpoint (OpenRouter's, by default) and records pass / partial / fail per feature:

  basic           a plain reply
  pin             the request is accepted with the registry's host pin (OpenRouter `provider`)
  forced_tool     tool_choice forces a tool; the call comes back well formed
  tool_loop       a multi-turn loop: call tools, read results, call more, then answer (GATE)
  parallel_tools  two independent calls in one assistant turn
  streaming       server-sent events, including a tool call whose JSON arrives in pieces
  caching         cache_control on a system block; the second request reads from cache
  thinking        a `thinking` request is accepted (Claude Code sends one on every call)
  claude_code_worker  a real headless Claude Code run fixes a bug and the test passes (GATE)

A model is ELIGIBLE only if both gates pass (UPGRADE-PLAN-v21.md, Phase 2).

Every request and response goes to a JSONL file; `report` builds docs/agent-compatibility.md
from those files alone, so a verdict can always be traced to a raw exchange. The key is sent in
a header and is never written to a log. Standard library only.

    OPENROUTER_API_KEY=... python docs/diagnostics/agent_compat.py run --out docs/agent-compat-logs/RUN
    python docs/diagnostics/agent_compat.py cap --out docs/agent-compat-logs/RUN     # no key needed
    python docs/diagnostics/agent_compat.py report --logs docs/agent-compat-logs/RUN
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_VERSION = "2023-06-01"
DEFAULT_BASE = "https://openrouter.ai/api"
FEATURES = ("basic", "pin", "forced_tool", "tool_loop", "parallel_tools", "streaming",
            "caching", "thinking", "claude_code_worker")
GATES = ("tool_loop", "claude_code_worker")
# Mirrors orchestra_mcp_server.py (a test fails if the two drift apart).
MODELS = {
    "meta/muse-spark-1.3": {"wire": "meta/muse-spark-1.3", "pin": "Meta"},
    "meta/muse-ember": {"wire": "meta/muse-ember", "pin": "Meta"},
    "moonshotai/kimi-k3.1": {"wire": "moonshotai/kimi-k3.1", "pin": "wafer/us"},
    "openai/gpt-terra-latest": {"wire": "~openai/gpt-terra-latest", "pin": None},
    "x-ai/grok-latest": {"wire": "~x-ai/grok-latest", "pin": None},
    "google/gemini-flash-latest": {"wire": "~google/gemini-flash-latest", "pin": None},
    "qwen/qwen4-max": {"wire": "qwen/qwen4-max", "pin": None},
    "tencent/hy4": {"wire": "tencent/hy4", "pin": None},
}
TOOLS = [
    {"name": "add", "description": "Add two integers.",
     "input_schema": {"type": "object", "properties": {"a": {"type": "integer"},
                                                       "b": {"type": "integer"}},
                      "required": ["a", "b"]}},
    {"name": "multiply", "description": "Multiply two integers.",
     "input_schema": {"type": "object", "properties": {"a": {"type": "integer"},
                                                       "b": {"type": "integer"}},
                      "required": ["a", "b"]}},
]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


# ---------------------------------------------------------------------------------------------
# Redaction, SSE parsing, HTTP
# ---------------------------------------------------------------------------------------------

def redact(text, key: str = "") -> str:
    s = str(text)
    if key and len(key) >= 8:
        s = s.replace(key, "[REDACTED]")
    s = re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "[REDACTED]", s)
    s = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", "[REDACTED]", s)
    return re.sub(r"(?i)((?:api[_-]?key|authorization|token|secret)[\"']?\s*[:=]\s*[\"']?)"
                  r"[A-Za-z0-9._\-]{8,}", lambda m: m.group(1) + "[REDACTED]", s)


def parse_sse(lines) -> list:
    """[(event, data)] from an SSE stream. Data that is not JSON is kept as a string."""
    events, name, data = [], None, []

    def flush():
        nonlocal name, data
        if name is not None or data:
            raw = "\n".join(data)
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = raw
            events.append((name or (payload.get("type") if isinstance(payload, dict) else None),
                           payload))
        name, data = None, []

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.rstrip("\r\n")
        if not line:
            flush()
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    flush()
    return events


class BudgetExceeded(Exception):
    pass


class Resp:
    def __init__(self, status, json_body, text, headers, elapsed, events=None, error=""):
        self.status, self.json, self.text = status, json_body, text
        self.headers, self.elapsed, self.events, self.error = headers, elapsed, events or [], error

    @property
    def ok(self):
        return self.status == 200 and not self.error


class Client:
    """One model's connection. Logs every exchange; never logs the key."""

    def __init__(self, base: str, key: str, log_path, max_requests: int = 80, timeout: int = 180):
        u = urllib.parse.urlparse(base)
        if u.scheme != "https" and not (u.scheme == "http" and u.hostname in LOOPBACK):
            raise ValueError(f"refusing {base!r}: the key would travel unencrypted "
                             f"(https, or http on loopback, only)")
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout
        self.max_requests, self.requests, self.tokens = max_requests, 0, {"in": 0, "out": 0}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), **row}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def post(self, model: str, feature: str, body: dict, stream: bool = False) -> Resp:
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"request budget of {self.max_requests} used up")
        self.requests += 1
        if stream:
            body = {**body, "stream": True}
        req = urllib.request.Request(
            f"{self.base}/v1/messages", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "anthropic-version": API_VERSION,
                     "Authorization": f"Bearer {self.key}"}, method="POST")
        t0, status, headers, text, events, error, parsed = time.time(), 0, {}, "", [], "", None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status, headers = r.status, dict(r.headers)
                if stream and "text/event-stream" in headers.get("Content-Type", ""):
                    events = parse_sse(r)
                    text = f"{len(events)} events"
                else:
                    text = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            status, headers = e.code, dict(e.headers or {})
            text = e.read().decode("utf-8", "replace")
        except Exception as e:                       # connection refused, timeout, TLS ...
            error = f"{type(e).__name__}: {e}"
        if text and not events:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        usage = (parsed or {}).get("usage") or {}
        self.tokens["in"] += int(usage.get("input_tokens") or 0)
        self.tokens["out"] += int(usage.get("output_tokens") or 0)
        elapsed = round(time.time() - t0, 2)
        self.log({"type": "request", "model": model, "feature": feature, "status": status,
                  "elapsed_s": elapsed, "stream": stream, "request": json.loads(redact(
                      json.dumps(body), self.key)) if len(json.dumps(body)) < 60000 else "(large)",
                  "response": redact(text[:20000], self.key) if not events else
                  [[n, d] for n, d in events][:400], "error": redact(error, self.key),
                  "headers": {k.lower(): redact(v, self.key) for k, v in headers.items()
                              if k.lower().startswith(("x-", "openrouter", "retry"))}})
        return Resp(status, parsed, text, headers, elapsed, events, error)


# ---------------------------------------------------------------------------------------------
# Features. Each returns (status, note); status is pass | partial | fail | skip.
# ---------------------------------------------------------------------------------------------

def blocks(resp: Resp) -> list:
    return (resp.json or {}).get("content") or []


def tool_uses(bs: list) -> list:
    return [b for b in bs if b.get("type") == "tool_use"]


def text_of(bs: list) -> str:
    return "".join(b.get("text", "") for b in bs if b.get("type") == "text")


def run_tool(name: str, args: dict):
    a, b = int(args.get("a")), int(args.get("b"))
    return a + b if name == "add" else a * b if name == "multiply" else "unknown tool"


def _why(resp: Resp) -> str:
    if resp.error:
        return resp.error[:160]
    return f"HTTP {resp.status}: {redact(resp.text, '')[:160]}"


def body_for(spec: dict, use_pin: bool, **kw) -> dict:
    b = {"model": spec["wire"], **kw}
    b.update(spec.get("extra", {}))
    if use_pin and spec.get("pin"):
        b["provider"] = {"only": [spec["pin"]], "allow_fallbacks": False}
    return b


def f_basic(c: Client, label, spec, use_pin):
    r = c.post(label, "basic", body_for(spec, use_pin, max_tokens=64, messages=[
        {"role": "user", "content": "Reply with the single word: pong"}]))
    if not r.ok:
        return "fail", _why(r)
    t = text_of(blocks(r)).strip()
    if not t:
        return "fail", "HTTP 200 with no text"
    return ("pass", t[:40]) if "pong" in t.lower() else ("partial", f"answered {t[:40]!r}")


def f_pin(c: Client, label, spec, use_pin):
    if not spec.get("pin"):
        return "skip", "no host pin for this model"
    r = c.post(label, "pin", body_for(spec, True, max_tokens=32, messages=[
        {"role": "user", "content": "Reply with the single word: pong"}]))
    if not r.ok:
        return "fail", "pinned request rejected: " + _why(r)
    served = next((v for k, v in r.headers.items() if "provider" in k.lower()), None) or \
        (r.json or {}).get("provider")
    if served is None:
        return "partial", "accepted, but the response does not name the serving host"
    return ("pass", f"served by {served}") if str(served).lower() == spec["pin"].lower() else \
        ("fail", f"pinned to {spec['pin']} but served by {served}")


def f_forced_tool(c: Client, label, spec, use_pin):
    r = c.post(label, "forced_tool", body_for(
        spec, use_pin, max_tokens=256, tools=TOOLS[:1],
        tool_choice={"type": "tool", "name": "add"},
        messages=[{"role": "user", "content": "Use the add tool to add 2 and 3."}]))
    if not r.ok:
        return "fail", _why(r)
    uses = tool_uses(blocks(r))
    if not uses:
        return "fail", "HTTP 200 but no tool_use block"
    u = uses[0]
    args = u.get("input") or {}
    if u.get("name") == "add" and args.get("a") == 2 and args.get("b") == 3 and u.get("id"):
        return "pass", "add(a=2, b=3)"
    return "fail", f"malformed call: {json.dumps(u)[:120]}"


def f_tool_loop(c: Client, label, spec, use_pin, max_turns: int = 6):
    msgs = [{"role": "user", "content": "Compute (7 + 5) * 3 using the tools, one step at a "
                                        "time. When you have the result, reply with just the "
                                        "number."}]
    calls, final = 0, ""
    for turn in range(max_turns):
        r = c.post(label, "tool_loop", body_for(spec, use_pin, max_tokens=512, tools=TOOLS,
                                                messages=msgs))
        if not r.ok:
            return "fail", f"turn {turn + 1}: {_why(r)}"
        bs = blocks(r)
        uses = tool_uses(bs)
        if not uses:
            final = text_of(bs)
            break
        calls += len(uses)
        msgs.append({"role": "assistant", "content": bs})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u.get("id"),
             "content": str(run_tool(u.get("name"), u.get("input") or {}))} for u in uses]})
    else:
        return "fail", f"no final answer within {max_turns} turns ({calls} tool calls)"
    if calls < 2:
        return "fail", f"used {calls} tool call(s); a 2-step sum needed the tools"
    return ("pass", f"{calls} tool calls, answered {final.strip()[:20]!r}") if "36" in final \
        else ("fail", f"{calls} tool calls but answered {final.strip()[:40]!r}")


def f_parallel_tools(c: Client, label, spec, use_pin):
    r = c.post(label, "parallel_tools", body_for(
        spec, use_pin, max_tokens=512, tools=TOOLS[:1],
        messages=[{"role": "user", "content": "Call the add tool twice at the same time, in one "
                                              "step: once for 1 and 2, once for 30 and 40."}]))
    if not r.ok:
        return "fail", _why(r)
    uses = tool_uses(blocks(r))
    if len(uses) >= 2:
        return "pass", f"{len(uses)} tool calls in one turn"
    if len(uses) == 1:
        return "partial", "one tool call per turn (sequential only)"
    return "fail", "no tool_use block"


def f_streaming(c: Client, label, spec, use_pin):
    t = c.post(label, "streaming", body_for(spec, use_pin, max_tokens=64, messages=[
        {"role": "user", "content": "Count from 1 to 5."}]), stream=True)
    if not t.ok or not t.events:
        return "fail", "text stream: " + (_why(t) if not t.ok else "no SSE events")
    names = [n for n, _ in t.events]
    text = "".join(d.get("delta", {}).get("text", "") for n, d in t.events
                   if isinstance(d, dict) and n == "content_block_delta")
    if "message_start" not in names or "message_stop" not in names or not text:
        return "fail", f"text stream incomplete: {sorted(set(map(str, names)))}"
    r = c.post(label, "streaming", body_for(
        spec, use_pin, max_tokens=256, tools=TOOLS[:1],
        tool_choice={"type": "tool", "name": "add"},
        messages=[{"role": "user", "content": "Use the add tool to add 2 and 3."}]), stream=True)
    if not r.ok or not r.events:
        return "partial", "text streams; tool stream: " + (_why(r) if not r.ok else "no events")
    pieces = "".join(d["delta"].get("partial_json", "") for n, d in r.events
                     if isinstance(d, dict) and n == "content_block_delta"
                     and isinstance(d.get("delta"), dict))
    try:
        args = json.loads(pieces)
    except ValueError:
        return "partial", f"text streams; tool arguments do not reassemble: {pieces[:60]!r}"
    return ("pass", "text and tool-call streams reassemble") if args.get("a") == 2 and \
        args.get("b") == 3 else ("partial", f"text streams; tool arguments wrong: {args}")


def f_caching(c: Client, label, spec, use_pin):
    system = [{"type": "text", "text": "Reference text. " * 450,
               "cache_control": {"type": "ephemeral"}}]
    reads = []
    for _ in range(2):
        r = c.post(label, "caching", body_for(spec, use_pin, max_tokens=16, system=system,
                                              messages=[{"role": "user", "content": "Say ok."}]))
        if not r.ok:
            return "fail", _why(r)
        u = (r.json or {}).get("usage") or {}
        reads.append((int(u.get("cache_creation_input_tokens") or 0),
                      int(u.get("cache_read_input_tokens") or 0)))
    if reads[1][1] > 0:
        return "pass", f"second request read {reads[1][1]} tokens from cache"
    if reads[0][0] > 0:
        return "partial", "cache written but not read back"
    return "fail", "no cache fields in usage (informational: not a gate)"


def f_thinking(c: Client, label, spec, use_pin):
    r = c.post(label, "thinking", body_for(
        spec, use_pin, max_tokens=2048, thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "What is 17 * 19? Think briefly, then answer."}]))
    if not r.ok:
        return "fail", "thinking request rejected: " + _why(r)
    n = sum(1 for b in blocks(r) if b.get("type") in ("thinking", "redacted_thinking"))
    return ("pass", f"{n} thinking block(s)") if n else \
        ("partial", "accepted, but no thinking block returned")


FEATURE_FUNCS = {"basic": f_basic, "pin": f_pin, "forced_tool": f_forced_tool,
                 "tool_loop": f_tool_loop, "parallel_tools": f_parallel_tools,
                 "streaming": f_streaming, "caching": f_caching, "thinking": f_thinking}


# ---------------------------------------------------------------------------------------------
# A real Claude Code worker, in isolation
# ---------------------------------------------------------------------------------------------

BROKEN = "def median(xs):\n    xs = sorted(xs)\n    return xs[len(xs) // 2]\n"
CHECK = ("from median import median\n"
         "assert median([3, 1, 2]) == 2\n"
         "assert median([7]) == 7\n"
         "assert median([4, 1, 3, 2]) == 2.5\n")
TASK = ("median() in median.py returns the wrong value for lists of even length. Fix it. "
        "Do not edit check.py. You can only read and edit files.")


def worker_env(key: str, base: str, wire: str, home: str, max_output=None) -> dict:
    """Everything the worker may see: a PATH, an empty HOME, and its one credential."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home,
           "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": "",
           "ANTHROPIC_MODEL": wire, "ANTHROPIC_SMALL_FAST_MODEL": wire,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    if max_output:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output)
    return env


def worker_command(claude: str, wire: str, task: str = TASK) -> list:
    return [claude, "-p", task, "--model", wire, "--tools", "Read,Edit",
            "--allowedTools", "Read,Edit", "--permission-mode", "acceptEdits",
            "--max-turns", "8", "--strict-mcp-config", "--setting-sources", "",
            "--output-format", "json"]


def run_checker(project: Path) -> int:
    """Runs check.py, which imports the worker-edited median.py: that executes the worker's
    code on this machine. Fine for a Read/Edit worker in a throwaway folder; a worker with
    shell access needs the container isolation of Phase 3. -I would drop the script's folder
    from sys.path, so -E -s -B (no environment, no user site, no bytecode) is used instead."""
    return subprocess.run([sys.executable, "-E", "-s", "-B", "check.py"], cwd=project,
                          capture_output=True, timeout=30).returncode


def run_worker(claude: str, base: str, key: str, wire: str, max_output=None, timeout: int = 300,
               runner=subprocess.run) -> dict:
    """Pass only if the checker fails before, passes after, median.py changed, check.py did not,
    and the worker exited cleanly. The checker is run here, never by the worker."""
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as home:
        project = Path(work)
        (project / "median.py").write_text(BROKEN)
        (project / "check.py").write_text(CHECK)
        before = run_checker(project)
        t0 = time.time()
        try:
            p = runner(worker_command(claude, wire), cwd=project, capture_output=True,
                       text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                       env=worker_env(key, base, wire, home, max_output))
            rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
        except subprocess.TimeoutExpired:
            rc, out, err = -1, "", "timed out"
        after = run_checker(project)
        changed = (project / "median.py").read_text() != BROKEN
        untouched = (project / "check.py").read_text() == CHECK
    detail = {"claude_exit": rc, "check_before": before, "check_after": after,
              "median_changed": changed, "check_untouched": untouched,
              "seconds": round(time.time() - t0, 1), "stdout": redact(out[:3000], key),
              "stderr": redact(err[:1000], key)}
    if before == 0:
        return {"status": "fail", "note": "the broken project already passed its check", **detail}
    if not untouched:
        return {"status": "fail", "note": "the worker edited check.py", **detail}
    if rc != 0:
        return {"status": "fail", "note": f"Claude Code exited {rc}", **detail}
    if not changed:
        return {"status": "fail", "note": "median.py is unchanged", **detail}
    if after != 0:
        return {"status": "fail", "note": "the check still fails", **detail}
    return {"status": "pass", "note": "check failed before and passes after", **detail}


# ---------------------------------------------------------------------------------------------
# What Claude Code sends, measured against a local recording server (no key, no network)
# ---------------------------------------------------------------------------------------------

class RecordingStub:
    """A loopback Messages server that records each request and answers 'ok'."""

    def __init__(self):
        self.requests, outer = [], self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                outer.requests.append({"path": self.path, "model": body.get("model"),
                                       "max_tokens": body.get("max_tokens"),
                                       "thinking": body.get("thinking"),
                                       "tools": len(body.get("tools") or [])})
                if "count_tokens" in self.path:
                    return self._json({"input_tokens": 10})
                msg = {"id": "msg_stub", "type": "message", "role": "assistant",
                       "model": body.get("model"), "content": [{"type": "text", "text": "ok"}],
                       "stop_reason": "end_turn", "usage": {"input_tokens": 10,
                                                            "output_tokens": 2}}
                if not body.get("stream"):
                    return self._json(msg)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                start = {**msg, "content": [], "stop_reason": None}
                for name, data in (
                        ("message_start", {"type": "message_start", "message": start}),
                        ("content_block_start", {"type": "content_block_start", "index": 0,
                                                 "content_block": {"type": "text", "text": ""}}),
                        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "ok"}}),
                        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                        ("message_delta", {"type": "message_delta",
                                           "delta": {"stop_reason": "end_turn"},
                                           "usage": {"output_tokens": 2}}),
                        ("message_stop", {"type": "message_stop"})):
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

            def _json(self, obj):
                out = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def claude_version(claude: str) -> str:
    try:
        return subprocess.run([claude, "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def run_cap_check(claude: str, wire: str = "meta/muse-spark-1.3",
                  caps=(None, 120000, 640000), runner=subprocess.run) -> list:
    """What max_tokens (and thinking mode) a worker sends for `wire`, with and without
    CLAUDE_CODE_MAX_OUTPUT_TOKENS. Measured, not documented behaviour."""
    rows, version = [], claude_version(claude)
    for cap in caps:
        stub = RecordingStub()
        try:
            with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as home:
                try:
                    runner(worker_command(claude, wire, "say ok"), cwd=work, capture_output=True,
                           text=True, timeout=120, stdin=subprocess.DEVNULL,
                           env=worker_env("dummy", stub.url, wire, home, cap))
                except Exception as e:
                    rows.append({"type": "cap", "claude_version": version, "wire": wire,
                                 "requested": cap, "error": f"{type(e).__name__}: {e}"})
                    continue
            seen = [r for r in stub.requests if "count_tokens" not in r["path"]]
            rows.append({"type": "cap", "claude_version": version, "wire": wire,
                         "requested": cap, "requests": len(seen),
                         "max_tokens_sent": sorted({r["max_tokens"] for r in seen
                                                    if r["max_tokens"] is not None}),
                         "thinking_sent": sorted({json.dumps(r["thinking"]) for r in seen}),
                         "tools_sent": sorted({r["tools"] for r in seen})})
        finally:
            stub.close()
    return rows


# ---------------------------------------------------------------------------------------------
# Running models, eligibility, the report
# ---------------------------------------------------------------------------------------------

def run_model(client: Client, label: str, spec: dict, features=FEATURES, worker=None) -> dict:
    """{feature: (status, note)}. A feature that raises is recorded as a failure, never skipped
    silently; running out of request budget stops the model and records what was not run."""
    results, use_pin = {}, False
    for feat in features:
        try:
            if feat == "claude_code_worker":
                if not worker:
                    status, note = "skip", "not run (needs the claude CLI and --worker)"
                else:
                    d = run_worker(**worker)
                    client.log({"type": "worker", "model": label, **d})
                    status, note = d["status"], d["note"]
            else:
                status, note = FEATURE_FUNCS[feat](client, label, spec, use_pin)
                if feat == "pin":
                    use_pin = status in ("pass", "partial")
        except BudgetExceeded as e:
            status, note = "skip", f"not run: {e}"
        except Exception as e:
            status, note = "fail", f"{type(e).__name__}: {redact(e, client.key)[:160]}"
        results[feat] = (status, note)
        client.log({"type": "verdict", "model": label, "feature": feat, "status": status,
                    "note": note})
    return results


def eligible(results: dict) -> bool:
    return all(results.get(g, ("skip",))[0] == "pass" for g in GATES)


def read_logs(log_dir) -> list:
    rows = []
    for p in sorted(Path(log_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def eligible_models(log_dir) -> list:
    """Registry ids whose logged verdicts pass both gates."""
    verdicts = {}
    for r in read_logs(log_dir):
        if r.get("type") == "verdict":
            verdicts.setdefault(r["model"], {})[r["feature"]] = (r["status"], r["note"])
    return sorted(m for m, res in verdicts.items() if eligible(res))


def build_report(log_dir) -> str:
    """The compatibility report, from the raw logs only."""
    rows = read_logs(log_dir)
    verdicts, caps, n_req, last = {}, [], 0, ""
    for r in rows:
        last = max(last, r.get("ts", ""))
        if r.get("type") == "verdict":
            verdicts.setdefault(r["model"], {})[r["feature"]] = (r["status"], r["note"])
        elif r.get("type") == "cap":
            caps.append(r)
        elif r.get("type") == "request":
            n_req += 1
    out = ["# Agent compatibility (Phase 2)", "",
           "Built by `docs/diagnostics/agent_compat.py report` from the raw logs in "
           f"`{Path(log_dir).as_posix()}`. {n_req} model requests logged"
           + (f"; last entry {last}." if last else "."), ""]
    out += ["## What Claude Code sends", ""]
    if caps:
        out += [f"Measured against a local recording server with Claude Code "
                f"`{caps[0].get('claude_version', '?')}` and the model id `{caps[0].get('wire')}`.",
                "", "| CLAUDE_CODE_MAX_OUTPUT_TOKENS | max_tokens sent | thinking sent | tools |",
                "|---|---|---|---|"]
        for c in caps:
            out.append(f"| {c.get('requested') or '(unset)'} | "
                       + (", ".join(map(str, c.get("max_tokens_sent", []))) or
                          c.get("error", "none seen")) + f" | "
                       + (", ".join(c.get("thinking_sent", [])) or "-") + " | "
                       + (", ".join(map(str, c.get("tools_sent", []))) or "-") + " |")
        out.append("")
    else:
        out += ["Not measured yet (`agent_compat.py cap`).", ""]
    out += ["## Feature matrix", ""]
    if not verdicts:
        out += ["No model has been run through the live matrix yet.", ""]
    else:
        out += ["| Model | " + " | ".join(FEATURES) + " | Eligible |",
                "|---|" + "---|" * (len(FEATURES) + 1)]
        for m, res in sorted(verdicts.items()):
            cells = [res.get(f, ("skip", ""))[0].upper() for f in FEATURES]
            out.append(f"| `{m}` | " + " | ".join(cells) + " | "
                       + ("**yes**" if eligible(res) else "no") + " |")
        out += ["", "### Notes", ""]
        for m, res in sorted(verdicts.items()):
            for f in FEATURES:
                if f in res and res[f][0] != "skip":
                    out.append(f"- `{m}` / {f}: {res[f][0].upper()} — {res[f][1]}")
        out.append("")
    out += ["## Rules", "",
            "- Eligible means both gates pass: `tool_loop` and `claude_code_worker`.",
            "- `caching`, `parallel_tools` and `thinking` are informational; they are not gates.",
            "- One run per model unless the logs show more. A pass is not a reliability figure.",
            "- `pin` checks the request is accepted with the registry's host pin; "
            "`partial` means the response did not say which host served it.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the live matrix (needs OPENROUTER_API_KEY)")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=DEFAULT_BASE)
    r.add_argument("--models", default=",".join(MODELS))
    r.add_argument("--features", default=",".join(f for f in FEATURES if f != "claude_code_worker"))
    r.add_argument("--worker", action="store_true", help="also run a real Claude Code worker")
    r.add_argument("--claude", default=shutil.which("claude") or "claude")
    r.add_argument("--max-output", type=int, default=120000)
    r.add_argument("--max-requests", type=int, default=60, help="per model")
    c = sub.add_parser("cap", help="measure what Claude Code sends (no key needed)")
    c.add_argument("--out", required=True)
    c.add_argument("--claude", default=shutil.which("claude") or "claude")
    c.add_argument("--wire", default="meta/muse-spark-1.3")
    p = sub.add_parser("report", help="build the report from existing logs")
    p.add_argument("--logs", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--eligible-json", default="", help="also write the models that pass both gates")
    a = ap.parse_args(argv)

    if a.cmd == "report":
        text = build_report(a.logs)
        if a.eligible_json:
            Path(a.eligible_json).write_text(json.dumps(
                {"models": eligible_models(a.logs), "built": time.strftime("%Y-%m-%d")}, indent=2),
                encoding="utf-8")
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "cap":
        with open(out / "_claude_code.jsonl", "a", encoding="utf-8") as f:
            for row in run_cap_check(a.claude, a.wire):
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                                    **row}) + "\n")
                print(row)
        return 0
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.stderr.write("OPENROUTER_API_KEY is not set; nothing was run.\n")
        return 2
    wanted = [m.strip() for m in a.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        sys.stderr.write(f"unknown model(s) {unknown}; known: {sorted(MODELS)}\n")
        return 2
    features = tuple(f.strip() for f in a.features.split(",") if f.strip())
    if a.worker and "claude_code_worker" not in features:
        features += ("claude_code_worker",)
    summary = {}
    for label in wanted:
        client = Client(a.base, key, out / f"{_safe(label)}.jsonl", max_requests=a.max_requests)
        worker = {"claude": a.claude, "base": a.base, "key": key,
                  "wire": MODELS[label]["wire"], "max_output": a.max_output} if a.worker else None
        summary[label] = run_model(client, label, MODELS[label], features, worker)
        print(f"{label}: " + ", ".join(f"{f}={s}" for f, (s, _) in summary[label].items())
              + f" | eligible={eligible(summary[label])} | requests={client.requests}"
              f" tokens in/out={client.tokens['in']}/{client.tokens['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
