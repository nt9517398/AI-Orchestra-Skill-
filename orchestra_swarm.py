#!/usr/bin/env python3
"""orchestra_swarm.py — run Claude Code agents whose model is a fleet model (stdlib only).

A subagent started from Claude Code always uses the session's own Claude model. A fleet agent is
therefore a separate headless `claude -p` process whose API calls go to OpenRouter's
Anthropic-compatible endpoint. This file does three things:

  fleet_count(n)  how many of n agents are fleet agents (60-70%, else 80-90%, else nearest)
  plan(...)       who is a fleet agent, which model drives it, which task it gets
  run             start the fleet agents, each in its own git worktree, and keep every result

Each fleet agent gets: a worktree of HEAD (tracked files only, so an untracked .env is never
there), a scrubbed environment (PATH, an empty HOME and its one OpenRouter credential), a tool
list, a turn cap and a time limit. Its changes come back as a diff that Claude reviews and
applies. Spend is NOT capped here (Claude Code has no spend flag): set an OpenRouter key limit.

TOOLS ARE PER AGENT. Every fleet agent has file tools. Bash is OFF unless the plan turns it on
for that agent, and then only for the four models in BASH_MODELS, only as the test-runner rule
BASH_RULE (never bare Bash), and only when the caller said the repo is public or a test project.
This is a permission rule, NOT isolation: see swarm.md. A Bash agent can still write and read
outside its worktree, so "never touches the real checkout" holds for file-only agents only.

    python orchestra_swarm.py run --plan results/<id>/plan.json --repo . --out results/<id>
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BANDS = ((0.60, 0.70), (0.80, 0.90))     # the owner's rule: 60-70%, and 80-90% when that is out of reach
TARGET = 0.65
FILE_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")     # every fleet agent; no Bash
# The only models that may be given Bash, kept in one place. Any other model gets file tools only,
# even if a plan asks for Bash (the launcher re-checks, so editing plan.json does not widen it).
BASH_MODELS = frozenset({"meta/muse-spark-1.3", "x-ai/grok-latest", "openai/gpt-terra-latest",
                         "qwen/qwen4-max"})
# The one shell permission: the test runner. Checked against Claude Code 2.1.332 with a stub model
# (bashprobe, 2026-11-25): this rule runs `python -m pytest ...`, and `env`, `curl`, `python -c`
# are denied in headless mode. It is never bare "Bash"; the assert makes a bad edit fail on import.
BASH_RULE = "Bash(python -m pytest:*)"
assert BASH_RULE.startswith("Bash(") and BASH_RULE.endswith(")") and len(BASH_RULE) > 10
BASH_GUARD = ("Bash is on for this plan. The public/test-repo flag is the ONLY guard on what the "
              "shell can reach: a permission rule is not isolation. A Bash agent can still read and "
              "write outside its worktree (verified offline), and a test run executes arbitrary "
              "code. With bubblewrap and socat installed the key is hidden from the shell and "
              "outside network is refused; without them every shell command fails closed. Full "
              "Bash stays blocked until container isolation exists.")
DEFAULT_BASE = "https://openrouter.ai/api"


def fleet_count(n: int) -> int:
    """Fleet agents among n. A single agent is not a swarm (0: Claude decides). Otherwise the
    count whose share falls in 60-70%; if no whole number does, in 80-90%; if neither, the
    share nearest 65%. At least one agent is always a Claude agent."""
    if n < 2:
        return 0
    for lo, hi in BANDS:
        mid = (lo + hi) / 2
        ok = [f for f in range(1, n) if lo - 1e-9 <= f / n <= hi + 1e-9]
        if ok:
            return min(ok, key=lambda f: (round(abs(f / n - mid), 9), -f))
    return min(range(1, n), key=lambda f: (round(abs(f / n - TARGET), 9), -f))


def band_of(n: int, f: int) -> str:
    share = f / n if n else 0
    for lo, hi in BANDS:
        if lo - 1e-9 <= share <= hi + 1e-9:
            return f"{int(lo * 100)}-{int(hi * 100)}%"
    return "outside both bands (no whole number fits)"


def _spread(models: list, k: int) -> list:
    """k models, round-robin across correlation classes so one lab's blind spot does not
    repeat across the swarm. `models`: dicts with id, klass."""
    by_class = {}
    for m in models:
        by_class.setdefault(m["klass"], []).append(m)
    queues = [list(v) for _, v in sorted(by_class.items())]
    out, i = [], 0
    while len(out) < k and any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out


def resolve_tools(agent: dict, public_or_test_repo: bool) -> dict:
    """What this fleet agent may use. Bash only when the agent asks, the caller vouched for the
    repo, and the model is in BASH_MODELS; otherwise file tools only, with the reason. Nothing
    in `agent` except `model` and `bash` is trusted: a `tools` list in a plan file is ignored."""
    tools, allowed, note = list(FILE_TOOLS), list(FILE_TOOLS), ""
    if agent.get("bash"):
        if public_or_test_repo is not True:
            note = "Bash refused: the plan does not say the repo is public or a test project"
        elif agent.get("model") not in BASH_MODELS:
            note = f"Bash refused: {agent.get('model')} is not in BASH_MODELS; file tools only"
        else:
            tools.append("Bash")
            allowed.append(BASH_RULE)
    return {"tools": tools, "allowed": allowed, "bash": "Bash" in tools, "bash_note": note}


def plan(n_agents: int, tasks=None, models=None, public_or_test_repo: bool = False) -> dict:
    """The swarm plan. `tasks`: strings, or dicts {"task": str, "claude_only": bool, "bash": bool}.
    A task marked claude_only (judgment, repository-wide context, anything sensitive) goes to a
    Claude agent. `bash: true` asks for the test-runner shell; it is refused unless
    `public_or_test_repo` is true, and only models in BASH_MODELS get it. `models`: the eligible
    fleet models, dicts with id, wire, klass, max_out."""
    models = list(models or [])
    n = int(n_agents)
    items = []
    for t in (tasks or []):
        d = t if isinstance(t, dict) else {"task": t}
        items.append({"task": str(d.get("task", "")), "claude_only": bool(d.get("claude_only")),
                      "bash": bool(d.get("bash")) and not d.get("claude_only")})
    if items and len(items) != n:
        return {"error": f"{len(items)} tasks for {n} agents: give one task per agent"}
    if n < 1:
        return {"error": "n_agents must be at least 1"}
    asks = [f"a{i + 1}" for i, d in enumerate(items) if d["bash"]]
    if asks and public_or_test_repo is not True:
        return {"error": f"Bash was requested for {', '.join(asks)} but the caller did not say the "
                         "repo is public or a test project (public_or_test_repo). Refused: that "
                         "flag is the only guard on what the shell can reach."}
    f = fleet_count(n)
    if f and not models:
        return {"error": "no eligible fleet model: run the compatibility harness first "
                         "(docs/diagnostics/agent_compat.py) or list one in swarm_eligible.json"}
    need_claude = [i for i, d in enumerate(items) if d["claude_only"]]
    if len(need_claude) > n - f:
        f = n - len(need_claude)              # judgment tasks win over the ratio
    chosen = _spread(models, f)
    fleet_slots = [i for i in range(n) if i not in need_claude][:f]
    bash_pool = [m for m in models if m["id"] in BASH_MODELS]
    agents, k, nb = [], 0, 0
    for i in range(n):
        a = {"id": f"a{i + 1}", "task": items[i]["task"] if items else ""}
        if i in fleet_slots:
            wants = bool(items and items[i]["bash"])
            if wants and bash_pool:                   # a Bash agent goes to a model that may have it
                m, nb = bash_pool[nb % len(bash_pool)], nb + 1
            else:
                m, k = chosen[k % len(chosen)], k + 1
            a.update(driver="fleet", model=m["id"], wire=m["wire"], klass=m["klass"],
                     max_out=min(int(m.get("max_out") or 64000), 120000), bash=wants)
            res = resolve_tools(a, public_or_test_repo)
            a.update(tools=[*FILE_TOOLS, *([BASH_RULE] if res["bash"] else [])], bash=res["bash"])
            if res["bash_note"]:
                a["bash_note"] = res["bash_note"]
        else:
            a.update(driver="claude")
        agents.append(a)
    out = {"n": n, "fleet": f, "claude": n - f, "share": round(f / n, 2),
           "band": band_of(n, f) if f else "none", "public_or_test_repo": public_or_test_repo is True,
           "agents": agents}
    if any(a.get("bash") for a in agents):
        out["bash_guard"] = BASH_GUARD
    return out


# ---------------------------------------------------------------------------------------------
# Running fleet agents
# ---------------------------------------------------------------------------------------------

def redact(text: str, key: str) -> str:
    s = str(text or "")
    if key and len(key) >= 8:
        s = s.replace(key, "[REDACTED]")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1[REDACTED]", s)


def agent_env(key: str, base: str, wire: str, home: str, max_out=None, bash: bool = False) -> dict:
    """Everything the agent may see. Nothing else from this process is passed on. For a Bash
    agent, CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 hides the credential from the shell; Claude Code
    needs bubblewrap and socat for it and otherwise fails every shell command (fails closed)."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home,
           "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": "",
           "ANTHROPIC_MODEL": wire, "ANTHROPIC_SMALL_FAST_MODEL": wire,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    if max_out:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_out)
    if bash:
        env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "1"
    return env


def agent_command(claude: str, wire: str, task: str, max_turns: int, res: dict = None) -> list:
    res = res or resolve_tools({}, False)
    return [claude, "-p", task, "--model", wire, "--tools", ",".join(res["tools"]),
            "--allowedTools", ",".join(res["allowed"]),
            "--permission-mode", "acceptEdits", "--max-turns", str(max_turns),
            "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"]


def _git(repo, *args, runner=subprocess.run):
    return runner(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)


def run_agent(agent: dict, repo, out_dir, *, claude="claude", base=DEFAULT_BASE, key="",
              timeout=900, max_turns=30, runner=subprocess.run,
              public_or_test_repo: bool = False) -> dict:
    """One fleet agent in its own worktree. Never raises; always writes <out_dir>/<id>.json
    (atomically), so a swarm that is killed keeps every agent that finished."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = resolve_tools(agent, public_or_test_repo)          # derived here; the plan's list is ignored
    rec = {"id": agent["id"], "model": agent["model"], "wire": agent["wire"],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "failed", "diff": "",
           "tools": res["allowed"], "bash": res["bash"],
           **({"bash_note": res["bash_note"]} if res["bash_note"] else {})}
    t0 = time.time()
    wt = tempfile.mkdtemp(prefix="orchestra-wt-")
    home = tempfile.mkdtemp(prefix="orchestra-home-")
    added = False
    try:
        r = _git(repo, "worktree", "add", "--detach", wt, "HEAD")
        if r.returncode != 0:
            rec["note"] = "could not create a worktree: " + redact(r.stderr, key)[:300]
            return rec
        added = True
        try:
            p = runner(agent_command(claude, agent["wire"], agent["task"], max_turns, res), cwd=wt,
                       capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL,
                       env=agent_env(key, base, agent["wire"], home, agent.get("max_out"),
                                     res["bash"]))
            rec.update(exit=p.returncode, stdout=redact(p.stdout, key)[:5000],
                       stderr=redact(p.stderr, key)[:1000])
        except subprocess.TimeoutExpired:
            rec.update(exit=-1, status="timeout", note=f"no finish within {timeout}s")
        except Exception as e:     # a crash in one agent must not lose the others
            rec.update(exit=-1, note=f"could not run Claude Code: {type(e).__name__}: {e}")
        _git(wt, "add", "-A")
        rec["diff"] = redact(_git(wt, "diff", "--cached", "HEAD").stdout, key)
        if rec["status"] != "timeout":
            if rec.get("exit") != 0:
                rec["status"] = "failed"
            else:
                rec["status"] = "changed" if rec["diff"].strip() else "no_change"
        return rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        if added:
            _git(repo, "worktree", "remove", "--force", wt)
        shutil.rmtree(wt, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)
        final = out_dir / f"{agent['id']}.json"
        tmp = final.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, final)


def run_swarm(plan_obj: dict, repo, out_dir, *, claude="claude", base=DEFAULT_BASE, key="",
              timeout=900, max_turns=30, parallel=4, runner=subprocess.run) -> list:
    """Run every fleet agent in the plan, `parallel` at a time. Returns the records."""
    fleet = [a for a in plan_obj.get("agents", []) if a.get("driver") == "fleet"]
    results, lock, sem = [], threading.Lock(), threading.Semaphore(max(1, parallel))

    def one(a):
        with sem:
            r = run_agent(a, repo, out_dir, claude=claude, base=base, key=key, timeout=timeout,
                          max_turns=max_turns, runner=runner,
                          public_or_test_repo=plan_obj.get("public_or_test_repo") is True)
        with lock:
            results.append(r)
    threads = [threading.Thread(target=one, args=(a,)) for a in fleet]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results, key=lambda r: r["id"])


def _key() -> str:
    """OPENROUTER_API_KEY from the environment, else from the .env next to this file."""
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if k:
        return k
    try:
        for line in (Path(__file__).with_name(".env")).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s#]+)", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the fleet agents of a swarm plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--repo", default=".")
    r.add_argument("--out", required=True)
    r.add_argument("--claude", default=shutil.which("claude") or "claude")
    r.add_argument("--base", default=DEFAULT_BASE)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--max-turns", type=int, default=30)
    r.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args(argv)
    key = _key()
    if not key:
        sys.stderr.write("OPENROUTER_API_KEY is not set; nothing was run.\n")
        return 2
    plan_obj = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    recs = run_swarm(plan_obj, a.repo, a.out, claude=a.claude, base=a.base, key=key,
                     timeout=a.timeout, max_turns=a.max_turns, parallel=a.parallel)
    for rec in recs:
        sys.stdout.write(f"{rec['id']} {rec['model']}: {rec['status']} "
                         f"({rec.get('seconds')}s, diff {len(rec['diff'])} chars)\n")
    return 0 if recs and all(r["status"] in ("changed", "no_change") for r in recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
