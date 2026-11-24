#!/usr/bin/env python3
"""orchestra_swarm.py — run Claude Code agents whose model is a fleet model (stdlib only).

A subagent started from Claude Code always uses the session's own Claude model. A fleet agent is
therefore a separate headless `claude -p` process whose API calls go to OpenRouter's
Anthropic-compatible endpoint. This file does three things:

  fleet_count(n)  how many of n agents are fleet agents (60-70%, else 80-90%, else nearest)
  plan(...)       who is a fleet agent, which model drives it, which task it gets
  run             start the fleet agents, each in its own git worktree, and keep every result

Each fleet agent gets: a worktree of HEAD (tracked files only, so an untracked .env is never
there), a scrubbed environment (PATH, an empty HOME and its one OpenRouter credential), file
tools only (no Bash), a turn cap and a time limit. It never touches the real checkout: what it
changed comes back as a diff that Claude reviews and applies. Spend is NOT capped here (Claude
Code has no spend flag): set an OpenRouter key limit as the backstop.

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
TOOLS = "Read,Edit,Write,Glob,Grep"      # no Bash: isolation is not decided yet (plan, decision 3)
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


def plan(n_agents: int, tasks=None, models=None, claude_only=()) -> dict:
    """The swarm plan. `tasks`: strings, or dicts {"task": str, "claude_only": bool}. A task
    marked claude_only (judgment, repository-wide context, anything sensitive) goes to a Claude
    agent. `models`: the eligible fleet models, dicts with id, wire, klass, max_out."""
    models = list(models or [])
    n = int(n_agents)
    items = []
    for t in (tasks or []):
        d = t if isinstance(t, dict) else {"task": t}
        items.append({"task": str(d.get("task", "")), "claude_only": bool(d.get("claude_only"))})
    if items and len(items) != n:
        return {"error": f"{len(items)} tasks for {n} agents: give one task per agent"}
    if n < 1:
        return {"error": "n_agents must be at least 1"}
    f = fleet_count(n)
    if f and not models:
        return {"error": "no eligible fleet model: run the compatibility harness first "
                         "(docs/diagnostics/agent_compat.py) or list one in swarm_eligible.json"}
    need_claude = [i for i, d in enumerate(items) if d["claude_only"]]
    if len(need_claude) > n - f:
        f = n - len(need_claude)              # judgment tasks win over the ratio
    chosen = _spread(models, f)
    fleet_slots = [i for i in range(n) if i not in need_claude][:f]
    agents = []
    for i in range(n):
        a = {"id": f"a{i + 1}", "task": items[i]["task"] if items else ""}
        if i in fleet_slots:
            m = chosen[fleet_slots.index(i) % len(chosen)]
            a.update(driver="fleet", model=m["id"], wire=m["wire"], klass=m["klass"],
                     max_out=min(int(m.get("max_out") or 64000), 120000))
        else:
            a.update(driver="claude")
        agents.append(a)
    return {"n": n, "fleet": f, "claude": n - f, "share": round(f / n, 2),
            "band": band_of(n, f) if f else "none", "agents": agents}


# ---------------------------------------------------------------------------------------------
# Running fleet agents
# ---------------------------------------------------------------------------------------------

def redact(text: str, key: str) -> str:
    s = str(text or "")
    if key and len(key) >= 8:
        s = s.replace(key, "[REDACTED]")
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1[REDACTED]", s)


def agent_env(key: str, base: str, wire: str, home: str, max_out=None) -> dict:
    """Everything the agent may see. Nothing else from this process is passed on."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home,
           "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": "",
           "ANTHROPIC_MODEL": wire, "ANTHROPIC_SMALL_FAST_MODEL": wire,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    if max_out:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_out)
    return env


def agent_command(claude: str, wire: str, task: str, max_turns: int) -> list:
    return [claude, "-p", task, "--model", wire, "--tools", TOOLS, "--allowedTools", TOOLS,
            "--permission-mode", "acceptEdits", "--max-turns", str(max_turns),
            "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"]


def _git(repo, *args, runner=subprocess.run):
    return runner(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)


def run_agent(agent: dict, repo, out_dir, *, claude="claude", base=DEFAULT_BASE, key="",
              timeout=900, max_turns=30, runner=subprocess.run) -> dict:
    """One fleet agent in its own worktree. Never raises; always writes <out_dir>/<id>.json
    (atomically), so a swarm that is killed keeps every agent that finished."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"id": agent["id"], "model": agent["model"], "wire": agent["wire"],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "failed", "diff": ""}
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
            p = runner(agent_command(claude, agent["wire"], agent["task"], max_turns), cwd=wt,
                       capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL,
                       env=agent_env(key, base, agent["wire"], home, agent.get("max_out")))
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
                          max_turns=max_turns, runner=runner)
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
