"""Offline probe (stdlib, no key, no network): does Claude Code honour a narrow Bash allow rule,
and what can a permitted shell still do? A stub model asks a real `claude -p` to run one command;
we record the tool_result it sends back (ran vs denied). Used for swarm.md's Bash section.

    python bash_probe.py "Read,Bash(python -m pytest:*)" "python -m pytest --version" "ls /" "env"

It RUNS the commands you give it on this machine, as the allowed rule permits. Run it on a
throwaway machine. For the env-scrub cases set CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 via env_extra
and install bubblewrap and socat first.
"""
import http.server, json, os, subprocess, sys, tempfile, threading, shutil
CMD = {"v": ""}
SEEN = []

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if "count_tokens" in self.path:
            out = json.dumps({"input_tokens": 10}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out); return
        results = [b for m in body.get("messages", []) if isinstance(m.get("content"), list)
                   for b in m["content"] if b.get("type") == "tool_result"]
        has_tools = bool(body.get("tools"))
        if results:
            SEEN.append(json.dumps(results[-1].get("content"))[:300]); blocks = [{"type": "text", "text": "done"}]; stop = "end_turn"
        elif has_tools:
            blocks = [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": CMD["v"]}}]; stop = "tool_use"
        else:
            blocks = [{"type": "text", "text": "x"}]; stop = "end_turn"
        msg = {"id": "m", "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
               "stop_reason": None, "usage": {"input_tokens": 5, "output_tokens": 5}}
        if not body.get("stream"):
            msg.update(content=blocks, stop_reason=stop)
            out = json.dumps(msg).encode(); self.send_response(200)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out)))
            self.end_headers(); self.wfile.write(out); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        ev = [("message_start", {"type": "message_start", "message": msg})]
        for i, b in enumerate(blocks):
            if b["type"] == "text":
                ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}),
                       ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})]
            else:
                ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}),
                       ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})]
            ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}),
               ("message_stop", {"type": "message_stop"})]
        for n, d in ev:
            self.wfile.write(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode())

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"

def run(cmd, allow, tools="Read,Edit,Write,Glob,Grep,Bash", extra=(), env_extra=None):
    CMD["v"] = cmd; SEEN.clear()
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as work:
        env = {"PATH": os.environ["PATH"], "HOME": home, "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": "k"*12,
               "ANTHROPIC_API_KEY": "", "ANTHROPIC_MODEL": "stub/model", "ANTHROPIC_SMALL_FAST_MODEL": "stub/model",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", **(env_extra or {})}
        c = ["claude", "-p", "go", "--model", "stub/model", "--tools", tools, "--allowedTools", allow,
             "--permission-mode", "acceptEdits", "--max-turns", "4", "--strict-mcp-config", "--setting-sources", "",
             "--output-format", "json", *extra]
        p = subprocess.run(c, cwd=work, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    return p.returncode, (SEEN[-1] if SEEN else "(no tool_result seen) " + p.stdout[:200] + p.stderr[:200])

if __name__ == "__main__":
    allow = sys.argv[1]
    for cmd in sys.argv[2:]:
        rc, res = run(cmd, allow)
        print(f"{cmd!r:60} rc={rc} -> {res}")
    srv.shutdown()
