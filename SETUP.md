# Setup

## What runs where

- `orchestra.skill` is instructions. Upload it to claude.ai (Settings, Capabilities, Skills). It calls nothing.
- `orchestra_mcp_server.py` does all the calling. It is a local MCP server: it runs on your machine under Claude Desktop (or a local Claude Code), not inside claude.ai and not inside a cloud session.

## Keys

The server reads its keys from **its own process environment or from a `.env` file next to the script**, and from nowhere else. Claude Desktop's sign-in and its Developer Mode gateway settings are separate things and do not reach the server. A missing key shows up as `[SKIPPED] OPENROUTER_API_KEY not set` (or the `ZHIPU_` / `DEEPSEEK_` equivalent).

1. Copy `.env.example` to `.env` in this folder and fill it in.
2. Hosts the server calls: `openrouter.ai`, `api.deepseek.com`, `open.bigmodel.cn`. A firewall, VPN or DNS filter that blocks them blocks the models behind them.

## Register the server in Claude Desktop

In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orchestra": {
      "command": "python",
      "args": ["/absolute/path/to/orchestra_mcp_server.py"]
    }
  }
}
```

Keys may instead go in an `"env": { "OPENROUTER_API_KEY": "..." }` block next to `args`. Restart Claude Desktop.

## Verify

```
python orchestra_mcp_server.py --check
```

It looks up each model's real limits and makes one tiny real call per model (about a cent in all). Exit 0: something was tested and nothing failed. Exit 1: a problem, with the fix printed. Exit 2: nothing could be tested because no keys are set; that is not a pass.

## Cloud sessions (Claude Code on the web)

A cloud session is a separate machine with its own network policy. To let it reach the models: in the environment's settings (the cloud environment menu in the session's title bar, then Edit), allow the three hosts above under Network access, and add the three keys as environment variables there. A new session picks them up. Do not paste keys into the chat.
