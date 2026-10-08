# Jev + LLM Agent (experimental prototype)

English · [简体中文](README.zh-CN.md)

**Status: runnable, but unreliable on real tasks.** This repository is primarily a research idea and an invitation to collaborate, not a claim that the agent outperforms native LLM tool calling in accuracy, speed, or cost.

The code is a rough first pass meant to make the idea tangible. I am publishing it for others to fork, question, and continue; it is not a production project or a promise of ongoing development.

## The idea: a feedback loop, not a one-shot tool call

The LLM reads the goal, a compact impression of prior context, and the **actual result or error of the previous action**. It reasons in ordinary language about what to do next. Jev uses that thought as evidence to choose **one** action from a fixed set of tools and exits. Code validates parameters, executes the action, and sends its real feedback back to the LLM. Then the LLM thinks again. Repeat until the goal is answered, blocked, needs user input, or hits the step budget.

```text
Goal + working impression + actual result/error
                    ↓
       LLM reasons about the next step in natural language
                    ↓
       Jev maps that thought to a fixed action or exit
          ↙                                  ↘
 Existing tool: fill parameters,              Missing capability:
 validate, execute                            cloud fallback / growth queue
          ↓                                  ↓
      Real feedback                   draft for review / blocked task
          └──────────→ next LLM reasoning turn
```

**The LLM plans; Jev maps natural-language intent to structured choices.** Jev is not asked to independently solve a complex task, and the LLM is not required to emit a particular keyword or a complete function-call envelope. Once a tool is chosen, the LLM still fills its parameters with the tool description, schema, and relevant context; code validates them. This split may reduce some formatting demands, but it does **not** eliminate reasoning errors, incorrect arguments, or Jev misclassification. It has not been fairly benchmarked against constrained native function calling.

The prototype exposes built-in read-only file tools and can discover MCP tools and their input schemas. MCP calls require confirmation. Chat history is compacted into a short impression while original turns remain available through `list_memory` and `recall_history`. A summary is not evidence that a tool actually ran.

## Self-growth: what happens when a tool is missing?

The **proposed second loop** starts only when Jev identifies a genuine missing capability. A cloud LLM should first check whether Jev missed an existing tool, an alternative route is available, or the task simply cannot be done. If a capability is truly absent, record the goal, current thought, available tools, and recent failures in a growth queue. A separate cloud LLM can asynchronously draft a tool description, argument schema, implementation, and tests. **Only after human review, testing, and authorization** should the new tool enter the catalog for future tasks.

**What exists today:** `MISSING_TOOL` records a local draft request. With `DEEPSEEK_API_KEY`, a background worker may generate a draft marked for review; without a key it waits. The current task stops when the tool is missing. There is **no immediate cloud takeover of the task**, no automatic tool installation, and no learned capability until someone reviews and connects a real implementation. A cloud model cannot conjure access to a nonexistent API. Wrong tool selection, invalid arguments, and transient failures should instead feed back into the ordinary decision loop, not trigger growth.

## Known limitations and useful contributions

Real tasks can select the wrong tool, declare completion too early, fill incorrect browser arguments, or get stuck on verification pages. Small-model advice can mislead Jev, and Jev itself can choose incorrectly. Error feedback and stop conditions do not guarantee recovery. This is a text-based state loop, not persistent latent-state cognition; faster reusable skills, visual actions, and speech are future ideas, not current features.

If this architecture is interesting, contributions are especially welcome on:

- Reproducible multi-step comparisons with the **same LLM using native tool calls**, measuring task success, selection accuracy, retries, latency, and cost.
- Candidate retrieval when there are many tools, without requiring the small LLM to output a rigid selection format.
- Compact per-step state that preserves the goal and key facts, with retrieval from original observations rather than trusting summaries as evidence.
- Safe missing-capability triage, reviewable tool drafts, parameter filling, browser observation, and recovery from repeated mistakes.

Please open an Issue with the original goal, observed choice, expected next action, and a minimal reproduction. PRs with reproducible examples are welcome. **Do not upload keys, logged-in pages, private tool results, or local traces.**

## Run locally

Requires Python 3.12+, Git, Ollama, and your own Jev API key. Browser MCP support additionally requires Node.js, Chrome, and `chrome-devtools-mcp`. The Jev HTTP client uses [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT). **No model weights or secrets are included**; check each external model's license yourself.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b
$env:TYPESAFE_API_KEY = Read-Host "Jev API key"
.\.venv\Scripts\python.exe -X utf8 .\web_ui.py --root .
```

The default is local Ollama `qwen3:4b`; set `ROBOT_MODEL` for another installed model. Local requests use a 48K context setting, which does not imply reliable long-context attention. CLI: `python agent.py --root . --chat`; `/goal <task>` starts a persistent goal; `/heartbeat on` opts into periodic thinking (off by default).

For browser tools, copy `mcp.chrome.example.json` to a private config and adjust the Node command for your machine (for example, `npx.cmd` on Windows), then pass `--mcp-config <file>`. The example launches an isolated, headless, logged-out browser. MCP servers run with their own OS permissions: confirmation is **not** a sandbox. Do not configure untrusted servers or enter account credentials into that browser.

Optionally set `AGENT_LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY`, or run `Start-Jev-DeepSeek.ps1` for interactive key entry. This sends requests and relevant tool context to the cloud and may incur charges. Local traces are written under the user's application-data directory (`JevAgent/trace.jsonl`) and may contain private content; do not commit them.

## Code map and license

- `agent.py`: think → choose → fill → execute → feedback, plus compact memory and retrieval.
- `mcp_bridge.py`: MCP tool discovery, input schemas, and execution.
- `growth.py`: missing-tool queue and review-only draft generation.
- `web_ui.py` and `web_assets/`: local web interface and trace display.
- `test_*.py` and `fake_mcp_server.py`: tests and a sample MCP server.

This project is MIT-licensed; see `LICENSE`. External models, MCP servers, and Jev dependencies retain their own licenses.
