# Jev + LLM Agent（实验原型）

**状态：能运行，但实际任务效果还不理想。** 这不是稳定的浏览器代理，也没有证据表明它比主流模型原生工具调用更准确、更快或更便宜。我公开的是一个可修改的研究起点，希望有人帮忙设计实验、找问题、继续开发。

> English: An experimental agent loop. An LLM proposes the next step in natural language; Jev maps that proposal onto a fixed action; code validates parameters, executes the tool, and returns real feedback. Tool selection, browser use, memory and recovery are not reliable yet. Contributions and controlled comparisons are welcome.

## 想法

传统工具调用通常让语言模型同时想下一步、选择工具、生成结构化参数。这里尝试拆开：

1. LLM 看到**当前目标、上一步真实结果、错误和简短印象**，用普通文字想“下一步做什么”。它不需要生成工具调用格式。
2. Jev 把这段自然语言映射到固定选项：某个工具、直接回答、询问用户、停下等。Jev 是快速的选项映射器，不负责独立完成复杂推理。
3. 选中工具后，LLM 再看到该工具的描述和参数规范，填写参数；代码按 JSON Schema 校验。MCP 工具调用需要用户确认。
4. 程序执行一小步，把**实际结果或报错**交回 LLM；LLM 更新想法，再由 Jev 选下一步。循环直到能回答、必须追问、缺工具、被阻塞或达到步数上限。

```
目标 + 短印象 + 真实反馈 → LLM 自然语言想法 → Jev 固定选项
                                      ↓
                       填参 → 校验 → 执行工具
                                      ↓
                              下一轮真实反馈
```

聊天第二轮起，旧内容会压成短印象；程序另外保留原始轮次和关键参数索引。需要旧细节时，可以选择 `list_memory`、`recall_history` 回查。摘要不是工具执行的证据。缺工具时可异步请求 DeepSeek 生成**待人工审核的草案**，不会自动安装或执行生成的代码。

这只是文字形式的循环状态，不是模型隐藏层的连续思维。未来的语音输出、视觉动作和更快的“技能”层是设想，**当前没有实现**。

## 当前能做什么 / 哪里不稳

- 有命令行和本地网页；内置少量只读文件工具，可从 MCP 工具目录自动读取描述与参数规范。
- 本地 Ollama 模型负责参谋、填参和回答；也能切换 DeepSeek。Jev 负责固定选项选择。
- 已有目标续跑、错误反馈、参数校验、工具确认、短印象与回查；定时心跳默认关闭，需要时手动开启。
- **真实任务仍经常选错工具、误判任务完成、填错浏览器参数或在网页验证页卡住。** 小模型的建议可能误导 Jev；Jev 自身也可能错选。错误反馈和停止条件不是成功率保证。
- 尚未完成与同一 LLM 原生 function calling 的公平对照。请不要把这个仓库当成性能或准确率的证明。

## 希望有人一起做

- 设计可复现的多步任务集，对照“LLM 原生工具调用”与“LLM + Jev”，分别量选工具正确率、任务成功率、延迟、调用成本。
- 改善工具很多时的候选召回，且不依赖小模型输出固定格式；测漏召回和错选。
- 研究每步短状态如何保留目标与关键事实，同时从原始记录按需回查；防止摘要幻觉被误认成事实。
- 改进浏览器观察、参数填充、错误恢复、重复动作检测和安全边界。欢迎提供失败案例、最小复现、文档或代码。

请先开 Issue 说明场景、原目标、实际选项和预期动作；提交 PR 时尽量附可复现用例。**不要上传包含密钥、登录态、私人网页或工具结果的日志。**

## 运行

需要 Python 3.12+、Git、Ollama；浏览器扩展还需要 Node.js、Chrome 和 `chrome-devtools-mcp`。Jev 的 HTTP 客户端复用 [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（MIT）；使用 Jev API 需自备 `TYPESAFE_API_KEY`。本仓库**不含模型权重或任何密钥**；请自行检查所用模型的许可证。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b
$env:TYPESAFE_API_KEY = Read-Host "Jev API key"
.\.venv\Scripts\python.exe -X utf8 .\web_ui.py --root .
```

默认通过本机 Ollama 的 `qwen3:4b` 运行，设置 `ROBOT_MODEL` 可换成自己已安装的模型。所有本地模型请求设为 48K 上下文；这不表示模型在长上下文中一定可靠。命令行可用 `python agent.py --root . --chat`；`/goal <任务>` 启动持续目标，`/heartbeat on` 手动启用心跳。

若想接入 Chrome MCP，复制 `mcp.chrome.example.json` 并按本机 Node.js 安装位置调整 `command`；例如 Windows 上可能需要改为 `npx.cmd` 或 Node 可执行文件，再使用 `--mcp-config <配置文件>`。该示例启动独立、无登录的无界面浏览器。MCP 服务具有自己的系统权限；每次调用前的确认**不是沙箱**。不要配置不可信的服务，也不要在隔离浏览器中输入账号。

可选：设置 `AGENT_LLM_PROVIDER=deepseek` 和 `DEEPSEEK_API_KEY`，或运行 `Start-Jev-DeepSeek.ps1` 并交互式输入密钥。这样会把请求、相关工具结果和参数发给云端并产生费用。不要把密钥写入仓库或公开日志。

本机 trace 默认写入用户的应用数据目录（`JevAgent/trace.jsonl`），可能含私人内容，不要提交。

## 代码位置

- `agent.py`：LLM → Jev → 填参 → 执行 → 反馈的循环、短印象和回查。
- `mcp_bridge.py`：从 MCP 协议读取工具、参数规范并执行调用。
- `web_ui.py`、`web_assets/`：本地网页和执行记录。
- `growth.py`：仅生成待审核的缺失工具草案。
- `test_*.py`、`fake_mcp_server.py`：可修改的测试与模拟服务。

## 许可证

本项目采用 MIT 许可证，见 `LICENSE`。外部模型、MCP 服务和 Jev 依赖分别遵守各自许可证。
