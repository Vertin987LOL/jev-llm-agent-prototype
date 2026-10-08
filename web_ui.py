from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from agent import (LiveTrace, MCPChatSession, compact_history, conversation_goal, heartbeat_decision,
                   run_turn, save_trace)


ASSETS = Path(__file__).with_name("web_assets")


class AgentUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.feed: list[dict] = []
        self.counter = 0
        self.busy = False
        self.approval: dict | None = None
        self.approvals: queue.Queue[str] = queue.Queue()
        self.history: list[dict] = []
        self.history_summary = ""
        self.history_archive: list[dict] = []
        self.goal: dict | None = None
        self.pending_task: str | None = None
        self.heartbeat = False
        self.interval = args.heartbeat_interval
        self.next_tick = time.monotonic() + self.interval
        self.session: MCPChatSession | None = None
        self.closed = threading.Event()
        threading.Thread(target=self._clock, daemon=True).start()

    def emit(self, kind: str, **data: object) -> None:
        with self.lock:
            self.counter += 1
            self.feed.append({"id": self.counter, "kind": kind, **data})

    def snapshot(self, after: int) -> dict:
        with self.lock:
            return {"events": [item for item in self.feed if item["id"] > after],
                    "busy": self.busy, "approval": self.approval,
                    "heartbeat": self.heartbeat, "interval": self.interval,
                    "goal": {"text": self.goal["text"], "status": self.goal["status"],
                             "steps": len(self.goal["events"])} if self.goal else None,
                    "pending_task": self.pending_task,
                    "provider": os.environ.get("AGENT_LLM_PROVIDER", "ollama").lower(),
                    "model": (os.environ.get("DEEPSEEK_MODEL", "deepseek-chat") + " / DeepSeek"
                              if os.environ.get("AGENT_LLM_PROVIDER", "ollama").lower() == "deepseek"
                              else os.environ.get("ROBOT_MODEL", "qwen3:4b")),
                    "mcp": bool(self.args.mcp_config)}

    def _begin(self, target, *arguments) -> None:
        with self.lock:
            if self.busy:
                raise ValueError("上一项工作还没结束")
            self.busy = True

        def worker():
            try:
                target(*arguments)
            except Exception as error:
                self.emit("error", text=str(error))
            finally:
                with self.lock:
                    self.busy = False
                    self.approval = None

        threading.Thread(target=worker, daemon=True).start()

    def _notice(self, item: dict) -> None:
        with self.lock:
            self.approval = item
        self.emit("approval", **item)

    def approve(self, allowed: bool) -> None:
        with self.lock:
            if self.approval is None:
                raise ValueError("目前没有待确认的工具")
            self.approval = None
        self.approvals.put("YES" if allowed else "NO")
        self.emit("approval_result", allowed=allowed)

    def command(self, action: str, text: str = "") -> None:
        text = text.strip()
        if action == "approve":
            self.approve(True)
            return
        if action == "deny":
            self.approve(False)
            return
        with self.lock:
            if self.busy:
                raise ValueError("上一项工作还没结束")
            if action == "heartbeat_on":
                self.heartbeat = True
                self.next_tick = time.monotonic() + self.interval
            elif action == "heartbeat_off":
                self.heartbeat = False
            elif action == "pending":
                if not text:
                    raise ValueError("先输入待办任务")
                self.pending_task = text
            elif action == "clear_pending":
                self.pending_task = None
            elif action == "stop_goal":
                self.goal = None
            elif action == "resume":
                if not self.goal or self.goal["status"] != "budget":
                    raise ValueError("当前没有因步数上限暂停的目标")
            elif action == "goal":
                if not text or self.goal:
                    raise ValueError("请填写目标，或先结束原目标")
                self.goal = {"text": text, "events": [], "thought": "", "status": "running"}
            elif action == "chat" and not text:
                raise ValueError("先输入消息")
            elif action not in ("chat", "heartbeat_once"):
                raise ValueError("未知操作")
        if action in ("chat", "goal", "resume"):
            self.emit("user", text=text or "继续目标", action=action)
            self._begin(self._run, action, text)
        elif action == "heartbeat_once":
            self._begin(self._beat)
        else:
            self.emit("status", text={"heartbeat_on": "心跳已开启", "heartbeat_off": "心跳已关闭",
                                      "pending": "待办已记录", "clear_pending": "待办已清除",
                                      "stop_goal": "目标已结束"}[action])

    def _run(self, action: str, text: str) -> None:
        goal_run = action in ("goal", "resume")
        with self.lock:
            current = self.goal if goal_run else None
            old_count = len(current["events"]) if current else 0
        target = (current["text"] if current else
                  conversation_goal(text, self.history, self.history_summary, self.history_archive))
        outcome: dict = {}
        trace = LiveTrace(lambda step: self.emit("trace", step=step), echo=False)
        args = argparse.Namespace(**{**vars(self.args), "goal": target, "tier": "deep",
                                     "max_steps": self.args.goal_max_steps if goal_run else 5,
                                     "initial_events": current["events"] if current else None,
                                     "initial_thought": current["thought"] if current else "",
                                     "memory_archive": self.history_archive,
                                     "outcome": outcome, "approval_queue": self.approvals,
                                     "approval_notice": self._notice})
        try:
            if args.mcp_config:
                if self.session is None:
                    self.session = MCPChatSession(args.mcp_config)
                result, events, tier = self.session.run(args, trace)
            else:
                result, events, tier = run_turn(args, target, trace)
            if current is not None:
                with self.lock:
                    current["events"] = events
                    current["status"] = outcome.get("status", "blocked")
                    thoughts = [step["text"] for step in trace if step["type"] == "advisor"]
                    if thoughts:
                        current["thought"] = thoughts[-1]
                    if current["status"] in ("completed", "needs_input"):
                        self.goal = None
                if old_count == len(events) and not result:
                    result = "未产生新结果"
            if not goal_run or current["status"] in ("completed", "needs_input"):
                turn = {"user": target if goal_run else text, "answer": result, "events": events}
                self.history_archive.append(turn)
                self.history.append(turn)
                self.history, self.history_summary = compact_history(self.history, self.history_summary)
            self.emit("answer", text=result, tier=tier,
                      status=current["status"] if current else "completed")
            save_trace(target, trace, events)
        except Exception as error:
            if current is not None:
                with self.lock:
                    current["events"] = outcome.get("events", current["events"])
                    current["status"] = "error"
            save_trace(target, trace, outcome.get("events", []), str(error))
            raise

    def _beat(self) -> None:
        with self.lock:
            goal = self.goal
            pending = self.pending_task
        thought, choice = heartbeat_decision(goal, datetime.now().astimezone(),
                                              can_resume=not bool(self.args.mcp_config),
                                              pending_task=pending)
        self.emit("heartbeat", thought=thought, choice=choice)
        if choice == "RESUME_GOAL":
            self._run("resume", "")
        elif choice == "START_TASK" and pending:
            with self.lock:
                self.pending_task = None
                self.goal = {"text": pending, "events": [], "thought": "", "status": "running"}
            self._run("goal", pending)

    def _clock(self) -> None:
        while not self.closed.wait(0.2):
            with self.lock:
                due = self.heartbeat and time.monotonic() >= self.next_tick
                if due and not self.busy:
                    self.next_tick = time.monotonic() + self.interval
                else:
                    due = False
            if due:
                try:
                    self._begin(self._beat)
                except ValueError:
                    pass

    def close(self) -> None:
        self.closed.set()
        with self.lock:
            if self.approval is not None:
                self.approval = None
                self.approvals.put("NO")
        if self.session is not None:
            self.session.close()


def serve(args: argparse.Namespace) -> None:
    app = AgentUI(args)
    token = secrets.token_urlsafe(32)

    server = make_server(args, app, token)
    url = f"http://127.0.0.1:{args.port}/#token={token}"
    print("Jev 网页界面：", url)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


def make_server(args: argparse.Namespace, app: AgentUI, token: str) -> ThreadingHTTPServer:

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            parsed = urlsplit(self.path)
            if self.headers.get("Host") not in (f"127.0.0.1:{args.port}", f"localhost:{args.port}"):
                return False
            if parsed.path in ("/", "/ui.css", "/ui.js") and self.command == "GET":
                return True
            return self.headers.get("X-Jev-Token") == token

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, item: dict) -> None:
            self._send(code, json.dumps(item, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._authorized():
                self._json(403, {"error": "未授权"})
                return
            route = urlsplit(self.path)
            if route.path == "/api/events":
                try:
                    after = max(0, int(parse_qs(route.query).get("after", ["0"])[0]))
                except ValueError:
                    self._json(400, {"error": "游标无效"})
                    return
                self._json(200, app.snapshot(after))
            elif route.path in ("/", "/ui.css", "/ui.js"):
                name = {"/": "index.html", "/ui.css": "ui.css", "/ui.js": "ui.js"}[route.path]
                mime = {"/": "text/html", "/ui.css": "text/css", "/ui.js": "text/javascript"}[route.path]
                self._send(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self._json(404, {"error": "找不到页面"})

        def do_POST(self) -> None:
            if not self._authorized() or self.headers.get("Origin") not in (
                    None, f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                self._json(403, {"error": "未授权"})
                return
            if urlsplit(self.path).path != "/api/command":
                self._json(404, {"error": "找不到操作"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16384:
                    raise ValueError("请求长度无效")
                item = json.loads(self.rfile.read(length))
                if not isinstance(item, dict) or not isinstance(item.get("action"), str) or not isinstance(item.get("text", ""), str):
                    raise ValueError("请求格式无效")
                app.command(item["action"], item.get("text", ""))
                self._json(200, {"ok": True})
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})

        def log_message(self, format: str, *arguments: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    args.port = server.server_port
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Jev browser interface")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mcp-config", type=Path)
    parser.add_argument("--goal-max-steps", type=int, default=25)
    parser.add_argument("--heartbeat-interval", type=float, default=60)
    arguments = parser.parse_args()
    if not (arguments.port == 0 or 1024 <= arguments.port <= 65535) or arguments.heartbeat_interval <= 0 or arguments.goal_max_steps < 1:
        parser.error("invalid port, interval or goal steps")
    serve(arguments)
