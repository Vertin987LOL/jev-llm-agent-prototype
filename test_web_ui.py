import argparse
import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent import advise, heartbeat_decision, run
from web_ui import AgentUI, make_server


def options():
    return argparse.Namespace(root=Path.cwd(), mcp_config=None, tier="deep",
                              goal_max_steps=5, heartbeat_interval=3600, port=0)


class WebUITests(unittest.TestCase):
    def test_web_history_uses_rolling_summary(self):
        app = AgentUI(options())
        try:
            app.history = [{"user": "问题0", "answer": "答复", "events": []}]
            with patch("web_ui.run_turn", return_value=("完成", [], "deep")), \
                 patch("web_ui.compact_history", return_value=(
                       [{"user": "当前问题", "answer": "完成", "events": []}], "更早的问题")) as compact, \
                 patch("web_ui.save_trace"):
                app.command("chat", "当前问题")
                deadline = time.monotonic() + 2
                while app.snapshot(0)["busy"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                compact.assert_called_once()
            self.assertEqual(app.history_summary, "更早的问题")
            self.assertEqual(len(app.history_archive), 1)
            from agent import conversation_goal
            context = json.loads(conversation_goal("下一问题", app.history,
                                                   app.history_summary, app.history_archive))
            self.assertEqual(context["previous_summary"], "更早的问题")
            self.assertEqual(context["memory_index"][0]["request"], "当前问题")
        finally:
            app.close()

    def test_snapshot_identifies_deepseek_provider(self):
        with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": "deepseek",
                                  "DEEPSEEK_MODEL": "deepseek-chat"}):
            app = AgentUI(options())
            try:
                state = app.snapshot(0)
                self.assertEqual(state["provider"], "deepseek")
                self.assertIn("DeepSeek", state["model"])
                self.assertFalse(state["heartbeat"])
            finally:
                app.close()

    def test_ambiguous_input_asks_user_without_tool_call(self):
        with patch("agent.clarification_answer", return_value="你想让我对 111 做什么？"):
            answer, events = run("111", Path.cwd(), "deep",
                                 pick=lambda question, criteria, state: "NEED_USER" if "NEED_USER" in criteria else "ASK_USER",
                                 planner=lambda goal, actions: "请用户说明目标",
                                 act=lambda root, tool, arguments: self.fail("no tool should run"))
        self.assertEqual(answer, "你想让我对 111 做什么？")
        self.assertEqual(events, [])

    def test_ambiguous_request_and_idle_heartbeat_prompts_do_not_invent_tasks(self):
        captured = []
        with patch("agent.local_model", side_effect=lambda messages, **kwargs: captured.append(messages) or "信息不足"):
            advise("111", [], {})
        self.assertIn("只是数字", captured[0][0]["content"])
        self.assertIn("不要为了展示能力编造", captured[0][0]["content"])
        seen = []
        heartbeat_decision(None, __import__("datetime").datetime.now().astimezone(), can_resume=True,
                           thinker=lambda messages: seen.append(messages) or "没有待办，休息",
                           pick=lambda question, criteria, state: "REST")
        self.assertIn("时间本身不会创造任务", seen[0][0]["content"])

    def test_web_stream_shows_unabridged_thought_tool_and_answer(self):
        app = AgentUI(options())
        long_result = "BEGIN" + "x" * 2200 + "END"

        def fake_turn(args, goal, trace):
            trace.append({"type": "advisor", "text": "先读取这个文件"})
            trace.append({"type": "decision", "turn": 1, "choice": "read_text", "description": "读取文件"})
            event = {"tool": "read_text", "args": {"path": "note.txt"}, "result": long_result}
            trace.append({"type": "tool", "event": event})
            return "完成", [event], "deep"

        try:
            with patch("web_ui.run_turn", side_effect=fake_turn), patch("web_ui.save_trace"):
                app.command("chat", "读 note.txt")
                deadline = time.monotonic() + 2
                while app.snapshot(0)["busy"] and time.monotonic() < deadline:
                    time.sleep(0.01)
            stream = app.snapshot(0)["events"]
            self.assertEqual([e["kind"] for e in stream], ["user", "trace", "trace", "trace", "answer"])
            self.assertEqual(stream[3]["step"]["event"]["result"], long_result)
            self.assertEqual(app.snapshot(stream[-1]["id"])["events"], [])
        finally:
            app.close()

    def test_http_requires_token_and_approves_only_pending_tool(self):
        args = options()
        app = AgentUI(args)
        server = make_server(args, app, "test-token")
        args.port = server.server_port
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{args.port}"
        try:
            with urlopen(base + "/", timeout=2) as response:
                self.assertIn(b"Jev", response.read())
            with self.assertRaises(HTTPError) as denied:
                urlopen(base + "/api/events?after=0", timeout=2)
            self.assertEqual(denied.exception.code, 403)
            with urlopen(Request(base + "/api/events?after=0", headers={"X-Jev-Token": "test-token"}), timeout=2) as response:
                self.assertFalse(json.load(response)["heartbeat"])
            request = Request(base + "/api/command", data=json.dumps({"action": "approve"}).encode(),
                              headers={"X-Jev-Token": "test-token", "Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as denied:
                urlopen(request, timeout=2)
            self.assertEqual(denied.exception.code, 400)
            app._notice({"type": "approval", "tool": "sample/read", "args": {"path": "x"}})
            with urlopen(request, timeout=2) as response:
                self.assertTrue(json.load(response)["ok"])
            self.assertEqual(app.approvals.get_nowait(), "YES")
        finally:
            server.shutdown()
            server.server_close()
            app.close()
