from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import run
from growth import missing_context, process_draft, queue_draft


class GrowthTests(unittest.TestCase):
    def test_only_explicit_missing_tool_queues(self):
        queued = []
        with tempfile.TemporaryDirectory() as directory:
            for choice in ("STOP", "ANSWER_DIRECTLY", "MISSING_TOOL"):
                trace = []
                with patch("agent.clarification_answer", return_value="请补充目标"):
                    result, events = run(
                        "搜索实时新闻", Path(directory), "guided",
                        pick=lambda question, criteria, state: choice,
                        planner=lambda goal, events: "需要网络搜索工具",
                        answer=lambda goal, events: "无法提供实时新闻",
                        on_missing=lambda context: queued.append(context) or (Path(directory) / "draft.json", True),
                        trace=trace,
                    )
                self.assertEqual(events, [])
                self.assertEqual(any(step["type"] == "draft" for step in trace), choice == "MISSING_TOOL")
                if choice == "MISSING_TOOL":
                    self.assertIn("草案请求", result)
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["current_thought"], "需要网络搜索工具")

    def test_context_excludes_history_and_tool_results(self):
        goal = json.dumps({"current_request": "查今天的新闻", "previous_turns": [{"answer": "private"}]})
        context = missing_context(goal, "请搜索", [{"tool": "read", "result": "secret", "args": {"key": "secret"}}],
                                  {"search": {"description": "Search the web"}})
        self.assertEqual(context["request"], "查今天的新闻")
        self.assertNotIn("secret", json.dumps(context))
        self.assertNotIn("private", json.dumps(context))

    def test_listing_tools_returns_to_loop_when_request_remains(self):
        decisions = iter(["list_available_tools", "MISSING_TOOL"])
        options_seen = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "搜索今天新闻", Path(directory), "guided",
                pick=lambda question, criteria, state: options_seen.append(criteria) or next(decisions),
                planner=lambda goal, events: "需要联网搜索，目前没有这个工具",
                verify=lambda goal, events: False,
                on_missing=lambda context: (Path(directory) / "draft.json", True),
            )
        self.assertIn("草案请求", result)
        self.assertEqual([event["tool"] for event in events], ["list_available_tools"])
        self.assertNotIn("list_available_tools", options_seen[1])

    def test_deduplicates_and_waits_without_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            path, created = queue_draft({"request": "search"}, Path(directory))
            again, duplicate = queue_draft({"request": "search"}, Path(directory))
            self.assertTrue(created)
            self.assertFalse(duplicate)
            self.assertEqual(path, again)
            self.assertEqual(process_draft(path)["status"], "waiting_for_key")

    def test_reworded_thought_does_not_start_another_request(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            context = {"request": "search", "available_tools": {"read": "read files"},
                       "recent_actions": [], "current_thought": "search web"}
            first, _ = queue_draft(context, Path(directory))
            second, created = queue_draft({**context, "current_thought": "look online"}, Path(directory))
            self.assertEqual(first, second)
            self.assertFalse(created)

    def test_mock_cloud_produces_pending_review_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}):
            path, _ = queue_draft({"request": "search"}, Path(directory), launch=False)
            record = process_draft(path, generate=lambda context: "Draft: web search MCP tool")
            self.assertEqual(record["status"], "pending_review")
            self.assertIn("web search", record["draft"])
            self.assertEqual(process_draft(path, generate=lambda context: self.fail("called twice")), record)

    def test_cloud_failure_remains_inspectable(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}):
            path, _ = queue_draft({"request": "search"}, Path(directory), launch=False)

            def failure(context):
                raise RuntimeError("API unavailable")

            record = process_draft(path, generate=failure)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["error"], "API unavailable")


if __name__ == "__main__":
    unittest.main()
