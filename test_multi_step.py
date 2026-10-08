import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import run
from mcp_bridge import MCPBridge


CONFIG = {"sample": {"command": sys.executable,
                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}


class MultiStepTests(unittest.IsolatedAsyncioTestCase):
    async def test_jev_repairs_repeated_bad_parameter_after_mcp_error(self):
        async with MCPBridge(CONFIG) as bridge:
            identifiers = {name: key for key, (_, name) in bridge.targets.items()}
            loop = asyncio.get_running_loop()

            def act(root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with tempfile.TemporaryDirectory() as directory:
                    result, events = await asyncio.to_thread(
                        run, "找出有今日晴天的文档并统计行数", Path(directory), "direct",
                        pick=lambda question, criteria, state: (
                            next(key for key, description in criteria.items() if "alpha.txt" in description)
                            if "NONE" in criteria else
                            identifiers["read_document"] if not state["events"] else
                            identifiers["count_document_lines"]),
                        planner=lambda goal, events: "已读 alpha.txt，需统计它",
                        fill=lambda goal, advice, tool, events: {
                            "name": "alpha.txt" if tool == identifiers["read_document"] else "今日晴天"},
                        act=act, tools=bridge.catalog,
                        verify=lambda goal, events: (events[-1]["tool"] == identifiers["count_document_lines"]
                                                     and bool(events[-1].get("result"))),
                        answer=lambda goal, events: events[-1]["result"],
                    )
            self.assertIn("2", result)
            self.assertEqual(events[-1]["args"], {"name": "alpha.txt"})

    async def test_mcp_error_then_new_decision_fills_document_name(self):
        async with MCPBridge(CONFIG) as bridge:
            identifier = next(key for key, target in bridge.targets.items()
                              if target[1] == "count_document_lines")
            loop = asyncio.get_running_loop()
            advice_seen = []

            def act(root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "统计包含宇宙计划的文档", Path(directory), "direct",
                    pick=lambda question, criteria, state: identifier,
                    planner=lambda goal, events: advice_seen.append(len(events)) or
                        "宇宙计划是正文，文档名是 beta.txt",
                    fill=lambda goal, advice, tool, events: {"name": "beta.txt" if advice else "宇宙计划"},
                    act=act, tools=bridge.catalog,
                    verify=lambda goal, events: bool(events[-1].get("result")),
                    answer=lambda goal, events: events[-1]["result"],
                )
            self.assertIn("3", result)
            self.assertEqual(len(events), 2)
            self.assertIn("error", events[0])
            self.assertEqual(events[1]["args"], {"name": "beta.txt"})
            self.assertEqual(advice_seen, [1])

    async def test_list_read_twice_count_and_answer(self):
        async with MCPBridge(CONFIG) as bridge:
            identifiers = {name: identifier for identifier, (_, name) in bridge.targets.items()}
            route = [identifiers[name] for name in
                     ("list_documents", "read_document", "read_document", "count_document_lines")]
            route.append("DONE")
            advice_at = []
            loop = asyncio.get_running_loop()

            def pick(question, criteria, state):
                index = len(state["events"])
                if index == 2:
                    self.assertIn("普通备忘", state["events"][-1]["result"])
                if index == 3:
                    self.assertIn("宇宙计划", state["events"][-1]["result"])
                return route[index]

            def fill(goal, advice, tool, events):
                if tool == identifiers["read_document"]:
                    return {"name": "alpha.txt" if len(events) == 1 else "beta.txt"}
                if tool == identifiers["count_document_lines"]:
                    return {"name": "beta.txt"}
                return {}

            def act(root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "找出有宇宙二字的文档并统计行数", Path(directory), "deep",
                    pick=pick, fill=fill, act=act, tools=bridge.catalog,
                    planner=lambda goal, events: advice_at.append(len(events)) or "继续查找",
                    answer=lambda goal, events: events[-1]["result"],
                )
            self.assertEqual([item["tool"] for item in events], route[:-1])
            self.assertEqual(advice_at, [0, 1, 2, 3, 4])
            self.assertIn("3", result)

    async def test_parameter_description_and_optional_integer_across_steps(self):
        async with MCPBridge(CONFIG) as bridge:
            identifiers = {name: identifier for identifier, (_, name) in bridge.targets.items()}
            loop = asyncio.get_running_loop()
            choices = iter([identifiers["list_documents"], identifiers["repeat_text"], "DONE"])
            schemas = []

            def model(messages, schema=None):
                schemas.append(schema)
                if "repeat_text" in messages[1]["content"]:
                    self.assertIn("list_documents", messages[1]["content"])
                    return '{"text":"ok","count":4}'
                return "{}"

            def act(root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with patch("agent.local_model", side_effect=model), tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "先列文档，再重复 ok 四次", Path(directory), "direct",
                    pick=lambda question, criteria, state: next(choices),
                    act=act, tools=bridge.catalog,
                    answer=lambda goal, events: events[-1]["result"],
                )
            self.assertEqual(len(events), 2)
            self.assertIn("okokokok", result)
            self.assertEqual(schemas[-1]["properties"]["count"]["type"], "integer")

    async def test_duplicate_names_remain_independent(self):
        async with MCPBridge({"first": CONFIG["sample"], "second": CONFIG["sample"]}) as bridge:
            matching = [(identifier, server) for identifier, (server, name) in bridge.targets.items()
                        if name == "list_documents"]
            self.assertEqual(len(matching), 2)
            self.assertNotEqual(matching[0][0], matching[1][0])
            for identifier, server in matching:
                self.assertIn(server, bridge.catalog[identifier]["description"])
                self.assertIn("beta.txt", await bridge.call(identifier, {}))

    async def test_server_error_is_recorded_not_faked_as_success(self):
        async with MCPBridge(CONFIG) as bridge:
            identifier = next(key for key, target in bridge.targets.items() if target[1] == "read_document")
            loop = asyncio.get_running_loop()

            def act(root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "读不存在的文档", Path(directory), "direct",
                    pick=lambda question, criteria, state: identifier,
                    fill=lambda goal, advice, tool, events: {"name": "missing.txt"},
                    act=act, tools=bridge.catalog,
                    max_steps=1,
                )
            self.assertIn("错误", result)
            self.assertEqual(len(events), 1)
            self.assertNotIn("result", events[0])
