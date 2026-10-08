import asyncio
import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import MCPChatSession, run, run_with_mcp, tool_arguments
from mcp_bridge import MCPBridge, load_servers


class MCPBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowlist_exposes_only_selected_tools(self):
        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))],
                                     "allowed_tools": ["list_documents", "read_document"]}}
        async with MCPBridge(configuration) as bridge:
            self.assertEqual({name for _, name in bridge.targets.values()},
                             {"list_documents", "read_document"})
            self.assertNotIn("repeat_text", str(bridge.catalog))

    async def test_allowlist_rejects_unknown_tool(self):
        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))],
                                     "allowed_tools": ["missing"]}}
        with self.assertRaisesRegex(ValueError, "unknown allowed_tools"):
            async with MCPBridge(configuration):
                pass

    async def test_argument_prompt_explains_tool_field_and_observed_values(self):
        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        async with MCPBridge(configuration) as bridge:
            identifiers = {name: key for key, (_, name) in bridge.targets.items()}
            listed = await bridge.call(identifiers["list_documents"], {})
            read = await bridge.call(identifiers["read_document"], {"name": "beta.txt"})
            events = [
                {"tool": identifiers["list_documents"], "name": bridge.catalog[identifiers["list_documents"]]["description"],
                 "args": {}, "result": listed},
                {"tool": identifiers["read_document"], "name": bridge.catalog[identifiers["read_document"]]["description"],
                 "args": {"name": "beta.txt"}, "result": read},
                {"tool": identifiers["count_document_lines"], "name": bridge.catalog[identifiers["count_document_lines"]]["description"],
                 "args": {"name": "宇宙计划"}, "error": "Unknown document"},
            ]
            captured = []

            def fake_model(messages, schema=None):
                captured.append(json.loads(messages[1]["content"]))
                return '{"name":"beta.txt"}'

            with patch("agent.local_model", side_effect=fake_model):
                arguments = tool_arguments("统计含宇宙计划的文档行数", "建议 count_document_lines，参数用 beta.txt", identifiers["count_document_lines"],
                                           events, bridge.catalog)
            self.assertEqual(arguments, {"name": "beta.txt"})
            prompt = captured[0]
            self.assertEqual(prompt["current_intent"], "建议 count_document_lines，参数用 beta.txt")
            self.assertIn("count_document_lines", prompt["selected_tool"]["name"])
            self.assertIn("Count the lines", prompt["selected_tool"]["purpose"])
            self.assertEqual(prompt["selected_tool"]["fields"][0]["name"], "name")
            self.assertEqual(prompt["selected_tool"]["fields"][0]["type"], "string")
            self.assertTrue(prompt["selected_tool"]["fields"][0]["required"])
            self.assertIn("beta.txt", [item["value"] for item in prompt["observed_values"]])
            self.assertNotIn("宇宙计划", [item["value"] for item in prompt["observed_values"]])
            self.assertEqual(prompt["failed_calls"][0]["arguments"], {"name": "宇宙计划"})

    async def test_configured_server_only_exposes_its_tools(self):
        from argparse import Namespace

        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        with patch("agent.load_servers", return_value=configuration), patch("agent.run") as run_mock:
            run_mock.return_value = ("complete", [])
            await run_with_mcp(Namespace(goal="list documents", root=Path.cwd(),
                                         tier="direct", mcp_config=Path("unused")))
        exposed = run_mock.call_args.kwargs["tools"]
        self.assertTrue(exposed)
        self.assertIn("list_available_tools", exposed)
        self.assertTrue(all(name.startswith("mcp_") for name in exposed if name != "list_available_tools"))

    async def test_chat_approval_uses_shared_input_queue(self):
        from argparse import Namespace

        configuration = {"sample": {"command": sys.executable,
                                    "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        approvals = queue.Queue()
        approvals.put("YES")

        def call_one(goal, root, tier, **kwargs):
            selected = next(name for name, entry in kwargs["tools"].items()
                            if entry.get("tool_name", "").endswith("/list_documents"))
            result = kwargs["act"](root, selected, {})
            return result, [{"tool": selected, "args": {}, "result": result}]

        with (patch("agent.load_servers", return_value=configuration),
              patch("agent.run", side_effect=call_one),
              patch("builtins.input", side_effect=AssertionError("direct stdin is unsafe in chat"))):
            result, events, tier = await run_with_mcp(
                Namespace(goal="list documents", root=Path.cwd(), tier="deep",
                          mcp_config=Path("unused"), approval_queue=approvals))
        self.assertEqual(len(events), 1)
        self.assertIn("alpha.txt", result)

    async def test_chat_session_keeps_same_mcp_bridge_across_turns(self):
        from argparse import Namespace

        configuration = {"sample": {"command": sys.executable,
                                    "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        approvals = queue.Queue()
        approvals.put("YES")
        approvals.put("YES")

        def call_one(goal, root, tier, **kwargs):
            selected = next(name for name, entry in kwargs["tools"].items()
                            if entry.get("tool_name", "").endswith("/list_documents"))
            result = kwargs["act"](root, selected, {})
            return result, [{"tool": selected, "args": {}, "result": result}]

        with (patch("agent.load_servers", return_value=configuration) as load,
              patch("agent.run", side_effect=call_one)):
            session = await asyncio.to_thread(MCPChatSession, Path("unused"))
            try:
                args = Namespace(goal="list documents", root=Path.cwd(), tier="deep",
                                 mcp_config=Path("unused"), approval_queue=approvals)
                first = await asyncio.to_thread(session.run, args, [])
                second = await asyncio.to_thread(session.run, args, [])
            finally:
                await asyncio.to_thread(session.close)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(first[1][0]["tool"], second[1][0]["tool"])
        self.assertIn("alpha.txt", second[0])

    async def test_discover_fill_select_call_and_return_result(self):
        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        async with MCPBridge(configuration) as bridge:
            identifier = next(key for key, target in bridge.targets.items() if target[1] == "repeat_text")
            entry = bridge.catalog[identifier]
            self.assertIn("repeat_text", entry["description"])
            self.assertEqual(entry["schema"]["properties"]["count"]["type"], "integer")
            self.assertNotIn("count", entry["schema"]["required"])
            messages_seen = []

            def fake_model(messages, schema=None):
                messages_seen.append((messages, schema))
                return '{"text":"ok","count":3}'

            with patch("agent.local_model", side_effect=fake_model):
                arguments = tool_arguments("重复三次 ok", "", identifier, [], bridge.catalog)
            self.assertEqual(arguments, {"text": "ok", "count": 3})
            self.assertIn("count", messages_seen[0][0][1]["content"])

            loop = asyncio.get_running_loop()
            choices = iter([identifier, "DONE"])

            def act(_root, tool, args):
                return asyncio.run_coroutine_threadsafe(bridge.call(tool, args), loop).result(timeout=10)

            with tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "重复三次 ok", Path(directory), "direct",
                    pick=lambda question, criteria, state: next(choices),
                    fill=lambda goal, advice, tool, events: arguments,
                    act=act, answer=lambda goal, events: events[-1]["result"],
                    tools=bridge.catalog,
                )
            self.assertEqual(len(events), 1)
            self.assertIn("okokok", result)

    async def test_bad_parameter_rejected_before_call(self):
        configuration = {"sample": {"command": sys.executable,
                                     "args": [str(Path(__file__).with_name("fake_mcp_server.py"))]}}
        async with MCPBridge(configuration) as bridge:
            identifier = next(key for key, target in bridge.targets.items() if target[1] == "repeat_text")
            with tempfile.TemporaryDirectory() as directory:
                result, events = await asyncio.to_thread(
                    run, "重复", Path(directory), "direct",
                    pick=lambda question, criteria, state: identifier,
                    fill=lambda goal, advice, tool, events: {"text": "ok", "count": "3"},
                    act=lambda root, tool, args: self.fail("Must not call MCP"),
                    tools=bridge.catalog,
                )
            self.assertIn("失败", result)
            self.assertEqual(len(events), 1)

    def test_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.json"
            path.write_text(json.dumps({"mcpServers": {"test": {"command": sys.executable}}}), encoding="utf-8")
            self.assertIn("test", load_servers(path))
            path.write_text(json.dumps({"mcpServers": {"bad": {"url": "file:///tmp"}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_servers(path)
