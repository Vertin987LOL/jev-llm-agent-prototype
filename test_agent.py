import argparse
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent import LiveTrace, advise, candidate_repair, chat, compact_history, completion_check, conversation_goal, earlier_observations, execute, heartbeat_decision, list_memory, local_model, memory_index, recall_history, recent_observations, run, show_trace
from mcp_bridge import MCPToolError


class AgentLoopTests(unittest.TestCase):
    def test_tool_list_is_from_registered_catalog_not_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "not_a_tool.py").write_text("pass", encoding="utf-8")
            result, events = run(
                "有哪些工具", Path(directory), "guided",
                pick=lambda question, criteria, state: "list_available_tools",
                planner=lambda goal, events: "查看当前代理真正注册的工具",
                fill=lambda goal, advice, tool, events: self.fail("Catalog needs no model arguments"),
                act=lambda root, tool, args: self.fail("Catalog is not a file operation"),
                answer=lambda goal, events: self.fail("Catalog answer must not be invented by a model"),
                verify=lambda goal, events: goal == "有哪些工具" and "list_available_tools" in events[0]["result"],
            )
        self.assertIn("list_available_tools", result)
        self.assertIn("read_text", result)
        self.assertIn("count_lines", result)
        self.assertNotIn("not_a_tool.py", result)
        self.assertEqual(events[0]["result"], result)

    def test_tool_list_displays_mcp_names_not_internal_ids(self):
        tools = {"mcp_0": {"description": "sample/find: Lookup records", "purpose": "Lookup records",
                           "tool_name": "sample/find"},
                 "list_available_tools": {"description": "列出可用工具"}}
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "有哪些工具", Path(directory), "direct", tools=tools,
                pick=lambda question, criteria, state: "list_available_tools",
            )
        self.assertIn("sample/find：Lookup records", result)
        self.assertNotIn("mcp_0", result)
        self.assertEqual(len(events), 1)

    def test_nonexistent_local_file_returns_to_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.txt").write_text("芯片报告", encoding="utf-8")
            decisions = iter(["read_text", "list_files", "read_text"])
            seen = []

            def pick(question, criteria, state):
                seen.append(state["events"])
                return next(decisions)

            result, events = run(
                "找到并读取本地报告", root, "direct", pick=pick,
                planner=lambda goal, events: "只用真实文件名；找不到就先列出文件",
                fill=lambda goal, advice, tool, events: (
                    {} if tool == "list_files" else
                    {"path": "chip_report.txt" if not events else "report.txt"}),
                verify=lambda goal, events: events[-1]["tool"] == "read_text"
                and "result" in events[-1],
                answer=lambda goal, events: events[-1]["result"],
                max_steps=4,
            )
        self.assertEqual(result, "芯片报告")
        self.assertEqual([event["tool"] for event in events],
                         ["read_text", "list_files", "read_text"])
        self.assertEqual(events[0]["args"], {"path": "chip_report.txt"})
        self.assertIn("File does not exist", events[0]["error"])
        self.assertIn("error", seen[1][0])

    def test_repeated_nonexistent_file_stops_without_retrying_action(self):
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "读取本地报告", Path(directory), "direct",
                pick=lambda question, criteria, state: "read_text",
                planner=lambda goal, events: "仍需找到真实文件名",
                fill=lambda goal, advice, tool, events: {"path": "guessed.txt"},
            )
        self.assertIn("死循环", result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["args"], {"path": "guessed.txt"})

    def test_advisor_prompt_focuses_on_one_next_action(self):
        from unittest.mock import patch

        captured = {}

        def model(messages, schema=None, *, advisory=False):
            captured["prompt"] = messages[0]["content"]
            captured["payload"] = json.loads(messages[1]["content"])
            captured["advisory"] = advisory
            return "先找本地报告"

        with patch("agent.local_model", side_effect=model):
            self.assertEqual(advise("先找报告再搜新闻", [], {"file_search": "查本地文件"}), "先找本地报告")
        self.assertIn("整个任务最终要完成什么", captured["prompt"])
        self.assertIn("眼下最值得先做的一件事", captured["prompt"])
        self.assertIn("没有记录就表示本轮还没执行任何动作", captured["prompt"])
        self.assertEqual(captured["payload"]["events"], [])
        self.assertTrue(captured["advisory"])

    def test_only_advisor_has_bounded_generation(self):
        from io import BytesIO
        from unittest.mock import patch

        bodies = []

        def fake_urlopen(request, timeout):
            bodies.append(json.loads(request.data))
            return BytesIO(b'{"message":{"content":"next action"}}')

        with patch("agent.urlopen", side_effect=fake_urlopen):
            self.assertEqual(local_model([{"role": "user", "content": "task"}], advisory=True), "next action")
            self.assertEqual(local_model([{"role": "user", "content": "task"}]), "next action")
        self.assertFalse(bodies[0]["think"])
        self.assertEqual(bodies[0]["options"]["num_predict"], 384)
        self.assertEqual(bodies[0]["options"]["num_ctx"], 49152)
        self.assertEqual(bodies[1]["options"]["num_ctx"], 49152)
        self.assertEqual(bodies[0]["model"], "qwen3:4b")
        self.assertNotIn("think", bodies[1])

    def test_deepseek_provider_uses_chat_and_json_mode(self):
        from io import BytesIO

        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return BytesIO(json.dumps({"choices": [{"message": {"content": '  {"pageId": 2}  '}}]}).encode())

        schema = {"type": "object", "properties": {"pageId": {"type": "integer"}},
                  "required": ["pageId"]}
        with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": "deepseek",
                                  "DEEPSEEK_API_KEY": "test-only-key"}), patch("agent.urlopen", side_effect=fake_urlopen):
            self.assertEqual(local_model([{"role": "user", "content": "pick page"}], schema),
                             '{"pageId": 2}')
            local_model([{"role": "user", "content": "next step"}], advisory=True)
        body = json.loads(requests[0].data)
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("JSON", body["messages"][0]["content"])
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-only-key")
        self.assertEqual(json.loads(requests[1].data)["max_tokens"], 384)

    def test_deepseek_requires_key_without_network_request(self):
        with patch.dict(os.environ, {"AGENT_LLM_PROVIDER": "deepseek",
                                  "DEEPSEEK_API_KEY": ""}), patch("agent.urlopen") as fake_urlopen:
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                local_model([{"role": "user", "content": "task"}])
            fake_urlopen.assert_not_called()

    def test_direct_answer_does_not_require_tool_evidence(self):
        trace = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "解释什么是行数", Path(directory), "guided",
                pick=lambda question, criteria, state: "ANSWER_DIRECTLY",
                planner=lambda goal, events: "这是普通问题，不需要工具",
                verify=lambda goal, events: self.fail("No-tool answer should not require tool verification"),
                answer=lambda goal, events: "一行文字算一行。",
                trace=trace,
            )
        self.assertEqual(result, "一行文字算一行。")
        self.assertEqual(events, [])
        self.assertEqual([step["type"] for step in trace], ["advisor", "decision"])
        self.assertEqual(trace[0]["text"], "这是普通问题，不需要工具")
        self.assertEqual(trace[1]["choice"], "ANSWER_DIRECTLY")

    def test_chat_keeps_previous_answer_and_tool_result(self):
        from unittest.mock import patch

        requests = iter(["统计 a.txt", "它有多少行？", "/exit"])
        goals = []

        def fake_turn(args, goal, trace):
            goals.append(goal)
            trace.append({"type": "advisor", "text": "读取已有结果"})
            trace.append({"type": "decision", "choice": "DONE"})
            return (("共 2 行。", [{"tool": "count_lines", "args": {"path": "a.txt"}, "result": "2"}], "guided")
                    if len(goals) == 1 else ("2 行。", [], "guided"))

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="guided", mcp_config=None, chat=True)
        with patch("builtins.input", side_effect=lambda prompt: next(requests)), \
             patch("agent.local_model", return_value="已统计 a.txt"), \
             patch("agent.run_turn", side_effect=fake_turn), \
             patch("agent.save_trace") as saved:
            chat(args)
        self.assertEqual(goals[0], "统计 a.txt")
        context = json.loads(goals[1])
        self.assertEqual(context["current_request"], "它有多少行？")
        self.assertEqual(context["previous_turns"][0]["events"][0]["result"], "2")
        self.assertEqual(context["previous_turns"][0]["answer"], "共 2 行。")
        self.assertEqual(saved.call_count, 2)
        self.assertEqual(saved.call_args_list[0].args[1][0]["text"], "读取已有结果")

    def test_chat_uses_short_impression_but_keeps_recall_index(self):
        requests = iter(["查 A", "查 B", "还记得 A 吗", "/exit"])
        goals = []

        def fake_turn(args, goal, trace):
            goals.append(goal)
            return "答复", [{"tool": "search", "args": {"query": "A"}, "result": "找到 A"}], "deep"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="deep", mcp_config=None, chat=True)
        with patch("builtins.input", side_effect=lambda prompt: next(requests)), \
             patch("agent.local_model", return_value="曾查过 A"), \
             patch("agent.run_turn", side_effect=fake_turn), patch("agent.save_trace"):
            chat(args)
        context = json.loads(goals[2])
        self.assertEqual(context["current_request"], "还记得 A 吗")
        self.assertIn("曾查过 A", context["previous_summary"])
        self.assertEqual(len(context["previous_turns"]), 1)
        self.assertEqual([item["turn_id"] for item in context["memory_index"]], [1, 2])
        self.assertEqual(context["memory_index"][0]["key_parameters"][0]["args"], {"query": "A"})

    def test_history_compaction_preserves_recent_turns_and_summary(self):
        history = [{"user": f"问题{index}", "answer": f"回答{index}", "events": [
            {"tool": "read_text", "args": {"path": f"{index}.txt"}, "result": f"内容{index}"}]}
                   for index in range(4)]
        seen = []

        def summarize(messages, *, advisory):
            seen.append(json.loads(messages[1]["content"]))
            return "旧目标：读取 0.txt"

        recent, summary = compact_history(history, summarizer=summarize)
        self.assertEqual(summary, "旧目标：读取 0.txt")
        self.assertEqual(len(recent), 1)
        self.assertEqual(len(seen[0]["old_turns"]), 3)
        self.assertEqual(seen[0]["old_turns"][0]["events"][0]["result"], "内容0")
        context = json.loads(conversation_goal("现在读取 4.txt", recent, summary))
        self.assertEqual(context["previous_summary"], summary)
        self.assertEqual(context["previous_turns"][0]["user"], "问题3")
        self.assertIn("以前的动作不等于本轮已执行", context["instruction"])
        self.assertEqual(len(history), 4)

    def test_history_compaction_rolls_forward_and_falls_back_on_failure(self):
        history = [{"user": str(index), "answer": "done", "events": []} for index in range(4)]
        recent, summary = compact_history(history, "previous", summarizer=lambda *args, **kwargs: "new")
        self.assertEqual(summary, "new")
        recent.append({"user": "4", "answer": "done", "events": []})

        def fail(*args, **kwargs):
            raise ConnectionError("model unavailable")

        recent, summary = compact_history(recent, summary, summarizer=fail)
        self.assertEqual(len(recent), 1)
        self.assertIn("new", summary)
        self.assertIn("用户：3", summary)

    def test_compacts_after_two_turns(self):
        history = [{"user": "先查 A", "answer": "查到了 A", "events": []},
                   {"user": "再查 B", "answer": "查到了 B", "events": []}]
        recent, summary = compact_history(history, summarizer=lambda *args, **kwargs: "已查 A")
        self.assertEqual(summary, "已查 A")
        self.assertEqual([turn["user"] for turn in recent], ["再查 B"])

    def test_earlier_events_are_visible_without_becoming_new_evidence(self):
        events = [{"tool": "search", "args": {"query": str(index)}, "result": "fact" * 100}
                  for index in range(7)]
        older = earlier_observations(events)
        self.assertEqual([event["args"]["query"] for event in older], ["0", "1"])
        self.assertEqual(len(older[0]["excerpt"]), 300)
        self.assertEqual(len(recent_observations(events)), 5)
        self.assertEqual(events[0]["result"], "fact" * 100)

    def test_memory_index_keeps_key_arguments_and_retrieves_raw_result(self):
        archive = [{"user": f"问题{index}", "answer": "答复", "events": []} for index in range(65)]
        archive[0]["events"] = [{"tool": "search", "args": {"query": "old phrase"},
                                   "result": "原始搜索结果"}]
        self.assertEqual(memory_index(archive)[0]["turn_id"], 36)
        self.assertEqual(memory_index(archive, 2)[0]["turn_id"], 1)
        self.assertEqual(json.loads(list_memory(archive, {"page": 2}))["turns"][0]
                         ["key_parameters"][0]["args"]["query"], "old phrase")
        self.assertEqual(json.loads(recall_history(archive, {"turn_id": 1}))
                         ["events"][0]["result"], "原始搜索结果")
        with self.assertRaises(ValueError):
            recall_history(archive, {"turn_id": 0})

    def test_recall_is_read_only_and_uses_archived_turn(self):
        archive = [{"user": "记住搜索词", "answer": "已搜", "events": [
            {"tool": "search", "args": {"query": "原词"}, "result": "证据"}]}]
        choices = iter(["recall_history", "DONE"])
        tools = {"recall_history": {"description": "取回档案", "parameters": {"turn_id": "integer"}}}
        answer, events = run("刚才搜的词是什么", Path.cwd(), "guided", tools=tools,
                             memory_archive=archive, max_steps=2,
                             pick=lambda question, criteria, state: next(choices),
                             planner=lambda goal, events: "回查上轮",
                             fill=lambda goal, advice, tool, events: {"turn_id": 1},
                             act=lambda root, tool, args: self.fail("No external tool should execute"),
                             answer=lambda goal, events: "原词")
        self.assertEqual(answer, "原词")
        self.assertEqual(json.loads(events[0]["result"])["events"][0]["result"], "证据")

    def test_clear_goal_retries_without_asking_user_to_plan_tools(self):
        choices = iter(["ASK_USER", "PROCEED", "list_files"])
        seen = []

        def choose_next(question, criteria, state):
            choice = next(choices)
            seen.append((choice, set(criteria)))
            return choice

        result, events = run("找目录里的报告", Path.cwd(), "guided",
                             pick=choose_next, planner=lambda goal, events: "先列文件",
                             act=lambda root, tool, args: "report.txt", max_steps=1)
        self.assertEqual(events[0]["tool"], "list_files")
        self.assertNotIn("ASK_USER", seen[-1][1])

    def test_verified_task_does_not_offer_done_without_tool_evidence(self):
        choices = []

        def pick(question, criteria, state):
            choices.append(criteria)
            return "list_files"

        result, events = run("列出文件", Path.cwd(), "deep", max_steps=1,
                             pick=pick, planner=lambda goal, events: "先列文件",
                             act=lambda root, tool, args: "report.txt",
                             verify=lambda goal, events: bool(events),
                             answer=lambda goal, events: "找到 report.txt")
        self.assertNotIn("DONE", choices[0])
        self.assertEqual(result, "找到 report.txt")

    def test_failed_chat_turn_records_partial_trace(self):
        from unittest.mock import patch

        args = argparse.Namespace(goal="测试", root=Path.cwd(), tier="guided", mcp_config=None, chat=True)

        def fail(args, goal, trace):
            trace.append({"type": "tier", "choice": "guided"})
            raise RuntimeError("connection failed")

        with patch("builtins.input", return_value="/exit"), \
             patch("agent.run_turn", side_effect=fail), \
             patch("agent.save_trace") as saved:
            chat(args)
        self.assertEqual(saved.call_args.args[0], "测试")
        self.assertEqual(saved.call_args.args[1][0]["choice"], "guided")
        self.assertEqual(saved.call_args.args[3], "connection failed")

    def test_completion_check_records_jev_verdict(self):
        from unittest.mock import patch

        trace = []
        with patch("agent.choose", return_value="CONTINUE"):
            complete = completion_check("统计文件", [{"tool": "list_files", "result": "a.txt"}],
                                        {"count_lines": {"description": "统计行数"}}, trace)
        self.assertFalse(complete)
        self.assertEqual(trace, [{"type": "completion", "after_events": 1, "choice": "CONTINUE"}])

    def test_recommendation_flows_to_jev_and_selected_tool_arguments(self):
        from unittest.mock import patch
        import json

        observed = {}

        def model(messages, schema=None, *, advisory=False):
            payload = json.loads(messages[1]["content"])
            if schema is None:
                observed["advisor_options"] = payload["decision_options"]
                return "建议 count 工具，name 填 beta.txt，因为目标要统计它"
            observed["argument_context"] = payload
            return '{"name":"beta.txt"}'

        def pick(question, criteria, state):
            observed["jev_options"] = criteria
            observed["jev_thought"] = state["thinking_state"]["text"]
            return "count"

        with tempfile.TemporaryDirectory() as directory, patch("agent.local_model", side_effect=model):
            result, events = run(
                "统计 beta.txt 的行数", Path(directory), "guided",
                tools={"count": {"description": "Count document lines", "schema": {
                    "type": "object", "properties": {"name": {"type": "string", "description": "Document name"}},
                    "required": ["name"], "additionalProperties": False}}},
                pick=pick, act=lambda root, tool, args: "3",
                verify=lambda goal, events: True, answer=lambda goal, events: "3",
            )
        self.assertEqual(result, "3")
        self.assertEqual(events[0]["args"], {"name": "beta.txt"})
        self.assertEqual(observed["advisor_options"], observed["jev_options"])
        self.assertEqual(observed["jev_thought"], observed["argument_context"]["current_intent"])
        self.assertEqual(observed["argument_context"]["selected_tool"]["fields"][0]["description"],
                         "Document name")

    def test_same_thought_reaches_jev_and_parameter_filler(self):
        thoughts = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "统计 beta.txt", Path(directory), "guided",
                tools={"count": {"description": "Count document lines"}},
                planner=lambda goal, events: "当前要统计 beta.txt",
                pick=lambda question, criteria, state: thoughts.append(state["thinking_state"]) or "count",
                fill=lambda goal, advice, tool, events: thoughts.append(advice) or {"name": "beta.txt"},
                act=lambda root, tool, args: "3",
                verify=lambda goal, events: True,
                answer=lambda goal, events: "3",
            )
        self.assertEqual(result, "3")
        self.assertEqual(thoughts[0]["text"], thoughts[1])
        self.assertEqual(thoughts[0]["based_on_events"], 0)

    def test_advisor_sees_available_tool_descriptions(self):
        from unittest.mock import patch
        import json

        messages = []

        def model(prompt, *, advisory=False):
            messages.extend(prompt)
            return "下一步读取文档"

        with patch("agent.local_model", side_effect=model):
            advice = advise("找文件", [], {"read": "Read document"})
        self.assertEqual(advice, "下一步读取文档")
        self.assertIn("Read document", json.loads(messages[1]["content"])["decision_options"]["read"])

    def test_advisor_and_jev_see_exact_same_actions(self):
        from unittest.mock import patch
        import json

        observed = {}

        def model(messages, schema=None, *, advisory=False):
            observed["advisor_options"] = json.loads(messages[1]["content"])["decision_options"]
            return "建议 count，参数取 beta.txt"

        def pick(question, criteria, state):
            observed["jev_options"] = criteria
            observed["thought"] = state["thinking_state"]["text"]
            return "count"

        with tempfile.TemporaryDirectory() as directory, patch("agent.local_model", side_effect=model):
            run("统计 beta.txt", Path(directory), "guided",
                tools={"count": {"description": "Count document lines"}},
                pick=pick, fill=lambda goal, advice, tool, events: {"name": "beta.txt"},
                act=lambda root, tool, args: "3", verify=lambda goal, events: True,
                answer=lambda goal, events: "3")
        self.assertEqual(observed["advisor_options"], observed["jev_options"])
        self.assertEqual(observed["thought"], "建议 count，参数取 beta.txt")
        self.assertNotIn("DONE", observed["advisor_options"])
        self.assertNotIn("ASK_ADVISOR", observed["advisor_options"])

    def test_done_after_tool_error_must_not_answer(self):
        decisions = iter(["count", "DONE", "count"])
        answers = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "统计文档行数", Path(directory), "direct",
                tools={"count": {"description": "Count document"}},
                pick=lambda question, criteria, state: next(decisions),
                planner=lambda goal, events: "先前工具失败，改用真正的文档名",
                fill=lambda goal, advice, tool, events: {"name": "alpha.txt" if advice else "bad"},
                act=lambda root, tool, args: "2" if args["name"] == "alpha.txt" else
                    (_ for _ in ()).throw(MCPToolError("Unknown document")),
                verify=lambda goal, events: bool(events[-1].get("result")),
                answer=lambda goal, events: answers.append(len(events)) or "完成",
                max_steps=3,
            )
        self.assertEqual(result, "完成")
        self.assertEqual(answers, [2])
        self.assertEqual(events[-1]["result"], "2")

    def test_jev_selects_grounded_candidate_for_one_parameter(self):
        chosen = []
        tools = {"count": {"description": "Count document lines", "schema": {
            "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
        events = [
            {"tool": "read", "name": "Read document", "args": {"name": "alpha.txt"},
             "result": '{"structuredContent":{"result":"今日晴天"}}'},
            {"tool": "read", "name": "Read document", "args": {"name": "beta.txt"},
             "result": '{"structuredContent":{"result":"宇宙计划"}}'},
            {"tool": "count", "name": "Count document lines", "args": {"name": "今日晴天"},
             "error": "Unknown document"},
        ]

        def pick(question, criteria, state):
            chosen.append((criteria, state))
            return next(key for key, description in criteria.items() if "alpha.txt" in description)

        result = candidate_repair("统计含今日晴天的文档", "count", {"name": "今日晴天"},
                                  events, tools, pick=pick)
        self.assertEqual(result, {"name": "alpha.txt"})
        self.assertIn("beta.txt", str(chosen[0][0]))
        self.assertIn("name", str(chosen[0][1]))

    def test_mcp_error_returns_to_decision_with_failure_evidence(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "统计 alpha", Path(directory), "direct",
                tools={"count": {"description": "Count document"}},
                pick=lambda question, criteria, state: seen.append(state["events"]) or "count",
                planner=lambda goal, events: "刚才的参数不对；改用 alpha.txt",
                fill=lambda goal, advice, tool, events: {"name": "alpha.txt" if advice else "alpha"},
                act=lambda root, tool, args: "2" if args["name"] == "alpha.txt" else
                    (_ for _ in ()).throw(MCPToolError("Unknown document")),
                verify=lambda goal, events: bool(events[-1].get("result")),
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(result, "完成")
        self.assertEqual(len(events), 2)
        self.assertIn("error", events[0])
        self.assertEqual(events[1]["args"], {"name": "alpha.txt"})
        self.assertIn("Unknown document", str(seen[1]))

    def test_incomplete_result_becomes_next_turn_intent(self):
        advice_seen = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "统计两份文件", Path(directory), "direct",
                tools={"count": {"description": "Count a document"}},
                pick=lambda question, criteria, state: advice_seen.append(state["advice"]) or "count",
                planner=lambda goal, events: "已统计 alpha；还要统计 beta",
                fill=lambda goal, advice, tool, events: {"name": "beta" if advice else "alpha"},
                act=lambda root, tool, args: "2" if args["name"] == "alpha" else "3",
                verify=lambda goal, events: len(events) == 2,
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(result, "完成")
        self.assertEqual([event["args"]["name"] for event in events], ["alpha", "beta"])
        self.assertEqual(advice_seen, ["", "已统计 alpha；还要统计 beta"])

    def test_completion_check_uses_evidence_and_catalog_not_advice(self):
        from unittest.mock import patch

        with patch("agent.choose", return_value="CONTINUE") as chooser:
            complete = completion_check("count the lines", [{"tool": "read", "result": "a\nb"}],
                                        {"count": {"description": "Count lines"}})
        self.assertFalse(complete)
        state = chooser.call_args.args[2]
        self.assertIn("count", state["available_tools"])
        self.assertNotIn("advice", state)

    def test_rejected_completion_reaches_advisor_and_requires_new_evidence(self):
        from unittest.mock import patch

        advisor_payloads = []
        choices = []

        def model(messages, schema=None, *, advisory=False):
            advisor_payloads.append(json.loads(messages[1]["content"]))
            return "读取后统计行数"

        def pick(question, criteria, state):
            choices.append((set(criteria), state.get("feedback", "")))
            if not state["events"]:
                return "read_text"
            return "DONE" if "DONE" in criteria else "count_lines"

        with patch("agent.local_model", side_effect=model), tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "读取 note.txt 第一行并统计行数", Path(directory), "guided",
                tools={"read_text": {"description": "读取文本"},
                       "count_lines": {"description": "统计文本行数"}},
                pick=pick, fill=lambda goal, advice, tool, events: {"path": "note.txt"},
                act=lambda root, tool, args: "Header\nNext\n" if tool == "read_text" else "2",
                verify=lambda goal, events: any(event["tool"] == "count_lines" for event in events),
                answer=lambda goal, events: "首行 Header，共 2 行",
            )
        self.assertEqual(result, "首行 Header，共 2 行")
        self.assertEqual([event["tool"] for event in events], ["read_text", "count_lines"])
        self.assertIn("DONE", choices[1][0])
        self.assertNotIn("DONE", choices[2][0])
        self.assertIn("count_lines", choices[2][1])
        self.assertIn("count_lines", advisor_payloads[-1]["feedback"])
        self.assertEqual(advisor_payloads[-1]["events"][0]["result"], "Header\nNext\n")

    def test_tool_error_feedback_reaches_next_decision(self):
        from unittest.mock import patch

        seen = []
        advisor_payloads = []

        def model(messages, schema=None, *, advisory=False):
            advisor_payloads.append(json.loads(messages[1]["content"]))
            return "读取文件"

        def pick(question, criteria, state):
            seen.append(state)
            return "read_text" if not state["events"] else "STOP"

        with patch("agent.local_model", side_effect=model), tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "读取 missing.txt", Path(directory), "guided", pick=pick,
                fill=lambda goal, advice, tool, events: {"path": "missing.txt"},
            )
        self.assertIn("无法安全继续", result)
        self.assertIn("File does not exist", seen[1]["feedback"])
        self.assertEqual(seen[1]["events"][0]["error"], events[0]["error"])
        self.assertEqual(advisor_payloads[1]["events"][0]["error"], events[0]["error"])
        self.assertIn("File does not exist", advisor_payloads[1]["feedback"])

    def test_repeat_argument_can_be_repaired_once(self):
        attempts = []
        advice = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "分别统计 alpha.txt 和 beta.txt", Path(directory), "direct",
                tools={"count": {"description": "Count lines", "schema": {
                    "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
                pick=lambda question, criteria, state: "count" if len(state["events"]) < 2 else "DONE",
                planner=lambda goal, events: advice.append(len(events)) or "已统计 alpha.txt，还需 beta.txt",
                fill=lambda goal, hint, tool, events: attempts.append(hint) or
                     {"name": "beta.txt" if hint else "alpha.txt"},
                act=lambda root, tool, args: "2" if args["name"] == "alpha.txt" else "3",
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(result, "完成")
        self.assertEqual([event["args"]["name"] for event in events], ["alpha.txt", "beta.txt"])
        self.assertEqual(advice, [1])

    def test_evidence_only_completion_gate_prevents_extra_call(self):
        decisions = iter(["read", "repeat", "read"])
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "读取并重复", Path(directory), "direct",
                tools={"read": {"description": "Read"}, "repeat": {"description": "Repeat"}},
                pick=lambda question, criteria, state: next(decisions),
                planner=lambda goal, events: "继续",
                fill=lambda goal, advice, tool, events: {},
                act=lambda root, tool, args: tool,
                verify=lambda goal, events: len(events) == 2,
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(result, "完成")
        self.assertEqual([event["tool"] for event in events], ["read", "repeat"])

    def test_deep_advises_after_every_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("one\ntwo\n", encoding="utf-8")
            picks = iter(["list_files", "count_lines", "DONE"])
            advised = []

            def planner(goal, events):
                advised.append(len(events))
                return "读出文件并给出行数"

            result, events = run(
                "找出 note.txt 并数行", root, "deep",
                pick=lambda question, criteria, state: next(picks),
                planner=planner,
                fill=lambda goal, advice, tool, events: {"path": "note.txt"} if tool == "count_lines" else {},
                answer=lambda goal, events: "完成：" + events[-1]["result"],
            )
            self.assertEqual(result, "完成：2")
            self.assertEqual(advised, [0, 1, 2])
            self.assertEqual([event["tool"] for event in events], ["list_files", "count_lines"])

    def test_deep_advises_after_final_result_too(self):
        advised = []
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "列出文件", Path(directory), "deep",
                pick=lambda question, criteria, state: "list_files" if not state["events"] else "DONE",
                planner=lambda goal, events: advised.append(len(events)) or "核对当前结果",
                fill=lambda goal, advice, tool, events: {},
                verify=lambda goal, events: True,
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(advised, [0, 1])
        self.assertEqual(len(events), 1)
        self.assertIsInstance(result, str)

    def test_goal_keeps_events_and_thought_when_continued(self):
        from unittest.mock import patch

        requests = iter(["/goal 找到并读取报告", "/goal", "/goal continue", "/exit"])
        calls = []

        def fake_turn(args, goal, trace):
            calls.append((goal, args.tier, args.max_steps, list(args.initial_events), args.initial_thought))
            trace.append({"type": "advisor", "text": f"当前想法{len(calls)}"})
            if len(calls) == 1:
                args.outcome["status"] = "budget"
                return "达到上限", [{"tool": "list_files", "args": {}, "result": "report.txt"}], "deep"
            args.outcome["status"] = "completed"
            return "报告内容", args.initial_events + [{"tool": "read_text", "result": "报告内容"}], "deep"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="guided", mcp_config=None,
                                  chat=True, goal_max_steps=2)
        with (patch("builtins.input", side_effect=lambda prompt: next(requests)),
              patch("agent.run_turn", side_effect=fake_turn),
              patch("agent.save_trace")):
            chat(args)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1:3], ("deep", 2))
        self.assertEqual(calls[1][3][0]["result"], "report.txt")
        self.assertEqual(calls[1][4], "当前想法1")

    def test_mcp_goal_continue_reuses_chat_session(self):
        requests = iter(["/goal 读页面", "/goal continue", "/exit"])
        sessions = []
        calls = []

        class FakeSession:
            def __init__(self, config):
                sessions.append(self)
                self.closed = False

            def close(self):
                self.closed = True

        def fake_turn(args, goal, trace):
            calls.append(args.mcp_session)
            args.outcome["status"] = "budget" if len(calls) == 1 else "completed"
            return "pause" if len(calls) == 1 else "done", [], "deep"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="deep", mcp_config=Path("config.json"),
                                  chat=True, goal_max_steps=1)
        with (patch("builtins.input", side_effect=lambda prompt: next(requests)),
              patch("agent.MCPChatSession", FakeSession),
              patch("agent.run_turn", side_effect=fake_turn),
              patch("agent.save_trace")):
            chat(args)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(calls, [sessions[0], sessions[0]])
        self.assertTrue(sessions[0].closed)

    def test_resume_run_reads_old_results_and_reports_status(self):
        previous = [{"tool": "list_files", "args": {}, "result": "report.txt"}]
        outcome = {}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "report.txt").write_text("内容", encoding="utf-8")
            result, events = run(
                "读取报告", Path(directory), "deep",
                pick=lambda question, criteria, state: "read_text",
                planner=lambda goal, events: "读报告" if events[0]["result"] == "report.txt" else "错误",
                fill=lambda goal, advice, tool, events: {"path": "report.txt"},
                verify=lambda goal, events: len(events) == 2,
                answer=lambda goal, events: events[-1]["result"],
                initial_events=previous, outcome=outcome,
            )
        self.assertEqual(result, "内容")
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(len(events), 2)
        self.assertEqual(len(previous), 1)

    def test_deep_loop_handles_six_actions_without_premature_done(self):
        thoughts = []
        outcome = {}
        parameters = iter(range(6))
        tools = {"observe": {"description": "inspect next item", "parameters": {"index": "integer"}}}
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "six observations", Path(directory), "deep", max_steps=7,
                pick=lambda question, criteria, state: "observe",
                planner=lambda goal, events: thoughts.append(len(events)) or "continue",
                fill=lambda goal, advice, tool, events: {"index": next(parameters)},
                act=lambda root, tool, arguments: "result",
                verify=lambda goal, events: len(events) >= 6,
                answer=lambda goal, events: "done",
                outcome=outcome, tools=tools,
            )
        self.assertEqual((result, outcome["status"]), ("done", "completed"))
        self.assertEqual(len(events), 6)
        self.assertEqual(thoughts, list(range(7)))

    def test_heartbeat_passes_time_and_evidence_to_inner_voice_then_jev(self):
        observed = {}
        goal = {"text": "读取报告", "status": "budget", "thought": "下一步读报告",
                "events": [{"tool": "list_files", "result": "report.txt"}]}

        def thinker(messages):
            observed["prompt"] = json.loads(messages[1]["content"])
            return "已经找到报告，继续读取"

        def picker(question, options, state):
            observed["options"] = options
            observed["state"] = state
            return "RESUME_GOAL"

        thought, decision = heartbeat_decision(
            goal, datetime(2026, 10, 6, 12, 30, tzinfo=timezone.utc),
            can_resume=True, thinker=thinker, pick=picker)
        self.assertEqual((thought, decision), ("已经找到报告，继续读取", "RESUME_GOAL"))
        self.assertEqual(observed["prompt"]["time"], "2026-10-06T12:30:00+00:00")
        self.assertEqual(observed["state"]["inner_voice"], thought)
        self.assertEqual(observed["prompt"]["recent_events"][0]["result"], "report.txt")
        self.assertIn("REST", observed["options"])

    def test_heartbeat_without_goal_can_only_rest(self):
        options_seen = []
        thought, decision = heartbeat_decision(
            None, datetime.now(timezone.utc), can_resume=True,
            thinker=lambda messages: "没有待办，休息",
            pick=lambda question, options, state: options_seen.append(options) or "REST")
        self.assertEqual(decision, "REST")
        self.assertEqual(list(options_seen[0]), ["REST"])

    def test_heartbeat_pending_task_can_wait_then_start(self):
        seen = []
        for choice in ("REST", "START_TASK"):
            thought, decision = heartbeat_decision(
                None, datetime.now(timezone.utc), can_resume=True, pending_task="18点读报告",
                thinker=lambda messages: "现在是否该读报告？",
                pick=lambda question, options, state: seen.append((options, state)) or choice)
            self.assertEqual(decision, choice)
            self.assertEqual(thought, "现在是否该读报告？")
        self.assertEqual(seen[0][1]["pending_task"], "18点读报告")
        self.assertIn("START_TASK", seen[1][0])
        options_without_auto_mcp = []
        heartbeat_decision(None, datetime.now(timezone.utc), can_resume=False,
                           pending_task="MCP 页面操作", thinker=lambda messages: "先等待",
                           pick=lambda question, options, state: options_without_auto_mcp.append(options) or "REST")
        self.assertEqual(list(options_without_auto_mcp[0]), ["REST"])

    def test_heartbeat_task_starts_normal_goal_only_after_jev_choice(self):
        requests = iter(["/heartbeat task 读报告", "/heartbeat once", "/heartbeat once", "/exit"])
        decisions = iter(["REST", "START_TASK"])
        calls = []

        def fake_turn(args, goal, trace):
            calls.append(goal)
            args.outcome["status"] = "completed"
            return "done", [], "deep"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="deep", mcp_config=None,
                                  chat=True, goal_max_steps=2)
        with (patch("builtins.input", side_effect=lambda prompt: next(requests)),
              patch("agent.run_turn", side_effect=fake_turn),
              patch("agent.heartbeat_decision", side_effect=lambda *args, **kwargs: ("想法", next(decisions))),
              patch("agent.save_trace")):
            chat(args)
        self.assertEqual(calls, ["读报告"])

    def test_long_observation_keeps_end_and_original(self):
        original = "start" + "x" * 2000 + "end"
        events = [{"tool": "read_text", "result": original}]
        summary = recent_observations(events)
        self.assertTrue(summary[0]["result"].startswith("start"))
        self.assertTrue(summary[0]["result"].endswith("end"))
        self.assertEqual(events[0]["result"], original)

    def test_live_trace_shows_full_thought_tool_schema_arguments_and_result_once(self):
        output = io.StringIO()
        trace = LiveTrace()
        result_text = "BEGIN" + "x" * 2000 + "END"
        tools = {"inspect": {"description": "Read the item", "schema": {
            "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            run("inspect it", Path(directory), "deep", tools=tools, trace=trace,
                planner=lambda goal, events: "read the item",
                pick=lambda question, criteria, state: "inspect",
                fill=lambda goal, advice, tool, events: {"name": "note.txt"},
                act=lambda root, tool, arguments: result_text,
                verify=lambda goal, events: True,
                answer=lambda goal, events: "done")
            show_trace(trace)
        displayed = output.getvalue()
        self.assertIn("本地模型参谋： read the item", displayed)
        self.assertIn("Jev 选择： inspect", displayed)
        self.assertIn('"required": [', displayed)
        self.assertIn('"name": "note.txt"', displayed)
        self.assertEqual(displayed.count(result_text), 1)

    def test_chat_heartbeat_is_off_by_default(self):
        ticked = threading.Event()

        def user_input(prompt):
            self.assertFalse(ticked.wait(0.05))
            return "/exit"

        def fake_heartbeat(*args, **kwargs):
            ticked.set()
            return "没有待办，休息", "REST"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="deep", mcp_config=None,
                                  chat=True, goal_max_steps=2, heartbeat_interval=0.01)
        with (patch("builtins.input", side_effect=user_input),
              patch("agent.heartbeat_decision", side_effect=fake_heartbeat),
              patch("agent.save_trace")):
            chat(args)
        self.assertFalse(ticked.is_set())

    def test_timed_heartbeat_can_resume_goal(self):
        requests = iter(["/goal 读报告", "/heartbeat on"])
        heartbeat_seen = threading.Event()
        calls = []

        def user_input(prompt):
            try:
                return next(requests)
            except StopIteration:
                self.assertTrue(heartbeat_seen.wait(2))
                return "/exit"

        def fake_turn(args, goal, trace):
            calls.append(list(args.initial_events))
            args.outcome["status"] = "budget" if len(calls) == 1 else "completed"
            return "pause" if len(calls) == 1 else "done", args.initial_events + [
                {"tool": "list_files", "args": {}, "result": "report.txt"}], "deep"

        def fake_heartbeat(goal, now, *, can_resume, pending_task):
            heartbeat_seen.set()
            self.assertEqual(goal["status"], "budget")
            return "继续", "RESUME_GOAL"

        args = argparse.Namespace(goal=None, root=Path.cwd(), tier="deep", mcp_config=None,
                                  chat=True, goal_max_steps=1, heartbeat_interval=0.01)
        with (patch("builtins.input", side_effect=user_input),
              patch("agent.run_turn", side_effect=fake_turn),
              patch("agent.heartbeat_decision", side_effect=fake_heartbeat),
              patch("agent.save_trace")):
            chat(args)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0]["result"], "report.txt")

    def test_direct_skips_advisor(self):
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "列出文件", Path(directory), "direct",
                pick=lambda question, criteria, state: "list_files" if not state["events"] else "DONE",
                planner=lambda goal, events: self.fail("advisor should not run"),
                fill=lambda goal, advice, tool, events: {},
                answer=lambda goal, events: "完成",
            )
            self.assertEqual(result, "完成")
            self.assertEqual(len(events), 1)

    def test_guided_asks_advisor_when_stuck(self):
        advised = []
        choices = iter(["list_files", "ASK_ADVISOR", "DONE"])
        with tempfile.TemporaryDirectory() as directory:
            run(
                "检查文件", Path(directory), "guided",
                pick=lambda question, criteria, state: next(choices),
                planner=lambda goal, events: advised.append(len(events)) or "继续",
                fill=lambda goal, advice, tool, events: {},
                answer=lambda goal, events: "完成",
            )
        self.assertEqual(advised, [0, 1])

    def test_repeated_action_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            result, events = run(
                "列出文件", Path(directory), "direct",
                pick=lambda question, criteria, state: "list_files",
                planner=lambda goal, events: "继续",
                fill=lambda goal, advice, tool, events: {},
            )
            self.assertIn("死循环", result)
            self.assertEqual(len(events), 1)

    def test_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                execute(Path(directory).resolve(), "read_text", {"path": "../secret.txt"})


if __name__ == "__main__":
    unittest.main()
