from __future__ import annotations

import json
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


class MCPToolError(ValueError):
    pass


def load_servers(path: Path) -> dict:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    servers = configuration.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError("Expected an mcpServers object")
    for name, settings in servers.items():
        if not isinstance(name, str) or not isinstance(settings, dict):
            raise ValueError("Invalid MCP server entry")
        if ("command" in settings) == ("url" in settings):
            raise ValueError(f"{name}: specify exactly one of command or url")
        if "command" in settings and (not isinstance(settings["command"], str)
                                      or not isinstance(settings.get("args", []), list)
                                      or not all(isinstance(arg, str) for arg in settings.get("args", []))
                                      or not isinstance(settings.get("env", {}), dict)):
            raise ValueError(f"{name}: invalid stdio configuration")
        if "url" in settings and (not isinstance(settings["url"], str)
                                  or not settings["url"].startswith(("http://", "https://"))):
            raise ValueError(f"{name}: invalid HTTP URL")
        allowed = settings.get("allowed_tools")
        if allowed is not None and (not isinstance(allowed, list)
                                    or not all(isinstance(tool, str) for tool in allowed)
                                    or len(set(allowed)) != len(allowed)):
            raise ValueError(f"{name}: allowed_tools must contain unique tool names")
    return servers


class MCPBridge:
    def __init__(self, servers: dict):
        self.servers = servers
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.catalog: dict[str, dict] = {}
        self.targets: dict[str, tuple[str, str]] = {}

    async def __aenter__(self) -> MCPBridge:
        try:
            for name, settings in self.servers.items():
                allowed = settings.get("allowed_tools")
                discovered = set()
                if "command" in settings:
                    parameters = StdioServerParameters(
                        command=settings["command"], args=settings.get("args", []),
                        env=settings.get("env"), cwd=settings.get("cwd"),
                    )
                    read_stream, write_stream = await self.stack.enter_async_context(stdio_client(parameters))
                else:
                    read_stream, write_stream, _ = await self.stack.enter_async_context(
                        streamable_http_client(settings["url"])
                    )
                session = await self.stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                self.sessions[name] = session
                cursor = None
                seen_cursors = set()
                while True:
                    page = await session.list_tools(cursor=cursor)
                    for tool in page.tools:
                        discovered.add(tool.name)
                        if allowed is not None and tool.name not in allowed:
                            continue
                        schema = tool.inputSchema
                        Draft202012Validator.check_schema(schema)
                        if schema.get("type") != "object":
                            raise ValueError(f"{name}/{tool.name}: inputSchema must be an object")
                        identifier = f"mcp_{len(self.targets)}"
                        self.targets[identifier] = (name, tool.name)
                        properties = schema.get("properties", {})
                        parameters = ", ".join(
                            f"{key}: {value.get('type', 'value')}"
                            f"{' (required)' if key in schema.get('required', []) else ''}"
                            f" {value.get('description', '')}"
                            for key, value in properties.items()
                        )
                        self.catalog[identifier] = {
                            "description": f"{name}/{tool.name}: {tool.description or tool.title or tool.name}; "
                                           f"parameters: {parameters[:800]}",
                            "tool_name": f"{name}/{tool.name}",
                            "purpose": tool.description or tool.title or tool.name,
                            "schema": schema,
                        }
                    cursor = page.nextCursor
                    if not cursor:
                        break
                    if cursor in seen_cursors:
                        raise ValueError(f"{name}: repeated tools/list cursor")
                    seen_cursors.add(cursor)
                if allowed is not None and set(allowed) - discovered:
                    raise ValueError(f"{name}: unknown allowed_tools: {sorted(set(allowed) - discovered)}")
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *_exc: object) -> None:
        await self.stack.aclose()

    async def call(self, identifier: str, arguments: dict) -> str:
        server, tool = self.targets[identifier]
        result = await self.sessions[server].call_tool(
            tool, arguments, read_timeout_seconds=timedelta(seconds=30),
        )
        if result.isError:
            raise MCPToolError(f"{server}/{tool}: {result.content}")
        output = {"content": [item.text[:8000] if item.type == "text" else
                              f"({item.type} result: not readable by the text model)"
                              for item in result.content]}
        if result.structuredContent is not None:
            output["structuredContent"] = result.structuredContent
        return json.dumps(output, ensure_ascii=False, default=str)[:12000]
