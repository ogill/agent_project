# mcp/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

JsonObj = Dict[str, Any]


@dataclass(frozen=True)
class McpToolRoute:
    server_alias: str
    server_tool_name: str


@dataclass
class McpRegistry:
    """
    Holds:
      - tool_specs: list of tool specs suitable for the planner (same shape as local tools)
      - routes: mapping exposed_tool_name -> McpToolRoute(server_alias, server_tool_name)
    """
    tool_specs: List[JsonObj]
    routes: Dict[str, McpToolRoute]

    def has_tool(self, exposed_name: str) -> bool:
        return exposed_name in self.routes

    def resolve(self, exposed_name: str) -> McpToolRoute:
        return self.routes[exposed_name]


def build_registry(server_alias: str, tools: List[JsonObj]) -> McpRegistry:
    """
    tools: raw tool definitions from MCP server list_tools response:
      { "name": str, "description": str, "input_schema": {...} , ... }

    Returns a registry that exposes each tool as:
      mcp.<server_alias>.<tool_name>
    """
    tool_specs: List[JsonObj] = []
    routes: Dict[str, McpToolRoute] = {}

    for t in tools:
        name = t.get("name")
        if not isinstance(name, str) or not name:
            # Skip malformed entries
            continue

        exposed_name = _exposed_name(server_alias, name)
        desc = t.get("description") if isinstance(t.get("description"), str) else ""
        input_schema = t.get("input_schema") if isinstance(t.get("input_schema"), dict) else {}

        # Planner tool spec format:
        # We keep this minimal and generic because Stage 6 is locked.
        # This mirrors common tool formats: name/description/parameters (JSON schema).
        spec: JsonObj = {
            "name": exposed_name,
            "description": desc,
            "parameters": input_schema,
        }

        tool_specs.append(spec)
        routes[exposed_name] = McpToolRoute(server_alias=server_alias, server_tool_name=name)

    return McpRegistry(tool_specs=tool_specs, routes=routes)


def _exposed_name(server_alias: str, tool_name: str) -> str:
    return f"mcp.{server_alias}.{tool_name}"