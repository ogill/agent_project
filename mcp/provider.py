# mcp/provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client import McpHttpServer, invoke, list_tools
from mcp.registry import McpRegistry, build_registry

JsonObj = Dict[str, Any]


@dataclass
class McpProvider:
    """
    Thin façade for the rest of the app.

    - Initializes MCP servers (discovery)
    - Exposes tool specs for planner
    - Executes MCP tools and normalizes results
    """
    servers: List[McpHttpServer]
    registries: Dict[str, McpRegistry]  # keyed by alias

    @classmethod
    def from_config(cls, *, enabled: bool, servers_cfg: List[JsonObj]) -> Optional["McpProvider"]:
        if not enabled:
            return None

        servers: List[McpHttpServer] = []
        for s in servers_cfg:
            try:
                alias = str(s["alias"])
                endpoint = str(s["endpoint"])
                timeout_ms = int(s.get("timeout_ms", 5000))
            except Exception:
                # Skip invalid config entries
                continue

            servers.append(McpHttpServer(alias=alias, endpoint=endpoint, timeout_ms=timeout_ms))

        provider = cls(servers=servers, registries={})
        provider._discover_all()
        return provider

    def _discover_all(self) -> None:
        # Discover tools for each server and build registry
        for s in self.servers:
            resp = list_tools(s)
            if resp.get("ok") is True:
                tools = resp.get("tools", [])
                if isinstance(tools, list):
                    self.registries[s.alias] = build_registry(s.alias, tools)
            else:
                # If discovery fails, keep server present but with no tools
                self.registries[s.alias] = McpRegistry(tool_specs=[], routes={})

    def get_tool_specs(self) -> List[JsonObj]:
        specs: List[JsonObj] = []
        for reg in self.registries.values():
            specs.extend(reg.tool_specs)
        return specs

    def has_tool(self, exposed_tool_name: str) -> bool:
        return any(reg.has_tool(exposed_tool_name) for reg in self.registries.values())

    def execute(self, exposed_tool_name: str, args: JsonObj) -> JsonObj:
        """
        Execute MCP tool by exposed name (mcp.<alias>.<tool>) and return a normalized tool result.
        This result format is designed to be compatible with existing Stage 6 tool failure handling.
        """
        # Find registry that contains this tool
        for alias, reg in self.registries.items():
            if reg.has_tool(exposed_tool_name):
                route = reg.resolve(exposed_tool_name)
                server = self._server_by_alias(alias)
                if server is None:
                    return _tool_error(
                        "MCP_TRANSPORT_ERROR",
                        f"No MCP server configured for alias '{alias}'",
                        {"tool": exposed_tool_name, "alias": alias},
                    )

                resp = invoke(server, route.server_tool_name, args)

                # Pass through ok/result or ok/error in a stable envelope
                if resp.get("ok") is True:
                    return {"ok": True, "result": resp.get("result")}
                return {
                    "ok": False,
                    "error": resp.get("error", {"type": "MCP_TOOL_ERROR", "message": "Unknown MCP error", "details": {}}),
                }

        return _tool_error(
            "MCP_TOOL_NOT_FOUND",
            f"Unknown MCP tool: {exposed_tool_name}",
            {"tool": exposed_tool_name},
        )

    def _server_by_alias(self, alias: str) -> Optional[McpHttpServer]:
        for s in self.servers:
            if s.alias == alias:
                return s
        return None


def _tool_error(err_type: str, message: str, details: JsonObj) -> JsonObj:
    return {"ok": False, "error": {"type": err_type, "message": message, "details": details}}
