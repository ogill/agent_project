# mcp/provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client import McpHttpServer, invoke, list_tools
from mcp.registry import McpRegistry, build_registry

JsonObj = Dict[str, Any]


@dataclass
class McpProvider:
    servers: List[McpHttpServer]
    registries: Dict[str, McpRegistry]  # alias -> registry

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
                continue
            servers.append(McpHttpServer(alias=alias, endpoint=endpoint, timeout_ms=timeout_ms))

        provider = cls(servers=servers, registries={})
        provider._discover_all()
        return provider

    def _discover_all(self) -> None:
        for server in self.servers:
            # Default to an empty registry for this alias
            self.registries[server.alias] = McpRegistry(tools_dict={}, routes={})

            try:
                resp = list_tools(server)
            except Exception as e:
                # Keep the empty registry, but preserve a hint for debugging
                try:
                    self.registries[server.alias].routes["_discover_error"] = {
                        "alias": server.alias,
                        "endpoint": getattr(server, "endpoint", None),
                        "error": repr(e),
                    }
                except Exception:
                    pass
                continue

            tools = resp.get("tools", [])

            if resp.get("ok") is True and isinstance(tools, list) and tools:
                self.registries[server.alias] = build_registry(
                    server_alias=server.alias,
                    tools=tools,
                    executor_fn=self.execute,
                )
            else:
                # Keep empty registry; optionally store the response for debugging
                try:
                    self.registries[server.alias].routes["_discover_response"] = {
                        "alias": server.alias,
                        "endpoint": getattr(server, "endpoint", None),
                        "ok": resp.get("ok"),
                        "error": resp.get("error"),
                        "tools_type": type(tools).__name__,
                    }
                except Exception:
                    pass
                
    def get_tools_dict(self) -> Dict[str, JsonObj]:
        out: Dict[str, JsonObj] = {}
        for reg in self.registries.values():
            out.update(reg.tools_dict)
        return out

    def execute(self, exposed_tool_name: str, args: JsonObj) -> Any:
        # Find matching registry
        for alias, reg in self.registries.items():
            if reg.has_tool(exposed_tool_name):
                route = reg.resolve(exposed_tool_name)
                server = self._server_by_alias(alias)
                if server is None:
                    return _soft_failure("MCP_TRANSPORT_ERROR", "No MCP server for alias", {"alias": alias})

                resp = invoke(server, route.server_tool_name, args)
                if resp.get("ok") is True:
                    # Return tool payload directly (e.g. {"result": 3.0})
                    return resp.get("result")
                return _soft_failure("MCP_TOOL_ERROR", "MCP tool call failed", {"error": resp.get("error", {})})

        return _soft_failure("MCP_TOOL_NOT_FOUND", "Unknown MCP tool", {"tool": exposed_tool_name})

    def _server_by_alias(self, alias: str) -> Optional[McpHttpServer]:
        for s in self.servers:
            if s.alias == alias:
                return s
        return None


def _soft_failure(err_type: str, message: str, details: JsonObj) -> JsonObj:
    return {
        "ok": False,
        "status": "failed",
        "reason": message,
        "retryable": False,
        "details": {"type": err_type, **details},
    }