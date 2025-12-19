# mcp/client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonObj = Dict[str, Any]


@dataclass(frozen=True)
class McpHttpServer:
    alias: str
    endpoint: str  # e.g. "http://localhost:8080/mcp"
    timeout_ms: int = 5000

    def tools_url(self) -> str:
        return _join(self.endpoint, "tools")

    def invoke_url(self) -> str:
        return _join(self.endpoint, "invoke")


def list_tools(server: McpHttpServer) -> JsonObj:
    """
    Calls GET {endpoint}/tools
    Expected response (server-defined but we will implement to match):
      { "tools": [ { "name": str, "description": str, "input_schema": {...}, ... }, ... ] }
    Returns:
      { "ok": True, "tools": [...] } on success
      { "ok": False, "error": {...} } on failure
    """
    try:
        status, data = _http_json(
            method="GET",
            url=server.tools_url(),
            body=None,
            timeout_ms=server.timeout_ms,
        )
        if status != 200:
            return _transport_error(
                message=f"Unexpected HTTP status {status} from list_tools",
                details={"alias": server.alias, "url": server.tools_url(), "status": status, "body": data},
            )

        tools = data.get("tools")
        if not isinstance(tools, list):
            return _transport_error(
                message="Invalid list_tools response: missing or non-list 'tools'",
                details={"alias": server.alias, "url": server.tools_url(), "body": data},
            )

        return {"ok": True, "tools": tools}

    except Exception as e:
        return _transport_error(
            message=str(e),
            details={"alias": server.alias, "url": server.tools_url()},
        )


def invoke(server: McpHttpServer, tool_name: str, args: JsonObj) -> JsonObj:
    """
    Calls POST {endpoint}/invoke
    Body:
      { "tool": "<tool_name>", "args": {...} }
    Expected response:
      { "ok": true, "result": {...} }
      OR
      { "ok": false, "error": {...} }
    Returns that response (validated / normalized) or transport error.
    """
    if not isinstance(args, dict):
        return _transport_error(
            message="invoke args must be a JSON object (dict)",
            details={"alias": server.alias, "tool": tool_name, "args_type": type(args).__name__},
        )

    try:
        status, data = _http_json(
            method="POST",
            url=server.invoke_url(),
            body={"tool": tool_name, "args": args},
            timeout_ms=server.timeout_ms,
        )
        if status != 200:
            return _transport_error(
                message=f"Unexpected HTTP status {status} from invoke",
                details={"alias": server.alias, "url": server.invoke_url(), "status": status, "body": data},
            )

        ok = data.get("ok")
        if ok is True:
            # Must have a result
            if "result" not in data:
                return _transport_error(
                    message="Invalid invoke response: ok=true but missing 'result'",
                    details={"alias": server.alias, "tool": tool_name, "body": data},
                )
            return {"ok": True, "result": data["result"]}

        if ok is False:
            # Must have an error object
            err = data.get("error")
            if not isinstance(err, dict):
                return _transport_error(
                    message="Invalid invoke response: ok=false but missing/invalid 'error'",
                    details={"alias": server.alias, "tool": tool_name, "body": data},
                )
            return {"ok": False, "error": err}

        # If server didn't include ok, treat as protocol error
        return _transport_error(
            message="Invalid invoke response: missing boolean 'ok'",
            details={"alias": server.alias, "tool": tool_name, "body": data},
        )

    except Exception as e:
        return _transport_error(
            message=str(e),
            details={"alias": server.alias, "url": server.invoke_url(), "tool": tool_name},
        )


def _http_json(method: str, url: str, body: Optional[JsonObj], timeout_ms: int) -> Tuple[int, JsonObj]:
    timeout_s = max(timeout_ms, 1) / 1000.0

    headers = {"Accept": "application/json"}
    data_bytes: Optional[bytes] = None

    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, method=method, headers=headers, data=data_bytes)

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read().decode("utf-8") if resp.readable() else ""
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                # We require JSON object responses
                return status, {"_raw": parsed}
            return status, parsed

    except HTTPError as e:
        # HTTPError is also a file-like response; try read JSON body if any
        try:
            raw = e.read().decode("utf-8") if hasattr(e, "read") else ""
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                parsed = {"_raw": parsed}
        except Exception:
            parsed = {"message": str(e)}
        return int(getattr(e, "code", 500)), parsed

    except URLError as e:
        raise RuntimeError(f"MCP transport error: {e.reason}") from e

    except json.JSONDecodeError as e:
        raise RuntimeError(f"MCP protocol error: invalid JSON response: {e}") from e


def _transport_error(message: str, details: JsonObj) -> JsonObj:
    return {
        "ok": False,
        "error": {
            "type": "MCP_TRANSPORT_ERROR",
            "message": message,
            "details": details,
        },
    }


def _join(base: str, suffix: str) -> str:
    base = base.rstrip("/")
    suffix = suffix.lstrip("/")
    return f"{base}/{suffix}"