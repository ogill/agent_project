from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List

app = FastAPI(title="MCP Math Server", version="0.1")


# ---- MCP models ----

class ToolDef(BaseModel):
    name: str
    description: str = ""
    input_schema: Dict[str, Any]


class ToolsResponse(BaseModel):
    tools: List[ToolDef]


class InvokeRequest(BaseModel):
    tool: str = Field(..., description="Server-local tool name")
    args: Dict[str, Any] = Field(default_factory=dict)


class InvokeOk(BaseModel):
    ok: bool = True
    result: Dict[str, Any]


class InvokeErr(BaseModel):
    ok: bool = False
    error: Dict[str, Any]


# ---- Tool definitions (contracts locked) ----

ADD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}

SUB_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}


@app.get("/mcp/tools", response_model=ToolsResponse)
def list_tools() -> ToolsResponse:
    return ToolsResponse(
        tools=[
            ToolDef(
                name="add_numbers",
                description="Add two numbers and return the sum.",
                input_schema=ADD_SCHEMA,
            ),
            ToolDef(
                name="subtract_numbers",
                description="Subtract b from a and return the difference.",
                input_schema=SUB_SCHEMA,
            ),
        ]
    )

@app.get("/mcp")
def mcp_root():
    return {"ok": True, "service": "mcp"}

@app.post("/mcp/invoke", response_model=InvokeOk | InvokeErr)
def invoke(req: InvokeRequest):
    tool = req.tool
    args = req.args or {}

    try:
        if tool == "add_numbers":
            a = args.get("a")
            b = args.get("b")
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                return InvokeErr(
                    ok=False,
                    error={
                        "type": "VALIDATION_ERROR",
                        "message": "Fields 'a' and 'b' must be numbers",
                        "details": {"args": args},
                    },
                )
            return InvokeOk(ok=True, result={"result": float(a) + float(b)})

        if tool == "subtract_numbers":
            a = args.get("a")
            b = args.get("b")
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                return InvokeErr(
                    ok=False,
                    error={
                        "type": "VALIDATION_ERROR",
                        "message": "Fields 'a' and 'b' must be numbers",
                        "details": {"args": args},
                    },
                )
            return InvokeOk(ok=True, result={"result": float(a) - float(b)})

        return InvokeErr(
            ok=False,
            error={
                "type": "TOOL_NOT_FOUND",
                "message": f"Unknown tool: {tool}",
                "details": {},
            },
        )

    except Exception as e:
        return InvokeErr(
            ok=False,
            error={
                "type": "TOOL_ERROR",
                "message": f"Unexpected error: {e!r}",
                "details": {},
            },
        )