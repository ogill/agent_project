# mcp/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field, create_model

JsonObj = Dict[str, Any]


@dataclass(frozen=True)
class McpToolRoute:
    server_alias: str
    server_tool_name: str


@dataclass
class McpRegistry:
    tools_dict: Dict[str, JsonObj]              # exposed_name -> TOOLS entry
    routes: Dict[str, McpToolRoute]             # exposed_name -> route

    def has_tool(self, exposed_name: str) -> bool:
        return exposed_name in self.routes

    def resolve(self, exposed_name: str) -> McpToolRoute:
        return self.routes[exposed_name]


def build_registry(
    *,
    server_alias: str,
    tools: List[JsonObj],
    executor_fn,  # callable(exposed_name: str, args: dict) -> Any
) -> McpRegistry:
    tools_dict: Dict[str, JsonObj] = {}
    routes: Dict[str, McpToolRoute] = {}

    for t in tools:
        name = t.get("name")
        if not isinstance(name, str) or not name:
            continue

        exposed_name = f"mcp.{server_alias}.{name}"
        desc = t.get("description") if isinstance(t.get("description"), str) else ""
        input_schema = t.get("input_schema") if isinstance(t.get("input_schema"), dict) else {}

        args_model = _pydantic_model_from_json_schema(exposed_name, input_schema)

        tools_dict[exposed_name] = {
            "description": desc,
            "args_model": args_model,
            "fn": (lambda _exposed=exposed_name, _model=args_model: (
                lambda **kwargs: executor_fn(_exposed, _model(**kwargs).model_dump())
            ))(),
        }

        routes[exposed_name] = McpToolRoute(server_alias=server_alias, server_tool_name=name)

    return McpRegistry(tools_dict=tools_dict, routes=routes)


def _pydantic_model_from_json_schema(exposed_name: str, schema: JsonObj) -> Type[BaseModel]:
    if schema.get("type") != "object":
        return create_model(_safe_model_name(exposed_name), __base__=BaseModel)

    props = schema.get("properties", {})
    required = set(schema.get("required", [])) if isinstance(schema.get("required", []), list) else set()

    fields: Dict[str, Tuple[Any, Any]] = {}
    if isinstance(props, dict):
        for field_name, field_schema in props.items():
            if not isinstance(field_name, str) or not isinstance(field_schema, dict):
                continue

            py_type = _jsonschema_type_to_py(field_schema.get("type"))
            field_desc = field_schema.get("description")
            field_kwargs = {}
            if isinstance(field_desc, str) and field_desc:
                field_kwargs["description"] = field_desc

            if field_name in required:
                fields[field_name] = (py_type, Field(..., **field_kwargs))
            else:
                fields[field_name] = (Optional[py_type], Field(None, **field_kwargs))

    return create_model(_safe_model_name(exposed_name), **fields)


def _jsonschema_type_to_py(t: Any) -> Any:
    if t == "number":
        return float
    if t == "integer":
        return int
    if t == "string":
        return str
    if t == "boolean":
        return bool
    if t == "object":
        return dict
    if t == "array":
        return list
    return Any


def _safe_model_name(exposed_name: str) -> str:
    return "McpArgs_" + exposed_name.replace(".", "_").replace("-", "_")