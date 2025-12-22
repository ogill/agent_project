from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class RoleSpec:
    """
    RoleSpec describes *how* a role is fulfilled.
    In 8.2.2 this is just an agent instance.
    Later it can hold prompt overrides, model choice, tool allowlists, etc.
    """
    name: str
    agent: Any


class RoleRegistry:
    def __init__(self, roles: Mapping[str, RoleSpec]) -> None:
        self._roles: Dict[str, RoleSpec] = dict(roles)

    def get_agent(self, role_name: str) -> Any:
        if role_name not in self._roles:
            raise KeyError(f"Unknown role: {role_name!r}. Known roles: {sorted(self._roles.keys())}")
        return self._roles[role_name].agent