# orchestrator/roles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class RoleSpec:
    """
    Stable, explicit role contract.

    Keeping this as a dataclass makes it easy to extend later
    (e.g., role-specific policies, timeouts, models, etc.)
    without changing call sites.
    """
    name: str
    agent: Any


class RoleRegistry:
    """
    Holds role -> RoleSpec mappings.

    Back-compat:
      - RoleRegistry({"generalist": RoleSpec(...), ...})
      - RoleRegistry({"generalist": SomeAgentWithRun(), ...})
      - RoleRegistry({"generalist": RoleLikeWithAgentAttr(), ...})
      - RoleRegistry(roles=...)  (keyword)
    """

    def __init__(self, roles: Optional[Mapping[str, Any]] = None, **kwargs) -> None:
        # Support RoleRegistry(roles=...) and RoleRegistry({...})
        if roles is None and "roles" in kwargs:
            roles = kwargs["roles"]

        roles = dict(roles or {})

        # Normalize to Dict[str, RoleSpec]
        normalized: Dict[str, RoleSpec] = {}
        for role_name, role_obj in roles.items():
            # Case 1: already a RoleSpec
            if isinstance(role_obj, RoleSpec):
                normalized[role_name] = role_obj
                continue

            # Case 2: stored an Agent directly (has .run)
            if hasattr(role_obj, "run"):
                normalized[role_name] = RoleSpec(name=str(role_name), agent=role_obj)
                continue

            # Case 3: stored a wrapper with .agent
            if hasattr(role_obj, "agent"):
                normalized[role_name] = RoleSpec(name=str(role_name), agent=role_obj.agent)
                continue

            raise TypeError(
                f"Role entry for {role_name!r} must be RoleSpec, an Agent (has .run), "
                f"or a Role-like wrapper (has .agent). Got: {type(role_obj).__name__}"
            )

        self.roles: Dict[str, RoleSpec] = normalized

    def get_agent(self, role_name: str) -> Any:
        if role_name not in self.roles:
            raise KeyError(f"Unknown role: {role_name!r}")
        return self.roles[role_name].agent

    def get(self, role_name: str) -> RoleSpec:
        """
        Prefer this in orchestrator code where you may later want role metadata.
        """
        if role_name not in self.roles:
            raise KeyError(f"Unknown role: {role_name!r}")
        return self.roles[role_name]