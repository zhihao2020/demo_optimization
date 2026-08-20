"""Infer CAES parameterization from checkpoint actor keys."""

from __future__ import annotations

from typing import Any, Mapping


def infer_parameterized_caes(
    actor_state: Mapping[str, Any] | None = None,
    *,
    explicit: bool | None = None,
) -> bool:
    """Resolve parameterized_caes from checkpoint field or actor key names.

    Args:
        actor_state: ``state_dict`` of the actor (optional if ``explicit`` set).
        explicit: Value from checkpoint ``parameterized_caes`` when present.

    Returns:
        True if mode+mag heads; False if projected continuous caes head.
    """
    if explicit is not None:
        return bool(explicit)
    if not actor_state:
        return True
    keys = set(actor_state.keys())
    if any(k.startswith("caes_mode_head") for k in keys):
        return True
    if any(k.startswith("caes_mean") or k.startswith("caes_log_std") for k in keys):
        return False
    if any(k == "caes_head.weight" or k.startswith("caes_head.") for k in keys) and not any(
        k.startswith("caes_mode_head") for k in keys
    ):
        return False
    return True
