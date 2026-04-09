"""Shared privacy masking for observability backends (Langfuse, MLflow)."""

from __future__ import annotations

import os
from typing import Any, Callable


def resolve_message_mask_fn() -> Callable[[Any], Any] | None:
    """Return a mask callable when LANGFUSE_MASK_MESSAGES requests redaction.

    Uses the same env var as Langfuse for a single privacy knob across backends.

    Values (case-insensitive):

    - Unset, ``true``, ``1``, ``yes``: **legacy** masking (same as pre–MLflow dual-backend:
      redact strings longer than 50 characters; ``metadata`` key passed through unchanged).
    - ``strict``: redact all non-empty strings and recursively sanitize inside ``metadata``.
    - ``false``, ``0``, ``no``: no masking.
    """
    raw = os.getenv("LANGFUSE_MASK_MESSAGES", "true").strip().lower()
    if raw in ("false", "0", "no"):
        return None
    if raw == "strict":
        return privacy_mask_message_data_strict
    return privacy_mask_message_data


def privacy_mask_message_data(data: Any, **kwargs: Any) -> Any:
    """Legacy mask: long strings and known content fields; ``metadata`` left as-is.

    Preserves historical Langfuse behaviour when masking is enabled (default).
    """
    if isinstance(data, str):
        if len(data) > 50:
            return "[REDACTED FOR PRIVACY]"
        return data
    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for key, value in data.items():
            if key in (
                "usage",
                "usage_details",
                "metadata",
                "model",
                "turn",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "total_tokens",
                "cost_usd",
                "duration_ms",
                "duration_api_ms",
                "num_turns",
                "session_id",
                "tool_id",
                "tool_name",
                "is_error",
                "level",
            ):
                masked[key] = value
            elif key in ("content", "text", "input", "output", "prompt", "completion"):
                if isinstance(value, str) and len(value) > 50:
                    masked[key] = "[REDACTED FOR PRIVACY]"
                else:
                    masked[key] = privacy_mask_message_data(value)
            else:
                masked[key] = privacy_mask_message_data(value)
        return masked
    if isinstance(data, list):
        return [privacy_mask_message_data(item) for item in data]
    return data


def privacy_mask_message_data_strict(data: Any, **kwargs: Any) -> Any:
    """Stricter mask for operators who set LANGFUSE_MASK_MESSAGES=strict.

    Redacts all non-empty strings and recursively processes ``metadata`` (no passthrough).
    """
    if isinstance(data, str):
        if data == "":
            return data
        return "[REDACTED FOR PRIVACY]"
    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for key, value in data.items():
            if key in (
                "usage",
                "usage_details",
                "model",
                "turn",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "total_tokens",
                "cost_usd",
                "duration_ms",
                "duration_api_ms",
                "num_turns",
                "session_id",
                "tool_id",
                "tool_name",
                "is_error",
                "level",
                "role",
            ):
                masked[key] = value
            elif key in ("content", "text", "input", "output", "prompt", "completion"):
                if isinstance(value, str):
                    masked[key] = "" if value == "" else "[REDACTED FOR PRIVACY]"
                else:
                    masked[key] = privacy_mask_message_data_strict(value)
            else:
                masked[key] = privacy_mask_message_data_strict(value)
        return masked
    if isinstance(data, list):
        return [privacy_mask_message_data_strict(item) for item in data]
    return data
