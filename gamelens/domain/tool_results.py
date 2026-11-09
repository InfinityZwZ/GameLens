"""Stable tool result and failure contracts shared across process boundaries."""

from __future__ import annotations

from typing import Any
import re

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolFailure(BaseModel):
    """Machine-readable failure metadata; ``message`` is safe, bounded context."""

    model_config = ConfigDict(extra="forbid")

    error_type: str = Field(min_length=1, max_length=120)
    error_code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False
    side_effects_started: bool = False
    status_code: int | None = Field(default=None, ge=100, le=599)


class InspectDataPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    time_standard: dict[str, Any]


class ReviewQueryPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: dict[str, Any]
    returned_count: int = Field(ge=0)
    reviews: list[dict[str, Any]]
    stats: dict[str, Any]


class EvidenceSearchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: dict[str, Any]
    returned_count: int = Field(ge=0)
    hits: list[dict[str, Any]]
    reranking: dict[str, Any]


class VerificationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    verified_files: int = Field(ge=0)
    scope: str = Field(min_length=1)


CRITICAL_TOOL_PAYLOADS: dict[str, type[BaseModel]] = {
    "inspect_data": InspectDataPayload,
    "query_reviews": ReviewQueryPayload,
    "search_game_evidence": EvidenceSearchPayload,
    "verify_investigation": VerificationPayload,
}


def classify_tool_exception(error: BaseException) -> ToolFailure:
    """Classify conservatively: unknown, local and permission errors never retry."""

    status = None
    retryable = False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        retryable = status in {408, 429, 500, 502, 503, 504}
        code = f"http_{status}"
    elif isinstance(error, (httpx.TimeoutException, TimeoutError)):
        status, retryable, code = 408, True, "request_timeout"
    elif isinstance(error, (httpx.TransportError, ConnectionError)):
        retryable, code = True, "network_read_failed"
    elif isinstance(error, PermissionError):
        code = "permission_denied"
    elif isinstance(error, ValidationError):
        code = "invalid_arguments"
    elif isinstance(error, FileNotFoundError):
        code = "source_missing"
    elif isinstance(error, ValueError):
        code = "invalid_tool_operation"
    else:
        code = "unknown_tool_error"
    message = " ".join(str(error).split())[:1000] or type(error).__name__
    message = re.sub(r'\bsk-[A-Za-z0-9_.-]+', '[REDACTED]', message)
    message = re.sub(r'(?i)bearer\s+\S+', 'Bearer [REDACTED]', message)
    message = re.sub(r'(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)[^\s,;]+',
                     r'\1[REDACTED]', message)
    return ToolFailure(
        error_type=type(error).__name__,
        error_code=code,
        message=message,
        retryable=retryable,
        side_effects_started=bool(getattr(error, "side_effects_started", False)),
        status_code=status,
    )

