"""Shared data contracts between the SDK, gateway, engine and dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["sdk", "gateway"]
EventType = Literal["llm_call", "tool_call", "tool_result", "run_end"]
Decision = Literal["allow", "flag", "block"]
Severity = Literal["low", "medium", "high"]
Framework = Literal["GDPR", "EU AI Act", "Company policy"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """One observable step of an agent run."""

    run_id: str
    step: int
    source: Source = "sdk"
    type: EventType
    tool: Optional[str] = None
    input: dict[str, Any] = Field(default_factory=dict)
    destination: Optional[str] = None  # e.g. "eu-central", "us-east"
    timestamp: datetime = Field(default_factory=_now)


class Verdict(BaseModel):
    """The engine's decision about one Event."""

    decision: Decision = "allow"
    severity: Severity = "low"
    framework: Optional[Framework] = None
    article: Optional[str] = None
    reason: str = ""
    redacted_summary: str = ""
    checked_by: Literal["rules", "judge", "none"] = "rules"
