"""All LLM traffic goes through Mozilla.ai's Otari gateway (OpenAI-compatible).

get_llm(role)       -> ChatOpenAI for "agent" or "judge", each with its own Otari virtual key.
get_llm_with_fallback(role, tools=None) -> same, with a client-side fallback model if configured.
otari_usage(role)   -> {"tokens", "cost_usd", "requests", "source"} for that role.

Config comes only from the environment (.env): OPENAI_BASE_URL, AGENT_LLM_KEY, JUDGE_LLM_KEY,
AGENT_MODEL, JUDGE_MODEL and optional AGENT_FALLBACK_MODEL / JUDGE_FALLBACK_MODEL.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

Role = Literal["agent", "judge"]
ROLES: tuple[Role, ...] = ("agent", "judge")


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def _role_key(role: Role) -> str:
    return _env(f"{role.upper()}_LLM_KEY")


def _base_url() -> str:
    return _env("OPENAI_BASE_URL").rstrip("/")


# ---------------------------------------------------------------- usage tracking
class _UsageTracker(BaseCallbackHandler):
    """Accumulates tokens and Otari's inline `usage.cost_usd` per role, in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.totals: dict[str, dict[str, float]] = {
            r: {"tokens": 0, "cost_usd": 0.0, "requests": 0} for r in ROLES
        }

    def add(self, role: str, usage: dict[str, Any]) -> dict[str, float]:
        delta = extract_usage(usage)
        with self._lock:
            t = self.totals.setdefault(role, {"tokens": 0, "cost_usd": 0.0, "requests": 0})
            t["tokens"] += delta["tokens"]
            t["cost_usd"] += delta["cost_usd"]
            t["requests"] += 1
        return delta


def extract_usage(usage: dict[str, Any] | None) -> dict[str, float]:
    """Normalise an OpenAI/Otari usage dict to {"tokens", "cost_usd"}."""
    usage = usage or {}
    tokens = int(usage.get("total_tokens") or 0)
    try:
        cost = float(usage.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return {"tokens": tokens, "cost_usd": cost}


def usage_from_result(response: LLMResult) -> dict[str, Any]:
    """Pull the raw usage dict (including Otari's cost_usd) out of a LangChain LLMResult."""
    out = (response.llm_output or {}).get("token_usage") or {}
    if out:
        return dict(out)
    # Fallback: usage_metadata on the message (no cost there)
    try:
        meta = response.generations[0][0].message.usage_metadata or {}
        return {"total_tokens": meta.get("total_tokens", 0)}
    except (AttributeError, IndexError):
        return {}


class _RoleCallback(BaseCallbackHandler):
    def __init__(self, role: Role) -> None:
        self.role = role

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        TRACKER.add(self.role, usage_from_result(response))


TRACKER = _UsageTracker()


# ---------------------------------------------------------------- LLM factories
def _make(role: Role, model: str, **kwargs: Any) -> ChatOpenAI:
    params: dict[str, Any] = {
        "model": model,
        "base_url": _base_url(),
        "api_key": _role_key(role),
        "timeout": 60,
        "max_retries": 2,
        "callbacks": [_RoleCallback(role)],
        "temperature": 0,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


def get_llm(role: Role, **kwargs: Any) -> ChatOpenAI:
    """ChatOpenAI routed through Otari with the virtual key for `role`."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return _make(role, _env(f"{role.upper()}_MODEL"), **kwargs)


def get_llm_with_fallback(role: Role, tools: list | None = None, **kwargs: Any):
    """Primary model, falling back to <ROLE>_FALLBACK_MODEL if the call fails.

    Hosted Otari routing policies are managed in the dashboard, so fallback is done
    client-side here. Tools are bound before wrapping so the result still supports tools.
    """
    primary: Any = get_llm(role, **kwargs)
    fb_model = os.getenv(f"{role.upper()}_FALLBACK_MODEL", "").strip()
    if tools is not None:
        primary = primary.bind_tools(tools)
    if not fb_model:
        return primary
    fallback: Any = _make(role, fb_model, **kwargs)
    if tools is not None:
        fallback = fallback.bind_tools(tools)
    return primary.with_fallbacks([fallback])


# ---------------------------------------------------------------- usage reporting
def otari_usage(role: Role) -> dict[str, Any]:
    """Tokens and cost for `role`.

    Tries Otari's per-key usage API first (needs the key's `usage:read` scope);
    otherwise returns what this process tracked from Otari's inline `usage.cost_usd`.
    """
    try:
        r = httpx.get(
            f"{_base_url()}/key/usage/summary",
            headers={"Authorization": f"Bearer {_role_key(role)}"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            totals = data.get("totals", data)
            return {
                "tokens": int(totals.get("total_tokens") or totals.get("tokens") or 0),
                "cost_usd": float(totals.get("cost_usd") or totals.get("total_cost_usd") or 0.0),
                "requests": int(totals.get("requests") or totals.get("request_count") or 0),
                "source": "otari-api",
            }
    except (httpx.HTTPError, RuntimeError, ValueError):
        pass
    t = TRACKER.totals.get(role, {"tokens": 0, "cost_usd": 0.0, "requests": 0})
    return {**t, "source": "tracked"}


if __name__ == "__main__":  # smoke test: one call per key
    for r in ROLES:
        reply = get_llm(r).invoke("Reply with the single word: ready")
        print(r, "->", reply.content.strip(), otari_usage(r))
