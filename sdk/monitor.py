"""AgentMonitor: a LangChain callback handler that checks every agent step live.

    from sdk import AgentMonitor
    agent.invoke(inputs, config={"callbacks": [AgentMonitor(run_id)]})

Every LLM call and tool call is sent to the engine's /check as an Event. If the engine
blocks a tool call (and block=True), ComplianceBlocked is raised before the tool runs;
the agent sees it as the tool's result, naming the article. When the run finishes,
/runs/{id}/end is called with the agent's Otari token/cost usage.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

log = logging.getLogger("agent_monitor.sdk")


class _HideIntendedBlocks(logging.Filter):
    """LangChain logs every callback exception; a ComplianceBlocked is intended, not an error."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "ComplianceBlocked" not in record.getMessage()


logging.getLogger("langchain_core.callbacks.manager").addFilter(_HideIntendedBlocks())

MAX_FIELD_CHARS = 4000


class ComplianceBlocked(Exception):
    """Raised when Agent Monitor blocks an action. The message names the article."""

    def __init__(self, verdict: dict[str, Any], tool: str | None = None) -> None:
        self.verdict = verdict
        self.tool = tool
        where = f"{verdict.get('framework')} {verdict.get('article')}".strip()
        super().__init__(
            f"BLOCKED by Agent Monitor ({where}): {verdict.get('reason', '')} "
            f"The action '{tool}' was NOT executed. Do not retry it with the same data; "
            f"continue with the remaining work."
        )


def handle_blocked(e: Exception) -> str:
    """Use as ToolNode(handle_tool_errors=handle_blocked) so blocks become tool messages."""
    if isinstance(e, ComplianceBlocked):
        return str(e)
    return f"Error: {e!r}"


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + "…"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value[:50]]
    return value


def _usage(response: LLMResult) -> tuple[int, float]:
    u = (response.llm_output or {}).get("token_usage") or {}
    tokens = int(u.get("total_tokens") or 0)
    try:
        cost = float(u.get("cost_usd") or 0.0)  # Otari returns cost inline
    except (TypeError, ValueError):
        cost = 0.0
    if not tokens:
        try:
            tokens = int((response.generations[0][0].message.usage_metadata or {}).get("total_tokens", 0))
        except (AttributeError, IndexError):
            pass
    return tokens, cost


class AgentMonitor(BaseCallbackHandler):
    """LangChain callback that checks every LLM and tool call against the policy engine."""

    raise_error = True  # let ComplianceBlocked propagate out of on_tool_start
    run_inline = True   # run in the caller's flow so the block happens before the tool

    def __init__(
        self,
        run_id: str,
        engine_url: str | None = None,
        block: bool = True,
        fail_closed: bool = False,
        timeout: float = 30.0,
        on_verdict: Optional[callable] = None,
    ) -> None:
        self.monitor_run_id = run_id
        self.engine_url = (engine_url or os.getenv("AGENT_MONITOR_URL", "http://localhost:8000")).rstrip("/")
        self.block = block
        self.fail_closed = fail_closed
        self.on_verdict = on_verdict
        self._client = httpx.Client(timeout=timeout)
        self._lock = threading.Lock()
        self._step = 0
        self._ended = False
        self.agent_tokens = 0
        self.agent_cost = 0.0
        self.verdicts: list[dict[str, Any]] = []

    # ------------------------------------------------------------ plumbing
    def _next_step(self) -> int:
        with self._lock:
            self._step += 1
            return self._step

    def _send(self, type_: str, input_: dict[str, Any], tool: str | None = None,
              destination: str | None = None) -> dict[str, Any] | None:
        event = {
            "run_id": self.monitor_run_id,
            "step": self._next_step(),
            "source": "sdk",
            "type": type_,
            "tool": tool,
            "input": _clip(input_),
            "destination": destination,
        }
        try:
            r = self._client.post(f"{self.engine_url}/check", json=event)
            r.raise_for_status()
            verdict = r.json()
        except httpx.HTTPError as exc:
            log.warning("Agent Monitor engine unreachable (%s)", type(exc).__name__)
            if self.fail_closed and type_ == "tool_call":
                verdict = {"decision": "block", "framework": "Agent Monitor", "article": "fail-closed",
                           "reason": "The compliance engine is unreachable."}
            else:
                return None
        verdict["step"], verdict["tool"], verdict["type"] = event["step"], tool, type_
        self.verdicts.append(verdict)
        if self.on_verdict:
            try:
                self.on_verdict(verdict)
            except Exception:  # never let a printer break the agent
                log.exception("on_verdict callback failed")
        return verdict

    # ------------------------------------------------------------ LLM calls
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        last = messages[0][-1] if messages and messages[0] else None
        content = getattr(last, "content", "") if last is not None else ""
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        self._send("llm_call", {"role": getattr(last, "type", "unknown"), "prompt": content,
                                "messages": len(messages[0]) if messages else 0})

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._send("llm_call", {"prompt": prompts[-1] if prompts else ""})

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        tokens, cost = _usage(response)
        with self._lock:
            self.agent_tokens += tokens
            self.agent_cost += cost

    # ------------------------------------------------------------ tools
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, inputs: dict[str, Any] | None = None,
                      **kwargs: Any) -> None:
        tool = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        args = inputs if isinstance(inputs, dict) else None
        if args is None:
            try:
                args = json.loads(input_str)
            except (TypeError, ValueError):
                args = {"input": input_str}
            if not isinstance(args, dict):
                args = {"input": args}
        destination = args.get("destination") or args.get("region")
        verdict = self._send("tool_call", args, tool=tool, destination=destination)
        if verdict and verdict.get("decision") == "block" and self.block:
            raise ComplianceBlocked(verdict, tool)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        content = getattr(output, "content", output)
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        self._send("tool_result", {"output": content}, tool=kwargs.get("name"))

    # ------------------------------------------------------------ end of run
    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        if parent_run_id is None:
            self.end()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None,
                       **kwargs: Any) -> None:
        if parent_run_id is None:
            self.end()

    def end(self) -> None:
        """Tell the engine the run is over (idempotent)."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
        try:
            self._client.post(f"{self.engine_url}/runs/{self.monitor_run_id}/end",
                              json={"agent_tokens": self.agent_tokens, "agent_cost_usd": self.agent_cost})
        except httpx.HTTPError as exc:
            log.warning("could not end run on Agent Monitor engine (%s)", type(exc).__name__)

    # ------------------------------------------------------------ summary
    @property
    def counts(self) -> dict[str, int]:
        c = {"checked": len(self.verdicts), "flag": 0, "block": 0}
        for v in self.verdicts:
            if v.get("decision") in ("flag", "block"):
                c[v["decision"]] += 1
        return c
