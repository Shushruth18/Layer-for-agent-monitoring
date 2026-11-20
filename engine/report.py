"""Render a run's compliance report as PDF (Jinja2 + WeasyPrint)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from engine.policy import CONFIG

TEMPLATES = Path(__file__).with_name("templates")
_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))


def _explanations() -> dict[str, str]:
    rules = list(CONFIG.get("rules", [])) + list(CONFIG.get("company_policy", []))
    return {r["article"]: " ".join(r["explanation"].split()) for r in rules}


def render_html(run: dict[str, Any], violations: list[dict[str, Any]]) -> str:
    expl = _explanations()
    rows = [{**v, "explanation": expl.get(v.get("article") or "", v.get("reason") or "")} for v in violations]
    agent_cost, judge_cost = float(run.get("agent_cost") or 0), float(run.get("judge_cost") or 0)
    overhead = f"{judge_cost / agent_cost * 100:.0f}%" if agent_cost > 0 else "n/a"
    return _env.get_template("report.html").render(
        run=run,
        violations=rows,
        overhead=overhead,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_pdf(run: dict[str, Any], violations: list[dict[str, Any]]) -> bytes:
    from weasyprint import HTML  # imported lazily: needs WEASYPRINT_DLL_DIRECTORIES from .env on Windows

    return HTML(string=render_html(run, violations), base_url=str(TEMPLATES)).write_pdf()
