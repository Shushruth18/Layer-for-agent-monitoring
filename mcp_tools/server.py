"""Fake sales tools for the demo agent (FastMCP, streamable HTTP on port 8100, path /mcp).

No real network calls: everything is in memory and printed to the terminal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

LEADS = json.loads((Path(__file__).resolve().parent.parent / "mocks" / "leads.json").read_text(encoding="utf-8"))
PORT = int(os.getenv("TOOLS_PORT", "8100"))

mcp = FastMCP("agent-monitor-demo-tools", host="127.0.0.1", port=PORT)

PROFILES: dict[str, dict] = {}
CRM: list[dict] = []
DECISIONS: dict[str, str] = {}
OUTBOX: list[dict] = []


def _log(msg: str) -> None:
    print(f"[tools] {msg}", flush=True)


@mcp.tool()
def search_leads(city: str, industry: str) -> str:
    """Search the lead database. Returns full lead records (id, name, title, company, email, phone,
    city, employees, fit_score, notes) as JSON."""
    hits = [l for l in LEADS if l["city"].lower() == city.lower()] if industry.lower() in ("fintech", "finance") else []
    _log(f"search_leads(city={city!r}, industry={industry!r}) -> {len(hits)} leads")
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
def save_profile(
    lead_id: str,
    name: str,
    email: str,
    title: Optional[str] = None,
    company: Optional[str] = None,
    phone: Optional[str] = None,
    fit_score: Optional[int] = None,
    notes: Optional[str] = None,
) -> str:
    """Save a lead's profile to our internal lead store."""
    PROFILES[lead_id] = {k: v for k, v in locals().items() if v is not None}
    _log(f"save_profile({lead_id}) stored {len(PROFILES[lead_id])} fields")
    return f"Profile {lead_id} saved."


@mcp.tool()
def send_to_crm(leads: list[dict], destination: str = "us-east") -> str:
    """Push lead records to the CRM. `leads` is a list of lead objects (lead_id, name, email, company).
    The CRM is hosted in region `destination` (default 'us-east')."""
    CRM.extend(leads)
    _log(f"send_to_crm({len(leads)} leads) -> region {destination}")
    return f"{len(leads)} leads pushed to CRM in region {destination}."


@mcp.tool()
def decide_on_lead(lead_id: str, decision: str, reason: str, human_review: bool = False) -> str:
    """Record a qualification decision for a lead: decision is 'approve' or 'reject'."""
    DECISIONS[lead_id] = decision
    _log(f"decide_on_lead({lead_id}) -> {decision} (human_review={human_review})")
    return f"Lead {lead_id} marked as {decision}."


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a lead."""
    OUTBOX.append({"to": to, "subject": subject})
    _log(f"send_email(to=<redacted>, subject={subject!r})")
    return f"Email sent ({len(OUTBOX)} in outbox)."


if __name__ == "__main__":
    _log(f"serving on http://127.0.0.1:{PORT}/mcp")
    mcp.run(transport="streamable-http")
