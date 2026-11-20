import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contracts.schemas import Event, Verdict
from engine import policy
from engine.main import app

EVENTS = json.loads((Path(__file__).resolve().parent.parent / "mocks" / "events.json").read_text(encoding="utf-8"))

# step -> (decision, article)
EXPECTED = {
    1: ("allow", None),
    2: ("allow", None),
    3: ("allow", None),
    4: ("block", "Art. 9"),
    5: ("block", "Art. 9"),
    6: ("block", "Art. 44"),
    7: ("allow", None),
    8: ("flag", "Art. 22 + AI Act Art. 14"),
    9: ("allow", None),
    10: ("flag", "Art. 5(1)(c)"),
    11: ("flag", "Art. 50"),
    12: ("allow", None),
    13: ("block", "Policy CP-1: no emails to competitors"),
    14: ("block", "Policy CP-2: discounts max 20%"),
    15: ("allow", None),
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: f"step{e['step']}-{e.get('tool') or e['type']}")
def test_check_mock_events(client, event):
    r = client.post("/check", json=event)
    assert r.status_code == 200
    v = Verdict(**r.json())
    decision, article = EXPECTED[event["step"]]
    assert v.decision == decision
    assert v.article == article


def test_only_violations_stored_and_redacted(client):
    for e in EVENTS:
        client.post("/check", json=e)
    run = client.get("/runs/mock-run").json()
    assert run["status"] == "ended"
    vs = client.get("/runs/mock-run/violations").json()
    assert {v["decision"] for v in vs} <= {"block", "flag"}
    stored = json.dumps(vs)
    for raw in ("mehmet.yilmaz@bezahlt.io", "Fatima", "anna.becker@kontoflow.de", "cancer", "Muslim"):
        assert raw not in stored, f"raw value {raw!r} leaked into storage"
    assert "<PERSON>" in stored and "<HEALTH_DATA>" in stored


def test_frameworks(client):
    verdicts = {e["step"]: client.post("/check", json=e).json() for e in EVENTS}
    assert verdicts[4]["framework"] == "GDPR"
    assert verdicts[11]["framework"] == "EU AI Act"
    assert verdicts[13]["framework"] == "Company policy"


def test_default_tool_destination_is_checked():
    e = Event(run_id="t", step=1, type="tool_call", tool="send_to_crm",
              input={"name": "Anna Becker", "email": "anna@example.de"})
    v, _ = policy.check(e)
    assert v.decision == "block" and v.article == "Art. 44"


def test_cache_hit_skips_work(monkeypatch):
    e = Event(run_id="t", step=1, type="tool_call", tool="decide_on_lead", input={"lead_id": "X", "decision": "reject"})
    first, _ = policy.check(e)
    monkeypatch.setattr(policy, "analyse", lambda *_: pytest.fail("should have been cached"))
    second, _ = policy.check(e)
    assert first == second


def test_judge_used_when_no_rule_but_personal_data(monkeypatch):
    monkeypatch.setattr(policy, "JUDGE_ENABLED", True)
    policy.CACHE.clear()
    called = {}

    def fake_judge(event, a, summary):
        called["redacted"] = a.redacted
        return Verdict(decision="flag", severity="medium", framework="GDPR", article="Art. 6",
                       reason="no lawful basis", redacted_summary=summary, checked_by="judge"), {"tokens": 10, "cost_usd": 0.0001}

    monkeypatch.setattr(policy, "ask_judge", fake_judge)
    e = Event(run_id="t", step=1, type="tool_call", tool="send_to_crm", destination="eu-central",
              input={"name": "Anna Becker", "email": "anna.becker@kontoflow.de"})
    v, usage = policy.check(e)
    assert v.checked_by == "judge" and v.article == "Art. 6" and usage["tokens"] == 10
    assert "anna.becker@kontoflow.de" not in called["redacted"]  # judge only sees redacted text


def test_falls_back_to_rules_when_otari_fails(monkeypatch):
    monkeypatch.setattr(policy, "JUDGE_ENABLED", True)
    policy.CACHE.clear()

    def boom(*_):
        raise ConnectionError("otari down")

    monkeypatch.setattr(policy, "ask_judge", boom)
    e = Event(run_id="t", step=2, type="tool_call", tool="send_to_crm", destination="eu-central",
              input={"name": "Jonas Richter", "email": "jonas@zinsbaer.de"})
    v, _ = policy.check(e)
    assert v.decision == "allow" and v.checked_by == "rules"
    # rules still block when Otari is down
    e2 = Event(run_id="t", step=3, type="tool_call", tool="send_to_crm", destination="us-east",
               input={"name": "Jonas Richter", "email": "jonas@zinsbaer.de"})
    assert policy.check(e2)[0].decision == "block"


def test_report_pdf(client):
    for e in EVENTS:
        client.post("/check", json=e)
    r = client.get("/runs/mock-run/report")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_unknown_run_404(client):
    assert client.get("/runs/nope").status_code == 404
