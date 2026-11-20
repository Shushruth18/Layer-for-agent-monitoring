import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from engine.main import app
from sdk import AgentMonitor, ComplianceBlocked

executed = []


@tool
def save_profile(lead_id: str, name: str, notes: str = "") -> str:
    """Save a lead profile."""
    executed.append(lead_id)
    return "saved"


@pytest.fixture
def monitor():
    with TestClient(app) as client:
        m = AgentMonitor("sdk-test", engine_url="http://testserver")
        m._client = client  # TestClient is an httpx.Client
        yield m, client


def test_block_raises_before_tool_runs(monitor):
    m, _ = monitor
    executed.clear()
    with pytest.raises(ComplianceBlocked) as exc:
        save_profile.invoke({"lead_id": "L003", "name": "Mehmet Yilmaz", "notes": "on sick leave after surgery"},
                            config={"callbacks": [m]})
    assert "Art. 9" in str(exc.value)
    assert executed == []


def test_clean_call_passes_and_run_ends(monitor):
    m, client = monitor
    executed.clear()
    assert save_profile.invoke({"lead_id": "L001", "name": "Anna Becker"}, config={"callbacks": [m]}) == "saved"
    assert executed == ["L001"]
    m.end()
    run = client.get("/runs/sdk-test").json()
    assert run["status"] == "ended" and run["checked"] >= 2


def test_monitor_off_mode_does_not_block(monitor):
    m, _ = monitor
    m.block = False
    executed.clear()
    save_profile.invoke({"lead_id": "L005", "name": "Fatima", "notes": "attends mosque"}, config={"callbacks": [m]})
    assert executed == ["L005"]
    assert m.counts["block"] == 1
