"""Agent Monitor policy engine (FastAPI, port 8000).

POST /check                  Event -> Verdict (blocks/flags are stored redacted; clean steps only counted)
GET  /runs                   recent runs with counters
GET  /runs/{id}              one run
GET  /runs/{id}/violations   violations (optional ?after=<id> for incremental polling)
GET  /runs/{id}/report       PDF report
POST /runs/{id}/end          mark run ended (+ agent LLM usage)
POST /demo/run               start the demo agent in the background (409 if one is running)
GET  /usage                  Otari usage for both roles (this process)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from contracts.schemas import Event, Verdict  # noqa: E402
from engine import policy  # noqa: E402
from engine.otari import otari_usage  # noqa: E402
from engine.report import render_pdf  # noqa: E402
from store import db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent_monitor.engine")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    # Warm up Presidio/spaCy so the first live check is fast.
    threading.Thread(target=lambda: policy.PII.analyze("warm up John Doe"), daemon=True).start()
    yield


app = FastAPI(title="Agent Monitor engine", version="0.1.0", lifespan=lifespan)

extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", *extra_origins],
    allow_origin_regex=r"https://([a-z0-9-]+\.)*vercel\.app|https://[a-z0-9-]+\.trycloudflare\.com",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "demo_running": _demo_running()}


@app.post("/check", response_model=Verdict)
def check(event: Event) -> Verdict:
    db.count_step(event.run_id)
    if event.type == "run_end":
        u = event.input.get("agent_usage") or {}
        db.end_run(event.run_id, u.get("tokens"), u.get("cost_usd"))
    verdict, judge_usage = policy.check(event)
    if judge_usage.get("tokens") or judge_usage.get("cost_usd"):
        db.add_usage(event.run_id, "judge", judge_usage["tokens"], judge_usage["cost_usd"])
    if verdict.decision != "allow":
        db.save_violation(event, verdict)
        # Never log raw input — only the decision and article.
        log.info("run=%s step=%s %s %s %s", event.run_id, event.step, verdict.decision.upper(),
                 verdict.framework, verdict.article)
    return verdict


@app.get("/runs")
def runs(limit: int = 50) -> list[dict]:
    return db.get_runs(limit)


def _run_or_404(run_id: str) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, f"run {run_id} not found")
    return run


@app.get("/runs/{run_id}")
def run(run_id: str) -> dict:
    return _run_or_404(run_id)


@app.get("/runs/{run_id}/violations")
def violations(run_id: str, after: int = 0) -> list[dict]:
    _run_or_404(run_id)
    return db.get_violations(run_id, after)


@app.get("/runs/{run_id}/report")
def report(run_id: str) -> Response:
    run = _run_or_404(run_id)
    pdf = render_pdf(run, db.get_violations(run_id))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="agent-monitor-{run_id}.pdf"'})


class EndRun(BaseModel):
    agent_tokens: Optional[int] = None
    agent_cost_usd: Optional[float] = None


@app.post("/runs/{run_id}/end")
def end(run_id: str, body: Optional[EndRun] = None) -> dict:
    body = body or EndRun()
    db.end_run(run_id, body.agent_tokens, body.agent_cost_usd)
    return _run_or_404(run_id)


@app.get("/usage")
def usage() -> dict:
    return {"agent": otari_usage("agent"), "judge": otari_usage("judge")}


# ---------------------------------------------------------------- live demo
_demo: dict = {"proc": None, "run_id": None}
_demo_lock = threading.Lock()


def _demo_running() -> bool:
    p = _demo["proc"]
    return p is not None and p.poll() is None


@app.post("/demo/run")
def demo_run() -> dict:
    with _demo_lock:
        if _demo_running():
            raise HTTPException(409, {"message": "a demo run is already in progress", "run_id": _demo["run_id"]})
        run_id = f"demo-{uuid.uuid4().hex[:8]}"
        log_file = open(ROOT / "store" / "demo-agent.log", "w", encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        _demo["proc"] = subprocess.Popen(
            [sys.executable, "-m", "agent.run", "--run-id", run_id],
            cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        _demo["run_id"] = run_id
        db.start_run(run_id)  # visible on the dashboard immediately
        return {"run_id": run_id, "status": "started"}


@app.get("/demo/status")
def demo_status() -> dict:
    return {"running": _demo_running(), "run_id": _demo["run_id"]}
