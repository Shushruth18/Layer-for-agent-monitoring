"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Download, Loader2, Play, ShieldAlert, ShieldCheck, ShieldX } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, API_URL, type Recording, type Run, type Violation } from "@/lib/api";
import { cn } from "@/lib/utils";

type Mode = "connecting" | "live" | "replay";

const POLL_MS = 2000;
const REPLAY_STEP_MS = 900;

function usd(n: number) {
  return `$${(n || 0).toFixed(4)}`;
}

function time(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return iso;
  }
}

function ArticleBadge({ v }: { v: Violation }) {
  const block = v.decision === "block";
  return (
    <Badge
      className={cn(
        "h-6 gap-1 rounded-md px-2 text-xs font-semibold text-white",
        block ? "bg-red-600" : "bg-amber-500",
      )}
    >
      {block ? <ShieldX className="size-3.5" /> : <ShieldAlert className="size-3.5" />}
      {v.framework} {v.article}
    </Badge>
  );
}

function Stat({ label, value, tone, sub }: { label: string; value: string | number; tone?: "red" | "amber"; sub?: string }) {
  return (
    <Card size="sm" className="gap-1">
      <CardContent>
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</div>
        <div
          className={cn(
            "mt-1 text-3xl font-semibold tabular-nums",
            tone === "red" && "text-red-600",
            tone === "amber" && "text-amber-600",
          )}
        >
          {value}
        </div>
        {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
      </CardContent>
    </Card>
  );
}

export default function Page() {
  const [mode, setMode] = useState<Mode>("connecting");
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [freshIds, setFreshIds] = useState<Set<number>>(new Set());
  const recording = useRef<Recording | null>(null);
  const replayTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const knownIds = useRef<Set<number>>(new Set());
  const followLatest = useRef(true);

  // ---------------------------------------------------------------- live mode
  const poll = useCallback(async () => {
    try {
      const rs = await api.runs();
      setRuns(rs);
      const target = followLatest.current ? rs[0]?.run_id ?? null : selected;
      if (target !== selected) setSelected(target);
      if (target) {
        const vs = await api.violations(target);
        const fresh = vs.filter((v) => !knownIds.current.has(v.id)).map((v) => v.id);
        vs.forEach((v) => knownIds.current.add(v.id));
        if (fresh.length) setFreshIds(new Set(fresh));
        setViolations(vs);
      } else {
        setViolations([]);
      }
    } catch {
      /* transient: keep last state, try again next tick */
    }
  }, [selected]);

  // ---------------------------------------------------------------- replay mode
  const startReplay = useCallback(() => {
    const rec = recording.current;
    if (!rec) return;
    if (replayTimer.current) clearInterval(replayTimer.current);
    const total = rec.violations.length;
    let i = 0;
    const base: Run = { ...rec.run, status: "running", checked: 0, flagged: 0, blocked: 0, agent_cost: 0, judge_cost: 0 };
    setRuns([base]);
    setSelected(rec.run.run_id);
    setViolations([]);
    replayTimer.current = setInterval(() => {
      i += 1;
      const shown = rec.violations.slice(0, i);
      const frac = i / Math.max(total, 1);
      setViolations(shown);
      setFreshIds(new Set([rec.violations[i - 1]?.id]));
      setRuns([
        {
          ...rec.run,
          status: i >= total ? "ended" : "running",
          checked: Math.round(rec.run.checked * frac),
          flagged: shown.filter((v) => v.decision === "flag").length,
          blocked: shown.filter((v) => v.decision === "block").length,
          agent_cost: rec.run.agent_cost * frac,
          judge_cost: rec.run.judge_cost * frac,
        },
      ]);
      if (i >= total && replayTimer.current) clearInterval(replayTimer.current);
    }, REPLAY_STEP_MS);
    if (total === 0) setRuns([rec.run]);
  }, []);

  // ---------------------------------------------------------------- boot: live or recorded?
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await api.health(); // 3 s timeout
        if (!cancelled) setMode("live");
      } catch {
        try {
          const r = await fetch("/demo-run.json", { cache: "no-store" });
          recording.current = (await r.json()) as Recording;
        } catch {
          recording.current = null;
        }
        if (!cancelled) {
          setMode("replay");
          startReplay();
        }
      }
    })();
    return () => {
      cancelled = true;
      if (replayTimer.current) clearInterval(replayTimer.current);
    };
  }, [startReplay]);

  useEffect(() => {
    if (mode !== "live") return;
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [mode, poll]);

  useEffect(() => {
    if (!freshIds.size) return;
    const t = setTimeout(() => setFreshIds(new Set()), 2500);
    return () => clearTimeout(t);
  }, [freshIds]);

  // ---------------------------------------------------------------- actions
  async function runDemo() {
    setNotice(null);
    if (mode === "replay") {
      startReplay();
      return;
    }
    setStarting(true);
    try {
      const res = await api.startDemo();
      if (res.conflict) setNotice("A demo run is already in progress, following it.");
      followLatest.current = true;
      if (res.run_id) setSelected(res.run_id);
      setViolations([]);
      setTimeout(poll, 300);
    } catch {
      setNotice("Could not start the demo. Is the engine running?");
    } finally {
      setStarting(false);
    }
  }

  function selectRun(id: string) {
    followLatest.current = id === runs[0]?.run_id;
    setSelected(id);
    knownIds.current = new Set();
    setViolations([]);
  }

  const run = useMemo(() => runs.find((r) => r.run_id === selected) ?? null, [runs, selected]);
  const feed = useMemo(() => [...violations].reverse(), [violations]);
  const reportHref = mode === "replay" ? "/demo-report.pdf" : run ? api.reportUrl(run.run_id) : undefined;
  const overhead = run && run.agent_cost > 0 ? Math.round((run.judge_cost / run.agent_cost) * 100) : null;

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      {/* ------------------------------------------------ header */}
      <header className="flex flex-col gap-4 border-b pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-7 text-foreground" />
            <h1 className="text-2xl font-semibold tracking-tight">Agent Monitor</h1>
            {mode === "live" && (
              <Badge variant="outline" className="gap-1.5">
                <span className="size-2 animate-pulse rounded-full bg-emerald-500" /> Live
              </Badge>
            )}
            {mode === "replay" && (
              <Badge variant="outline" className="gap-1.5 border-amber-300 bg-amber-50 text-amber-800">
                <span className="size-2 rounded-full bg-amber-500" /> Recorded demo
              </Badge>
            )}
            {mode === "connecting" && <Badge variant="outline">Connecting…</Badge>}
          </div>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            A compliance layer inside your AI agent. It checks every action live against GDPR, the EU AI Act and
            company policy, blocking clearly illegal actions and flagging borderline ones, each with the exact article.
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button size="lg" onClick={runDemo} disabled={starting || mode === "connecting"}>
            {starting ? <Loader2 className="animate-spin" /> : <Play />}
            {mode === "replay" ? "Replay demo" : "Run live demo"}
          </Button>
          <Button
            size="lg"
            variant="outline"
            disabled={!reportHref}
            render={<a href={reportHref} target="_blank" rel="noreferrer" download />}
            nativeButton={false}
          >
            <Download /> Download report
          </Button>
        </div>
      </header>

      {notice && <div className="mt-4 rounded-lg border bg-muted/50 px-3 py-2 text-sm">{notice}</div>}
      {mode === "replay" && (
        <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          The live engine at <code className="font-mono text-xs">{API_URL}</code> did not answer, so this is a
          recording of a real run.
        </div>
      )}

      {/* ------------------------------------------------ counters */}
      <section className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Steps checked" value={run?.checked ?? 0} sub={run ? (run.status === "running" ? "running…" : "run ended") : "no run yet"} />
        <Stat label="Flagged" value={run?.flagged ?? 0} tone="amber" sub="borderline, allowed" />
        <Stat label="Blocked" value={run?.blocked ?? 0} tone="red" sub="stopped before execution" />
        <Stat
          label="LLM cost via Otari"
          value={usd((run?.agent_cost ?? 0) + (run?.judge_cost ?? 0))}
          sub={run ? `agent ${usd(run.agent_cost)} · compliance ${usd(run.judge_cost)}${overhead !== null ? ` (${overhead}%)` : ""}` : "agent vs compliance"}
        />
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-[260px_1fr]">
        {/* ------------------------------------------------ runs */}
        <Card size="sm" className="h-fit">
          <CardHeader>
            <CardTitle className="text-sm">Runs</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-1">
            {runs.length === 0 && <p className="text-sm text-muted-foreground">No runs yet. Click “Run live demo”.</p>}
            {runs.map((r) => (
              <button
                key={r.run_id}
                onClick={() => selectRun(r.run_id)}
                className={cn(
                  "flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted",
                  r.run_id === selected && "bg-muted",
                )}
              >
                <span className="min-w-0">
                  <span className="block truncate font-mono text-xs">{r.run_id}</span>
                  <span className="text-xs text-muted-foreground">
                    {time(r.started_at)} · {r.status === "running" ? "running" : "ended"}
                  </span>
                </span>
                <span className="flex shrink-0 gap-1 text-xs tabular-nums">
                  <span className="rounded bg-red-100 px-1.5 text-red-700">{r.blocked}</span>
                  <span className="rounded bg-amber-100 px-1.5 text-amber-800">{r.flagged}</span>
                </span>
              </button>
            ))}
          </CardContent>
        </Card>

        {/* ------------------------------------------------ live feed */}
        <Card size="sm">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="size-4" /> Violations
              {run?.status === "running" && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
            </CardTitle>
            <span className="text-xs text-muted-foreground">Only violations are stored, redacted. Clean steps are only counted.</span>
          </CardHeader>
          <CardContent>
            {feed.length === 0 ? (
              <div className="rounded-lg border border-dashed py-12 text-center text-sm text-muted-foreground">
                {run ? (run.status === "running" ? "Watching the agent…" : "No violations in this run.") : "Waiting for a run."}
              </div>
            ) : (
              <ul className="flex flex-col divide-y">
                {feed.map((v) => (
                  <li
                    key={v.id}
                    className={cn(
                      "flex flex-col gap-1.5 py-3 transition-colors duration-1000",
                      freshIds.has(v.id) && "animate-in fade-in slide-in-from-top-2 rounded-md bg-muted/60 px-2",
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <ArticleBadge v={v} />
                      <span className={cn("text-xs font-semibold uppercase", v.decision === "block" ? "text-red-700" : "text-amber-700")}>
                        {v.decision === "block" ? "Blocked" : "Flagged"}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        step {v.step} · {v.tool ?? v.type}
                        {v.destination ? ` → ${v.destination}` : ""}
                      </span>
                      <span className="ml-auto text-xs text-muted-foreground">
                        {v.checked_by === "judge" ? "judge LLM" : "rule"} · {time(v.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm">{v.reason}</p>
                    <p className="break-all font-mono text-xs text-muted-foreground">{v.redacted_summary}</p>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>

      <footer className="mt-10 text-center text-xs text-muted-foreground">
        All LLM calls are routed through Mozilla.ai&apos;s Otari gateway · PII redacted with Microsoft Presidio
      </footer>
    </main>
  );
}
