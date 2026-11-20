export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Run = {
  run_id: string;
  started_at: string;
  ended_at: string | null;
  status: "running" | "ended";
  checked: number;
  flagged: number;
  blocked: number;
  agent_tokens: number;
  agent_cost: number;
  judge_tokens: number;
  judge_cost: number;
  judge_calls: number;
};

export type Violation = {
  id: number;
  run_id: string;
  step: number;
  timestamp: string;
  source: "sdk" | "gateway";
  type: string;
  tool: string | null;
  destination: string | null;
  decision: "flag" | "block";
  severity: "low" | "medium" | "high";
  framework: string | null;
  article: string | null;
  reason: string;
  redacted_summary: string;
  checked_by: string;
};

export type Recording = { run: Run; violations: Violation[] };

async function get<T>(path: string, timeoutMs = 5000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_URL}${path}`, { signal: ctrl.signal, cache: "no-store" });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return (await r.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  health: () => get<{ status: string; demo_running: boolean }>("/health", 3000),
  runs: () => get<Run[]>("/runs?limit=25"),
  violations: (runId: string) => get<Violation[]>(`/runs/${encodeURIComponent(runId)}/violations`),
  reportUrl: (runId: string) => `${API_URL}/runs/${encodeURIComponent(runId)}/report`,
  startDemo: async (): Promise<{ run_id?: string; conflict?: boolean }> => {
    const r = await fetch(`${API_URL}/demo/run`, { method: "POST" });
    if (r.status === 409) {
      const body = await r.json().catch(() => ({}));
      return { run_id: body?.detail?.run_id, conflict: true };
    }
    if (!r.ok) throw new Error(`demo start failed: ${r.status}`);
    return r.json();
  },
};
