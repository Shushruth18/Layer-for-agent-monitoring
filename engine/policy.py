"""Policy engine: Presidio-based rules from rules.yaml, then a judge LLM via Otari.

check(event) -> (Verdict, judge_usage)
Never logs or returns raw input: redacted_summary is Presidio-anonymised text.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from contracts.schemas import Event, Verdict

log = logging.getLogger("agent_monitor.policy")

RULES_PATH = Path(__file__).with_name("rules.yaml")
DECISION_RANK = {"allow": 0, "flag": 1, "block": 2}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

# Tests switch this off to stay deterministic and offline.
JUDGE_ENABLED = os.getenv("MONITOR_JUDGE", "on").lower() not in ("off", "0", "false")


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_rules()


# ---------------------------------------------------------------- Presidio
class PII:
    """Lazily-built Presidio analyzer + anonymizer with Art. 9 deny-list recognizers."""

    _lock = threading.Lock()
    _analyzer = None
    _anonymizer = None

    @classmethod
    def _build(cls) -> None:
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()  # spaCy en_core_web_lg
        for entity, terms in CONFIG["special_categories"].items():
            analyzer.registry.add_recognizer(
                PatternRecognizer(supported_entity=entity, deny_list=terms, name=f"{entity}_deny_list")
            )
        cls._analyzer, cls._anonymizer = analyzer, AnonymizerEngine()

    @classmethod
    def analyze(cls, text: str):
        with cls._lock:
            if cls._analyzer is None:
                cls._build()
        if not text.strip():
            return []
        entities = list(CONFIG["personal_data_entities"]) + list(CONFIG["special_categories"])
        results = cls._analyzer.analyze(text=text, language="en", entities=entities)
        return [r for r in results if r.score >= CONFIG.get("min_score", 0.4)]

    @classmethod
    def anonymize(cls, text: str, results) -> str:
        if not results:
            return text
        return cls._anonymizer.anonymize(text=text, analyzer_results=results).text


def flatten(value: Any, prefix: str = "") -> str:
    """Turn a nested input dict into 'key: value' lines for Presidio."""
    if isinstance(value, dict):
        return "\n".join(flatten(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}") for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(flatten(v, prefix) for v in value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":  # tool args sometimes arrive as JSON strings
            try:
                return flatten(json.loads(stripped), prefix)
            except ValueError:
                pass
    return f"{prefix}: {value}" if prefix else str(value)


@dataclass
class Analysis:
    text: str
    redacted: str
    entities: set[str] = field(default_factory=set)

    @property
    def has_personal_data(self) -> bool:
        pd = set(CONFIG["personal_data_entities"]) | set(CONFIG["special_categories"])
        return bool(self.entities & pd)


def analyse(event: Event) -> Analysis:
    text = flatten(event.input)
    results = PII.analyze(text)
    return Analysis(text=text, redacted=PII.anonymize(text, results), entities={r.entity_type for r in results})


def summarise(event: Event, a: Analysis, limit: int = 400) -> str:
    head = f"{event.type}" + (f" {event.tool}" if event.tool else "")
    if event.destination:
        head += f" -> {event.destination}"
    body = " | ".join(line.strip() for line in a.redacted.splitlines() if line.strip())
    out = f"{head}: {body}" if body else head
    return out if len(out) <= limit else out[: limit - 1] + "…"


# ---------------------------------------------------------------- rules
def _field(event: Event, name: str) -> Any:
    return event.input.get(name)


def _destination(event: Event) -> str | None:
    if event.destination:
        return event.destination
    for key in ("destination", "region"):
        if isinstance(event.input.get(key), str):
            return event.input[key]
    return CONFIG.get("tool_destinations", {}).get(event.tool or "")


def _rule_fires(rule: dict[str, Any], event: Event, a: Analysis) -> str | None:
    """Return a short detail string if the rule fires, else None."""
    if event.type not in rule.get("applies_to", ["tool_call"]):
        return None
    if rule.get("tools") and event.tool not in rule["tools"]:
        return None
    kind = rule["kind"]

    if kind == "special_category":
        hit = sorted(a.entities & set(rule["entities"]))
        return f"detected {', '.join(hit)}" if hit else None

    if kind == "transfer":
        dest = _destination(event)
        eu = {d.lower() for d in CONFIG["eu_destinations"]}
        if dest and dest.lower() not in eu and a.has_personal_data:
            return f"personal data to non-EU destination '{dest}'"
        return None

    if kind == "automated_decision":
        value = str(_field(event, rule["field"]) or "").strip().lower()
        reviewed = _field(event, rule.get("human_review_field", "human_review"))
        if value in rule["negative_values"] and not reviewed:
            return f"decision '{value}' without human review"
        return None

    if kind == "data_minimisation":
        extra = sorted(k for k in event.input if k not in set(rule["allowed_fields"]))
        return f"unneeded fields: {', '.join(extra)}" if extra else None

    if kind == "ai_disclosure":
        body = str(_field(event, rule["field"]) or "")
        if body and not re.search(rule["disclosure_pattern"], body, re.IGNORECASE):
            return "no AI disclosure in message"
        return None

    if kind == "blocked_recipient_domain":
        to = str(_field(event, rule["field"]) or "").lower()
        bad = [d for d in rule["domains"] if to.endswith("@" + d) or to.endswith("." + d)]
        return f"recipient domain {bad[0]}" if bad else None

    if kind == "max_discount":
        text = a.text.lower()
        if not re.search(r"discount|off\b|rabatt|reduction|% off", text):
            return None
        pcts = [float(p) for p in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text)]
        over = [p for p in pcts if p > rule["max_percent"]]
        return f"{max(over):g}% discount offered (max {rule['max_percent']}%)" if over else None

    log.warning("unknown rule kind %s", kind)
    return None


def _verdict_from_rule(rule: dict[str, Any], detail: str, summary: str) -> Verdict:
    return Verdict(
        decision=rule["decision"],
        severity=rule["severity"],
        framework=rule["framework"],
        article=rule["article"],
        reason=f"{rule['explanation'].strip()} ({detail})",
        redacted_summary=summary,
        checked_by="rules",
    )


def _rank(v: Verdict) -> tuple[int, int]:
    return DECISION_RANK[v.decision], SEVERITY_RANK[v.severity]


def run_rules(event: Event, a: Analysis, summary: str) -> list[Verdict]:
    hits = []
    for rule in list(CONFIG.get("rules", [])) + list(CONFIG.get("company_policy", [])):
        detail = _rule_fires(rule, event, a)
        if detail:
            hits.append(_verdict_from_rule(rule, detail, summary))
    return sorted(hits, key=_rank, reverse=True)


# ---------------------------------------------------------------- judge
JUDGE_PROMPT = """You are a strict EU compliance judge for an AI sales agent.
Decide whether this single agent action violates GDPR, the EU AI Act or company policy.
Personal data has been redacted to placeholders like <PERSON>; judge the action itself.
Context: B2B sales outreach to business contacts at companies has a legitimate-interest basis (GDPR Art. 6(1)(f)),
and other checks already cover special-category data, transfers, automated rejection, minimisation and AI disclosure.
Only flag or block if you see a concrete, specific problem in THIS action.

Action type: {type}
Tool: {tool}
Destination: {destination}
Redacted input:
{redacted}

Answer with ONLY a JSON object, no prose:
{{"decision": "allow" | "flag" | "block",
  "severity": "low" | "medium" | "high",
  "framework": "GDPR" | "EU AI Act" | "Company policy" | null,
  "article": "e.g. Art. 6" or null,
  "reason": "one plain-English sentence"}}
Use "block" only for clearly illegal actions, "flag" for borderline ones, otherwise "allow"."""


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("judge returned no JSON")
    return json.loads(m.group(0))


def ask_judge(event: Event, a: Analysis, summary: str) -> tuple[Verdict, dict[str, float]]:
    from engine.otari import extract_usage, get_llm_with_fallback

    llm = get_llm_with_fallback("judge")
    msg = llm.invoke(
        JUDGE_PROMPT.format(
            type=event.type, tool=event.tool or "-", destination=_destination(event) or "-", redacted=a.redacted[:3000]
        )
    )
    usage = extract_usage((msg.response_metadata or {}).get("token_usage"))
    data = _parse_json(str(msg.content))
    decision = data.get("decision") if data.get("decision") in DECISION_RANK else "allow"
    framework = data.get("framework") if data.get("framework") in ("GDPR", "EU AI Act", "Company policy") else None
    v = Verdict(
        decision=decision,
        severity=data.get("severity") if data.get("severity") in SEVERITY_RANK else "medium",
        framework=framework if decision != "allow" else None,
        article=(data.get("article") or None) if decision != "allow" else None,
        reason=str(data.get("reason") or "")[:500],
        redacted_summary=summary,
        checked_by="judge",
    )
    return v, usage


# ---------------------------------------------------------------- cache + entry point
class _LRU(OrderedDict):
    def __init__(self, size: int = 2048) -> None:
        super().__init__()
        self.size = size
        self.lock = threading.Lock()

    def get_(self, key: str) -> Verdict | None:
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        return None

    def put(self, key: str, value: Verdict) -> None:
        with self.lock:
            self[key] = value
            self.move_to_end(key)
            while len(self) > self.size:
                self.popitem(last=False)


CACHE = _LRU()
NO_USAGE = {"tokens": 0, "cost_usd": 0.0}


def _cache_key(event: Event) -> str:
    raw = json.dumps([event.type, event.tool, event.destination, event.input], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def check(event: Event) -> tuple[Verdict, dict[str, float]]:
    """Evaluate one event. Returns the verdict and the judge's Otari usage for this call."""
    if event.type == "run_end":
        return Verdict(decision="allow", reason="run ended", checked_by="none"), NO_USAGE

    key = _cache_key(event)
    cached = CACHE.get_(key)
    if cached is not None:
        return cached, NO_USAGE

    a = analyse(event)
    summary = summarise(event, a)
    hits = run_rules(event, a, summary)
    usage = NO_USAGE

    if hits:
        verdict = hits[0]
        if len(hits) > 1:
            also = ", ".join(f"{h.framework} {h.article}" for h in hits[1:])
            verdict = verdict.model_copy(update={"reason": f"{verdict.reason} Also: {also}."})
    elif (
        JUDGE_ENABLED
        and a.has_personal_data
        and event.type in CONFIG.get("judge", {}).get("applies_to", ["tool_call"])
    ):
        try:
            verdict, usage = ask_judge(event, a, summary)
        except Exception as exc:  # Otari down, bad JSON, budget exceeded...
            log.warning("judge unavailable, falling back to rules: %s", type(exc).__name__)
            verdict = Verdict(decision="allow", reason="No rule fired; judge unavailable (rules-only fallback).",
                              redacted_summary=summary, checked_by="rules")
            CACHE.put(key, verdict)  # don't hammer a failing gateway with the same event
            return verdict, usage
    else:
        verdict = Verdict(decision="allow", reason="No rule fired.", redacted_summary=summary, checked_by="rules")

    CACHE.put(key, verdict)
    return verdict, usage
