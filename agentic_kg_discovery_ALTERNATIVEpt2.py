"""
step7_agentic_kg_writer.py
────────────────────────────
A write-capable extension of step6_agentic_kg_discovery.py.

DIFFERENCE FROM step6
──────────────────────
step6 gives the model query_graph / search_web / fetch_url / finish, and your
Python code decides whether to act on the model's final answer. The model
never touches the graph.

This module adds a FIFTH tool, update_policy, which the model can call
directly to replace an outdated policy in the knowledge graph. The model
decides WHEN to call it. But — and this is the important part — the tool
itself is not a thin wrapper that blindly does whatever the model asks. It
enforces real guardrails in code, not just in the prompt:

  1. SCOPE LOCK — the tool is bound to ONE manufacturer_iri for the entire
     run. The model cannot pass a different IRI and write to some other
     manufacturer's node. There is no "target" parameter on the tool at all.

  2. EVIDENCE REQUIREMENT — update_policy refuses to run unless the model
     has ALREADY called fetch_url at least once in this session AND the
     fetched text is what gets written — not text the model retypes from
     memory. The tool takes a `fetched_text_ref` token (returned by
     fetch_url) rather than freeform text, so the model cannot fabricate
     policy text and have it accepted as if it were real.

  3. DATE GATE — update_policy independently re-runs the same deterministic
     date comparison (extract_policy_date_agent + compare_policy_dates)
     used elsewhere in your codebase. If that comparison disagrees with the
     model ("should_update" is False), the write is refused regardless of
     how confident the model's tool call sounded. The model's confidence is
     advisory; the date math is the actual gate.

  4. SINGLE WRITE — once update_policy succeeds, it is removed from the
     model's available tools for the rest of that session, so a confused
     model cannot loop and overwrite the same node repeatedly in one run.

  5. EVERY WRITE IS LOGGED — before/after text length, the date evidence,
     and the model's own stated reasoning are all written to an audit
     log file, separate from the ontology itself.

WHAT STILL DOESN'T CHANGE
──────────────────────────
The actual graph mutation is still performed by step4a_timestamps.upsert_policy()
— this module does not reimplement that logic, it calls it. The old policy
is still preserved via hasPreviousPolicy, exactly as before. Nothing here
changes how upsert_policy() itself behaves.

USAGE
─────
    import step7_agentic_kg_writer as akw

    result = akw.run_agentic_discovery_and_update(
        g=g,
        company_name="Acurite",
        manufacturer_iri=URIRef(...),
        policy_prop=POLICY_PROP,
        groq_client=groq_client,
        onto_path=str(ONTO_PATH),   # pass None to skip auto-serialize
    )
    # result["wrote_update"], result["reasoning_trace"], result["audit_entry"]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rdflib import Graph, URIRef

import step4a_timestamps as ts
import agentic_policy_update_crawler as policy_crawler
import agentic_kg_discovery_ALTERNATIVE as akg 

logger = logging.getLogger("agentic_kg_writer")

MAX_AGENT_STEPS = 10
AUDIT_LOG_PATH = "agentic_write_audit_log.json"


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT LOGGING — append-only, separate file from the ontology itself
# ═══════════════════════════════════════════════════════════════════════════

def _append_audit_log(entry: dict[str, Any]):
    p = Path(AUDIT_LOG_PATH)
    log = []
    if p.exists():
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Could not read existing audit log, starting fresh")
            log = []
    log.append(entry)
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# THE WRITE TOOL — this is the one new capability vs step6.
# Bound to a single manufacturer_iri/policy_prop pair via closure, so the
# model has no parameter that could redirect the write elsewhere.
# ═══════════════════════════════════════════════════════════════════════════

class GraphWriteSession:
    """
    Holds the mutable state for one discovery+write run: which fetches have
    happened (so update_policy can verify evidence exists), and whether the
    single allowed write has already been used.
    """

    def __init__(self, g: Graph, manufacturer_iri: URIRef, policy_prop: URIRef,
                 company_name: str, groq_client: Any):
        self.g = g
        self.manufacturer_iri = manufacturer_iri
        self.policy_prop = policy_prop
        self.company_name = company_name
        self.groq_client = groq_client
        self.fetched_pages: dict[str, dict[str, Any]] = {}  # ref_token -> fetch result
        self.write_used = False
        self.write_result: Optional[dict[str, Any]] = None

    def record_fetch(self, url: str, fetch_result: dict[str, Any]) -> str:
        """Stores a fetched page under a short reference token the model can
        cite later, instead of having the model retype policy text itself."""
        ref = f"fetch_{len(self.fetched_pages) + 1}"
        self.fetched_pages[ref] = {"url": url, **fetch_result}
        return ref

    def update_policy(self, fetched_text_ref: str, model_confidence: str,
                       model_reasoning: str) -> dict[str, Any]:
        if self.write_used:
            return {"error": "A write has already been performed in this session. "
                              "Only one update_policy call is permitted per run."}

        fetch = self.fetched_pages.get(fetched_text_ref)
        if not fetch:
            return {"error": f"Unknown fetched_text_ref '{fetched_text_ref}'. "
                              "You must call fetch_url first and cite its returned ref."}

        # The preview stored by fetch_url is truncated for context-window reasons.
        # Re-fetch the FULL text here so what gets written to the KG is complete,
        # not the truncated preview the model saw.
        url = fetch["url"]
        try:
            refetched = policy_crawler.fetch_policy_source(url)
            full_text = policy_crawler.extract_policy_text(refetched)
        except Exception as e:
            return {"error": f"Could not re-fetch full text from {url} for the final write: {e}"}

        if len(full_text) < 500:
            return {"error": "Fetched page does not contain enough readable text "
                              "to be confidently treated as a real privacy policy. Refusing to write."}

        # DATE GATE — re-run the deterministic comparison independently of
        # whatever the model believes. This is the actual gate, not the
        # model's stated confidence.
        live_date, llm_used = policy_crawler.extract_policy_date_agent(
            full_text,
            last_modified_header=refetched.get("last_modified_header"),
            llm_client=self.groq_client,
        )
        stored_date = policy_crawler.get_stored_company_date(self.g, self.manufacturer_iri)
        decision = policy_crawler.compare_policy_dates(stored_date, live_date)

        if not decision["should_update"]:
            return {
                "error": "Refused: deterministic date comparison does not support an update.",
                "status": decision["status"],
                "reason": decision["reason"],
                "stored_date": str(stored_date.parsed),
                "live_date": str(live_date.parsed),
            }

        # All checks passed — perform the actual write via the SAME function
        # used everywhere else in the codebase. Old text is archived via
        # hasPreviousPolicy inside upsert_policy(), not deleted.
        update_info = ts.upsert_policy(self.g, self.manufacturer_iri, self.policy_prop, full_text)

        self.write_used = True
        self.write_result = {
            "status": "updated",
            "url": url,
            "live_date": str(live_date.parsed),
            "stored_date_before": str(stored_date.parsed),
            "new_text_length": len(full_text),
            "previous_text_length": len(update_info.get("previous_policy") or ""),
            "model_confidence": model_confidence,
            "model_reasoning": model_reasoning,
            "upsert_info": update_info,
        }
        return self.write_result


# ═══════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS — step6's three read tools + this module's write tool.
# fetch_url is overridden here to also register the fetch in the session
# (so update_policy can verify it later) and return a ref token.
# ═══════════════════════════════════════════════════════════════════════════

def _build_tool_schemas() -> list[dict]:
    schemas = [s for s in akg.TOOL_SCHEMAS if s["function"]["name"] != "finish"]
    schemas.append({
        "type": "function",
        "function": {
            "name": "update_policy",
            "description": (
                "Replace the stored privacy policy for this manufacturer in the knowledge graph "
                "with the text from a page you already fetched. You must cite the fetched_text_ref "
                "returned by a prior fetch_url call — you cannot supply policy text directly. "
                "This will be refused if the underlying date comparison does not actually support "
                "an update, even if you are confident it should. Only one write is allowed per run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fetched_text_ref": {
                        "type": "string",
                        "description": "The ref token (e.g. 'fetch_1') returned by a previous fetch_url call.",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string", "description": "Why you believe this page is the correct, current policy."},
                },
                "required": ["fetched_text_ref", "confidence", "reasoning"],
            },
        },
    })
    schemas.append({
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you are done — either after a successful update_policy call, "
                "or after concluding no update is warranted/possible. This ends the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "1-3 sentence summary of the outcome."},
                },
                "required": ["summary"],
            },
        },
    })
    return schemas


SYSTEM_PROMPT = """You are a compliance maintenance agent with WRITE access to one specific \
manufacturer's record in a knowledge graph. You may replace their stored privacy-policy text if, \
and only if, you find clear evidence their live policy is newer than what is stored.

Tools available: query_graph (read-only), search_web, fetch_url, update_policy, finish.

Process expectations (not a rigid script — use judgment):
  - Check the knowledge graph for the stored last-modified date before searching the web.
  - Search for and fetch the manufacturer's current official policy page. Verify it actually
    looks like a privacy policy (not a cookie notice, blog post, or unrelated page) before
    trusting it.
  - Only call update_policy if you have fetched a page that you believe is genuinely newer.
    update_policy will independently re-check the dates and refuse if the evidence doesn't
    actually support an update — your confidence alone is not sufficient, so don't try to call
    it speculatively.
  - If you cannot find clear evidence of an update, do NOT call update_policy — just call finish
    and explain why.
  - You may only write once per run. Use query_graph and fetch_url as much as you need BEFORE
    committing to a write.

Always end by calling finish()."""


# ═══════════════════════════════════════════════════════════════════════════
# THE LOOP — structurally the same shape as step6's loop, with the
# write tool's execution routed through the session object above.
# ═══════════════════════════════════════════════════════════════════════════

def run_agentic_discovery_and_update(
    g: Graph,
    company_name: str,
    manufacturer_iri: URIRef,
    policy_prop: URIRef,
    groq_client: Any,
    onto_path: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    max_steps: int = MAX_AGENT_STEPS,
) -> dict[str, Any]:
    session = GraphWriteSession(g, manufacturer_iri, policy_prop, company_name, groq_client)
    tool_schemas = _build_tool_schemas()
    trace: list[dict[str, Any]] = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Manufacturer name: {company_name}\n"
            f"Manufacturer graph IRI: {manufacturer_iri}\n\n"
            f"Check whether their stored privacy policy is outdated, and update it if and only "
            f"if you find clear evidence it should be."
        )},
    ]

    finish_summary = "Agent did not reach a conclusion within the step limit."

    for step in range(max_steps):
        # Once a write has happened, drop update_policy from the available
        # tools so the model structurally cannot call it again this run.
        active_schemas = tool_schemas
        if session.write_used:
            active_schemas = [s for s in tool_schemas if s["function"]["name"] != "update_policy"]

        resp = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=active_schemas,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=600,
        )
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or "",
                          "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            trace.append({"step": step, "type": "text_no_tool_call", "content": msg.content})
            messages.append({"role": "user", "content": "Please continue using tool calls, "
                                                          "and call finish() when done."})
            continue

        finished = False
        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fargs = {}

            if fname == "finish":
                finish_summary = fargs.get("summary", "")
                trace.append({"step": step, "type": "finish", "args": fargs})
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": fname,
                                  "content": json.dumps({"status": "task_ended"})})
                finished = True
                continue

            elif fname == "query_graph":
                tool_result = akg.query_graph(g, fargs.get("sparql", ""))

            elif fname == "search_web":
                tool_result = akg.search_web(fargs.get("query", ""))

            elif fname == "fetch_url":
                raw_result = akg.fetch_url(fargs.get("url", ""))
                if "error" not in raw_result:
                    ref = session.record_fetch(fargs.get("url", ""), raw_result)
                    tool_result = {**raw_result, "fetched_text_ref": ref}
                else:
                    tool_result = raw_result

            elif fname == "update_policy":
                tool_result = session.update_policy(
                    fetched_text_ref=fargs.get("fetched_text_ref", ""),
                    model_confidence=fargs.get("confidence", ""),
                    model_reasoning=fargs.get("reasoning", ""),
                )

            else:
                tool_result = {"error": f"Unknown tool {fname}"}

            trace.append({"step": step, "type": "tool_call", "tool": fname,
                          "args": fargs, "result_preview": str(tool_result)[:500]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": fname,
                              "content": json.dumps(tool_result, default=str)[:6000]})

        if finished:
            break
    else:
        trace.append({"step": max_steps, "type": "step_limit_reached"})

    # Serialize the graph to disk ONLY if a write actually happened — no
    # point re-saving an unchanged file, and it keeps "did we touch disk"
    # easy to reason about from the return value alone.
    serialized = False
    if session.write_used and onto_path:
        try:
            g.serialize(destination=str(onto_path), format="xml")
            serialized = True
        except Exception:
            logger.exception("Graph write succeeded in memory but serialize-to-disk failed")

    audit_entry = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "company": company_name,
        "manufacturer_iri": str(manufacturer_iri),
        "wrote_update": session.write_used,
        "serialized_to_disk": serialized,
        "write_result": session.write_result,
        "finish_summary": finish_summary,
        "tool_calls_made": sum(1 for t in trace if t["type"] == "tool_call"),
    }
    _append_audit_log(audit_entry)

    return {
        "company": company_name,
        "wrote_update": session.write_used,
        "serialized_to_disk": serialized,
        "write_result": session.write_result,
        "finish_summary": finish_summary,
        "reasoning_trace": trace,
        "audit_entry": audit_entry,
    }
