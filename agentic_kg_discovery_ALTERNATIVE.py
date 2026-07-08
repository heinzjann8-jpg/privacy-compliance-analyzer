"""
step6_agentic_kg_discovery.py
──────────────────────────────
A genuinely agentic version of policy discovery, as distinct from the
fixed pipeline in agentic_policy_update_crawler.py.

WHAT'S DIFFERENT FROM THE EXISTING PIPELINE
────────────────────────────────────────────
discover_policy_url_agent() in agentic_policy_update_crawler.py always
runs the same hardcoded sequence:
    search -> fetch candidates -> validate -> ask LLM to pick one

The LLM only ever gets to make ONE decision (which candidate to pick).
Every other step — how many searches, which pages to fetch, whether to
consult the knowledge graph at all — is decided by your Python code,
not the model.

This module instead gives the model a small toolbox:

    query_graph(sparql)   -- read-only SPARQL query against the live KG
    search_web(query)     -- Brave search, returns titles/snippets/urls
    fetch_url(url)        -- fetches a page and returns cleaned text

...and runs a loop where the MODEL decides which tool to call, how many
times, in what order, until it has enough information to answer. This
is the "agent reasons over the graph" pattern: the KG becomes something
the model can consult mid-task, not just a place results get written
to afterward.

WHY THIS MATTERS FOR YOUR USE CASE SPECIFICALLY
─────────────────────────────────────────────────
The model can now do things the fixed pipeline structurally cannot:
  - Check what regulatory classes a manufacturer is currently scored
    against BEFORE deciding how to search, so it can specifically look
    for whether THOSE sections changed (rather than treating policy
    text as one undifferentiated blob).
  - Check the manufacturer's last recorded modification date and
    skip an expensive multi-query search if it already has strong
    context from a prior run.
  - Decide on its own that one search wasn't enough and issue a
    second, differently-phrased one — instead of always running the
    same fixed 4 query templates regardless of how the first one went.

SAFETY NOTE
───────────
query_graph() is READ-ONLY by construction — it rejects anything other
than SELECT/ASK/CONSTRUCT/DESCRIBE queries. The model can look at the
graph, but cannot use this tool to write to it. All writes still go
through step4a_timestamps.upsert_policy(), called explicitly by your
code after the loop ends — same as before. Giving the model direct
write access to the ontology would be a significantly bigger trust
boundary than is appropriate here, given it's selecting among LLM-
suggested actions based on web content it does not fully control.

USAGE
─────
    import step6_agentic_kg_discovery as akg

    result = akg.run_agentic_discovery(
        g=g,                          # your rdflib Graph
        company_name="Acurite",
        manufacturer_iri=URIRef(...), # the company's node in the KG
        groq_client=groq_client,
        class_labels=class_labels,    # from app.py's startup block
        class_to_laws=class_to_laws,
        law_to_class_idxs=law_to_class_idxs,
        LAWS=LAWS,
    )
    # result["final_url"], result["reasoning_trace"], result["tool_calls_made"]
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from rdflib import Graph, URIRef, RDFS, Literal
from rdflib.plugins.sparql import prepareQuery

import agentic_policy_update_crawler as policy_crawler

logger = logging.getLogger("agentic_kg_discovery")

MAX_AGENT_STEPS = 8  # hard cap so a confused model can't loop forever / burn quota
DCTERMS_MODIFIED = URIRef("http://purl.org/dc/terms/modified")
DCTERMS_CREATED = URIRef("http://purl.org/dc/terms/created")


#query graph

_ALLOWED_QUERY_FORMS = ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE")


def query_graph(g: Graph, sparql: str, row_limit: int = 25) -> dict[str, Any]:
    """
    Execute a read-only SPARQL query against the live ontology.

    Refuses anything that isn't SELECT/ASK/CONSTRUCT/DESCRIBE — this is
    the boundary that keeps the model able to LOOK at the graph without
    being able to WRITE to it. (rdflib's query() method itself only runs
    read queries — UPDATE goes through g.update() instead — but we also
    reject suspicious tokens defensively in case of future refactors.)
    """
    stripped = sparql.strip()
    upper = stripped.upper()

    if not any(upper.startswith(form) or f"\n{form}" in upper[:50].upper() for form in _ALLOWED_QUERY_FORMS):
        # Also tolerate leading PREFIX lines before the query form.
        lines_upper = [l.strip().upper() for l in stripped.splitlines() if l.strip()]
        body_starts_ok = any(
            l.startswith(form) for l in lines_upper for form in _ALLOWED_QUERY_FORMS
        )
        if not body_starts_ok:
            return {"error": "Only SELECT, ASK, CONSTRUCT, or DESCRIBE queries are permitted."}

    if any(kw in upper for kw in ("INSERT", "DELETE", "DROP", "CLEAR", "LOAD", "CREATE GRAPH")):
        return {"error": "Write operations are not permitted via this tool."}

    try:
        results = g.query(stripped)
    except Exception as e:
        return {"error": f"SPARQL query failed: {e}"}

    rows = []
    for i, row in enumerate(results):
        if i >= row_limit:
            rows.append({"_truncated": True, "note": f"Result set truncated at {row_limit} rows."})
            break
        if hasattr(row, "asdict"):
            rows.append({k: str(v) for k, v in row.asdict().items()})
        else:
            rows.append([str(v) for v in row])

    return {"row_count": len(rows), "rows": rows}


def build_manufacturer_context_tools_payload(
    manufacturer_iri: URIRef,
    class_labels: list[str],
    class_to_laws: dict[str, set],
    law_to_class_idxs: dict[str, list[int]],
    LAWS: list[dict],
) -> dict[str, Any]:
    """
    A convenience helper (NOT a model-facing tool) that precomputes which
    regulatory classes/laws are relevant to this manufacturer, so the
    system prompt can mention it without the model needing several
    query_graph round-trips just to discover the obvious starting point.
    This mirrors how a human analyst would already know "this company is
    scored against Oregon + NISTIR" before going to check their policy.
    """
    law_lookup = {l["id"]: l["label"] for l in LAWS}
    laws_seen = set()
    for c_iri, lids in class_to_laws.items():
        laws_seen.update(lids)
    return {
        "applicable_law_labels": [law_lookup.get(lid, lid) for lid in sorted(laws_seen)],
    }


#web seearch and url get

def search_web(query: str, max_results: int = 6) -> dict[str, Any]:
    try:
        results = policy_crawler.web_search_privacy_policy_urls(query, max_results=max_results)
    except RuntimeError as e:
        return {"error": str(e)}
    return {
        "results": [
            {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("snippet", "")}
            for r in results
        ]
    }


def fetch_url(url: str, max_chars: int = 4000) -> dict[str, Any]:
    try:
        fetched = policy_crawler.fetch_policy_source(url)
        text = policy_crawler.extract_policy_text(fetched)
    except Exception as e:
        return {"error": str(e)}
    return {
        "final_url": fetched.get("url", url),
        "last_modified_header": fetched.get("last_modified_header"),
        "text_preview": text[:max_chars],
        "text_length": len(text),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS — what we hand to Groq's tools= parameter
# ═══════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_graph",
            "description": (
                "Run a read-only SPARQL query against the privacy-compliance knowledge graph. "
                "Use this to check what you already know before searching the web — e.g. what "
                "regulatory classes this manufacturer is scored against, when their policy was "
                "last modified, or what their previously stored policy URL was."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "A SELECT, ASK, CONSTRUCT, or DESCRIBE SPARQL query. No INSERT/DELETE.",
                    }
                },
                "required": ["sparql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for candidate privacy-policy pages for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'Acurite official privacy policy United States'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page and return its cleaned visible text (truncated preview) plus an HTTP Last-Modified header if present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you have identified the official privacy policy URL with "
                "reasonable confidence, OR when you are confident none can be found. "
                "This ends the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_policy_url": {"type": ["string", "null"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                    "reasoning_summary": {
                        "type": "string",
                        "description": "1-3 sentences explaining the decision, for the audit log.",
                    },
                },
                "required": ["confidence", "reasoning_summary"],
            },
        },
    },
]

TOOL_IMPLS = {
    "search_web": lambda args, ctx: search_web(args["query"]),
    "fetch_url": lambda args, ctx: fetch_url(args["url"]),
    "query_graph": lambda args, ctx: query_graph(ctx["g"], args["sparql"]),
}


SYSTEM_PROMPT = """You are a compliance research agent working over a knowledge graph of IoT \
manufacturers and the privacy regulations they're scored against.

Your task: find the official, current privacy-policy URL for a given manufacturer.

You have three information-gathering tools (query_graph, search_web, fetch_url) and one tool \
to end the task (finish). Decide for yourself which tools to use, in what order, and how many \
times — there is no fixed sequence you must follow. Good practice, but not a strict rule:
  - It's often useful to check the knowledge graph first (query_graph) to see what's already
    known about this manufacturer (prior policy URL, last-modified date, which regulations
    they're scored against) before spending a web search on it.
  - If a search result looks ambiguous (e.g. could be a cookie-notice page, a blog post about
    privacy, or a terms-of-service page rather than the actual policy), fetch it and check the
    text before trusting the title/snippet alone.
  - Stop and call finish() once you're reasonably confident, or once you've made a genuine
    effort and concluded nothing reliable exists — don't loop indefinitely.

Always end by calling finish() with your decision."""


def _build_kg_facts_prefix(facts: dict[str, Any]) -> str:
    if not facts.get("applicable_law_labels"):
        return ""
    return ("Known from the system (not from the model): this manufacturer is currently scored "
            f"against: {', '.join(facts['applicable_law_labels'])}.\n\n")


# ═══════════════════════════════════════════════════════════════════════════
# THE AGENTIC LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_agentic_discovery(
    g: Graph,
    company_name: str,
    manufacturer_iri: URIRef,
    groq_client: Any,
    class_labels: Optional[list[str]] = None,
    class_to_laws: Optional[dict[str, set]] = None,
    law_to_class_idxs: Optional[dict[str, list]] = None,
    LAWS: Optional[list[dict]] = None,
    model: str = "llama-3.3-70b-versatile",
    max_steps: int = MAX_AGENT_STEPS,
) -> dict[str, Any]:
    """
    Runs the tool-calling loop until the model calls finish() or max_steps
    is hit. Returns a dict with the final decision plus a full trace of
    every tool call made, for auditing/demo purposes.
    """
    ctx = {"g": g}
    trace: list[dict[str, Any]] = []

    kg_facts = {}
    if class_to_laws is not None and LAWS is not None:
        kg_facts = build_manufacturer_context_tools_payload(
            manufacturer_iri, class_labels or [], class_to_laws, law_to_class_idxs or {}, LAWS or [])

    user_prompt = (
        f"{_build_kg_facts_prefix(kg_facts)}"
        f"Manufacturer name: {company_name}\n"
        f"Manufacturer graph IRI: {manufacturer_iri}\n\n"
        f"Find the official current privacy-policy URL for this manufacturer."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    final_result = {
        "selected_policy_url": None,
        "confidence": "none",
        "reasoning_summary": "Agent did not reach a conclusion within the step limit.",
    }

    for step in range(max_steps):
        resp = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=600,
        )
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or "",
                          "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            # Model responded with plain text instead of a tool call — nudge it.
            trace.append({"step": step, "type": "text_no_tool_call", "content": msg.content})
            messages.append({"role": "user", "content": "Please continue using tool calls, "
                                                          "and call finish() once you have a decision."})
            continue

        finished = False
        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fargs = {}

            if fname == "finish":
                final_result = {
                    "selected_policy_url": fargs.get("selected_policy_url"),
                    "confidence": fargs.get("confidence", "none"),
                    "reasoning_summary": fargs.get("reasoning_summary", ""),
                }
                trace.append({"step": step, "type": "finish", "args": fargs})
                finished = True
                # Still need a tool result message for this call id, even though
                # we're ending — the API expects every tool_call to be answered.
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": fname,
                    "content": json.dumps({"status": "task_ended"}),
                })
                continue

            impl = TOOL_IMPLS.get(fname)
            if impl is None:
                tool_result = {"error": f"Unknown tool {fname}"}
            else:
                try:
                    tool_result = impl(fargs, ctx)
                except Exception as e:
                    tool_result = {"error": str(e)}

            trace.append({"step": step, "type": "tool_call", "tool": fname,
                          "args": fargs, "result_preview": str(tool_result)[:500]})

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "name": fname,
                "content": json.dumps(tool_result)[:6000],  # keep context bounded
            })

        if finished:
            break
    else:
        trace.append({"step": max_steps, "type": "step_limit_reached"})

    return {
        "company": company_name,
        "final_url": final_result["selected_policy_url"],
        "confidence": final_result["confidence"],
        "reasoning_summary": final_result["reasoning_summary"],
        "tool_calls_made": sum(1 for t in trace if t["type"] == "tool_call"),
        "reasoning_trace": trace,
    }
