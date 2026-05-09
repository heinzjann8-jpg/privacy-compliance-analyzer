"""
step3_agentic_weighted_grader.py
──────────────────────────
Three-agent weighted grading system for IoT compliance classes.

The flat compute_law_coverage() in app.py treats every regulatory class
as equal — a missing "Transducer Element" is penalised the same as a missing
"Authentication Mechanism".  This module replaces that with agent-determined
weights that reflect real-world legislative intent.

AGENTS
──────
Agent 1 — Legislation Analyst
    Reads all classes belonging to a law and assigns each a criticality
    weight (0.0–1.0) by reasoning about what the law actually cares about.
    Output: {class_label -> weight} per law.

Agent 2 — Cross-Law Comparator
    Looks across all laws and elevates classes that appear in multiple laws
    (consensus = more critical) and flags law-specific classes that are
    uniquely important.
    Output: merged global weight table {class_label -> final_weight}.

Agent 3 — Weighted Grader
    Applies the final weights to the manufacturer's BERT similarity scores
    and computes a weighted compliance score that replaces the flat percent.
    Produces per-law and overall weighted grades with a letter grade.

ENDPOINT
────────
POST /weighted_grade
Body: { "iri": "<manufacturer_iri>", "state": "all", "use_cache": true }

Cache: weights are expensive to compute (3 LLM calls).  They are cached
in memory keyed by frozenset(law_ids) so re-running for a different
manufacturer with the same laws reuses the weights.

INTEGRATION (add to app.py)
────────────────────────────
    import step_3agentic_weighted_grader as wgrader
    app.register_blueprint(wgrader.bp)

    # after all globals are ready:
    wgrader.init_grader(
        groq_client      = groq_client,
        manufacturers    = manufacturers,
        class_iris       = class_iris,
        class_labels     = class_labels,
        class_descs      = class_descs,
        class_to_laws    = class_to_laws,
        law_to_class_idxs= law_to_class_idxs,
        LAWS             = LAWS,
        COVERAGE_THRESHOLD = COVERAGE_THRESHOLD,
        rank_classes_for_policy = rank_classes_for_policy,
        state_detector   = state_detector,
    )
"""

from __future__ import annotations

import json
import time
import hashlib
from typing import Optional

from flask import Blueprint, request, jsonify

bp = Blueprint("wgrader", __name__)

# ── shared state ──────────────────────────────────────────────────────────────
_S: dict = {}

# in-memory weight cache  {cache_key: weight_table}
_weight_cache: dict[str, dict] = {}


def init_grader(**kwargs):
    _S.update(kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 800, temp: float = 0.15) -> str:
    resp = _S["groq_client"].chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict | list:
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def _cache_key(law_ids: list[str]) -> str:
    return hashlib.md5("|".join(sorted(law_ids)).encode()).hexdigest()


def _letter_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 55: return "D"
    return "F"


def _get_active_laws(state: str) -> list[dict]:
    LAWS = _S["LAWS"]
    if state and state != "all":
        applicable = _S["state_detector"].applicable_laws([state], LAWS)
        ids = {l["id"] for l in applicable}
        return [l for l in LAWS if l["id"] in ids]
    return LAWS


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Legislation Analyst
# Determines class criticality weights for a single law
# ══════════════════════════════════════════════════════════════════════════════

def agent1_legislation_analyst(law: dict, class_labels_for_law: list[str]) -> dict[str, float]:
    """
    Returns {class_label: weight (0.0-1.0)} for all classes under this law.

    Reasoning heuristic given to the LLM:
      - Security-critical classes (auth, encryption, patching) → high weight
      - Data governance classes (collection, retention) → medium-high
      - Structural/reporting classes (logging, documentation) → medium
      - Peripheral device classes (transducers, sensors) → lower unless law emphasises them
    """
    if not class_labels_for_law:
        return {}

    class_list = "\n".join(f"- {lbl}" for lbl in class_labels_for_law)

    system = """You are a legislative compliance analyst specialising in IoT privacy and security law.
Your job: assign a criticality weight (0.0 to 1.0) to each regulatory class for a specific law.

Weight guidelines:
  1.0  — Core security requirement explicitly mandated with enforcement consequences
         (e.g. unique default passwords, patch/update obligations, encryption of data in transit)
  0.8  — Strong data-protection or access-control requirement central to the law's purpose
         (e.g. data minimisation, user consent, access logging)
  0.6  — Important operational requirement that supports compliance
         (e.g. incident response, configuration management, network interface security)
  0.4  — Recommended or best-practice class referenced but not strictly mandated
         (e.g. device identity, transducer security, platform monitoring)
  0.2  — Peripheral or administrative class with minimal direct enforcement weight
         (e.g. documentation metadata, ancillary device components)

Return ONLY a JSON object: {"ClassName": weight, ...}
No markdown, no explanation, no extra keys."""

    user = f"""LAW: {law['label']}

REGULATORY CLASSES FOR THIS LAW:
{class_list}

Assign weights. Return JSON only."""

    try:
        raw = _llm(system, user, max_tokens=600)
        weights = _parse_json(raw)
        # Normalise: clamp to [0.1, 1.0], default 0.5 for unrecognised
        result = {}
        for lbl in class_labels_for_law:
            w = float(weights.get(lbl, 0.5))
            result[lbl] = max(0.1, min(1.0, w))
        return result
    except Exception as e:
        # Fallback: uniform weights
        return {lbl: 0.5 for lbl in class_labels_for_law}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Cross-Law Comparator
# Merges law weights into a global table, elevating consensus classes.
# ex if multiple legislations mention the same policy then it becomes more importnt
# ══════════════════════════════════════════════════════════════════════════════

def agent2_cross_law_comparator(per_law_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    per_law_weights: {law_id: {class_label: weight}}

    Strategy:
      - Classes appearing in N laws get their average weight boosted by log(N)*0.1
      - The LLM then reviews the merged table and can adjust classes it considers
        systematically under- or over-weighted given cross-law consensus.
    """
    import math

    # Step 1: accumulate
    merged: dict[str, list[float]] = {}
    for law_id, weights in per_law_weights.items():
        for lbl, w in weights.items():
            merged.setdefault(lbl, []).append(w)

    # Step 2: compute preliminary global weight with consensus boost
    preliminary: dict[str, float] = {}
    for lbl, ws in merged.items():
        avg = sum(ws) / len(ws)
        boost = math.log(len(ws) + 1) * 0.08  # ~0.08 for 1 law, ~0.14 for 3 laws
        preliminary[lbl] = round(min(1.0, avg + boost), 3)

    # Step 3: ask LLM to sanity-check and refine the top 20 most impactful classes
    top_classes = sorted(preliminary.items(), key=lambda x: x[1], reverse=True)[:20]
    class_summary = "\n".join(
        f"- {lbl}: preliminary weight {w:.2f} (appears in {len(merged[lbl])} law(s))"
        for lbl, w in top_classes
    )

    system = """You are a senior IoT compliance standards expert reviewing a preliminary class weight table.
The weights were computed by averaging per-law criticality scores and adding a consensus boost for classes appearing in multiple laws.

Your task: review and output a REFINED weight table.
Rules:
  - You may adjust any weight by at most ±0.15
  - Authentication, unique passwords, patch availability should NEVER be below 0.75
  - Peripheral hardware-only classes (transducer, antenna) should NEVER exceed 0.55
  - If a class appears in 3+ laws, its weight should be at least 0.65
  - Return ONLY a JSON object: {"ClassName": weight, ...}
  - Include EVERY class listed, unchanged if you agree."""

    user = f"""PRELIMINARY WEIGHT TABLE (top classes):
{class_summary}

Refine and return ALL of them as JSON."""

    try:
        raw = _llm(system, user, max_tokens=700)
        refined = _parse_json(raw)
        # Merge refined into preliminary (refined overrides where present)
        final = dict(preliminary)
        for lbl, w in refined.items():
            if lbl in final:
                final[lbl] = round(max(0.1, min(1.0, float(w))), 3)
        return final
    except Exception:
        return preliminary


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Weighted Grader
# applies weight table to BERT sims and computes weighted compliance scores.
# ══════════════════════════════════════════════════════════════════════════════

def agent3_weighted_grader(
    manufacturer: dict,
    sims,
    global_weights: dict[str, float],
    active_laws: list[dict],
    state: str,
) -> dict:
    """
    Replaces compute_law_coverage() with a weighted alternative.

    Weighted score formula per law:
        score = Σ(w_i * min(1, sim_i / threshold)) / Σ(w_i)

    where:
        w_i   = agent-determined weight for class i
        sim_i = BERT cosine similarity for class i
        threshold = COVERAGE_THRESHOLD (0.27 in your config)

    This means:
        - A class with sim=0.27 (exactly at threshold) contributes its full weight
        - A class with sim=0.54 (2× threshold) is capped at 1.0 — prevents over-rewarding
        - A class with sim=0.0 contributes 0
        - A missing high-weight class hurts the score far more than a missing low-weight one
    """
    THRESHOLD = _S["COVERAGE_THRESHOLD"]
    class_iris    = _S["class_iris"]
    class_labels  = _S["class_labels"]
    law_to_class_idxs = _S["law_to_class_idxs"]

    law_results = []
    total_weighted_sum = 0.0
    total_weight_sum   = 0.0

    for law in active_laws:
        lid = law["id"]
        class_idxs = law_to_class_idxs.get(lid, [])
        if not class_idxs:
            law_results.append({
                "id": lid, "label": law["label"],
                "weighted_score": 0.0, "flat_score": 0.0,
                "weight_sum": 0.0, "num_classes": 0,
                "grade": "N/A", "class_detail": [],
            })
            continue

        w_sum   = 0.0
        ws_sum  = 0.0
        details = []

        for idx in class_idxs:
            lbl = class_labels[idx]
            sim = float(sims[idx])
            w   = global_weights.get(lbl, 0.5)
            contribution = w * min(1.0, sim / THRESHOLD) if THRESHOLD > 0 else 0.0

            w_sum  += w
            ws_sum += contribution
            details.append({
                "class_label":  lbl,
                "bert_sim":     round(sim, 4),
                "weight":       round(w, 3),
                "contribution": round(contribution, 4),
                "covered":      sim >= THRESHOLD,
            })

        weighted_pct = round(100.0 * ws_sum / w_sum, 2) if w_sum > 0 else 0.0
        flat_above   = sum(1 for d in details if d["covered"])
        flat_pct     = round(100.0 * flat_above / len(details), 2)

        law_results.append({
            "id": lid, "label": law["label"],
            "weighted_score":  weighted_pct,
            "flat_score":      flat_pct,
            "score_delta":     round(weighted_pct - flat_pct, 2),
            "weight_sum":      round(w_sum, 3),
            "num_classes":     len(class_idxs),
            "grade":           _letter_grade(weighted_pct),
            "class_detail":    sorted(details, key=lambda x: x["weight"], reverse=True),
        })

        total_weighted_sum += ws_sum
        total_weight_sum   += w_sum

    overall_weighted = round(100.0 * total_weighted_sum / total_weight_sum, 2) if total_weight_sum > 0 else 0.0

    # Identify the highest-impact missing classes (weighted penalty = weight * (1 - sim/threshold))
    missed_weighted = []
    for law in active_laws:
        lid = law["id"]
        for idx in law_to_class_idxs.get(lid, []):
            lbl = class_labels[idx]
            sim = float(sims[idx])
            w   = global_weights.get(lbl, 0.5)
            if sim < THRESHOLD:
                penalty = w * (1.0 - sim / THRESHOLD) if THRESHOLD > 0 else w
                missed_weighted.append({
                    "class_label": lbl,
                    "law_label":   law["label"],
                    "weight":      round(w, 3),
                    "bert_sim":    round(sim, 4),
                    "penalty":     round(penalty, 4),
                })
    missed_weighted.sort(key=lambda x: x["penalty"], reverse=True)

    return {
        "manufacturer":        manufacturer["name"],
        "state_filter":        state,
        "overall_weighted_score": overall_weighted,
        "overall_grade":          _letter_grade(overall_weighted),
        "law_scores":          law_results,
        "top_weighted_gaps":   missed_weighted[:10],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs all three agents in sequence
# ══════════════════════════════════════════════════════════════════════════════

def run_weighted_grading(manufacturer: dict, state: str, use_cache: bool = True) -> dict:
    active_laws  = _get_active_laws(state)
    law_ids      = [l["id"] for l in active_laws]
    cache_key    = _cache_key(law_ids)

    rank = _S["rank_classes_for_policy"]
    _, sims = rank(manufacturer["policy"], state)
    if sims is None:
        return {"error": "Could not compute similarity scores."}

    t0 = time.time()

    # ── Check weight cache ────────────────────────────────────────────────────
    if use_cache and cache_key in _weight_cache:
        global_weights = _weight_cache[cache_key]
        cached = True
        per_law_weights = {}      # not re-run
        agent1_detail = None
    else:
        cached = False

        # ── Agent 1: per-law criticality ──────────────────────────────────────
        law_to_class_idxs = _S["law_to_class_idxs"]
        class_labels      = _S["class_labels"]
        per_law_weights: dict[str, dict[str, float]] = {}

        agent1_detail = {}
        for law in active_laws:
            lid = law["id"]
            idxs = law_to_class_idxs.get(lid, [])
            labels_for_law = [class_labels[i] for i in idxs]
            weights = agent1_legislation_analyst(law, labels_for_law)
            per_law_weights[lid] = weights
            agent1_detail[lid] = {
                "law_label": law["label"],
                "class_weights": weights,
            }

        # ── Agent 2: cross-law merge + refinement ─────────────────────────────
        global_weights = agent2_cross_law_comparator(per_law_weights)
        _weight_cache[cache_key] = global_weights

    # ── Agent 3: weighted scoring ─────────────────────────────────────────────
    grade_result = agent3_weighted_grader(manufacturer, sims, global_weights, active_laws, state)

    elapsed = round(time.time() - t0, 2)

    return {
        **grade_result,
        "weight_table":    global_weights,
        "weights_cached":  cached,
        "agent1_per_law":  agent1_detail,
        "elapsed_seconds": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE
# ══════════════════════════════════════════════════════════════════════════════

@bp.post("/weighted_grade")
def weighted_grade():
    data      = request.get_json(force=True) or {}
    iri       = (data.get("iri") or "").strip()
    state     = (data.get("state") or "all").strip()
    use_cache = data.get("use_cache", True)

    manufacturers = _S.get("manufacturers", [])
    manufacturer  = next((m for m in manufacturers if m["iri"] == iri), None)

    if not manufacturer:
        return jsonify({"error": "Manufacturer not found."}), 404
    if not manufacturer.get("policy", "").strip():
        return jsonify({"error": "No policy text to grade."}), 400

    result = run_weighted_grading(manufacturer, state, use_cache=use_cache)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result), 200


@bp.get("/weight_table")
def weight_table():
    """
    Returns the current cached weight table (or triggers generation).
    GET /weight_table?state=all
    Useful for inspecting agent-determined weights without grading a specific manufacturer.
    """
    state    = request.args.get("state", "all")
    active   = _get_active_laws(state)
    law_ids  = [l["id"] for l in active]
    key      = _cache_key(law_ids)

    if key in _weight_cache:
        return jsonify({
            "cached": True,
            "state": state,
            "laws": [l["label"] for l in active],
            "weight_table": _weight_cache[key],
        })

    # Trigger Agent 1 + 2 without grading
    law_to_class_idxs = _S["law_to_class_idxs"]
    class_labels      = _S["class_labels"]
    per_law = {}
    for law in active:
        idxs = law_to_class_idxs.get(law["id"], [])
        labels = [class_labels[i] for i in idxs]
        per_law[law["id"]] = agent1_legislation_analyst(law, labels)

    global_weights = agent2_cross_law_comparator(per_law)
    _weight_cache[key] = global_weights

    return jsonify({
        "cached": False,
        "state": state,
        "laws": [l["label"] for l in active],
        "weight_table": global_weights,
    })


@bp.delete("/weight_cache")
def clear_weight_cache():
    """Force re-generation of weights on next call. DELETE /weight_cache"""
    _weight_cache.clear()
    return jsonify({"cleared": True})
