"""
app.py — IoT Privacy Policy Compliance Analyzer
================================================
Single-source-of-truth architecture:

  compute_scores(policy, state)  →  one dict with sims + flat + weighted scores
       ↓                                    ↓
  /detail sends it to the browser      /chat uses the same dict to build the
  (UI renders weighted + flat)          LLM context → chatbot matches the UI

KG access: thin SPARQL reads (rdfs:label, rdfs:comment, hasLaw annotations). //analyze more scores*****U&**&*&**&**&
No intermediate text blobs.  No alias expansion.
"""

import re, os, json, hashlib
from pathlib import Path
from functools import lru_cache

import numpy as np
from flask import Flask, render_template, request, jsonify
from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from groq import Groq

import step2_state_detector as state_detector
import step4a_timestamps as ts
import step4b_regulation_update as reg_update
import step1_add_regulation as reg_add
import step3_agentic_weighted_grader as wgrader

# =============================================================================
# FLASK
# =============================================================================
app = Flask(__name__)
app.register_blueprint(wgrader.bp)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() == "pdf"

# =============================================================================
# CONFIG
# =============================================================================
CFG           = json.load(open("config.json", encoding="utf-8"))
ONTO_PATH     = Path(os.environ.get("ONTO_PATH", "") or CFG["ontology_path"]).resolve()
MFG_CLS       = URIRef(CFG["manufacturer_class_iri"])
POLICY_PROP   = URIRef(CFG["policy_property_iri"])
LAW_PREDS     = [URIRef(p) for p in CFG.get("law_annotation_predicates", [])]
LAWS          = CFG.get("laws", [])
THRESHOLD     = float(CFG.get("coverage_threshold", 0.27))
EMBED_MODEL   = CFG.get("embedding_model_name", "all-MiniLM-L6-v2")

USER_ADDED_PROP       = URIRef("http://example.org/onto.owl#userAddedManufacturer")
APPLIES_TO_STATE_PROP = URIRef("http://example.org/onto.owl#appliesToState")

# =============================================================================
# STATE NORMALISATION  (frontend value → STATE_CATALOG ID)
# =============================================================================
_STATE_MAP = {
    "all": "all", "oregon": "OR", "california": "CA",
    "texas": "TX",
    # Keep federal sources separate so selecting/asking for NISTIR does not
    # also pull Public Law 116-207 into the chatbot response.
    "nistir": "NISTIR_8259",
    "plaw": "IoT_Cyber_Act_2020",
    "or": "OR", "ca": "CA", "tx": "TX", "us_fed": "US_FED",
    "nist": "NISTIR_8259", "8259": "NISTIR_8259",
    "public_law": "IoT_Cyber_Act_2020", "pl_116_207": "IoT_Cyber_Act_2020",
}
def norm_state(raw):
    return _STATE_MAP.get((raw or "").strip().lower(), "all")

# =============================================================================
# LOAD ONTOLOGY
# =============================================================================
print(f"[load] {ONTO_PATH}")
g = Graph()
g.parse(str(ONTO_PATH))
print(f"[load] {len(g)} triples")

def local_name(iri):
    s = str(iri).rsplit("#", 1)[-1].rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).replace("_", " ").replace("-", " ")

# =============================================================================
# BUILD REGULATORY CLASS CORPUS (directly from KG annotations)
# =============================================================================
_LABEL_FIXES = {
    "iot device": "IoT Device", "io tdevice": "IoT Device",
    "iotdevice": "IoT Device", "network interface": "Network Interface",
}

def _class_label(c):
    lbls = [str(o) for o in g.objects(c, RDFS.label)]
    raw = lbls[0] if lbls else local_name(str(c))
    return _LABEL_FIXES.get(raw.lower().strip(), raw)


#!!!! searches for hasLaw annots
def _law_ids_for_class(c):
    ids = set()
    for pred in LAW_PREDS:
        for obj in g.objects(c, pred):
            txt = str(obj).lower()
            for law in LAWS:
                if any(kw.lower() in txt for kw in law.get("keywords", [])):
                    ids.add(law["id"])
    return ids

def _annotation(c):
    parts = []
    for pred in LAW_PREDS:
        for obj in g.objects(c, pred):
            parts.append(str(obj).strip())
    return " | ".join(parts)

# Build parallel lists
class_iris, class_labels, class_texts = [], [], []
class_descs, class_to_laws = {}, {}

for c in g.subjects(RDF.type, OWL.Class):
    if not isinstance(c, URIRef):
        continue
    law_ids = _law_ids_for_class(c)
    if not law_ids:
        continue
    label   = _class_label(c)
    comment = " ".join(str(o) for o in g.objects(c, RDFS.comment)).strip()
    ann     = _annotation(c)
    class_iris.append(str(c))
    class_labels.append(label)
    class_texts.append(f"{label} {comment} {ann} {local_name(str(c))}")
    class_descs[str(c)]   = comment or ann[:200]
    class_to_laws[str(c)] = law_ids

# Deduplicate by label
seen, keep = set(), []
for i, lbl in enumerate(class_labels):
    if lbl.lower() not in seen:
        seen.add(lbl.lower()); keep.append(i)
class_iris    = [class_iris[i]   for i in keep]
class_labels  = [class_labels[i] for i in keep]
class_texts   = [class_texts[i]  for i in keep]
class_descs   = {class_iris[j]: class_descs[class_iris[keep[j]]] for j in range(len(keep))}
class_to_laws = {class_iris[j]: class_to_laws[class_iris[keep[j]]] for j in range(len(keep))}

if not class_iris:
    raise SystemExit("No regulatory classes found in ontology.")

law_to_class_idxs = {law["id"]: [] for law in LAWS}
for idx, c_iri in enumerate(class_iris):
    for lid in class_to_laws[c_iri]:
        if lid in law_to_class_idxs:
            law_to_class_idxs[lid].append(idx)

for law in LAWS:
    print(f"[init] {law['id']}: {len(law_to_class_idxs[law['id']])} classes")

print(f"[init] Encoding {len(class_texts)} classes with BERT...")
_bert = SentenceTransformer(EMBED_MODEL)
class_embeddings = _bert.encode(class_texts, convert_to_numpy=True, normalize_embeddings=True)

# =============================================================================
# LOAD MANUFACTURERS
# =============================================================================
_ALLOWED = {"ADT","Emerson","Fitbit",
            "Panasonic","Ring","Vivint"}
_NAME_FIX = {"adt":"ADT"}

def clean_name(n):
    return _NAME_FIX.get(n.lower().strip(), n.title())

manufacturers = []
for row in g.query(f"SELECT DISTINCT ?i WHERE {{ ?i a ?t . ?t <{RDFS.subClassOf}>* <{MFG_CLS}> . }}"):
    inst = row.i
    lbls = [str(o) for o in g.objects(inst, RDFS.label)]
    name = clean_name(lbls[0] if lbls else local_name(str(inst)))
    pols = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
    pol  = max(pols, key=len) if pols else ""
    added = any(True for _ in g.objects(inst, USER_ADDED_PROP))
    manufacturers.append({"iri": str(inst), "name": name, "policy": pol, "user_added": added})

manufacturers.sort(key=lambda m: m["name"].lower())
_fil = [m for m in manufacturers if m["user_added"] or any(a.lower() in m["name"].lower() for a in _ALLOWED)]
manufacturers = _fil if _fil else manufacturers
print(f"[init] {len(manufacturers)} manufacturers")

# =============================================================================
# GROQ + WGRADER
# =============================================================================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY",
    "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"))

wgrader.init_grader(
    groq_client=groq_client, manufacturers=manufacturers,
    class_iris=class_iris, class_labels=class_labels,
    class_descs=class_descs, class_to_laws=class_to_laws,
    law_to_class_idxs=law_to_class_idxs, LAWS=LAWS,
    COVERAGE_THRESHOLD=THRESHOLD,
    rank_classes_for_policy=lambda p, s="all": ([], None),
    state_detector=state_detector,
)

# =============================================================================
# CORE SCORING  —  single function used by BOTH /detail AND /chat
# =============================================================================
def _active_laws(state):
    if state == "all":
        return LAWS

    # Direct single-law filters used by the frontend for NISTIR and Public Law.
    # This prevents a NISTIR-only question from being expanded to every federal law.
    direct_law_ids = {l["id"] for l in LAWS}
    if state in direct_law_ids:
        return [l for l in LAWS if l["id"] == state]

    ids = {l["id"] for l in state_detector.applicable_laws([state], LAWS)}
    return [l for l in LAWS if l["id"] in ids]


def compute_scores(policy, state):
    """
    Returns dict:
      sims         np.ndarray | None
      law_coverage list[{id, label, coverage_percent, num_classes, num_above}]
      overall      float
      covered      list[{class_label, law_labels, bert_sim, desc, annotation}]
      missing      list[{class_label, law_label, bert_sim, gap, desc, annotation}]
      weighted     dict | None  (from wgrader cache)
    """
    if not (policy or "").strip():
        return {"sims": None, "law_coverage": [], "overall": 0.0,
                "covered": [], "missing": [], "weighted": None}

    q_emb = _bert.encode([policy], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims  = np.dot(class_embeddings, q_emb)

    active     = _active_laws(state)
    active_ids = {l["id"] for l in active}

    # Flat coverage per law
    law_coverage, total_cls, total_above = [], 0, 0
    for law in active:
        lid  = law["id"]
        idxs = law_to_class_idxs.get(lid, [])
        if not idxs:
            law_coverage.append({"id": lid, "label": law["label"],
                                  "coverage_percent": 0.0, "num_classes": 0, "num_above": 0})
            continue
        above = sum(1 for i in idxs if float(sims[i]) >= THRESHOLD)
        pct   = round(100.0 * above / len(idxs), 2)
        law_coverage.append({"id": lid, "label": law["label"],
                              "coverage_percent": pct, "num_classes": len(idxs), "num_above": above})
        total_cls += len(idxs); total_above += above
    overall = round(100.0 * total_above / total_cls, 2) if total_cls else 0.0

    # Covered classes
    covered = []
    for idx, sim in enumerate(sims):
        sim = float(sim)
        if sim < THRESHOLD:
            continue
        c_iri = class_iris[idx]
        law_ids = class_to_laws.get(c_iri, set()) & active_ids
        if not law_ids:
            continue
        covered.append({
            "class_label": class_labels[idx],
            "law_labels":  [l["label"] for l in LAWS if l["id"] in law_ids],
            "bert_sim":    round(sim, 4),
            "desc":        class_descs.get(c_iri, ""),
            "annotation":  _annotation(URIRef(c_iri)),
        })
    covered.sort(key=lambda x: x["bert_sim"], reverse=True)

    # Missing classes — pull annotation text per law from KG
    missing = []
    for law in active:
        lid = law["id"]
        for idx in law_to_class_idxs.get(lid, []):
            sim = float(sims[idx])
            if sim >= THRESHOLD:
                continue
            c_iri = class_iris[idx]
            node  = URIRef(c_iri)
            ann_parts = []
            for pred in LAW_PREDS:
                for obj in g.objects(node, pred):
                    raw = str(obj)
                    if any(kw.lower() in raw.lower() for kw in law.get("keywords", [])):
                        ann_parts.append(raw.strip())
                        break
            missing.append({
                "class_label": class_labels[idx],
                "law_label":   law["label"],
                "bert_sim":    round(sim, 4),
                "gap":         round(THRESHOLD - sim, 4),
                "desc":        class_descs.get(c_iri, ""),
                "annotation":  " | ".join(ann_parts[:2]),
            })
    missing.sort(key=lambda x: x["gap"], reverse=True)

    # Weighted score from cache (do NOT run agents here; use /weighted_grade_trigger)
    weighted = None
    try:
        from step3_agentic_weighted_grader import (_cache_key, _weight_cache,
                                                   _get_active_laws, agent3_weighted_grader)
        al = _get_active_laws(state)
        ck = _cache_key([l["id"] for l in al])
        if ck in _weight_cache:
            weighted = agent3_weighted_grader(
                {"name": "?", "policy": policy}, sims, _weight_cache[ck], al, state)
    except Exception:
        pass

    return {"sims": sims, "law_coverage": law_coverage, "overall": overall,
            "covered": covered, "missing": missing, "weighted": weighted}


# =============================================================================
# CHAT CONTEXT — built from compute_scores(), reads KG directly
# =============================================================================
def _trim(text, n=260):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= n else text[:n].rstrip() + "..."


def _law_annotation_for_class(c_iri, law):
    """Return only the rdfs:hasLaw-style annotation text that belongs to one law."""
    node = URIRef(c_iri)
    hits = []
    keywords = [law.get("label", ""), law.get("id", "").replace("_", " ")] + law.get("keywords", [])
    keywords = [k.lower() for k in keywords if k]
    for pred in LAW_PREDS:
        for obj in g.objects(node, pred):
            raw = str(obj).strip()
            low = raw.lower()
            if any(k in low for k in keywords):
                hits.append(raw)
    return " | ".join(dict.fromkeys(hits))


def _section_hint(text):
    """Best-effort section/reference extraction from KG law annotation text."""
    if not text:
        return "No section identifier found in KG annotation"
    patterns = [
        r"(?:section|sec\.|§)\s*[0-9A-Za-z_.:-]+",
        r"ORS\s*[0-9A-Za-z_.:-]+",
        r"Cal\.?\s+Civ\.?\s+Code\s*§?\s*[0-9A-Za-z_.:-]+",
        r"NISTIR\s*8259(?:A)?",
        r"PL\s*116-207|Public Law\s*116-207",
        r"HB\s*2395|SB-327|HB\s*4",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(0)
    return "No section identifier found in KG annotation"

#!!! collects annots from requested law from the user 
def law_rules_from_kg(laws=None):
    """Build a clean law -> rule/class list directly from rdfs:hasLaw annotations."""
    laws = laws or LAWS
    out = {}
    for law in laws:
        rules = []
        for idx in law_to_class_idxs.get(law["id"], []):
            c_iri = class_iris[idx]
            ann = _law_annotation_for_class(c_iri, law)
            rules.append({
                "class_label": class_labels[idx],
                "section_hint": _section_hint(ann),
                "annotation": ann or class_descs.get(c_iri, ""),
            })
        # de-duplicate by class label while preserving order
        seen = set()
        clean = []
        for r in rules:
            key = r["class_label"].lower()
            if key not in seen:
                seen.add(key)
                clean.append(r)
        out[law["id"]] = {"id": law["id"], "label": law["label"], "rules": clean}
    return out


def law_comparison_from_kg(laws=None):
    """Return Law A has X / Law B has Y / common Z based on KG law annotations."""
    rules_by_law = law_rules_from_kg(laws)
    label_to_laws = {}
    for lid, law_data in rules_by_law.items():
        for r in law_data["rules"]:
            label_to_laws.setdefault(r["class_label"], set()).add(lid)

    common = sorted([lbl for lbl, lids in label_to_laws.items() if len(lids) >= 2])
    unique = {}
    for lid, law_data in rules_by_law.items():
        unique[lid] = sorted([
            r["class_label"] for r in law_data["rules"]
            if len(label_to_laws.get(r["class_label"], set())) == 1
        ])

    return {"rules_by_law": rules_by_law, "common_rules": common, "unique_rules": unique}


def _explicit_laws_from_question(question):
    """Return only laws explicitly named/aliased in the user's question."""
    q = (question or "").lower()
    wanted = []
    for law in LAWS:
        if law.get("id", "").lower() in q or law.get("label", "").lower() in q:
            wanted.append(law)
            continue
        aliases = {
            "OR_HB_2395": ["oregon", "hb 2395", "oregon hb"],
            "CA_SB_327": ["california", "sb-327", "sb 327"],
            "TX_HB_4": ["texas", "hb 4", "tx hb"],
            "IoT_Cyber_Act_2020": ["iot cybersecurity", "iot cybersecurity improvement", "pl 116-207", "pl 116 207", "public law", "plaw", "federal"],
            "NISTIR_8259": ["nist", "nistir", "8259"],
        }.get(law.get("id", ""), [])
        if any(a in q for a in aliases):
            wanted.append(law)
    return wanted


def _wanted_laws_from_question(question, state="all"):
    """Keep chat context small by only sending laws the user asked about."""
    explicit = _explicit_laws_from_question(question)
    if explicit:
        return explicit
    if state and state != "all":
        return _active_laws(state)
    return LAWS

#!!! guides what user is asking for (not llm due to token calls
def _question_intent(question):
    q = (question or "").lower()
    return {
        "comparison": any(w in q for w in ["compare", "difference", "different", "common", "overlap", "versus", " vs ", "same"]),
        "rules": any(w in q for w in ["rules", "rule", "requirements", "classes", "make up", "specified", "legislation annotations"]),
        "missing": any(w in q for w in ["missing", "non-compliant", "non compliant", "does not comply", "fails", "gap", "weakly"]),
        "score": any(w in q for w in ["score", "coverage", "percent", "grade"]),
    }


def build_chat_context(mfg, scores, state, question=""):
    """
    Small, intent-aware context for Groq's 6k TPM limit.
    Instead of always sending every hasLaw annotation, only send the sections
    needed for the user's question.
    """
    name    = mfg["name"]
    covered = scores["covered"]
    missing = scores["missing"]
    lc      = scores["law_coverage"]
    overall = scores["overall"]
    w       = scores["weighted"]
    scope   = state if state != "all" else "all regulations"
    intent  = _question_intent(question)
    wanted_laws = _wanted_laws_from_question(question, state)
    explicit_laws_for_context = _explicit_laws_from_question(question)
    wanted_law_ids = {law["id"] for law in wanted_laws}
    wanted_law_labels = {law["label"] for law in wanted_laws}

    # Default to the most useful context when the question is vague.
    if not any(intent.values()):
        intent["score"] = True
        intent["missing"] = True

    out = [f"Manufacturer: {name} | Active compliance scope: {scope}"]

    if intent["score"] or intent["missing"]:
        out.append("\nCompliance scores for the selected manufacturer:")
        display_lc = [lw for lw in lc if not explicit_laws_for_context or lw.get("id") in wanted_law_ids or lw.get("label") in wanted_law_labels]
        for lw in display_lc:
            out.append(f"  {lw['label']}: {lw['coverage_percent']}% ({lw['num_above']}/{lw['num_classes']} requirements covered)")
        if not explicit_laws_for_context:
            out.append(f"  Overall flat coverage: {overall}%")
            if w:
                out.append(f"  Overall weighted: {w.get('overall_weighted_score','?')}% | Grade: {w.get('overall_grade','?')}")

    # Law rules from KG: compact by default; annotations only for the requested law(s).
    if intent["rules"]:
        rules = law_rules_from_kg(wanted_laws)
        out.append("\nRules found in the knowledge graph for the requested legislation:")
        for law in wanted_laws:
            law_data = rules.get(law["id"], {"rules": []})
            out.append(f"  {law['label']} includes these requirements:")
            for r in law_data["rules"][:25]:
                ann = _trim(r.get("annotation", ""), 150)
                out.append(f"    - Requirement name: {r['class_label']} | Legal text note: {ann}")

    # Clean law comparison: only class names, no long annotations.
    if intent["comparison"]:
        compare_laws = wanted_laws if len(wanted_laws) >= 2 else LAWS
        kg_compare = law_comparison_from_kg(compare_laws)
        out.append("\nLegislation comparison based on rules in the knowledge graph:")
        out.append("  Requirements shared by the requested laws: " + (", ".join(kg_compare["common_rules"][:40]) if kg_compare["common_rules"] else "None found"))
        for law in compare_laws:
            rules = kg_compare["rules_by_law"].get(law["id"], {}).get("rules", [])
            unique = kg_compare["unique_rules"].get(law["id"], [])
            out.append(f"  {law['label']} includes: " + (", ".join([r["class_label"] for r in rules[:25]]) if rules else "None found"))
            out.append(f"  Requirements mainly found in {law['label']}: " + (", ".join(unique[:20]) if unique else "None found"))

    # Missing details: for a specific law, include EVERY missing item for that law.
    # For broad/all-regulation questions, keep a top-N fallback to avoid Groq TPM errors.
    if intent["missing"]:
        explicit_laws = explicit_laws_for_context
        law_labels_filter = {law["label"] for law in explicit_laws}

        if law_labels_filter:
            missing_for_context = [m for m in missing if m.get("law_label") in law_labels_filter]
            header_scope = ", ".join(sorted(law_labels_filter))
            out.append(f"\nMissing requirements for the selected manufacturer policy — all items for {header_scope} ({len(missing_for_context)}):")
        elif state and state != "all":
            # compute_scores() is already scoped by state, so this is safe to show in full.
            missing_for_context = missing
            out.append(f"\nMissing requirements for the selected manufacturer policy — all items for the active filter ({len(missing_for_context)}):")
        else:
            missing_for_context = missing[:10]
            out.append(f"\nMissing requirements for the selected manufacturer policy — top {min(len(missing), 10)} of {len(missing)}:")

        for m in missing_for_context:
            out.append(f"  ✗ Requirement name: {m['class_label']} | Law: {m['law_label']}")
            if m.get("desc"):
                out.append(f"    Plain meaning / requirement: {_trim(m['desc'], 180)}")
            if m.get("annotation"):
                out.append(f"    Legal text note from KG: {_trim(m['annotation'], 260)}")
        if not missing_for_context:
            out.append("  None — all requirements appear addressed for this law/filter.")

    # Include a tiny covered sample only if useful and there is room.
    if (intent["missing"] or intent["score"]) and covered:
        covered_for_context = []
        for c in covered:
            law_labels = [label for label in c.get("law_labels", []) if not explicit_laws_for_context or label in wanted_law_labels]
            if law_labels:
                item = dict(c)
                item["law_labels"] = law_labels
                covered_for_context.append(item)
        if covered_for_context:
            out.append(f"\nCovered examples — top {min(len(covered_for_context), 6)}:")
            for c in covered_for_context[:6]:
                out.append(f"  ✓ Requirement name: {c['class_label']} | Law: {', '.join(c['law_labels'])}")

    return "\n".join(out)


# =============================================================================
# LLM
# =============================================================================
SYSTEM_PROMPT = (
    "You are a concise IoT privacy compliance assistant. "
    "Answer only from the structured DATA provided from the knowledge graph. Never invent law sections, rules, or facts.\n\n"
    "Rules:\n"
    "• Do not print internal headings such as KG LAW COMPARISON, KG LEGISLATION RULES, DATA, or COMPLIANCE SCORES.\n"
    "• Do not use phrases like gap score, section/reference, BERT, KG annotation, class IRI, or technical ontology terms in the final answer.\n"
    "• When explaining legislation rules, translate each requirement name into natural language for a non-technical reader. Explain what the law is asking a company to do, not just the class name.\n"
    "• When comparing two laws, answer naturally: first say what both laws share, then say what each law uniquely emphasizes. Do not use a rigid template or internal labels.\n"
    "• For missing privacy-policy coverage, list every missing item provided in DATA. For each item, use about two plain-English sentences: what the law expects, and what the selected policy does not clearly say.\n"
    "• Only reference the specific law or laws provided in the DATA. If only NISTIR 8259 is provided, do not mention Public Law 116-207 or any other federal law.\n"
    "• Scores → one sentence, prefer weighted score when available. Nothing missing → one sentence, stop.\n"
    "• Keep answers focused; usually under 300 words unless the user asks for all rules."
)
def _call_llm(question, context):
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": f"DATA:\n{context}\n\nQUESTION:\n{question}"}],
            temperature=0.1, max_tokens=700)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

@lru_cache(maxsize=256)
def _cached_llm(question, ctx_hash, context):
    return _call_llm(question, context)

def ask_llm(question, context):
    h = hashlib.md5(context.encode()).hexdigest()
    return _cached_llm(question, h, context)


# =============================================================================
# HELPERS
# =============================================================================
def _gen_iri(name):
    base = str(MFG_CLS).split("#")[0] + "#"
    slug = re.sub(r"\W+", "_", name.strip()) or "Manufacturer"
    cand = URIRef(base + slug)
    i = 1
    while (cand, None, None) in g:
        cand = URIRef(base + f"{slug}_{i}"); i += 1
    return cand

def _find_mfg(name):
    t = name.lower().strip()
    return next((m for m in manufacturers if m["name"].lower().strip() == t), None)


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/list")
def list_mfg():
    return jsonify([{"iri": m["iri"], "name": m["name"]} for m in manufacturers])


@app.get("/detail")
def detail():
    """Returns flat + weighted scores, covered, missing classes — one payload."""
    iri   = request.args.get("iri")
    state = norm_state(request.args.get("state", "all"))
    if not iri:
        return jsonify({"error": "missing iri"}), 400

    mfg = next((m for m in manufacturers if m["iri"] == iri), None)
    if not mfg:
        inst = URIRef(iri)
        lbls = [str(o) for o in g.objects(inst, RDFS.label)]
        pols = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
        mfg  = {"iri": iri,
                "name":   clean_name(lbls[0] if lbls else local_name(iri)),
                "policy": max(pols, key=len) if pols else ""}

    scores = compute_scores(mfg["policy"], state)
    return jsonify({
        "iri":          mfg["iri"],
        "name":         mfg["name"],
        "policy":       mfg["policy"],
        "state":        state,
        "overall":      scores["overall"],
        "law_coverage": scores["law_coverage"],
        "covered":      scores["covered"],
        "missing":      scores["missing"],
        "weighted":     scores["weighted"],
    })


@app.post("/chat")
def chat():
    """Uses compute_scores() — same as /detail — so numbers always match."""
    data     = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    iri      = (data.get("iri") or "").strip()
    state    = norm_state(data.get("state", "all"))

    if not question or not iri:
        return jsonify({"answer": "Please select a manufacturer and ask a question."}), 400
    mfg = next((m for m in manufacturers if m["iri"] == iri), None)
    if not mfg:
        return jsonify({"answer": "Manufacturer not found."}), 404

    scores  = compute_scores(mfg["policy"], state)
    context = build_chat_context(mfg, scores, state, question)
    answer  = ask_llm(question, context)
    return jsonify({"answer": answer}), 200


@app.post("/add_manufacturer")
def add_manufacturer():
    data      = request.get_json(force=True) or {}
    name      = (data.get("name") or "").strip()
    policy    = (data.get("policy") or "").strip()
    sel_states = data.get("selected_states") or []
    if not name or not policy:
        return jsonify({"error": "name and policy required"}), 400

    existing = _find_mfg(name)
    inst_iri = URIRef(existing["iri"]) if existing else _gen_iri(name)
    if not existing:
        g.add((inst_iri, RDF.type, MFG_CLS))
        g.add((inst_iri, RDFS.label, Literal(name)))
        g.add((inst_iri, USER_ADDED_PROP, Literal(True)))

    ts_info = ts.upsert_policy(g, inst_iri, POLICY_PROP, policy)
    auto    = state_detector.detect_states_from_text(policy)
    states  = sel_states or [s["id"] for s in auto["detected"]]

    for old in list(g.objects(inst_iri, APPLIES_TO_STATE_PROP)):
        g.remove((inst_iri, APPLIES_TO_STATE_PROP, old))
    for sid in states:
        g.add((inst_iri, APPLIES_TO_STATE_PROP, Literal(sid)))

    entry = {"iri": str(inst_iri), "name": clean_name(name),
             "policy": policy, "user_added": True}
    if existing:
        manufacturers[:] = [m for m in manufacturers if m["iri"] != str(inst_iri)]
    manufacturers.append(entry)
    manufacturers.sort(key=lambda m: m["name"].lower())

    try:
        g.serialize(destination=str(ONTO_PATH), format="xml")
    except Exception as e:
        return jsonify({"error": f"Memory OK, file write failed: {e}"}), 500

    return jsonify({"iri": entry["iri"], "name": entry["name"],
                    "action": ts_info["action"],
                    "created_at": ts_info["created_at"],
                    "modified_at": ts_info["modified_at"],
                    "auto_detected_states": auto, "selected_states": states,
                    "applicable_laws": state_detector.applicable_laws(states, LAWS),
                    }), 200 if existing else 201


@app.post("/extract_pdf")
def extract_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f or not allowed_file(f.filename):
        return jsonify({"error": "PDF only"}), 400
    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
    f.save(path)
    try:
        text = "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages).strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except: pass
    return jsonify({"text": text}), 200


@app.post("/upload_regulation")
def upload_regulation():
    f         = request.files.get("file")
    law_id    = (request.form.get("law_id") or "").strip()
    law_label = (request.form.get("law_label") or "").strip()
    if not f or not allowed_file(f.filename) or not law_id or not law_label:
        return jsonify({"error": "PDF, law_id, law_label required"}), 400
    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
    f.save(path)
    try:
        text  = reg_add.read_pdf_text(path)
        sm    = SentenceTransformer(EMBED_MODEL)
        matched, new_cands = reg_add.extract_candidate_classes(text, sm)
        base  = str(MFG_CLS).split("#")[0] + "#"
        annotated, created = [], []
        for m in matched:
            ex = reg_add.find_existing_class(g, m["label"])
            if ex:
                reg_add.annotate_existing_class(g, ex, law_label, m["example_sentences"])
                annotated.append(m["label"])
        for n in new_cands:
            if not reg_add.find_existing_class(g, n["label"]):
                reg_add.create_new_class(g, base, n["label"], law_label, n["example_sentences"])
                created.append(n["label"])
        reg_add.register_law_in_config("config.json", law_id, law_label,
                                       [law_label, law_id.replace("_", " ")])
        g.serialize(destination=str(ONTO_PATH), format="xml")
        return jsonify({"annotated": annotated, "created": created})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except: pass


@app.post("/update_regulation")
def update_regulation():
    data = request.get_json(force=True) or {}
    for k in ["law_id", "law_label", "new_version", "updated_classes"]:
        if k not in data:
            return jsonify({"error": f"Missing {k}"}), 400
    base   = str(MFG_CLS).split("#")[0] + "#"
    result = reg_update.update_regulation(g, base, **{k: data[k] for k in
                ["law_label","law_id","new_version","updated_classes"]})
    try:
        g.serialize(destination=str(ONTO_PATH), format="xml")
    except Exception as e:
        return jsonify({"error": str(e), **result}), 500
    return jsonify(result)


@app.get("/manufacturer_history")
def manufacturer_history():
    iri = request.args.get("iri")
    if not iri:
        return jsonify({"error": "missing iri"}), 400
    return jsonify(ts.get_policy_history(g, URIRef(iri), POLICY_PROP))


@app.post("/detect_states")
def detect_states():
    data   = request.get_json(force=True) or {}
    policy = (data.get("policy") or "").strip()
    if not policy:
        return jsonify({"error": "policy required"}), 400
    auto = state_detector.detect_states_from_text(policy)
    return jsonify({"auto_detection": auto,
                    "applicable_laws": state_detector.applicable_laws(
                        [s["id"] for s in auto["detected"]], LAWS)})


@app.get("/weighted_grade_trigger")
def weighted_grade_trigger():
    """
    Pre-warms the weighted score cache for a manufacturer.
    Call once after selecting a manufacturer; /detail will use the cache.
    GET /weighted_grade_trigger?iri=<iri>&state=all
    """
    iri   = request.args.get("iri")
    state = norm_state(request.args.get("state", "all"))
    if not iri:
        return jsonify({"error": "missing iri"}), 400
    mfg = next((m for m in manufacturers if m["iri"] == iri), None)
    if not mfg:
        return jsonify({"error": "not found"}), 404
    try:
        result = wgrader.run_weighted_grading(mfg, state, use_cache=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
