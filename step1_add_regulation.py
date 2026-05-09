"""

What this script does, step by step:
  1. Reads a regulation PDF (e.g. Texas HB 2395, Oregon HB 2395,
     California SB-327, NIST IR 8259, etc.)
  2. Uses the existing semantic extractor pattern to pull out
     entities (candidate classes) and relationships (triples).
  3. For each candidate class:
       - If a class with that label ALREADY EXISTS in the ontology,
         we simply ANNOTATE it with a hasLaw literal tying it to the
         new regulation (no duplicate class is created).
       - If the class DOES NOT EXIST, we CREATE it as a new
         owl:Class and add the triples describing it.
  4. Writes the new law ID into config.json so the rest of the
     app (law coverage, dropdown, etc.) picks it up automatically.

How to run (standalone demo):
    python step1_add_regulation.py --pdf HB2395.pdf \
        --law-id TX_HB_2395 --law-label "Texas HB 2395"

This is the SAME pattern the previous students used for NIST and
IoT Act — just generalized so any new regulation can be plugged in.
"""

import argparse
import json
import os
import re
from pathlib import Path

from PyPDF2 import PdfReader
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
CONFIG_PATH = "config.json"
HAS_LAW_PRED = URIRef("http://www.w3.org/2000/01/rdf-schema#hasLaw")

# Seed ontology entities — same list used by semantic_policy_extractor.py.
# When a candidate class name from the PDF is semantically close to one
# of these, we treat it as a MATCH (annotate), otherwise we create a NEW class.
SEED_ENTITIES = [
    "IoT Device", "Manufacturer", "Federal Agency", "Cybersecurity Standard",
    "Personal Information", "Service Provider", "Customer", "Website",
    "Privacy Policy", "Data Breach", "Children", "Email", "Transaction",
    "Cookie", "User", "Data Processor", "Data Collector", "Law",
    "Policy", "Affiliate", "Subsidiary", "Division", "Franchisee",
    "Stakeholder", "Regulatory Body", "Security Feature", "Network Interface",
]


# ---------------------------------------------------------------
# STEP 1a: Read the PDF text
# ---------------------------------------------------------------
def read_pdf_text(pdf_path: str) -> str:
    """Extract raw text from the uploaded regulation PDF."""
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            chunks.append(t)
    return "\n\n".join(chunks)


# ---------------------------------------------------------------
# STEP 1b: Pull candidate entities out of the regulation text
# ---------------------------------------------------------------
def extract_candidate_classes(text: str, model: SentenceTransformer, top_n: int = 15):
    """
    Very lightweight candidate extraction:
      - Split the regulation into sentences.
      - For each sentence, find the closest seed entity by embedding.
      - Keep a short list of the most-referenced entities.
      - Also pull out capitalized multi-word noun phrases that DON'T
        match any seed — those are candidates for NEW classes.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return [], []

    # --- Matched seed entities (these will be annotated, not recreated) ---
    sent_emb = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    seed_emb = model.encode(SEED_ENTITIES, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(sent_emb, seed_emb)  # shape: (num_sentences, num_seeds)
    hit_counts = {e: 0 for e in SEED_ENTITIES}
    hit_sentences = {e: [] for e in SEED_ENTITIES}

    for i, sentence in enumerate(sentences):
        best_idx = int(sims[i].argmax())
        best_score = float(sims[i][best_idx])
        if best_score >= 0.35:  # tunable threshold
            entity = SEED_ENTITIES[best_idx]
            hit_counts[entity] += 1
            if len(hit_sentences[entity]) < 3:
                hit_sentences[entity].append(sentence)

    matched_existing = [
        {"label": e, "example_sentences": hit_sentences[e]}
        for e, c in sorted(hit_counts.items(), key=lambda kv: -kv[1])
        if c > 0
    ][:top_n]

    # --- Candidate NEW classes: capitalized phrases not near any seed ---
    phrase_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
    candidate_new = {}
    seed_lower = {s.lower() for s in SEED_ENTITIES}

    for sentence in sentences:
        for m in phrase_re.findall(sentence):
            key = m.strip()
            if key.lower() in seed_lower:
                continue
            # Skip phrases that are common but useless
            if key.lower() in {"united states", "privacy policy", "section one"}:
                continue
            candidate_new.setdefault(key, []).append(sentence)

    # Keep phrases that appear in 2+ sentences (avoid one-off noise)
    new_classes = [
        {"label": k, "example_sentences": v[:3]}
        for k, v in candidate_new.items()
        if len(v) >= 2
    ][:top_n]

    return matched_existing, new_classes


# ---------------------------------------------------------------
# STEP 1c: Apply to the knowledge graph
# ---------------------------------------------------------------
def slug(label: str) -> str:
    return re.sub(r"\W+", "_", label.strip())


def find_existing_class(g: Graph, label: str):
    """
    Return the IRI of an existing owl:Class whose rdfs:label OR local
    name matches `label` (case-insensitive). Otherwise None.
    """
    target = label.lower().strip()
    for c in g.subjects(RDF.type, OWL.Class):
        if not isinstance(c, URIRef):
            continue
        # Check labels
        for lbl in g.objects(c, RDFS.label):
            if str(lbl).lower().strip() == target:
                return c
        # Check local name
        local = str(c).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        local_clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", local).replace("_", " ").lower().strip()
        if local_clean == target:
            return c
    return None


def annotate_existing_class(g: Graph, class_iri: URIRef, law_label: str, snippets: list):
    """
    Class already exists → just attach a hasLaw literal that mentions
    the new law and a couple of example sentences. This is the
    'annotate' path from the requirements.
    """
    text = f"{law_label}. " + " ".join(snippets[:2])
    g.add((class_iri, HAS_LAW_PRED, Literal(text)))


def create_new_class(g: Graph, base_iri: str, label: str, law_label: str, snippets: list):
    """
    Class doesn't exist → create a new owl:Class with label, comment,
    and a hasLaw annotation pointing at the new regulation.
    This is the 'encode into the KG through triples' path.
    """
    new_iri = URIRef(base_iri + slug(label))
    g.add((new_iri, RDF.type, OWL.Class))
    g.add((new_iri, RDFS.label, Literal(label)))
    if snippets:
        g.add((new_iri, RDFS.comment, Literal(snippets[0])))
    text = f"{law_label}. " + " ".join(snippets[:2])
    g.add((new_iri, HAS_LAW_PRED, Literal(text)))
    return new_iri


# ---------------------------------------------------------------
# STEP 1d: Register the new law in config.json so the rest of the
# app knows about it (dropdowns, coverage, etc.)
# ---------------------------------------------------------------
def register_law_in_config(config_path: str, law_id: str, law_label: str, keywords: list):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    existing_ids = {law["id"] for law in cfg.get("laws", [])}
    if law_id in existing_ids:
        print(f"[config] Law {law_id} already registered — skipping.")
        return

    cfg.setdefault("laws", []).append({
        "id": law_id,
        "label": law_label,
        "keywords": keywords,
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] Registered new law: {law_label} ({law_id})")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the regulation PDF")
    ap.add_argument("--law-id", required=True, help="e.g. TX_HB_2395")
    ap.add_argument("--law-label", required=True, help="e.g. 'Texas HB 2395'")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    onto_path = Path(cfg["ontology_path"]).resolve()
    base_iri = cfg["manufacturer_class_iri"].split("#")[0] + "#"

    print(f"\n[1] Reading PDF: {args.pdf}")
    text = read_pdf_text(args.pdf)
    print(f"    Extracted {len(text):,} characters")

    print("\n[2] Loading semantic model + pulling candidate classes")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    matched, new_candidates = extract_candidate_classes(text, model)
    print(f"    Found {len(matched)} existing-class matches, "
          f"{len(new_candidates)} candidate NEW classes")

    print(f"\n[3] Loading ontology: {onto_path}")
    g = Graph()
    g.parse(str(onto_path))
    before = len(g)

    # --- Annotate classes that already exist ---
    print("\n[4] Annotating existing classes with new law…")
    for m in matched:
        existing = find_existing_class(g, m["label"])
        if existing is not None:
            annotate_existing_class(g, existing, args.law_label, m["example_sentences"])
            print(f"    + annotated existing class: {m['label']}")

    # --- Create new classes for anything missing ---
    print("\n[5] Creating new classes via triples…")
    for n in new_candidates:
        # double-check it truly doesn't exist already
        if find_existing_class(g, n["label"]) is not None:
            continue
        new_iri = create_new_class(g, base_iri, n["label"], args.law_label, n["example_sentences"])
        print(f"    + created new class: {n['label']}  ->  {new_iri}")

    after = len(g)
    print(f"\n[6] Graph grew from {before:,} to {after:,} triples (+{after - before})")

    print(f"\n[7] Saving ontology back to: {onto_path}")
    g.serialize(destination=str(onto_path), format="xml")

    # --- Register law in config ---
    print("\n[8] Registering law in config.json")
    keywords = [args.law_label, args.law_id.replace("_", " ")]
    register_law_in_config(args.config, args.law_id, args.law_label, keywords)

    print("\nDone. The new regulation is now part of the knowledge graph.")


if __name__ == "__main__":
    main()
