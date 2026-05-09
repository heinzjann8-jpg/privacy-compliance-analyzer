"""


What this module does:
  - When a privacy policy for a manufacturer is uploaded, we write
    a dcterms:modified (and dcterms:created, the first time) literal
    onto the manufacturer individual.
  - If the same manufacturer is re-uploaded:
        * the OLD policy_description is REMOVED
        * the NEW policy_description is written in its place
        * dcterms:modified is refreshed to 'now'
        * the previous version is kept as a hasPreviousPolicy
          annotation so we don't lose history — this is useful
          for the demo because you can show the professor that
          previous versions are still visible.

This file is pure Python helpers — no Flask — so you can run the
demo script at the bottom to see the before/after triples.

Run as a standalone demo:
    python step4a_timestamps.py
"""

from datetime import datetime, timezone
from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD


# ---------------------------------------------------------------
# Predicates we use for versioning.
# Using dcterms (Dublin Core) because it is the standard vocabulary
# for created/modified timestamps in RDF.
# ---------------------------------------------------------------
DCTERMS_CREATED = URIRef("http://purl.org/dc/terms/created")
DCTERMS_MODIFIED = URIRef("http://purl.org/dc/terms/modified")

# Custom predicate to keep a copy of the previous policy text so the
# demo can show "here's what it was before, here's what it is now".
PREV_POLICY_PRED = URIRef("http://example.org/onto.owl#hasPreviousPolicy")


def _now_iso() -> str:
    """Return an ISO-8601 UTC timestamp, e.g. 2026-04-22T14:30:00+00:00"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------
# CORE: upsert policy with timestamps
# ---------------------------------------------------------------
def upsert_policy(g: Graph, manufacturer_iri: URIRef, policy_prop: URIRef,
                  new_policy_text: str) -> dict:
    """
    Insert or update the policy for a manufacturer, managing timestamps.

    Returns an info dict:
        {
            "action": "created" | "updated",
            "created_at": "...",   # when the manufacturer was first added
            "modified_at": "...",  # when this upload happened
            "previous_policy": "...or None"
        }
    """
    now = Literal(_now_iso(), datatype=XSD.dateTime)

    # ---- Look up existing policy + created timestamp ----
    existing_policies = list(g.objects(manufacturer_iri, policy_prop))
    existing_created = list(g.objects(manufacturer_iri, DCTERMS_CREATED))

    previous_policy_text = None

    if existing_policies:
        # RE-UPLOAD PATH
        # Keep the most recent previous policy as an annotation (for history).
        previous_policy_text = str(existing_policies[0])
        g.add((manufacturer_iri, PREV_POLICY_PRED,
               Literal(f"[{_now_iso()}] {previous_policy_text}")))

        # Remove the old policy literal(s) so there's only one "current"
        for old in existing_policies:
            g.remove((manufacturer_iri, policy_prop, old))

        action = "updated"
    else:
        # FIRST-TIME UPLOAD PATH
        g.add((manufacturer_iri, DCTERMS_CREATED, now))
        action = "created"

    # Write the new policy text
    g.add((manufacturer_iri, policy_prop, Literal(new_policy_text)))

    # Always refresh modified timestamp
    # (first remove any stale modified triples so there's only one)
    for old_mod in list(g.objects(manufacturer_iri, DCTERMS_MODIFIED)):
        g.remove((manufacturer_iri, DCTERMS_MODIFIED, old_mod))
    g.add((manufacturer_iri, DCTERMS_MODIFIED, now))

    created_literal = existing_created[0] if existing_created else now

    return {
        "action": action,
        "created_at": str(created_literal),
        "modified_at": str(now),
        "previous_policy": previous_policy_text,
    }


def get_policy_history(g: Graph, manufacturer_iri: URIRef, policy_prop: URIRef) -> dict:
    """
    Return everything the UI needs to display the history panel:
        - current policy + modified timestamp
        - created timestamp
        - list of previous policies (each prefixed with the timestamp
          when they were replaced)
    """
    current = list(g.objects(manufacturer_iri, policy_prop))
    created = list(g.objects(manufacturer_iri, DCTERMS_CREATED))
    modified = list(g.objects(manufacturer_iri, DCTERMS_MODIFIED))
    previous = [str(o) for o in g.objects(manufacturer_iri, PREV_POLICY_PRED)]

    return {
        "current_policy": str(current[0]) if current else None,
        "created_at": str(created[0]) if created else None,
        "modified_at": str(modified[0]) if modified else None,
        "previous_policies": previous,
    }


# ---------------------------------------------------------------
# STANDALONE DEMO
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Minimal ontology for the demo
    g = Graph()
    BASE = "http://example.org/onto.owl#"
    MFG_CLS = URIRef(BASE + "Manufacturer")
    POLICY_PROP = URIRef(BASE + "policy_description")
    DEMO = URIRef(BASE + "DemoCorp")

    g.add((DEMO, RDF.type, MFG_CLS))
    g.add((DEMO, RDFS.label, Literal("DemoCorp")))

    print("=== First upload ===")
    info1 = upsert_policy(g, DEMO, POLICY_PROP,
                          "We collect data from Oregon residents. v1")
    print(info1)

    print("\n=== Re-upload (simulating an updated policy) ===")
    import time; time.sleep(1)  # so the timestamp actually differs
    info2 = upsert_policy(g, DEMO, POLICY_PROP,
                          "We now collect data from Oregon AND Texas residents. v2")
    print(info2)

    print("\n=== Full history for DemoCorp ===")
    hist = get_policy_history(g, DEMO, POLICY_PROP)
    for k, v in hist.items():
        print(f"  {k}: {v}")
