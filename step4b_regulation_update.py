"""


Scenario:
  A regulation (say, "Oregon HB 2395") was added to the KG six
  months ago. Ten of its classes are now sitting in the ontology
  with rdfs:label, rdfs:comment, hasLaw annotations, etc. Other
  parts of the KG link to those classes (e.g. a manufacturer's
  policy was matched to three of them).

  Now Oregon amends the law. We need to:
    - ADD any brand-new classes that didn't exist before.
    - REFRESH the hasLaw annotation on the existing classes so it
      reflects the amended text.
    - NOT delete or rename any existing class IRI — that would
      break every link from manufacturers to those classes.

This module gives you three surgical functions that do exactly that.

Run as a standalone demo:
    python step4b_regulation_update.py
"""

import re
from datetime import datetime, timezone
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL, XSD


HAS_LAW_PRED = URIRef("http://www.w3.org/2000/01/rdf-schema#hasLaw")
DCTERMS_MODIFIED = URIRef("http://purl.org/dc/terms/modified")
LAW_VERSION_PRED = URIRef("http://example.org/onto.owl#lawVersion")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(label: str) -> str:
    return re.sub(r"\W+", "_", label.strip())


def _find_class_by_label(g: Graph, label: str):
    """Case-insensitive lookup by rdfs:label OR local name."""
    t = label.lower().strip()
    for c in g.subjects(RDF.type, OWL.Class):
        if not isinstance(c, URIRef):
            continue
        for lbl in g.objects(c, RDFS.label):
            if str(lbl).lower().strip() == t:
                return c
        local = str(c).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", local).replace("_", " ").lower().strip()
        if clean == t:
            return c
    return None


# ---------------------------------------------------------------
# CORE: refresh an existing regulation
# ---------------------------------------------------------------
def update_regulation(
    g: Graph,
    base_iri: str,
    law_label: str,
    law_id: str,
    new_version: str,
    updated_classes: list,
) -> dict:
    """
    Apply an incremental update for a regulation.

    Args:
      g               : the rdflib Graph
      base_iri        : ontology namespace, e.g. "http://example.org/onto.owl#"
      law_label       : human label of the law ("Oregon HB 2395")
      law_id          : short id                      ("OR_HB_2395")
      new_version     : version tag for this amendment ("2026-amendment")
      updated_classes : list of dicts, each:
                         { "label": "...", "snippet": "amended text ..." }

    Behavior per class:
      - If a class with that label EXISTS:
          * replace only the hasLaw literals that mention `law_label`
            with a new one containing `snippet` + the new version tag.
          * leave all other triples (subClassOf, comments, links from
            manufacturers, other laws' hasLaw, etc.) UNTOUCHED.
      - If the class DOES NOT exist:
          * create a new owl:Class with label, comment and a hasLaw.

    Returns a summary dict showing what changed.
    """
    now = _now_iso()
    refreshed, created = [], []

    for entry in updated_classes:
        label = entry["label"]
        snippet = entry.get("snippet", "")
        existing = _find_class_by_label(g, label)

        new_law_text = (
            f"{law_label} [version: {new_version}] [updated: {now}]. {snippet}"
        )

        if existing is not None:
            # ---------- REFRESH ----------
            # Only remove hasLaw triples that reference THIS law,
            # so other laws' annotations (NIST, CA, etc.) survive.
            to_remove = []
            for lit in g.objects(existing, HAS_LAW_PRED):
                s = str(lit).lower()
                if law_label.lower() in s or law_id.lower().replace("_", " ") in s:
                    to_remove.append(lit)
            for lit in to_remove:
                g.remove((existing, HAS_LAW_PRED, lit))

            # Add the refreshed annotation
            g.add((existing, HAS_LAW_PRED, Literal(new_law_text)))

            # Touch a "modified" stamp on the class itself
            for m in list(g.objects(existing, DCTERMS_MODIFIED)):
                g.remove((existing, DCTERMS_MODIFIED, m))
            g.add((existing, DCTERMS_MODIFIED, Literal(now, datatype=XSD.dateTime)))

            refreshed.append(label)
        else:
            # ---------- CREATE NEW ----------
            new_iri = URIRef(base_iri + _slug(label))
            g.add((new_iri, RDF.type, OWL.Class))
            g.add((new_iri, RDFS.label, Literal(label)))
            if snippet:
                g.add((new_iri, RDFS.comment, Literal(snippet)))
            g.add((new_iri, HAS_LAW_PRED, Literal(new_law_text)))
            g.add((new_iri, DCTERMS_MODIFIED, Literal(now, datatype=XSD.dateTime)))
            created.append(label)

    # Record a version marker for the law itself (so we can track
    # "what version of HB 2395 is in the graph right now?").
    law_marker = URIRef(base_iri + f"Law_{_slug(law_id)}")
    # Clear out any old lawVersion triples so only the newest stays
    for v in list(g.objects(law_marker, LAW_VERSION_PRED)):
        g.remove((law_marker, LAW_VERSION_PRED, v))
    g.add((law_marker, RDF.type, OWL.NamedIndividual))
    g.add((law_marker, RDFS.label, Literal(law_label)))
    g.add((law_marker, LAW_VERSION_PRED, Literal(new_version)))
    g.add((law_marker, DCTERMS_MODIFIED, Literal(now, datatype=XSD.dateTime)))

    return {
        "law_id": law_id,
        "law_label": law_label,
        "new_version": new_version,
        "updated_at": now,
        "classes_refreshed": refreshed,
        "classes_created": created,
    }


# ---------------------------------------------------------------
# STANDALONE DEMO
# ---------------------------------------------------------------
if __name__ == "__main__":
    BASE = "http://example.org/onto.owl#"
    g = Graph()

    # --- Seed: pretend the KG already has two Oregon classes in it
    data_breach = URIRef(BASE + "Data_Breach")
    g.add((data_breach, RDF.type, OWL.Class))
    g.add((data_breach, RDFS.label, Literal("Data Breach")))
    g.add((data_breach, HAS_LAW_PRED,
           Literal("Oregon HB 2395. Companies must notify users within 45 days.")))
    g.add((data_breach, HAS_LAW_PRED,
           Literal("NISTIR 8259. Manufacturers should log breach events.")))

    manufacturer = URIRef(BASE + "Manufacturer")
    g.add((manufacturer, RDF.type, OWL.Class))
    g.add((manufacturer, RDFS.label, Literal("Manufacturer")))
    g.add((manufacturer, HAS_LAW_PRED,
           Literal("Oregon HB 2395. Manufacturers must provide reasonable security.")))

    # --- Link a fake "company" to Manufacturer so we can prove that
    #     updating the regulation does NOT break the link.
    acme = URIRef(BASE + "AcmeCorp")
    g.add((acme, RDF.type, manufacturer))

    print("=== BEFORE update ===")
    print(f"  Total triples: {len(g)}")
    print("  Manufacturer hasLaw values:")
    for lit in g.objects(manufacturer, HAS_LAW_PRED):
        print(f"    - {lit}")
    print(f"  AcmeCorp type link still present? {(acme, RDF.type, manufacturer) in g}")

    # --- Apply an amendment to Oregon HB 2395 ---
    result = update_regulation(
        g,
        base_iri=BASE,
        law_label="Oregon HB 2395",
        law_id="OR_HB_2395",
        new_version="2026-amendment",
        updated_classes=[
            {"label": "Data Breach",
             "snippet": "2026 amendment tightens breach notification to 30 days."},
            {"label": "Manufacturer",
             "snippet": "2026 amendment adds IoT device security requirements."},
            {"label": "Biometric Identifier",
             "snippet": "NEW class added in 2026 amendment."},
        ],
    )

    print("\n=== Update result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== AFTER update ===")
    print(f"  Total triples: {len(g)}")
    print("  Manufacturer hasLaw values (note NIST annotation is still there):")
    for lit in g.objects(manufacturer, HAS_LAW_PRED):
        print(f"    - {lit}")
    print(f"  AcmeCorp type link still present? {(acme, RDF.type, manufacturer) in g}")

    # Confirm the brand-new class got created
    new_cls = _find_class_by_label(g, "Biometric Identifier")
    print(f"  New class 'Biometric Identifier' created at: {new_cls}")
