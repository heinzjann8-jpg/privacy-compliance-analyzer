"""


The web page needs two things:
  (a) A place the user can CHECK off which states the company
      collects data from (manual override).
  (b) An AUTOMATIC scanner that reads the privacy policy text
      and guesses which states apply based on keywords/phrases.

This module exposes:

    detect_states_from_text(policy_text)  ->  dict
    applicable_laws(states, all_laws)     ->  list of law dicts

It is pure Python — no Flask — so it can be unit-tested and
demoed on its own before we plug it into app.py.

Run as a standalone demo:
    python step2_state_detector.py
"""

import re
from typing import Dict, List


# ---------------------------------------------------------------
# State keyword catalog.
# Each entry has:
#   - "id"       : short code used in config and law mapping
#   - "label"    : display name
#   - "phrases"  : strings we look for in the policy text
#                  (case-insensitive; word-boundary match)
#   - "laws"     : IDs of regulations that apply to this state
#                  (must match IDs in config.json -> "laws")
# ---------------------------------------------------------------
STATE_CATALOG = [
    {
        "id": "OR",
        "label": "Oregon",
        "phrases": [
            "oregon", "oregon resident", "oregon user",
            "oregon consumer", "or residents",
        ],
        "laws": ["OR_HB_2395"],
    },
    {
        "id": "CA",
        "label": "California",
        "phrases": [
            "california", "california resident", "california consumer",
            "ccpa", "cpra", "california privacy rights act",
            "california consumer privacy act",
        ],
        "laws": ["CA_SB_327"],
    },
    {
        "id": "TX",
        "label": "Texas",
        "phrases": [
            "texas", "texas resident", "texas consumer",
            "texas data privacy and security act", "tdpsa",
        ],
        # If you register a Texas law in config (e.g. TX_HB_4),
        # put its ID here. Leaving empty is fine.
        "laws": [],
    },
    {
        "id": "US_FED",
        "label": "United States (Federal)",
        "phrases": [
            "united states", "u.s.", "us residents", "federal",
            "coppa", "children's online privacy",
        ],
        "laws": ["IoT_Cyber_Act_2020", "NISTIR_8259"],
    },
]


# ---------------------------------------------------------------
# (a) AUTOMATIC detection from raw policy text
# ---------------------------------------------------------------
def detect_states_from_text(policy_text: str) -> Dict:
    """
    Scan the policy text and return:

        {
            "detected": [
                {"id": "OR", "label": "Oregon",
                 "matched_phrases": ["oregon resident", "oregon user"]},
                ...
            ],
            "not_detected": [
                {"id": "TX", "label": "Texas"},
                ...
            ]
        }

    A state is considered "detected" if at least one of its phrases
    appears in the policy text.
    """
    text = (policy_text or "").lower()
    detected, not_detected = [], []

    for state in STATE_CATALOG:
        hits = []
        for phrase in state["phrases"]:
            # word-boundary match so "oregon" doesn't match "oregonite"
            if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text):
                hits.append(phrase)

        if hits:
            detected.append({
                "id": state["id"],
                "label": state["label"],
                "matched_phrases": hits,
            })
        else:
            not_detected.append({"id": state["id"], "label": state["label"]})

    return {"detected": detected, "not_detected": not_detected}


# ---------------------------------------------------------------
# (b) Given a list of selected states, return which laws apply
# ---------------------------------------------------------------
def applicable_laws(state_ids: List[str], all_laws: List[Dict]) -> List[Dict]:
    """
    state_ids   : list of state codes the user selected (or auto-detected)
                  e.g. ["OR", "TX"]
    all_laws    : the full "laws" list from config.json

    Returns the subset of all_laws whose IDs are referenced by any
    of the selected states.
    """
    law_ids_needed = set()
    for s in STATE_CATALOG:
        if s["id"] in state_ids:
            law_ids_needed.update(s["laws"])

    return [law for law in all_laws if law["id"] in law_ids_needed]


# ---------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------
def analyze(policy_text: str, all_laws: List[Dict], user_selected_states=None):
    """
    One call that returns everything the UI needs.

    user_selected_states: if the user manually checks boxes on the
    web page, pass those here to override the auto-detection.
    """
    auto = detect_states_from_text(policy_text)
    auto_ids = [s["id"] for s in auto["detected"]]

    # If the user manually picked states, trust them. Otherwise use auto.
    final_ids = user_selected_states if user_selected_states else auto_ids
    final_ids = list(final_ids) if final_ids else []

    return {
        "auto_detection": auto,
        "selected_states": final_ids,
        "applicable_laws": applicable_laws(final_ids, all_laws),
    }


# ---------------------------------------------------------------
# STANDALONE DEMO
# ---------------------------------------------------------------
if __name__ == "__main__":
    import json

    demo_policy_oregon_only = (
        "This privacy policy applies to our services. We collect data "
        "from Oregon residents who use our website. Oregon HB 2395 "
        "governs our data breach notifications."
    )
    demo_policy_oregon_and_texas = (
        "We collect information from Oregon residents and Texas residents "
        "who interact with our connected devices. We comply with applicable "
        "state law."
    )
    demo_policy_california = (
        "California consumers have rights under the California Consumer "
        "Privacy Act (CCPA). Residents of California may opt out of sale."
    )

    with open("config.json", "r", encoding="utf-8") as f:
        laws = json.load(f).get("laws", [])

    for name, txt in [
        ("Oregon only", demo_policy_oregon_only),
        ("Oregon + Texas", demo_policy_oregon_and_texas),
        ("California", demo_policy_california),
    ]:
        print(f"\n--- {name} ---")
        result = analyze(txt, laws)
        print(f"  Detected states : {[s['label'] for s in result['auto_detection']['detected']]}")
        print(f"  Applicable laws : {[l['label'] for l in result['applicable_laws']]}")
