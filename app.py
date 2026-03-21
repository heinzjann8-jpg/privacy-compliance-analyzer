from flask import Flask, render_template, request, jsonify
from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal
from pathlib import Path
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv          
from groq import Groq                   
load_dotenv()                           

app = Flask(__name__)
# ---- PDF Upload Settings ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_MB = 10

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Config --------------------
CFG = json.load(open("config.json", "r", encoding="utf-8"))

ONTO_PATH = Path(CFG["ontology_path"]).resolve()
MANUFACTURER_CLS = URIRef(CFG["manufacturer_class_iri"])
POLICY_PROP = URIRef(CFG["policy_property_iri"])

TOP_K = int(CFG.get("top_k", 8))
LAW_PREDICATES = [URIRef(p) for p in CFG.get("law_annotation_predicates", [])]
LAWS = CFG.get("laws", [])
COVERAGE_THRESHOLD = float(CFG.get("coverage_threshold", 0.0))
MISSING_CLASS_SIM_THRESHOLD = COVERAGE_THRESHOLD
SIMILARITY_METHOD = CFG.get("similarity_method", "tfidf").lower()
EMBEDDING_MODEL_NAME = CFG.get("embedding_model_name", "all-MiniLM-L6-v2")
USER_ADDED_PROP = URIRef("http://example.org/onto.owl#userAddedManufacturer")

# Only show these manufacturers in dropdown
ALLOWED_MANUFACTURERS = {
    "ADT", "ATT", "Brainly", "Bloomberg",
    "Dodge", "Emerson", "EOS", "Fitbit",
    "NYTimes", "Panasonic", "Puma", "Ring",
    "UPS", "Verizon", "Vivint"
}
SPECIAL_NAME_FIXES = {
    "att": "AT&T",
    "adt": "ADT",
    "ups": "UPS",
    "nytimes": "NYTimes",
    "eos": "EOS",
    "ebay": "eBay"
}

def clean_manufacturer_name(name: str) -> str:
    """Normalize manufacturer labels so they look neat."""
    lower = name.lower().strip()
    if lower in SPECIAL_NAME_FIXES:
        return SPECIAL_NAME_FIXES[lower]
    return name.title()

def clean_class_label(label: str) -> str:
    """Fix known class capitalization issues."""
    fixes = {
        "network interface": "Network Interface",
        "iot device": "IoT Device",
        "io tdevice": "IoT Device",
        "iotdevice": "IoT Device",
    }
    lower = label.lower().strip()
    return fixes.get(lower, label)

def generate_manufacturer_iri(name: str) -> URIRef:
    """Generate a unique IRI for a new manufacturer based on its name."""
    safe_name = name.lower().replace(" ", "_").replace("&", "and")
    safe_name = re.sub(r'[^a-z0-9_]', '', safe_name)
    base_iri = "http://example.org/onto.owl#"
    return URIRef(f"{base_iri}{safe_name}")

# -------------------- Load graph --------------------
SWRL_COVERS = URIRef("http://example.org/onto.owl#swrl_covers")
INFERRED_PATH = ONTO_PATH.parent / "inferred_merged.rdf"
SWRL_ACTIVE = False

def run_swrl_reasoner() -> bool:
    """
    Manual SWRL-equivalent reasoning using rdflib.
    Fires keyword rules against policy_description and writes
    swrl_covers triples into the inferred ontology.
    Bypasses owlready2 entirely.
    """
    try:
        import shutil

        # SWRL rules — keyword → class IRI mapping
        SWRL_RULES = [
            ("authentication",      "http://yourdomain.org/ontology/laws#Authentication"),
            ("unique password",     "http://yourdomain.org/ontology/laws#Authentication"),
            ("encrypt",             "http://example.org/onto.owl#Security_Mechanism"),
            ("unauthorized access", "http://yourdomain.org/ontology/laws#Unauthorized_Access_Element"),
            ("software update",     "http://yourdomain.org/ontology/laws#Updates"),
            ("patch",               "http://example.org/onto.owl#PatchAvailability"),
            ("configuration",       "http://example.org/onto.owl#ConfigurationManagement"),
            ("data collection",     "http://yourdomain.org/ontology/laws#Data_Collection_Practice"),
            ("third party",         "http://yourdomain.org/ontology/laws#Data_Sharing_Practice"),
            ("personal information","http://yourdomain.org/ontology/laws#Personal_Information"),
            ("network",             "http://example.org/onto.owl#networkInterface"),
            ("secure development",  "http://example.org/onto.owl#SecureDevelopment"),
            ("cybersecurity",       "http://example.org/onto.owl#CybersecurityCapability"),
            ("security standard",   "http://example.org/onto.owl#MinimumSecurityStandards"),
            ("compliance",          "http://yourdomain.org/ontology/laws#Violation"),
        ]

        # Load raw ontology into working graph
        inferred_g = Graph()
        inferred_g.parse(str(ONTO_PATH))
        print(f"[swrl] Loaded {len(inferred_g)} triples from raw ontology")

        # Fire rules against every manufacturer
        rules_fired = 0
        for inst in inferred_g.subjects(RDF.type, MANUFACTURER_CLS):
            if not isinstance(inst, URIRef):
                continue
            policies = [
                str(o) for o in inferred_g.objects(inst, POLICY_PROP)
                if isinstance(o, Literal)
            ]
            if not policies:
                continue
            policy_text = max(policies, key=len).lower()

            for keyword, class_iri in SWRL_RULES:
                if keyword.lower() in policy_text:
                    inferred_g.add((
                        inst,
                        SWRL_COVERS,
                        URIRef(class_iri)
                    ))
                    rules_fired += 1

        print(f"[swrl] Fired {rules_fired} rule inferences")

        # Save inferred graph
        inferred_g.serialize(destination=str(INFERRED_PATH), format="xml")
        print(f"[swrl] Inferred ontology saved → {INFERRED_PATH}")
        return True

    except Exception as e:
        print(f"[swrl] Reasoner failed: {e}")
        return False

# -------------------- Load graph (with SWRL) --------------------
def is_valid_rdf(path: Path) -> bool:
    """Check if an RDF file exists and is non-empty and parseable."""
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 100:
            print(f"[swrl] {path.name} is empty or too small — discarding")
            path.unlink()
            return False
        test = Graph()
        test.parse(str(path))
        return len(test) > 0
    except Exception as e:
        print(f"[swrl] {path.name} failed validation: {e}")
        try:
            path.unlink()
        except Exception:
            pass
        return False

g = Graph()
SWRL_ACTIVE = False

if is_valid_rdf(INFERRED_PATH):
    print(f"[load] Loading valid inferred ontology → {INFERRED_PATH}")
    g.parse(str(INFERRED_PATH))
    SWRL_ACTIVE = True
else:
    print(f"[load] No valid inferred ontology found — attempting SWRL reasoning")
    success = run_swrl_reasoner()
    if success and is_valid_rdf(INFERRED_PATH):
        print(f"[load] Loading freshly inferred ontology → {INFERRED_PATH}")
        g.parse(str(INFERRED_PATH))
        SWRL_ACTIVE = True
    else:
        print(f"[load] Falling back to raw ontology → {ONTO_PATH}")
        g.parse(str(ONTO_PATH))
        SWRL_ACTIVE = False

print(f"[swrl] SWRL layer active: {SWRL_ACTIVE}")

# -------------------- Helpers --------------------
def local_name(iri: str) -> str:
    """Get a readable local name from a full IRI."""
    s = iri
    if "#" in s:
        s = s.rsplit("#", 1)[1]
    s = s.rstrip("/").rsplit("/", 1)[-1]
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ")
    return s

#----------look
def get_swrl_covered_classes(manufacturer_iri: str) -> set:
    """Returns set of class IRIs confirmed by SWRL rules."""
    if not SWRL_ACTIVE:
        return set()
    inst = URIRef(manufacturer_iri)
    return {str(obj) for obj in g.objects(inst, SWRL_COVERS)}

def get_hybrid_tier(bert_covered: bool, swrl_covered: bool) -> dict:
    """
    Returns hybrid confidence tier based on SWRL + BERT agreement.
    Tier 1: both agree covered       → HIGH
    Tier 2: SWRL yes, BERT no        → keyword present, weak context
    Tier 3: BERT yes, SWRL no        → semantic match, no keyword
    Tier 4: both agree not covered   → confirmed gap
    """
    if bert_covered and swrl_covered:
        return {"tier": 1, "label": "HIGH", "explanation": "Both SWRL and BERT confirm coverage"}
    elif swrl_covered and not bert_covered:
        return {"tier": 2, "label": "MEDIUM-SWRL", "explanation": "Keyword present but weak semantic context"}
    elif bert_covered and not swrl_covered:
        return {"tier": 3, "label": "MEDIUM-BERT", "explanation": "Semantic match without explicit keyword"}
    else:
        return {"tier": 4, "label": "NONE", "explanation": "Confirmed gap — neither method confirms coverage"}

def extract_law_segments(value: str):
    """Extract law-specific segments from text."""
    segments_by_law = {law["id"]: [] for law in LAWS}
    if not value:
        return {}

    pieces = re.split(r'(?<=[.!?])\s+|[\n\r]+|;+', value)
    for piece in pieces:
        plow = piece.lower().strip()
        if not plow:
            continue
        for law in LAWS:
            lid = law["id"]
            for kw in law.get("keywords", []):
                if kw.lower() in plow:
                    segments_by_law[lid].append(piece.strip())
                    break

    return {lid: segs for lid, segs in segments_by_law.items() if segs}

def filter_description_by_laws(description: str, allowed_law_ids: set) -> str:
        """Filter a description to only include text from specified laws."""
        if not description or not allowed_law_ids:
            return description
    
    # Law patterns to identify and extract
        law_patterns = {
        'CA_SB_327': r'California SB[- ]?327[^.]*\.',
        'OR_HB_2395': r'Oregon HB[- ]?2395[^.]*\.',
        'NISTIR_8259': r'NISTIR 8259[^.]*\.',
        'PL_116-207': r'(?:PL 116-207|Public Law 116-207)[^.]*\.'
    }
    
    # Extract only sentences that mention allowed laws
        filtered_parts = []
        for law_id in allowed_law_ids:
            if law_id in law_patterns:
                pattern = law_patterns[law_id]
            matches = re.finditer(pattern, description, re.IGNORECASE)
            for match in matches:
                filtered_parts.append(match.group(0))
    
        return ' '.join(filtered_parts) if filtered_parts else description

# -------------------- Build class corpus --------------------
class_iris = []
class_labels = []
class_texts = []
class_descs = {}
class_to_laws = {}

for c in g.subjects(RDF.type, OWL.Class):
    if not isinstance(c, URIRef):
        continue

    c_iri = str(c)
    law_ids_for_class = set()
    law_snippets = []

    for pred in LAW_PREDICATES:
        for obj in g.objects(c, pred):
            text = str(obj)
            segments_by_law = extract_law_segments(text)
            for lid, segs in segments_by_law.items():
                law_ids_for_class.add(lid)
                law_snippets.extend(segs)

    if not law_ids_for_class:
        continue

    labels = [str(o) for o in g.objects(c, RDFS.label)]
    comments = [str(o) for o in g.objects(c, RDFS.comment)]

    if not labels:
        labels = [clean_class_label(local_name(c_iri))]

    label_text = clean_class_label(" / ".join(labels))
    combined = " ".join(labels + comments + law_snippets + [local_name(c_iri)])

    if comments:
        class_desc = " ".join(comments).strip()
    else:
        class_desc = " ".join(law_snippets).strip()

    class_iris.append(c_iri)
    class_labels.append(label_text)
    class_texts.append(combined)
    class_descs[c_iri] = class_desc
    class_to_laws[c_iri] = law_ids_for_class

if not class_texts:
    raise SystemExit("No eligible classes found with law annotations matching configured keywords.")

# -------------------- DEDUPLICATE CLASSES --------------------
seen = set()
dedup_iris = []
dedup_labels = []
dedup_texts = []
dedup_descs = {}
dedup_laws = {}

for i, c_iri in enumerate(class_iris):
    label = class_labels[i]
    clean_label = label.lower().strip()

    if clean_label in seen:
        continue

    seen.add(clean_label)
    dedup_iris.append(c_iri)
    dedup_labels.append(label)
    dedup_texts.append(class_texts[i])
    dedup_descs[c_iri] = class_descs[c_iri]
    dedup_laws[c_iri] = class_to_laws[c_iri]

class_iris = dedup_iris
class_labels = dedup_labels
class_texts = dedup_texts
class_descs = dedup_descs
class_to_laws = dedup_laws

# -------------------- Class → laws & law → classes --------------------
iri_to_idx = {c_iri: idx for idx, c_iri in enumerate(class_iris)}
law_to_class_idxs = {law["id"]: [] for law in LAWS}

for c_iri, law_ids in class_to_laws.items():
    idx = iri_to_idx.get(c_iri)
    if idx is None:
        continue
    for lid in law_ids:
        if lid in law_to_class_idxs:
            law_to_class_idxs[lid].append(idx)

for law in LAWS:
    lid = law["id"]
    print(f"[init] Law {lid} has {len(law_to_class_idxs.get(lid, []))} related classes")

# -------------------- Build manufacturers list --------------------
manufacturers = []
for inst in g.subjects(RDF.type, MANUFACTURER_CLS):
    if not isinstance(inst, URIRef):
        continue
    
    inst_iri = str(inst)
    user_added = any(g.objects(inst, USER_ADDED_PROP))
    
    labels = [str(o) for o in g.objects(inst, RDFS.label)]
    if not labels:
        name = clean_manufacturer_name(local_name(inst_iri))
    else:
        name = clean_manufacturer_name(labels[0])
    
    if name not in ALLOWED_MANUFACTURERS and not user_added:
        continue
    
    policies = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
    policy = max(policies, key=len) if policies else ""
    
    manufacturers.append({
        "iri": inst_iri,
        "name": name,
        "policy": policy,
        "user_added": user_added
    })

manufacturers.sort(key=lambda m: m["name"].lower())
print(f"[init] Loaded {len(manufacturers)} manufacturers")

# -------------------- Prepare similarity model --------------------
if SIMILARITY_METHOD == "tfidf":
    print("[init] Using TF-IDF similarity")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000
    )
    class_matrix = vectorizer.fit_transform(class_texts)

elif SIMILARITY_METHOD == "bert":
    print(f"[init] Using BERT embeddings with model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    class_embeddings = model.encode(
        class_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

else:
    raise ValueError(f"Unknown similarity_method: {SIMILARITY_METHOD}")

# -------------------- Ranking --------------------
def rank_classes_for_policy(policy_text: str, state="all"):
    """Rank classes by similarity to policy text."""
    if not policy_text or not policy_text.strip():
        return [], None

    if SIMILARITY_METHOD == "tfidf":
        query_vec = vectorizer.transform([policy_text])
        sims = cosine_similarity(query_vec, class_matrix)[0]

    elif SIMILARITY_METHOD == "bert":
        q_emb = model.encode(
            [policy_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        sims = np.dot(class_embeddings, q_emb)

    else:
        raise ValueError(f"Unknown similarity_method: {SIMILARITY_METHOD}")

# Filter laws based on state
    filtered_laws = LAWS
    if state != "all":
        state_lower = state.lower()
        filtered_laws = []
        
        for law in LAWS:
            law_id_lower = law["id"].lower()
            law_label_lower = law.get("label", "").lower()
            
            # Match based on state filter
            if state_lower in law_id_lower or state_lower in law_label_lower:
                filtered_laws.append(law)
            elif state_lower == "california" and ("ca" in law_id_lower or "sb-327" in law_id_lower or "sb327" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "oregon" and ("or" in law_id_lower or "hb-2395" in law_id_lower or "hb2395" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "nistir" and ("nistir" in law_id_lower or "8259" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "plaw" and ("pl" in law_id_lower or "116-207" in law_id_lower or "public law" in law_label_lower):
                filtered_laws.append(law)

        # Get class indices that belong to filtered laws
    allowed_class_indices = set()
    if state == "all":
        allowed_class_indices = set(range(len(class_iris)))
    else:
        for law in filtered_laws:
            lid = law["id"]
            class_idxs = law_to_class_idxs.get(lid, [])
            allowed_class_indices.update(class_idxs)

    idxs = [i for i, s in enumerate(sims) if float(s) >= COVERAGE_THRESHOLD and i in allowed_class_indices]
    idxs.sort(key=lambda i: sims[i], reverse=True)

    top_classes = []
    for i in idxs:
        c_iri = class_iris[i]
        sim_score = float(sims[i])
        
        # Find which law(s) this class belongs to
        class_laws = class_to_laws.get(c_iri, set())
        law_labels = [law["label"] for law in LAWS if law["id"] in class_laws]
        
         # Filter description to only show text from filtered laws
        original_desc = class_descs.get(c_iri, "")
        filtered_desc = original_desc
        if state != "all" and class_laws:
            # Only show description text from laws that match the filter
            relevant_laws = class_laws.intersection({law["id"] for law in filtered_laws})
            filtered_desc = filter_description_by_laws(original_desc, relevant_laws)
        
        top_classes.append({
            "class_iri": c_iri,
            "class_label": class_labels[i],
            "class_desc": class_descs.get(c_iri, ""),
            "laws": list(class_laws),
            "law_labels": law_labels,
        })

    return top_classes, sims

def compute_law_coverage(sims, state="all"):
    """Compute coverage per law and which laws are missing entirely."""
    if sims is None or not LAWS:
        return [], [law["id"] for law in LAWS], 0.0

    # FIXED: Better state filtering logic
    filtered_laws = LAWS
    if state != "all":
        # Try multiple matching strategies
        state_lower = state.lower()
        filtered_laws = []
        
        for law in LAWS:
            law_id_lower = law["id"].lower()
            law_label_lower = law.get("label", "").lower()
            
            # Match if state name appears in ID or label
            if state_lower in law_id_lower or state_lower in law_label_lower:
                filtered_laws.append(law)
            # Also check for common abbreviations
            elif state_lower == "california" and ("ca" in law_id_lower or "sb-327" in law_id_lower or "sb327" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "oregon" and ("or" in law_id_lower or "hb-2395" in law_id_lower or "hb2395" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "nistir" and ("nistir" in law_id_lower or "8259" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "plaw" and ("pl" in law_id_lower or "116-207" in law_id_lower or "public law" in law_label_lower):
                filtered_laws.append(law)
    
    # DEBUG: Print what laws were found
    print(f"[DEBUG] State filter: {state}")
    print(f"[DEBUG] Filtered to {len(filtered_laws)} laws: {[law['id'] for law in filtered_laws]}")
    
    # If no laws matched the filter, return 0% coverage
    if not filtered_laws:
        print(f"[WARNING] No laws matched state filter '{state}'")
        return [], [], 0.0
    
    law_coverage = []
    missing_laws = []
    total_above = 0
    total_classes = 0
    
    for law in filtered_laws:
        lid = law["id"]
        class_idxs = law_to_class_idxs.get(lid, [])
        if not class_idxs:
            law_coverage.append({
                "id": lid,
                "label": law["label"],
                "coverage_percent": 0.0,
                "num_classes": 0,
                "num_above_threshold": 0,
            })
            missing_laws.append(lid)
            continue
        
        scores = [float(sims[i]) for i in class_idxs]
        above = [s for s in scores if s >= COVERAGE_THRESHOLD]
        coverage_percent = 100.0 * len(above) / len(class_idxs)
        
        total_above += len(above)
        total_classes += len(class_idxs)
        
        law_coverage.append({
            "id": lid,
            "label": law["label"],
            "coverage_percent": round(coverage_percent, 2),
            "num_classes": len(class_idxs),
            "num_above_threshold": len(above),
        })
        
        if len(above) == 0:
            missing_laws.append(lid)
    
    overall_percent = round(100.0 * total_above / total_classes, 2) if total_classes > 0 else 0.0
    
    # DEBUG: Print coverage calculation
    print(f"[DEBUG] Total classes for filtered laws: {total_classes}")
    print(f"[DEBUG] Classes above threshold: {total_above}")
    print(f"[DEBUG] Overall coverage: {overall_percent}%")
    
    return law_coverage, missing_laws, overall_percent

def compute_missing_classes(sims, state="all"):
    """Find missing classes (similarity below threshold)."""
    if sims is None:
        return []
    
    # FIXED: Use same filtering logic as compute_law_coverage
    filtered_laws = LAWS
    if state != "all":
        state_lower = state.lower()
        filtered_laws = []
        
        for law in LAWS:
            law_id_lower = law["id"].lower()
            law_label_lower = law.get("label", "").lower()
            
            if state_lower in law_id_lower or state_lower in law_label_lower:
                filtered_laws.append(law)
            elif state_lower == "california" and ("ca" in law_id_lower or "sb-327" in law_id_lower or "sb327" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "oregon" and ("or" in law_id_lower or "hb-2395" in law_id_lower or "hb2395" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "nistir" and ("nistir" in law_id_lower or "8259" in law_id_lower):
                filtered_laws.append(law)
            elif state_lower == "plaw" and ("pl" in law_id_lower or "116-207" in law_id_lower or "public law" in law_label_lower):
                filtered_laws.append(law)
    
    missing = []
    for law in filtered_laws:
        lid = law["id"]
        llabel = law["label"]
        class_idxs = law_to_class_idxs.get(lid, [])
        for idx in class_idxs:
            score = float(sims[idx])
            if score < COVERAGE_THRESHOLD:
                c_iri = class_iris[idx]

                # Filter description to only show text from filtered laws
                original_desc = class_descs.get(c_iri, "")
                filtered_desc = original_desc
                if state != "all":
                    class_laws = class_to_laws.get(c_iri, set())
                    relevant_laws = class_laws.intersection({law["id"] for law in filtered_laws})
                    filtered_desc = filter_description_by_laws(original_desc, relevant_laws)
                
                missing.append({
                    "class_iri": c_iri,
                    "class_label": class_labels[idx],
                    "class_desc": class_descs.get(c_iri, ""),
                    "law_id": lid,
                    "law_label": llabel,
                })
    return missing

# -------------------- SPARQL Validation --------------------
def sparql_validate_manufacturer(manufacturer_iri: str, sims) -> dict:
    """
    Cross-validates BERT decisions against keyword queries.
    Returns agreement rate and disagreement breakdown.
    """
    if sims is None:
        return {"error": "No sims", "agreement_rate": 0, "total_pairs": 0,
                "agreed": 0, "both_covered": 0, "both_not_covered": 0,
                "bert_only": 0, "sparql_only": 0}

    # Get policy text directly
    inst = URIRef(manufacturer_iri)
    policies = [str(o) for o in g.objects(inst, POLICY_PROP)
                if isinstance(o, Literal)]
    policy_text = max(policies, key=len).lower() if policies else ""

    if not policy_text:
        return {"error": "No policy", "agreement_rate": 0, "total_pairs": 0,
                "agreed": 0, "both_covered": 0, "both_not_covered": 0,
                "bert_only": 0, "sparql_only": 0}

    # Keyword map defined ONCE outside the loop
    keyword_map = {
        "authentication":            ["authentication", "password", "credential", "login", "unique", "verify", "identity"],
        "unauthorized access":       ["unauthorized", "access control", "breach", "intrusion", "restrict", "prohibited"],
        "security mechanism":        ["encryption", "encrypt", "secure", "cryptographic", "tls", "ssl", "protect", "safeguard"],
        "updates":                   ["update", "patch", "upgrade", "firmware", "maintenance", "release", "version"],
        "patch":                     ["patch", "update", "fix", "vulnerability", "hotfix", "software update"],
        "configuration management":  ["configuration", "config", "setting", "parameter", "setup", "manage"],
        "data collection practice":  ["data collection", "collect", "gather", "personal data", "information we collect"],
        "data sharing practice":     ["third party", "sharing", "disclose", "transfer", "partners", "affiliates", "vendor"],
        "personal information":      ["personal information", "personal data", "pii", "personally identifiable", "your information"],
        "network interface":         ["network", "internet", "connectivity", "wifi", "wireless", "bluetooth", "protocol"],
        "secure development":        ["secure development", "security testing", "vulnerability", "audit", "penetration"],
        "cybersecurity capability":  ["cybersecurity", "security", "cyber", "protection", "threat", "safeguard"],
        "minimum security standards":["security standard", "minimum security", "baseline", "comply", "requirement", "standard"],
        "core baseline":             ["baseline", "minimum", "standard", "requirement", "foundational", "essential"],
        "iot device":                ["device", "iot", "connected", "sensor", "smart device", "product", "hardware"],
        "iot platform":              ["platform", "cloud", "service", "infrastructure", "system", "backend", "ecosystem"],
        "violation":                 ["compliance", "comply", "violation", "enforce", "regulation", "legal", "obligation"],
        "non compliant":             ["non-compliant", "violation", "breach", "failure", "penalty", "fine"],
        "cybersecurity risk":        ["risk", "threat", "vulnerability", "exposure", "mitigation", "cyber risk"],
        "secure software":           ["software", "application", "code", "program", "secure code", "development"],
        "patch availability":        ["patch", "update available", "security fix", "software patch", "release"],
        "conventional it device":    ["device", "computer", "server", "hardware", "equipment", "machine"],
        "transducer":                ["sensor", "transducer", "actuator", "detector", "measurement", "monitor"],
        "standards":                 ["standard", "nist", "iso", "compliance", "framework", "guideline", "requirement"],
    }

    agreed = 0
    total = 0
    both_covered = 0
    both_not_covered = 0
    bert_only = 0
    sparql_only = 0

    for idx, c_iri in enumerate(class_iris):
        label = class_labels[idx]
        raw_label = label.lower().replace("/", " ").strip()
        bert_score = float(sims[idx])
        bert_decision = bert_score >= COVERAGE_THRESHOLD

        # Get keywords for this class
        expanded = keyword_map.get(raw_label, None)
        if not expanded:
            for key in keyword_map:
                if key in raw_label or raw_label in key:
                    expanded = keyword_map[key]
                    break
        if not expanded:
            words = [w for w in raw_label.split() if len(w) > 3]
            expanded = words if words else [raw_label[:10]]

        # Search policy text directly using Python string matching
        sparql_result = any(kw in policy_text for kw in expanded)

        # Count results
        total += 1
        if bert_decision == sparql_result:
            agreed += 1

        if bert_decision and sparql_result:
            both_covered += 1
        elif not bert_decision and not sparql_result:
            both_not_covered += 1
        elif bert_decision and not sparql_result:
            bert_only += 1
        else:
            sparql_only += 1

    agreement_rate = round(100.0 * agreed / total, 2) if total > 0 else 0.0

    return {
        "total_pairs": total,
        "agreed": agreed,
        "agreement_rate": agreement_rate,
        "both_covered": both_covered,
        "both_not_covered": both_not_covered,
        "bert_only": bert_only,
        "sparql_only": sparql_only
    }

# -------------------- Routes --------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/list")
def list_instances():
    """Return all manufacturers for the dropdown."""
    return jsonify([{"iri": m["iri"], "name": m["name"]} for m in manufacturers])

@app.get("/detail")
def detail():
    iri = request.args.get("iri")
    state = request.args.get("state", "all")

    if not iri:
        return jsonify({"error": "missing iri"}), 400

    match = next((m for m in manufacturers if m["iri"] == iri), None)
    if not match:
        inst = URIRef(iri)
        labels = [str(o) for o in g.objects(inst, RDFS.label)]
        name = clean_manufacturer_name(labels[0]) if labels else clean_manufacturer_name(local_name(iri))
        policies = [str(o) for o in g.objects(inst, POLICY_PROP) if isinstance(o, Literal)]
        policy = max(policies, key=len) if policies else ""
        match = {"iri": iri, "name": name, "policy": policy}

    top_classes, sims = rank_classes_for_policy(match["policy"], state)
    law_coverage, missing_laws, overall_percent = compute_law_coverage(sims, state)
    missing_classes = compute_missing_classes(sims, state)

    # SWRL hybrid tiers for top classes
    swrl_covered = get_swrl_covered_classes(iri)
    for cls in top_classes:
        bert_covered = True  # already above threshold to be in top_classes
        swrl_decision = cls["class_iri"] in swrl_covered
        cls["hybrid"] = get_hybrid_tier(bert_covered, swrl_decision)

    # SPARQL auto-validation — runs on every detail call
    sparql_validation = sparql_validate_manufacturer(iri, sims)

    return jsonify({
        "iri": match["iri"],
        "name": match["name"],
        "policy": match["policy"],
        "suggested_classes": top_classes,
        "law_coverage": law_coverage,
        "missing_laws": missing_laws,
        "missing_classes": missing_classes,
        "overall_coverage": overall_percent,
        "sparql_validation": sparql_validation,
        "swrl_active": SWRL_ACTIVE,
    })

@app.post("/add_manufacturer")
def add_manufacturer():
    """Create a new Manufacturer individual in the ontology."""
    data = request.get_json(force=True) or {}

    name = (data.get("name") or "").strip()
    policy = (data.get("policy") or "").strip()

    if not name or not policy:
        return jsonify({"error": "Both name and policy are required."}), 400

    inst_iri = generate_manufacturer_iri(name)

    g.add((inst_iri, RDF.type, MANUFACTURER_CLS))
    g.add((inst_iri, RDFS.label, Literal(name)))
    g.add((inst_iri, POLICY_PROP, Literal(policy)))
    g.add((inst_iri, USER_ADDED_PROP, Literal(True)))

    new_entry = {
        "iri": str(inst_iri),
        "name": clean_manufacturer_name(name),
        "policy": policy,
        "user_added": True,
    }
    manufacturers.append(new_entry)
    manufacturers.sort(key=lambda m: m["name"].lower())

    try:
        g.serialize(destination=str(ONTO_PATH), format="xml")
    except Exception as e:
        print("[error] Failed to save ontology:", e)
        return jsonify(
            {"error": "Manufacturer added in memory, but failed to update ontology file."}
        ), 500

    return jsonify({"iri": new_entry["iri"], "name": new_entry["name"]}), 201

@app.post("/extract_pdf")
def extract_pdf():
    """Accept a PDF upload and return extracted text as JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file field found."}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
        extracted = "\n\n".join(texts).strip()
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {e}"}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    if not extracted:
        return jsonify({"error": "No text could be extracted from that PDF (it may be scanned images)."}), 200

    return jsonify({"text": extracted}), 200

@app.get("/sparql_validate_all")
def sparql_validate_all():
    """
    Run SPARQL vs BERT cross-validation for ALL manufacturers.
    Use this to generate your paper's agreement rate table.
    """
    results = []
    for m in manufacturers:
        _, sims = rank_classes_for_policy(m["policy"], "all")
        if sims is None:
            continue
        result = sparql_validate_manufacturer(m["iri"], sims)
        result["name"] = m["name"]
        results.append(result)

    # Compute overall agreement rate across all manufacturers
    total_pairs = sum(r["total_pairs"] for r in results)
    total_agreed = sum(r["agreed"] for r in results)
    overall_rate = round(
        100.0 * total_agreed / total_pairs, 2
    ) if total_pairs > 0 else 0.0

    return jsonify({
        "overall_agreement_rate": overall_rate,
        "total_pairs_evaluated": total_pairs,
        "total_agreed": total_agreed,
        "manufacturers": results
    }), 200

# ------------------------------------------------------------------aliases--------------------------------------------------------------------------
def expand_question_with_aliases(question: str) -> str:
    """Expand user questions with domain-specific aliases."""
    aliases = {
        # Password & Authentication
        "password": "password authentication credential login unique default",
        "login": "login authentication credential password access",
        "credential": "credential password authentication login",
        "auth": "authentication credential password login verification",
        
        # Updates & Patching
        "update": "update patch upgrade firmware software maintenance",
        "patch": "patch update upgrade fix software",
        "upgrade": "upgrade update patch firmware software",
        
        # Encryption & Security
        "encryption": "encryption encrypted secure crypto cryptographic protect",
        "encrypt": "encryption encrypted secure crypto cryptographic",
        "secure": "secure security encryption encrypted protection",
        "security": "security cybersecurity protection secure safe mechanism",
        
        # Data & Privacy
        "data protection": "data protection privacy security information safeguard compliance",
        "data privacy": "data privacy protection information personal confidential",
        "personal data": "personal data privacy information user sensitive",
        "data collection": "data collection gathering information privacy personal",
        "data": "data information privacy collection personal user",
        "privacy": "privacy data information personal protection collection",
        "personal protection": "personal data privacy information user",
        "collection": "collection data information gathering privacy",
        "protection": "protection security safeguard privacy data secure",
        
        # Network & Communication
        "network": "network communication interface connectivity protocol",
        "communication": "communication network interface connectivity protocol",
        "interface": "interface network communication connectivity",
        
        # Device & IoT
        "device": "device iot connected smart thing sensor transducer",
        "iot": "iot device connected smart internet things",
        "sensor": "sensor device transducer iot measurement",
        "smart": "smart iot device connected intelligent",
        
        # Access & Control
        "access control": "access control permission authorization authentication security",
        "access": "access control permission authorization authentication",
        "control": "control access management permission authorization",
    }
    
    expanded = question.lower()
    
    # Sort by length (longest first) to match multi-word phrases before single words
    sorted_keys = sorted(aliases.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        if key in expanded:
            expanded += " " + aliases[key]
    
    return expanded

def detect_question_intent(question: str) -> str:
    """Detects what kind of question the user is asking."""
    q = question.lower()

    if any(w in q for w in [
        "compare", "versus", "vs", "better", "worse", "difference"
    ]):
        return "compare"

    if any(w in q for w in [
        "missing", "gap", "fail", "lack", "not cover",
        "improve", "fix", "add to"
    ]):
        return "missing"

    if any(w in q for w in [
        "score", "percent", "coverage", "how much",
        "overall", "compliant", "compliance"
    ]):
        return "score"

    if any(w in q for w in [
        "what is", "explain", "what does", "require",
        "mean", "define", "describe"
    ]):
        return "explain"

    if any(w in q for w in [
        "which law", "what law", "under what", "which regulation"
    ]):
        return "law_lookup"

    if any(w in q for w in [
        "how many", "list", "all manufacturers", "which companies"
    ]):
        return "general_kg"

    return "class_search"  # default — falls through to BERT


def route_chat_by_intent(
    intent: str,
    question: str,
    manufacturer: dict,
    sims
) -> str:
    """
    Returns a dynamic answer string based on detected intent.
    Returns None if intent is class_search — lets existing
    BERT matching in /chat handle it instead.
    """
    name = manufacturer["name"]

    if intent == "score":
        law_coverage, _, overall = compute_law_coverage(
            sims, "all"
        )
        breakdown = ", ".join([
            f"{lc['label']}: {lc['coverage_percent']}%"
            for lc in law_coverage
        ])
        return (
            f"{name} has an overall compliance score of "
            f"{overall}%. Breakdown by law: {breakdown}."
        )

    elif intent == "missing":
        missing = compute_missing_classes(sims, "all")
        if not missing:
            return (
                f"{name} has no missing compliance classes "
                f"above the current threshold."
            )
        items = "\n".join([
            f"• {m['class_label']} ({m['law_label']})"
            for m in missing[:5]
        ])
        return (
            f"{name} is missing {len(missing)} requirements. "
            f"Top gaps:\n{items}"
        )

    elif intent == "general_kg":
        return (
            f"The knowledge graph contains "
            f"{len(manufacturers)} manufacturers, "
            f"{len(class_iris)} regulatory classes, "
            f"and {len(LAWS)} laws: "
            f"{', '.join(law['label'] for law in LAWS)}."
        )

    elif intent == "law_lookup":
        for law in LAWS:
            if law["label"].lower() in question.lower():
                idxs = law_to_class_idxs.get(law["id"], [])
                classes = [class_labels[i] for i in idxs[:5]]
                return (
                    f"{law['label']} covers "
                    f"{len(idxs)} regulatory classes "
                    f"including: {', '.join(classes)}."
                )
        return (
            "Please specify which law you are asking about. "
            f"Available laws: "
            f"{', '.join(law['label'] for law in LAWS)}."
        )

    return None  # class_search let BERT handle it

# -------------------- Ollama KG Context --------------------
def build_kg_context(question: str, manufacturer: dict, sims) -> str:
    """
    Assembles knowledge graph context for LLaMA3.
    Includes BERT scores, SWRL facts, law coverage, gaps.
    """
    name = manufacturer["name"]
    iri = manufacturer["iri"]

    # Law coverage
    law_coverage, _, overall = compute_law_coverage(sims, "all")
    coverage_lines = "\n".join([
        f"  {lc['label']}: {lc['coverage_percent']}% "
        f"({lc['num_above_threshold']}/{lc['num_classes']} classes)"
        for lc in law_coverage
    ])

    # Missing classes (top 10)
    missing = compute_missing_classes(sims, "all")
    missing_lines = "\n".join([
        f"  - {m['class_label']} ({m['law_label']})"
        for m in missing[:10]
    ]) or "  None"

    # SWRL confirmed classes
    swrl_covered = get_swrl_covered_classes(iri)
    swrl_labels = []
    for idx, c_iri in enumerate(class_iris):
        if c_iri in swrl_covered:
            swrl_labels.append(class_labels[idx])
    swrl_lines = ", ".join(swrl_labels[:10]) or "None confirmed"

    # Top BERT covered classes
    top_idxs = [i for i, s in enumerate(sims) if float(s) >= COVERAGE_THRESHOLD]
    top_idxs = sorted(top_idxs, key=lambda i: sims[i], reverse=True)[:10]
    bert_lines = ", ".join([class_labels[i] for i in top_idxs]) or "None"

    # Policy excerpt
    policy_excerpt = manufacturer["policy"][:600] if manufacturer["policy"] else "Not available"

    context = f"""
MANUFACTURER: {name}
OVERALL COMPLIANCE SCORE: {overall}%

LAW COVERAGE BREAKDOWN:
{coverage_lines}

BERT CONFIRMED COMPLIANT CLASSES:
{bert_lines}

SWRL RULE-CONFIRMED CLASSES:
{swrl_lines}

NON-COMPLIANT CLASSES (gaps):
{missing_lines}

POLICY EXCERPT:
{policy_excerpt}

AVAILABLE LAWS: {', '.join(law['label'] for law in LAWS)}
COVERAGE THRESHOLD: {COVERAGE_THRESHOLD}
"""
    return context.strip()


groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_llm(question: str, kg_context: str) -> str:
    """
    Send question + KG context to LLaMA3 via Groq cloud API.
    Always on — no local Ollama needed.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a friendly IoT privacy compliance assistant helping 
non-technical users understand how well a company protects customer data.

When answering questions:
- Never use technical terms like 'class', 'ontology', 'IRI', 'triple', or 'node'
- Instead of 'class' say 'requirement' or 'area'
- Instead of 'Cybersecurity Capability' say 'cybersecurity protections'
- Instead of 'Patch Availability' say 'keeping software up to date'
- Instead of 'IoT Platform' say 'connected device platform security'
- Instead of 'Configuration Management' say 'device settings security'
- Instead of 'networkInterface' say 'network security'
- Instead of 'PatchAvailability' say 'software update practices'
- Explain what each missing requirement actually means in plain English
- Reference laws by their common name: 'California privacy law', 
  'Oregon security law', 'federal IoT security law', or 'NIST guidelines'
- Keep answers conversational and easy to understand
- Answer ONLY using the knowledge graph context provided
- Do not make up information not present in the context
- Keep all answers to 3 sentences maximum
- No analogies or metaphors — just state the facts directly"""
                },
                {
                    "role": "user",
                    "content": f"""KNOWLEDGE GRAPH CONTEXT:
{kg_context}

QUESTION: {question}

Please answer in plain English as if explaining to someone with no technical background.
ANSWER:"""
                }
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Chat service error: {str(e)}"


@app.post("/chat")
def chat():
    """
    Ad-hoc compliance chatbot powered by LLaMA3 via Ollama.
    Uses BERT scores + SWRL facts + SPARQL validation as context.
    """
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    iri = (data.get("iri") or "").strip()

    if not question or not iri:
        return jsonify({"answer": "Please select a manufacturer and ask a question."}), 400

    match = next((m for m in manufacturers if m["iri"] == iri), None)
    if not match:
        return jsonify({"answer": "Manufacturer not found."}), 404

    # Get BERT sims for this manufacturer
    _, sims = rank_classes_for_policy(match["policy"], "all")
    if sims is None:
        return jsonify({"answer": "Could not analyze this manufacturer's policy."}), 500

    # Build KG context and send to LLaMA3
    kg_context = build_kg_context(question, match, sims)
    answer = ask_llm(question, kg_context)

    return jsonify({
        "answer": answer,
        "show_card": False
    }), 200

@app.get("/compare")
def compare():
    """
    Compare two manufacturers side by side.
    GET /compare?iri1=<iri>&iri2=<iri>&state=all
    """
    iri1 = request.args.get("iri1")
    iri2 = request.args.get("iri2")
    state = request.args.get("state", "all")

    m1 = next((m for m in manufacturers if m["iri"] == iri1), None)
    m2 = next((m for m in manufacturers if m["iri"] == iri2), None)

    if not m1 or not m2:
        return jsonify({"error": "One or both manufacturers not found"}), 404

    _, sims1 = rank_classes_for_policy(m1["policy"], state)
    _, sims2 = rank_classes_for_policy(m2["policy"], state)

    coverage1, _, overall1 = compute_law_coverage(sims1, state)
    coverage2, _, overall2 = compute_law_coverage(sims2, state)

    missing1 = compute_missing_classes(sims1, state)
    missing2 = compute_missing_classes(sims2, state)

    # Find classes one covers but the other doesn't
    covered1_iris = {
        class_iris[i] for i in range(len(class_iris))
        if sims1 is not None and float(sims1[i]) >= COVERAGE_THRESHOLD
    }
    covered2_iris = {
        class_iris[i] for i in range(len(class_iris))
        if sims2 is not None and float(sims2[i]) >= COVERAGE_THRESHOLD
    }

    m1_advantage = [
        class_labels[class_iris.index(iri)]
        for iri in covered1_iris - covered2_iris
    ]
    m2_advantage = [
        class_labels[class_iris.index(iri)]
        for iri in covered2_iris - covered1_iris
    ]

    return jsonify({
        "manufacturer1": {
            "name": m1["name"],
            "overall": overall1,
            "law_coverage": coverage1,
            "missing_count": len(missing1),
            "advantages_over_other": m1_advantage[:5]
        },
        "manufacturer2": {
            "name": m2["name"],
            "overall": overall2,
            "law_coverage": coverage2,
            "missing_count": len(missing2),
            "advantages_over_other": m2_advantage[:5]
        }
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
