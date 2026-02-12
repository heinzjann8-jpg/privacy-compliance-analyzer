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
g = Graph()
print(f"[load] {ONTO_PATH}")
g.parse(str(ONTO_PATH))

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
    """
    For a selected manufacturer:
    - Return its policy text
    - Top class matches
    - Law coverage & missing laws
    - Missing classes
    """
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

    return jsonify({
        "iri": match["iri"],
        "name": match["name"],
        "policy": match["policy"],
        "suggested_classes": top_classes,
        "law_coverage": law_coverage,
        "missing_laws": missing_laws,
        "missing_classes": missing_classes,
        "overall_coverage": overall_percent,
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
