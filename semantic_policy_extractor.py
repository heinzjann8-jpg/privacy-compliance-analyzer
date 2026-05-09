"""
We have list of company documents listed under the folder "company_list" containing IoT device manufacturers and their policy guidelines related to enhancing cybersecurity capabilities.

Your task is to extract:
1. Key Entities or Classes: Identify the main categories or entities (e.g., "IoT Devices," "Manufacturers," "Federal Agencies," "Cybersecurity Standards," etc.) that the document focuses on in each section.
2. Relationships: Identify in triples how these entities are connected or interact with one another as per the document. For example:
   - "Manufacturers must adhere to Cybersecurity Standards."

Output Format:
1. Entity/Class Name:
2. Relationships: in triples of [subject, predicate, object]
      [Entity A → Relationship → Entity B]
      [Entity C → Relationship → Entity D]

Context: {context}

Please go through all documents and extensively identify the key entities or classes and their respective relationships based on the provided context from the policy sentences in all files. Create another column for entity and relationship identified respectively for each sentence in the files.
"""


import os
import csv
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example ontology entities and relationships (should be loaded from ontology in production)
ontology_entities = [
    'IoT Device', 'Manufacturer', 'Federal Agency', 'Cybersecurity Standard', 'Personal Information', 'Service Provider', 'Customer', 'Website', 'Privacy Policy', 'Data Breach', 'Children', 'Email', 'Transaction', 'Cookie', 'User', 'Data Processor', 'Data Collector', 'Law', 'Policy', 'Affiliate', 'Subsidiary', 'Division', 'Franchisee'
]
ontology_relationships = [
    'must adhere to', 'collects', 'uses', 'shares', 'protects', 'provides', 'requires', 'complies with', 'notifies', 'markets to', 'removes', 'processes', 'stores', 'encrypts', 'discloses', 'tracks', 'offers', 'includes', 'agrees to', 'controls', 'enforces', 'covers', 'describes', 'receives', 'removes', 'contacts', 'owns', 'operates', 'contains', 'is subject to', 'is governed by', 'is responsible for', 'is not responsible for', 'is regulated by', 'is enforced by', 'is collected by', 'is shared with', 'is used by', 'is protected by', 'is provided by', 'is required by', 'is complied with by', 'is notified by', 'is marketed to by', 'is removed by', 'is processed by', 'is stored by', 'is encrypted by', 'is disclosed by', 'is tracked by', 'is offered by', 'is included by', 'is agreed to by', 'is controlled by', 'is enforced by', 'is covered by', 'is described by', 'is received by', 'is contacted by', 'is owned by', 'is operated by'
]

company_list_dir = 'company_list'

def process_file(filename):
    if not filename.endswith('.csv'):
        return None
    filepath = os.path.join(company_list_dir, filename)
    rows = []
    with open(filepath, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        # Ensure header contains required fields
        required_fields = ['company_name', 'policy_sentence', 'entity', 'relationship']
        header = reader.fieldnames if reader.fieldnames else []
        # If header is missing or incomplete, fix it
        if not header or 'policy_sentence' not in header:
            # Try to read as plain CSV and reconstruct rows
            infile.seek(0)
            raw_rows = list(csv.reader(infile))
            for raw_row in raw_rows:
                if len(raw_row) < 2:
                    continue
                row = {'company_name': raw_row[0], 'policy_sentence': raw_row[1]}
                rows.append(row)
        else:
            for row in reader:
                rows.append(row)
        # Now process each row for entity/relationship
    rows = []
    expected_fields = ['company_name', 'policy_sentence', 'entity', 'relationship']
    with open(filepath, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        # If header is missing or malformed, fix it
        if reader.fieldnames is None or any(f not in reader.fieldnames for f in expected_fields[:2]):
            # Try to recover: assume first two columns are company_name, policy_sentence
            # Read all lines, prepend header
            infile.seek(0)
            lines = infile.readlines()
            if lines:
                # Write back with correct header
                with open(filepath, 'w', encoding='utf-8', newline='') as outfile:
                    outfile.write(','.join(expected_fields) + '\n')
                    for line in lines:
                        outfile.write(line)
            # Re-open and re-read
            with open(filepath, 'r', encoding='utf-8', newline='') as infile2:
                reader = csv.DictReader(infile2)
        for row in reader:
            sentence = row.get('policy_sentence', '')
            # Semantic search for entity
            entity_scores = util.cos_sim(model.encode(sentence, convert_to_tensor=True), model.encode(ontology_entities, convert_to_tensor=True))[0]
            best_entity_idx = int(entity_scores.argmax())
            best_entity = ontology_entities[best_entity_idx]
            # Semantic search for relationship
            rel_scores = util.cos_sim(model.encode(sentence, convert_to_tensor=True), model.encode(ontology_relationships, convert_to_tensor=True))[0]
            best_rel_idx = int(rel_scores.argmax())
            best_rel = ontology_relationships[best_rel_idx]
            # For triple extraction, try to find another entity in the sentence
            sorted_entity_indices = entity_scores.argsort(descending=True)
            if len(sorted_entity_indices) > 1:
                second_entity = ontology_entities[int(sorted_entity_indices[1])]
            else:
                second_entity = ''
            triple = f"[{best_entity} → {best_rel} → {second_entity}]" if second_entity else ''
            # Only keep expected fields
            clean_row = {k: row.get(k, '') for k in expected_fields}
            clean_row['entity'] = best_entity
            clean_row['relationship'] = triple
            rows.append(clean_row)
    # Write back with new columns
    if rows:
        with open(filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=expected_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return filename
    return filename

files = [f for f in os.listdir(company_list_dir) if f.endswith('.csv')]
batch_size = 32
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, filename) for filename in files]
    for future in as_completed(futures):
        fname = future.result()
        if fname:
            print(f"Processed: {fname}")
print('Semantic entity and relationship extraction complete for all company policy files.')
