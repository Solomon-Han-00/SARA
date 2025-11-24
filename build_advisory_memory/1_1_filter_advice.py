import json

def load_jsonl_in_json(file_path):
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file.
        
    Returns:
        list: List of dictionaries loaded from the JSONL file.
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Ignore lines that start with '#' or are empty
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            records.append(json.loads(line.strip()))
    return records

# configuration
# Note: The paths to the proactive and reactive advice files are hardcoded here.
valid_ids_path = "test_valid_ids.json"
#proactive_file = "./proactive_advice_train_3k_no_overlap.jsonl"
reactive_file = "./reactive_advice_train_3k_no_overlap.jsonl"
#output_proactive_file = "./filtered_proactive_advice_3k.jsonl"
output_reactive_file = "./filtered_reactive_advice_3k.jsonl"

# step 1: get the valid ids
with open(valid_ids_path, 'r', encoding='utf-8') as f:
    valid_ids = json.load(f)
    valid_ids_s = set(valid_ids['success_ids'])
    valid_ids_f = set(valid_ids['failure_ids'])
print(f"Loaded {len(valid_ids_s)} success IDs and {len(valid_ids_f)} failure IDs from {valid_ids_path}")

# step 2: load proactive and reactive advice records
#proactive_records = load_jsonl_in_json(proactive_file)
reactive_records = load_jsonl_in_json(reactive_file)
#print(f"Loaded {len(proactive_records)} proactive records from {proactive_file}")
print(f"Loaded {len(reactive_records)} reactive records from {reactive_file}")

# step 3: filter proactive and reactive records based on valid IDs
#filtered_proactive = []
filtered_reactive = []


# visit reactive records
for record in reactive_records:
    if record.get('_id') in valid_ids_s:
        record['consequence'] = 'success'
        filtered_reactive.append(record)
    elif record.get('_id') in valid_ids_f:
        record['consequence'] = 'failure'
        filtered_reactive.append(record)

# Print the number of filtered records
print(f"Filtered {len(filtered_reactive)} reactive records")

# step 4: save the filtered records to new JSONL files
with open(output_reactive_file, 'w', encoding='utf-8') as f:
    for record in filtered_reactive:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')