import json
import argparse
import os
from utils_build import (
    add_to_advice_kb,
    find_similar_advice,
    load_jsonl_in_json,
    dump_jsonline
)

def build_advice_kb(reactive_final, output_file):
    """
    Build advice knowledge base from proactive and reactive advice tracking files.
    
    Args:
        proactive_final (list): List of proactive advice records.
        reactive_final (list): List of reactive advice records.
        output_file (str): Path to output knowledge base file
    """
    
    # Initialize knowledge base for advice
    advice_kb = {
        'reactive_keys': [],  # List of current_situation texts (keys for reactive advice)
        'reactive_key_embs': [],  # List of current_situation embeddings
        'reactive_records': []  # List of reactive advice records
    }
    
    
    # Process reactive advice records
    for i, record in enumerate(reactive_final):
        situation = record.get('current_situation', '')
        advice = record.get('final_advice', '')
        consequence = record.get('consequence', '')
        
        if not situation or not advice:
            print(f"Skipping reactive record {i} due to missing fields.")
            continue
        
        if consequence == "failure":
            print(f"Skipping reactive record {i} due to failure consequence.")
            continue
        
        # Add to reactive knowledge base
        add_to_advice_kb(record, advice_kb)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(reactive_final)} reactive records")
    
    # Print summary
    print(f"\nKnowledge Base Summary:")
    print(f"  Proactive advice records: {len(advice_kb['proactive_records'])}")
    print(f"  Reactive advice records: {len(advice_kb['reactive_records'])}")
    print(f"  Total unique proactive keys: {len(advice_kb['proactive_keys'])}")
    print(f"  Total unique reactive keys: {len(advice_kb['reactive_keys'])}")
    
    # Save the knowledge base
    print(f"\nSaving knowledge base to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f_kb:
        json.dump(advice_kb, f_kb, ensure_ascii=False, indent=4)
    
    print("Knowledge base saved successfully!")


def integrate_dict_list(advice_records, reflection_records):
    # extract the common '_id' field all three records
    _id_advice = [record.get('_id', '') for record in advice_records]
    _id_reflection = [record.get('_id', '') for record in reflection_records]
    common_ids = set(_id_advice) & set(_id_reflection)
    
    # Integrate records based on common '_id'
    integrated_records = []
    for _id in common_ids:
        advice_record = next((record for record in advice_records if record.get('_id') == _id), {})
        reflection_record = next((record for record in reflection_records if record.get('_id') == _id), {})

        integrated_record = advice_record.copy()
        integrated_record.update(reflection_record)
        integrated_records.append(integrated_record)
        
    return integrated_records

if __name__ == "__main__":
    # conversion to hardcoded values
    reactive_analyzed = "./analyzed_reactive_advice_3k.jsonl"
    reactive_reflected = "./reactive_reflected_results.jsonl"
    output_file = "advisory_memory.json"
    
    # load reactive advice files
    with open(reactive_analyzed, 'r', encoding='utf-8') as f:
        reactive_records_analyzed = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(reactive_records_analyzed)} reactive advice records from {reactive_analyzed}")
    with open(reactive_reflected, 'r', encoding='utf-8') as f:
        reactive_records_reflected = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(reactive_records_reflected)} reflected reactive advice records from {reactive_reflected}")
    
    # Integrate records
    reactive_final = integrate_dict_list(reactive_records_analyzed, reactive_records_reflected)
    print(f"Integrated {len(reactive_final)} reactive records.")

    # Build the memory
    build_advice_kb(reactive_final, output_file)
    print("Advisory memory built successfully!")
    print(f"Advisory memory saved to {output_file}")

