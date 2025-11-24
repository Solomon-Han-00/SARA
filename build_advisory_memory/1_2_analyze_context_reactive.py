from models import SnowClient
import json

# configs
model_name = 'llama3.1-70b'
param_path = './snow_params.json'
with open(param_path, 'r', encoding='utf-8') as f:
    params = json.load(f)
snow_client = SnowClient(connection_params=params)

def analyze_context_reactive(situation, model=model_name):
    """
    Analyze the context proactively using the SnowClient.
    
    Args:
        query (str): The query to analyze.
        model (str): The model to use for analysis.
    
    Returns:
        str: The response from the model.
    """
    
    prompt_format = '''Analyze the given context, which consists of a query and its reasoning trajectory (termed as "current situation"), to provide a high-level semantic summary. The analysis should capture underlying logical structures, critical reasoning issues, core needs, and overall strategic intent present in the trajectory. This summary will serve as a key for future retrieval, aiding in identifying similar logical reasoning trajectories.

Your analysis should:

Clearly summarize the main logical reasoning path and its purpose.

Identify critical logical issues or points of reasoning that arise.

Highlight the primary needs or objectives driving the reasoning trajectory.

Categorize the trajectory into relevant logical or strategic patterns.

Specify any implicit assumptions, contextual constraints, or underlying knowledge necessary to interpret the trajectory fully.

Context:
Current Situation (Reasoning Trajectory):
"{given_situation}"
    '''
    prompt = prompt_format.format(given_situation=situation)
    # Call the SnowClient with the formatted prompt
    response = snow_client(prompt, model=model)
    return response

if __name__ == "__main__":
    # config
    data_path = "./filtered_reactive_advice_3k.jsonl"
    sample_size = 9999  # Number of samples to analyze
    save_as_file = True  # Whether to save the analysis results to a file
    
    # Load reactive advice records
    with open(data_path, 'r', encoding='utf-8') as f:
        reactive_records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(reactive_records)} reactive records from {data_path}")

    # Sample reactive records
    sampled_records = reactive_records[:sample_size]
    print(f"Sampling {len(sampled_records)} reactive records for analysis.")

    # Analyze each sampled record
    for i, record in enumerate(sampled_records):
        situation = record.get('current_situation', '')
        #print(f"Analyzing query: {situation}")
        analysis_result = analyze_context_reactive(situation)
        #print(f"Analysis result: {analysis_result}\n")
        #print("="*50 + "\n")
        # Optionally, you can save the analysis result back to the record or a file
        record['analysis'] = analysis_result
        # print progress
        print(f"Record {i+1}/{len(sampled_records)} analyzed: {record['_id'] if '_id' in record else 'No ID'}")
    
    if save_as_file:
        # Save the analyzed records back to a file
        output_file = "./analyzed_reactive_advice_3k.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in sampled_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Analysis results saved to {output_file}")
        print("Reactive advice analysis completed.")
    else:
        print("Analysis results not saved to file. Set 'save_as_file' to True to save results.")
        print("Reactive advice analysis completed without saving.")