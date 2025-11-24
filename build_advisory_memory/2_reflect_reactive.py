from models import SnowClient
import json
import re

# configs
model_name = 'llama3.1-70b'
param_path = './snow_params.json'
with open(param_path, 'r', encoding='utf-8') as f:
    params = json.load(f)
snow_client = SnowClient(connection_params=params)

def integrate_dict_list(dict_list1, dict_list2):
    """
    Integrate two lists of dictionaries by merging them based on the '_id' key.
    
    Args:
        dict_list1: First list of dictionaries
        dict_list2: Second list of dictionaries
    
    Returns:
        List of merged dictionaries
    """
    merged_dict = {d['_id']: d for d in dict_list1}
    for d in dict_list2:
        if d['_id'] in merged_dict:
            merged_dict[d['_id']].update(d)
    return list(merged_dict.values())

def reflect_reactive(given_situation, given_advice, given_trajectory, given_consequence, given_truth, model=model_name):
    """
    Analyze the context proactively using the SnowClient.
    
    Args:
        query (str): The query to analyze.
        model (str): The model to use for analysis.
    
    Returns:
        str: The response from the model.
    """
    
    prompt_format = '''Given the following inputs:

1. **Situation**: The current situation or reasoning trajectory of the agent, which includes the query and the reasoning steps taken.
2. **Advice to Agent**: Specific recommendation or instruction provided to guide the agent.
3. **Full Trajectory of the Agent**: Detailed sequence of actions, decisions, and intermediate states taken by the agent following the advice.
4. **Consequence of the Advice**: The final outcome of the agent's actions explicitly indicated as either "success" or "failure."

Perform the following analysis and provide your response strictly in the JSON format below:

{{
  "causality": true or false,
  "consequence": "success" or "failure",
  "rationale": "Brief reasoning on how or why the advice influenced or did not influence the outcome.",
  "causal_feature": "Clearly defined critical factor, if causality is true; otherwise, null.",
  "feature_explanation": "Brief explanation connecting the feature explicitly to the consequence, if causality is true; otherwise, null."
}}

Ensure the fields are filled out as follows:

* causality: Boolean value indicating if the given advice directly caused the outcome. This should be true only if the consequence is "success" and the advice played an instrument role for success, or the consequence is "failure" and the advice was misguiding the agent, leading to the failure.
* consequence: String, either "success" or "failure".
* rationale: Short explanation clearly relating the advice to the trajectory and outcome.
* causal_feature: String if advice was causal, otherwise null.
* feature_explanation: Short explanation connecting the causal feature explicitly to the consequence, otherwise null.

Keep in mind that there may appear multiple advices in the trajectory, but you should focus on the causality of the given advice particularly.

The input arguments are as follows:
Situation: {given_situation}
Advice to Agent: {given_advice}
Full Trajectory of the Agent: {given_trajectory}
Ground Truth Answer: {given_truth}
Consequence of the Advice: {given_consequence}
    '''
    prompt = prompt_format.format(given_situation=given_situation,
                                given_advice=given_advice,
                                given_trajectory=given_trajectory,
                                given_truth=given_truth,
                                given_consequence=given_consequence)
    # Call the SnowClient with the formatted prompt
    response = snow_client(prompt, model=model)
    return response

if __name__ == "__main__":
    # config
    data_adv_path = "./filtered_reactive_advice_3k.jsonl"
    data_traj_path = "../results/sanity_check/hotpotqa_train_no_overlap_70b_claude_reactive.jsonl"
    sample_size = 9999  # Number of samples to reflect
    save_as_file = True  # Whether to save the reflection results to a file
    
    # load data
    with open(data_adv_path, 'r', encoding='utf-8') as f:
        advice_records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(advice_records)} reactive advice records from {data_adv_path}")
    with open(data_traj_path, 'r', encoding='utf-8') as f:
        trajectory_records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(trajectory_records)} reactive trajectory records from {data_traj_path}")
    
    # integrate data
    integrated_records = integrate_dict_list(advice_records, trajectory_records)
    print(f"Integrated {len(integrated_records)} records from both advice and trajectory data.")
    
    # Sample reactive records
    sampled_records = integrated_records[:sample_size]
    print(f"Sampling {len(sampled_records)} reactive records for reflection.")
    
    # Reflect on each sampled record
    for i, record in enumerate(sampled_records):
        situation = record.get('current_situation', '')
        advice = record.get('final_advice', '')
        trajectory = record.get('agent_output', '')
        consequence = record.get('consequence', '')
        truth = record.get('answer', '')
        _id = record.get('_id', 'No ID')
        
        reflection_result = reflect_reactive(situation, advice, trajectory, consequence, truth)
        
        # parse the reflection result as json
        try:
            # extract the JSON part from the response
            # find the first occurrence of a JSON object
            json_match = re.search(r'\{.*?\}', reflection_result, re.DOTALL)
            if json_match:
                reflection_result = json.loads(json_match.group(0))
                reflection_result['_id'] = _id
            else:
                print("No valid JSON found in reflection result, skipping this record.")
                continue
        except json.JSONDecodeError:
            print("Error decoding JSON from reflection result, skipping this record.")
            continue
        
        #print(f"Reflecting on situation: {situation}")
        #print(f"Reflection Result: {reflection_result}")
        #print("="*50 + "\n")
        
        # print progress
        print(f"Record {i+1}/{len(sampled_records)} reflected: {record['_id'] if '_id' in record else 'No ID'}")
        
        if save_as_file and reflection_result["causality"]:
            with open("reactive_reflected_results.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(reflection_result) + '\n')
    print("Reflection finished.")
    