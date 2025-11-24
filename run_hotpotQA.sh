# ProductQA & MedMCQA
lm_name=llama3.1-70b
lm_codename=70b
diff=easy # easy / medium / hard
python3 src/hotpot_agent_sara.py \
    --test_file data/hotpotqa/test/${diff}_samples.jsonl \
    --output_file results/sara_${lm_codename}_${diff}.jsonl \
    --reflection \
    --use_memory \
    --model $lm_name \
    --advisor_model $lm_name \
    --advisory_memory_path build_advisory_memory/advisory_memory.json \
    --agent_prompt src/prompt/agent_for_hotpot_minimal_report \
    --sample_limit -1 \
    --start_index 0 \
    --snowflake_connection_params snow_params.json \
    --main_agent_input_tracking_file token_tracking/log_${lm_codename}/${diff}/main_agent_inputs.txt \
    --main_agent_output_tracking_file token_tracking/log_${lm_codename}/${diff}/main_agent_outputs.txt \
    --advisor_input_tracking_file token_tracking/log_${lm_codename}/${diff}/advisor_inputs.txt \
    --advisor_output_tracking_file token_tracking/log_${lm_codename}/${diff}/advisor_outputs.txt


# HotPotQa
#python3 agile/hotpot_agent.py