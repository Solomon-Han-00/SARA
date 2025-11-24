import json

# LLama70
#path_control = "../results/test_run/hotpot/hotpot_70b_base_bolaa_stable_noseek_0615_d8_h.jsonl"
#path_treat = "../results/test_run/hotpot_kb_update/hotpot_70b_claude_1k_h_rad_reactive_up05_no_ref.jsonl"

#LLM experiment
lm_codename = "70b"
diff = "easy"
path = f"../results/sara_{lm_codename}_{diff}.jsonl"


with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# helper function: SQuAD-style f1 score
import re
def f1_score(prediction, ground_truth):
    prediction_tokens = re.split(r'\s+', prediction.lower())
    ground_truth_tokens = re.split(r'\s+', ground_truth.lower())
    
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def get_f1_score(d):
    if 'resp_answer' not in d or 'answer' not in d:
        return 0.0
    pred = d['resp_answer']
    gt = d['answer']
    return f1_score(pred, gt)

# step 1: get the ids of data for the cases where seekadvice is True
treat_ids = [i for i, data in enumerate(data) if data.get('seekadvice', False)]
print(f"Number of IDs with advice: {len(treat_ids)}")


# step 2: get the F1 score of the control data and treatment data
f1_scores = [get_f1_score(d) for d in data]
avg_f1 = sum(f1_scores) / len(f1_scores)
print(f"Overall F1 score: {avg_f1:.2%}")

print("====="* 20)

# step 3: get the F1 score of the advice-requested cases
f1_scores_treat = [f1_scores[i] for i in treat_ids]
avg_f1_treat = sum(f1_scores_treat) / len(f1_scores_treat)
print(f"F1 score for advice-requested cases: {avg_f1_treat:.2%}")