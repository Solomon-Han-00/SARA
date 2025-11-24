import json
import re
from typing import List, Dict, Any

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate SQuAD-style F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        F1 score between 0.0 and 1.0
    """
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

def get_f1_score(data: Dict[str, Any]) -> float:
    """
    Extract F1 score from a data entry.
    
    Args:
        data: Dictionary containing 'resp_answer' and 'answer' fields
    
    Returns:
        F1 score for the entry
    """
    if 'resp_answer' not in data or 'answer' not in data:
        return 0.0
    pred = data['resp_answer']
    gt = data['answer']
    return f1_score(pred, gt)

def evaluate_dataset(data_path: str, f1_threshold: float = 0.5, output_path: str = None) -> Dict[str, Any]:
    """
    Evaluate a dataset and collect IDs where F1 score exceeds threshold.
    
    Args:
        data_path: Path to the JSONL file containing evaluation data
        f1_threshold: F1 score threshold (default: 0.5)
        output_path: Path to save the list of high F1 score IDs (optional)
    
    Returns:
        Dictionary containing evaluation results
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} entries from {data_path}")
    
    # Calculate accuracy (eval field)
    accuracy = sum(1 for entry in data if entry.get('eval', False)) / len(data)
    
    # Calculate F1 scores
    f1_scores = [get_f1_score(entry) for entry in data]
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    # Collect IDs where F1 score exceeds threshold
    success_ids = []
    for i, entry in enumerate(data):
        if f1_scores[i] > f1_threshold and '_id' in entry:
            if entry.get('gpt_eval', False):
                success_ids.append(entry['_id'])
    failure_ids = []
    for i, entry in enumerate(data):
        if not entry.get('eval', False) and '_id' in entry:
            if not entry.get('gpt_eval', False):
                failure_ids.append(entry['_id'])
    
    # Results
    results = {
        'dataset_path': data_path,
        'total_entries': len(data),
        'accuracy': accuracy,
        'average_f1': avg_f1,
        'f1_threshold': f1_threshold,
        'success_ids': success_ids,
        'failure_ids': failure_ids,
    }
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Dataset: {data_path}")
    print(f"Total entries: {len(data)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average F1 score: {avg_f1:.2%}")
    print(f"F1 threshold: {f1_threshold}")
    print(f"success IDs : {len(success_ids)}")
    print(f"failure IDs : {len(failure_ids)}")
    
    # Save high F1 IDs if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'f1_threshold': f1_threshold,
                'success_ids': success_ids,
                'failure_ids': failure_ids
            }, f, indent=2)
        print(f"\nsuccess / failure IDs saved to: {output_path}")
    
    return results

def main():
    # Hardcoded configuration values
    data_path = "../results/sanity_check/past_experience_with_advice.jsonl"  # Path to your JSONL file
    f1_threshold = 0.5  # F1 score threshold
    output_path = "test_valid_ids.json"  # Output path for success and failure IDs
    
    evaluate_dataset(data_path, f1_threshold, output_path)

if __name__ == "__main__":
    main()
