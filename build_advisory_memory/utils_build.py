import json
import hashlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models import SnowClient, EmbeddingModel

# Initialize embedding model
embed_model = EmbeddingModel()

# ============================================================================
# EMBEDDING AND SIMILARITY FUNCTIONS
# ============================================================================

def text2emb(text):
    """
    Convert text to embedding using the embedding model.
    
    Args:
        text (str): Input text to embed
        
    Returns:
        list: Text embedding
    """
    results = embed_model([text])
    return results[0]

def search_past_similar(question, history_qa_embed, memory_embed, qa_thred=0.53, memory_thred=0.42):
    """
    Search for similar past questions and memories based on embedding similarity.
    
    Args:
        question (str): Current question
        history_qa_embed (list): List of historical QA embeddings
        memory_embed (list): List of memory embeddings
        qa_thred (float): Threshold for QA similarity
        memory_thred (float): Threshold for memory similarity
        
    Returns:
        tuple: (qa_similar_count, memory_similar_count)
    """
    question_emb = text2emb(question)
    
    if len(history_qa_embed) == 0:
        qs_num = 0
    else:
        dis = F.cosine_similarity(torch.tensor(history_qa_embed), torch.tensor(question_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        qs_num = len([a for a in dis if a > qa_thred])
    
    if len(memory_embed) == 0:
        ms_num = 0
    else:
        dis = F.cosine_similarity(torch.tensor(memory_embed), torch.tensor(question_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        ms_num = len([a for a in dis if a > memory_thred])
    
    return qs_num, ms_num

def search_k_memory(memory, memory_emb, query, tokenizer=None, target_num=1e8, target_length=512):
    """
    Search and retrieve the most similar memories based on query.
    
    Args:
        memory (list): List of memory data
        memory_emb (list): List of memory embeddings
        query (str): Query text
        tokenizer: Tokenizer for length calculation
        target_num (int): Maximum number of memories to retrieve
        target_length (int): Maximum total length of retrieved memories
        
    Returns:
        str: Concatenated selected memory data
    """
    assert len(memory) == len(memory_emb)
    
    if len(memory_emb) > 0:
        dis = []
        query_emb = text2emb(query)
        dis = F.cosine_similarity(torch.tensor(memory_emb), torch.tensor(query_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        idx = range(len(dis))
        dis = list(zip(idx, dis))
        dis = sorted(dis, key=lambda x: x[1], reverse=True)
        data_sort = [memory[x[0]] for x in dis]
        
        if tokenizer is None:
            data_sort_decode = [x for x in data_sort]
        else:
            data_sort_decode = [tokenizer.encode(x) for x in data_sort]
        
        current_len, current_num, ans = 0, 0, ''
        for i in range(len(dis)):
            if current_num >= target_num:
                break
            if current_len + len(data_sort_decode[i]) > target_length:
                continue
            current_len += len(data_sort_decode[i])
            ans += data_sort[i]
            current_num += 1
        return ans
    else:
        return ''

# ============================================================================
# ADVICE KNOWLEDGE BASE FUNCTIONS
# ============================================================================

def add_to_advice_kb(advice_record, advice_kb):
    """
    Add advice record and its embedding to the knowledge base.
    
    Args:
        advice_record (dict): The advice record to add
        advice_kb (dict): The knowledge base to add to
    """
    advice_type = advice_record['advice_type']
    
    if advice_type == 'proactive':
        # use analysis as the key for proactive advice
        key = advice_record['analysis']
        if key not in advice_kb['proactive_keys']:
            advice_kb['proactive_keys'].append(key)
            advice_kb['proactive_key_embs'].append(text2emb(key))
            advice_kb['proactive_records'].append(advice_record)
    
    elif advice_type == 'reactive':
        # use analysis as the key for reactive advice
        key = advice_record['analysis']
        if key not in advice_kb['reactive_keys']:
            advice_kb['reactive_keys'].append(key)
            advice_kb['reactive_key_embs'].append(text2emb(key))
            advice_kb['reactive_records'].append(advice_record)

def find_similar_advice(query_text, advice_type='proactive', top_k=1, advice_kb=None):
    """
    Find the most similar advice based on the query text.
    
    Args:
        query_text (str): The query text to find similar advice for
        advice_type (str): Either 'proactive' or 'reactive' to specify which type of advice to search
        top_k (int): Number of similar advice records to return
        advice_kb (dict): The advice knowledge base
        
    Returns:
        list: List of dictionaries containing the most similar advice records
    """
    query_emb = text2emb(query_text)
    
    if advice_type == 'proactive':
        keys = advice_kb['proactive_keys']
        key_embs = advice_kb['proactive_key_embs']
        records = advice_kb['proactive_records']
    else:  # reactive
        keys = advice_kb['reactive_keys']
        key_embs = advice_kb['reactive_key_embs']
        records = advice_kb['reactive_records']
    
    if not key_embs:
        return []
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(key_embs):
        sim = F.cosine_similarity(
            torch.tensor(query_emb).unsqueeze(0),
            torch.tensor(emb).unsqueeze(0)
        ).item()
        similarities.append((i, sim))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in similarities[:top_k]]
    
    return [records[i] for i in top_k_indices]

def find_similar_example(query_text, reflection_type='full', top_k=1, reflection_kb=None):
    """
    Find the most similar example based on the query text.
    
    Args:
        query_text (str): The query text to find similar examples for
        reflection_type (str): Either 'full' or 'recent' to specify which type of reflection to search
        top_k (int): Number of similar examples to return
        reflection_kb (dict): The reflection knowledge base
        
    Returns:
        list: List of dictionaries containing the most similar examples
    """
    query_emb = text2emb(query_text)
    
    if reflection_type == 'full':
        reflections = reflection_kb['full_reflections']
        reflection_embs = reflection_kb['full_reflection_embs']
    else:
        reflections = reflection_kb['recent_reflections']
        reflection_embs = reflection_kb['recent_reflection_embs']
    
    if not reflection_embs:
        return []
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(reflection_embs):
        sim = F.cosine_similarity(
            torch.tensor(query_emb).unsqueeze(0),
            torch.tensor(emb).unsqueeze(0)
        ).item()
        similarities.append((i, sim))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in similarities[:top_k]]
    
    return [reflection_kb['examples'][i] for i in top_k_indices]

# ============================================================================
# DATA PROCESSING AND UTILITY FUNCTIONS
# ============================================================================

def get_group_split(group):
    """
    Determine if a group belongs to test or train split.
    
    Args:
        group (str): Group name
        
    Returns:
        str: 'test' or 'train'
    """
    return 'test' if group in ['all_pans', 'camera_cases', 'leggings', 'motherboards', 'rifle_scopes', 'rollerball_pens'] else 'train'

def get_schema_description(schema):
    """
    Generate the schema description for the prompt using from schema file.
    
    Args:
        schema (dict): Schema dictionary
        
    Returns:
        str: Formatted schema description
    """
    schema_des = ''
    for key in schema:
        schema_des = schema_des + key
        if 'unit' in schema[key]:
            schema_des = schema_des + '[' + schema[key]['unit'] + ']'
        if schema[key]['type'] == 'choice':
            schema_des = schema_des + '('
            for value in schema[key]['choices']:
                schema_des = schema_des + value + ','
            schema_des = schema_des[:-1] + ')'
        schema_des = schema_des + '|'
    return schema_des[:-1]

def gen_hash_id(data):
    """
    Generate a hash ID from data.
    
    Args:
        data: Data to hash
        
    Returns:
        int: Hash ID
    """
    if not isinstance(data, bytes):
        data = str(data).encode('utf-8')
    m = hashlib.md5(data)
    md5sum = m.hexdigest()
    hash_id = int(md5sum, 16) % (2 ** 63)
    return hash_id

# ============================================================================
# PARSING AND EXTRACTION FUNCTIONS
# ============================================================================

def get_action(resp):
    """
    Parse the action from the response.
    
    Args:
        resp (str): Response text
        
    Returns:
        str: Extracted action
    """
    seekadvice_idx = resp.lower().find("seekadvice")
    searchproduct_idx = resp.lower().find("searchproduct")
    predictanswer_idx = resp.lower().find("predictanswer")
    idx_list = [[seekadvice_idx, " [SeekAdvice]\n"], [searchproduct_idx, " [SearchProduct]\n"], [predictanswer_idx, " [PredictAnswer]\n"]]
    action = [x for x in idx_list if x[0] >= 0]
    if len(action) == 0:  # 提取失败则默认predictanswer
        action = [[predictanswer_idx, " [PredictAnswer]\n"]]
    action.sort(key=lambda x: x[0])
    return action[0][1]

def parse_answer(text, token):
    """
    Extract the long and short answer.
    
    Args:
        text (str): Text to parse
        token (str): Token to search for
        
    Returns:
        str: Extracted answer
    """
    begin_idx = text.find(token)
    text = text[begin_idx + len(token):]
    while len(text) > 0 and not ((text[0] >= 'a' and text[0] <= 'z') or (text[0] >= 'A' and text[0] <= 'Z')):
        text = text[1:]
    end_idx = text.find('[')
    if end_idx != -1:
        text = text[: end_idx]
    text = ''.join(text.split('\n'))
    return text

def get_sql(text):
    """
    Extract SQL from text.
    
    Args:
        text (str): Text containing SQL
        
    Returns:
        tuple: (sql_string, found_boolean)
    """
    token = 'SELECT'
    s = text.split('\n')
    for line in s:
        idx = line.find(token)
        if idx == -1:
            continue
        return line[idx:], True
    return '', False

def str2dict(s):
    """
    Convert string to dictionary, handling various formats.
    
    Args:
        s (str): String to convert
        
    Returns:
        dict: Parsed dictionary
    """
    try:
        d = json.loads(s)
        return d
    except:
        try:
            d = json.loads(s[8:-3])
            return d
        except:
            pass

    begin_idx = s.find('```')
    if begin_idx < 0:
        print(s)
        return dict()

    end_idx = s[begin_idx + 3:].find('```')
    if end_idx < 0:
        print(s)
        return dict()

    while s[begin_idx] != '{':
        begin_idx += 1

    end_idx += 3
    while s[end_idx] != '}':
        end_idx -= 1

    try:
        d = json.loads(''.join(s[begin_idx: end_idx + 1].split('\n')))
        return d
    except:
        pass
    
    print(s)
    return dict()

# ============================================================================
# AGENT PROMPT PARSING FUNCTIONS
# ============================================================================

def parse_agent_prompt(prompt):
    """
    Parse agent prompt to extract components.
    
    Args:
        prompt (str): Agent prompt
        
    Returns:
        tuple: (asin, question, schema_metadata, product_category)
    """
    asin, question, schema_metadata = None, None, None
    asin = prompt.split("Answer users' [Question] about product")[1].split("based on")[0].strip()
    product_category = prompt.split("The schema of the")[1].split("database")[0].strip()
    question = prompt.split("[Question]:")[1].split("\n")[0].strip()
    schema_metadata = prompt.split("(in the format field[unit](value1, value2, ...)).")[1].split("In addition to")[0].strip()
    return asin, question, schema_metadata, product_category

def parse_agent_prompt_hotpot(prompt):
    """
    Parse HotPotQA agent prompt.
    
    Args:
        prompt (str): HotPotQA agent prompt
        
    Returns:
        str: Extracted question
    """
    question = prompt.split("[Question]:")[1].split("\n")[0].strip()
    return question

def parse_agent_trajectory(prompt):
    """
    Parse agent trajectory from prompt.
    
    Args:
        prompt (str): Agent trajectory prompt
        
    Returns:
        tuple: (header, trajectory_steps)
    """
    steps = prompt.split("\n==========================\n")
    header = steps[0].strip()
    trajectory = steps[1:] if len(steps) > 1 else []
    return header, trajectory

def get_trajectory_str(trajectory):
    """
    Convert trajectory list to string.
    
    Args:
        trajectory (list): List of trajectory steps
        
    Returns:
        str: Trajectory string
    """
    delimiter = "\n==========================\n"
    trajectory_str = delimiter.join(trajectory)
    return trajectory_str

def get_recent_steps(trajectory):
    """
    Get the most recent steps from trajectory.
    
    Args:
        trajectory (list): List of trajectory steps
        
    Returns:
        tuple: (previous_step, latest_step)
    """
    if len(trajectory) >= 2:
        prev_step = trajectory[-2]
        latest_step = trajectory[-1]
    else:
        prev_step = "Empty (no previous step)"
        latest_step = trajectory[0]
    return prev_step, latest_step

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def eval_search(response, short_answer):
    """
    Evaluate the search result.
    
    Args:
        response (list): Search response
        short_answer (list): Expected short answer
        
    Returns:
        bool: Evaluation result
    """
    if type(short_answer) is not list:
        # wrong action, wrong result
        return False
    if len(short_answer) == 0:
        if len(response) == 0:
            return True
        else:
            return False
    if len(response) == 0 and len(short_answer) != 0:
        return False
    answer_asin = [x["asin"] for x in short_answer]
    for i in response:
        if i not in answer_asin:
            return False
    return True

def eval_predict(response, short_answer, all_products):
    """
    Evaluate the predict result in short answer.
    
    Args:
        response (str): Model response
        short_answer (list or str): Expected answer
        all_products (list): All available products
        
    Returns:
        bool: Evaluation result
    """
    if type(short_answer) is list:
        # comparison_qa but prediction answer
        response = response.lower()
        answer_asin = [x["asin"].lower() for x in short_answer]
        has_gold, not_has_not_gold = False, True  # correct iff the answer contains gold and does not include not gold
        for i in short_answer:
            if i["asin"].lower() in response or i["title"].lower() in response:
                has_gold = True
                break
        for i in all_products:
            if i["asin"] in answer_asin:
                continue
            if i["asin"].lower() in response or i["title"].lower() in response:
                not_has_not_gold = False
                break
        return has_gold and not_has_not_gold
    else:
        return short_answer.lower().strip() == str(response).lower().strip()

def eval_predict_long(question, response, long_answer, prompt_file, snow_client):
    """
    Evaluate the predict result in long answer.
    Note that this function needs Claude-Sonnet.
    
    Args:
        question (str): Question
        response (str): Model response
        long_answer (str): Reference long answer
        prompt_file (str): Path to evaluation prompt file
        snow_client: Snowflake client
        
    Returns:
        tuple: (evaluation_result, detailed_result)
    """
    prompt = load_prompt(prompt_file).format(question=question, reference=long_answer, response=str(response))
    result = snow_client(prompt, "claude-3-5-sonnet", 500)
    if 'Yes' in result:
        return True, result
    return False, result

# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def load_prompt(file):
    """
    Load the prompt from file.
    
    Args:
        file (str): Path to prompt file
        
    Returns:
        str: Prompt content
    """
    prompt = ''
    with open(file) as fin:
        for line in fin:
            prompt = prompt + line
    return prompt

def load_jsonl_in_json(json_file_path, encoding='utf-8', **kwargs):
    """
    Load data from a JSONL file.
    
    Args:
        json_file_path (str): Path to the JSONL file
        encoding (str): File encoding
        **kwargs: Additional arguments for json.loads
        
    Returns:
        list: List of JSON objects
    """
    data = []
    with open(json_file_path, "r", encoding=encoding) as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip(), **kwargs))
    return data

def dump_jsonline(json_file_path, data, encoding="utf-8"):
    """
    Save data to a JSONL file.
    
    Args:
        json_file_path (str): Path to the output JSONL file
        data (list): List of data to save
        encoding (str): File encoding
        
    Returns:
        int: 0 on success
    """
    with open(json_file_path, "wt", encoding=encoding) as fout:
        for ins in data:
            fout.write(f"{json.dumps(ins, ensure_ascii=False)}\n")
    fout.close()
    return 0
