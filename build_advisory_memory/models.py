import requests
import time

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from typing import Dict, List, Union
from transformers import AutoTokenizer, AutoModel

from snowflake.snowpark import Session
from snowflake.cortex import complete


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) # [bsz, hid_dim]


class EmbeddingModel:
    def __init__(self, model_dir="sentence-transformers/all-MiniLM-L6-v2"):
        # load model
        # the recommend model is all-MiniLM-L6-v2-sent2emb https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        print("loading embedding model all-MiniLM-L6-v2 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.mode = "cuda:5"  # default to CPU
        if self.mode != "cpu":
            self.model = self.model.cuda(self.mode)
        print('finish loading embedding model all-MiniLM-L6-v2')

    def __call__(
        self, sentences
    ):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.mode)

        with torch.no_grad():
            output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(output, encoded_input['attention_mask']) # [bsz, hid_dim]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1) # [bsz, hid_dim]

        return sentence_embeddings.cpu().numpy().tolist()


class AzureClient:
    def __init__(self, api_key, repeat_num=50):
        self.url = api_key
        self.repeat_num = repeat_num
    
    def __call__(self, prompt, model='gpt-4', max_tokens=1000):
        for _ in range(self.repeat_num):
            try:
                resp = requests.post(
                    self.url,
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': model,
                        'messages': [{
                            'role': 'user',
                            'content': prompt,
                        }],
                        'max_tokens': max_tokens,
                    },
                )
                rst = resp.json()['choices'][0]['message']['content']
                if model =='gpt-4':
                    pass
                else:
                    pass
                return rst
            except:
                time.sleep(1)
                continue
        
        return ''

class SnowClient:
    def __init__(self, connection_params, repeat_num=50):
        # Ensure connection_params is a dictionary with required fields
        if not isinstance(connection_params, dict):
            raise ValueError("connection_params must be a dictionary")
        
        required_params = ['account', 'user', 'password', 'warehouse', 'database', 'schema']
        missing_params = [param for param in required_params if param not in connection_params]
        if missing_params:
            raise ValueError(f"Missing required connection parameters: {', '.join(missing_params)}")
        
        self.connection_params = connection_params
        self.repeat_num = repeat_num
        self.snowpark_session = Session.builder.configs(connection_params).create()
    
    def __call__(self, prompt, model='llama3.2-1b', max_tokens=1000):
        for _ in range(self.repeat_num):
            try:
                messages = [{'role': 'user', 'content': prompt}]

                options = {
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "top_p": 0.95,
                }

                resp = complete(
                    model=model,
                    prompt=messages,
                    options=options,
                )
                return resp
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
        
        return ''