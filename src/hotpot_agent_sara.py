import json
import re
import torch
import torch.nn.functional as F
from utils_general import embed_model
from models import SnowClient
import copy
import os
import datetime
from utils_sara import find_similar_advice

class AdvisorAgent:
    def __init__(
        self,
        model="gpt-4",
        snow_client=None,
        max_questions=1,  # Maximum number of clarifying questions allowed
        advisory_memory_path = "agile/advisory_memory.json",
        advisor_input_tracking_file=None,
        advisor_output_tracking_file=None,
    ):
        self.model = model
        self.snow_client = snow_client
        self.max_questions = max_questions
        
        # Token tracking files
        self.advisor_input_tracking_file = advisor_input_tracking_file
        self.advisor_output_tracking_file = advisor_output_tracking_file
        
        # Internal variables for the advisor
        self.top_k = 2
        self.advisor_log = []

        # advice kb
        self.advisory_memory = None
        self.advisory_memory_path = advisory_memory_path
        # load advice kb
        try:
            with open(self.advisory_memory_path, 'r') as f:
                self.advisory_memory = json.load(f)
                print("Loaded advice kb from {}".format(self.advisory_memory_path))
        except FileNotFoundError:
            print("Warning: Advice KB not found at {}. Will use empty KB.".format(self.advisory_memory_path))
            self.advisory_memory = {
                'reactive_keys': [],
                'reactive_key_embs': [],
                'reactive_records': []
            }

        # Define prompts directly in the code
        self.proactive_advice_prompt = """You are an Advisor Agent specialized in strategic planning and decomposition of complex reasoning questions.

You will be given with a question that the Base Agent finds initially challenging, along with the analysis of the question.
Your task is to proactively generate a concise, step-by-step strategic advice for how to answer this question.

Here is the description of your main agent, which will carry out the actions as your advice:
Main agent is a question answering agent with the ability to search knowledge.
Main agent can analyze the solution steps based on the problem and known information.
Known information includes the internal knowledge of the main agent, search results, and the content of advice you provide.
For missing information, Main agent can use search tools by output `Action: [Search] ([entity])`.
If there is enough information, Main agent can output `Action: [PredictAnswer] ([answer])` to answer the question directly.

Important: Do NOT directly provide the final answer to the question. Your job is purely strategic planning to guide the Main Agent's reasoning trajectory.


{few_shot_examples}

Your Question: {question}
Analysis: {analysis}"""

        self.reactive_advice_prompt = """You are an intelligent advisor agent that helps the main agent solve complex questions successfully. Your role is to:
1. Understand the main agent's current situation based on the dialogue history
2. Provide helpful advice using `Action: [GiveAdvice]` if you are confident about the advice
3. Leverage your factual knowledge and reasoning abilities to assist the main agent in answering the question
4. Do not ask questions, but instead provide best-effort advice for the main agent to proceed
5. Assume that the searched documents are always factually correct, and there is no factual misunderstanding in given question

You will be given with a situation where the Base Agent finds initially challenging, along with the analysis of the situation.

Here is the description of your main agent:
Main agent is an intelligent question answering agent with the ability to search knowledge.
Main agent can analyze the solution steps based on the problem and known information.
Known information includes the internal knowledge of the main agent, search results, and the content of advice you provide.
For missing information, Main agent can use search tools by output `Action: [Search] ([entity])`.
If there is enough information, Main agent can output `Action: [PredictAnswer] ([answer])` to answer the question directly.

{few_shot_examples}

Your current situation:
{current_situation}

Analysis: {analysis}

Select the next action from the following options and provide the required information:  
- `Action: [GiveAdvice] (your advice here)`

Remember to always use the exact format `Action: [ActionType] (content)` for your response.
The ActionType should be enclosed by square brackets, and the argument should be in enclosed in parentheses."""

    def clear_internals(self):
        self.advisor_log = []

    def call_model(self, prompt, max_token=1000):
        # Track input to advisor agent
        if self.advisor_input_tracking_file:
            with open(self.advisor_input_tracking_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== ADVISOR INPUT ===\n{prompt}")
        
        response = self.snow_client(prompt, self.model, max_token)
        
        # Track output from advisor agent
        if self.advisor_output_tracking_file:
            with open(self.advisor_output_tracking_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== ADVISOR OUTPUT ===\n{response}")
        
        return response
    
    def analyze_context(self, input, advice_type='proactive'):
        """
        Analyze the context using the SnowClient.
        
        Args:
            input (str): The input to analyze.
            advice_type (str): The type of advice ('proactive' or 'reactive').
        
        Returns:
            str: The response from the model.
        """
        
        prompt_format_proactive = '''Analyze the following query to extract high-level semantic features. The analysis should identify key concepts, intent, underlying themes, contextual categories, and user objectives inherent in the query. Avoid superficial details or exact string matching. Instead, focus on deeper semantic understanding to enable robust retrieval of semantically similar queries in the future.

Your analysis should:

Clearly summarize the main intent of the query.

List core concepts and entities involved.

Categorize the query into relevant semantic or topical areas.

Provide potential user goals or objectives behind the query.

Note any implicit assumptions or contexts necessary to fully interpret the query.

Query:
"{given_query}"
    '''
    
        prompt_format_reactive = '''Analyze the given context, which consists of a query and its reasoning trajectory (termed as "current situation"), to provide a high-level semantic summary. The analysis should capture underlying logical structures, critical reasoning issues, core needs, and overall strategic intent present in the trajectory. This summary will serve as a key for future retrieval, aiding in identifying similar logical reasoning trajectories.

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
    
        if advice_type == 'proactive':
            prompt = prompt_format_proactive.format(given_query=input)
        elif advice_type == 'reactive':
            prompt = prompt_format_reactive.format(given_situation=input)
        else:
            raise ValueError("Invalid advice_type. Use 'proactive' or 'reactive'.")
        
        # Call the SnowClient with the formatted prompt
        response = self.snow_client(prompt, model=self.model)
        return response.strip()
    
    def format_few_shot_examples(self, examples, advice_type='proactive'):
        """Format few-shot examples for the prompt"""
        if not examples:
            return ""
        
        output_str = "\nHere are some similar examples from past successful experience, including analysis and advice. In examples, the high-level feature and reflection on the advice will be also provided. (high-level feature and reflection will be given in examples only) \n\n"
        for i, example in enumerate(examples):
            if advice_type == 'proactive':
                content = f"Example {i+1}:\n"
                content += f"Question: {example['query'][:200]}...\n"
                content += f"Analysis: {example['analysis']}\n"
                content += f"Advice: {example['advice']}\n"
                content += f"High-level Feature: {example['causal_feature']}\n"
                content += f"Reflection: {example['feature_explanation']}\n"
                
            else:  # reactive
                content = f"Example {i+1}:\n"
                content += f"Situation: {example['current_situation']}...\n"
                content += f"Analysis: {example['analysis']}\n"
                content += f"Advice: {example['final_advice']}\n"
                content += f"High-level Feature: {example['causal_feature']}\n"
                content += f"Reflection: {example['feature_explanation']}\n"
            content += "----------------------------------------\n"
            output_str += content
        output_str += "(END OF EXAMPLES)\n"
        return output_str
    
    def get_action(self, res):
        answer_pattern = r"Action: \[GiveAdvice\] \(([^)]*)\)"
        matches = re.findall(answer_pattern, res, re.DOTALL)
        if len(matches) < 1:
            return None, "", ""
        param = matches[0].strip()
        prefix = f"Action: [GiveAdvice] ({param})"
        return "GiveAdvice", param, prefix
        
    def interact_single_step(self, current_situation):
        # Retrieve similar reactive advice examples
        similar_examples = []
        # Analyze the current situation to get high-level features
        analysis = self.analyze_context(current_situation, advice_type='reactive')
        
        if self.advisory_memory and len(self.advisory_memory['reactive_records']) > 0:
            similar_examples = find_similar_advice(
                analysis,
                advice_type='reactive', 
                top_k=self.top_k, 
                advisory_memory=self.advisory_memory
            )
        
        # Format few-shot examples
        few_shot_examples = self.format_few_shot_examples(similar_examples, 'reactive')
        
        # Generate reactive advice based on the current situation and examples
        reactive_prompt = self.reactive_advice_prompt.format(
            current_situation=current_situation,
            analysis=analysis,
            few_shot_examples=few_shot_examples
        )

        response = self.call_model(reactive_prompt)
        action, param, prefix = self.get_action(response)
        

        return {
                "action": action,
                "param": param,
                "prefix": prefix,
                "full_response": response
            }

    def interact(self, current_situation):
        self.clear_internals()
        single_step_result = self.interact_single_step(current_situation)
        self.advisor_log.append(single_step_result)
        print("Advisor action: {}".format(single_step_result["action"]))
        single_step_result["advisor_log"] = copy.deepcopy(self.advisor_log)
        return single_step_result

class HotpotAgent:
    def __init__(
        self,
        model="gpt-4",
        reflection=True,
        seek_advice=True,
        use_memory=True,
        extra="1",
        agent_prompt="",
        agent_reflection_prompt="",
        similarity_thred=0.46,
        snowflake_connection_params=None,
        advisor_model="claude-3-5-sonnet",
        advisory_memory_path="agile/advisory_memory.json",
        main_agent_input_tracking_file=None,
        main_agent_output_tracking_file=None,
        advisor_input_tracking_file=None,
        advisor_output_tracking_file=None,
    ):
        self.model = model
        self.reflection = reflection
        self.seek_advice = seek_advice
        self.use_memory = use_memory
        self.similarity_thred = similarity_thred
        self.agent_prompt = agent_prompt
        self.agent_reflection_prompt = agent_reflection_prompt
        self.proactive_thr = 4  # Threshold for proactive advice
        self.snow_client = SnowClient(connection_params=snowflake_connection_params)
        
        # Token tracking files
        self.main_agent_input_tracking_file = main_agent_input_tracking_file
        self.main_agent_output_tracking_file = main_agent_output_tracking_file
        
        self.advisor = AdvisorAgent(
            model=advisor_model, 
            snow_client=self.snow_client, 
            advisory_memory_path=advisory_memory_path,
            advisor_input_tracking_file=advisor_input_tracking_file,
            advisor_output_tracking_file=advisor_output_tracking_file
        )
        self.eval_prompt = """Based on the provided question and reference answer, please determine if the response is correct or incorrect. Begin by articulating your rationale, and conclude with a single word judgment: 'Yes' for correct or 'No' for incorrect.

question: {question}
reference answer: {reference}
response: {response}"""
        self.cnt_proactive_advice = 0
        self.cnt_reactive_advice = 0

        try:
            self.extra = int(extra)
            if self.extra < 0:
                self.extra = -1
        except:
            self.extra = -1
        print("Retrieve {} results, -1 for filling 512 tokens".format(self.extra))

    def reset_storage(self):
        self.history = {}
        self.history_emb = {}
        self.history_question_emb = {}
        self.memory = []
        self.memory_emb = []
        self.total = {}

    def call_model(self, prompt, max_token=1000):
        # Track input to main agent
        if self.main_agent_input_tracking_file:
            with open(self.main_agent_input_tracking_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== MAIN AGENT INPUT ===\n{prompt}")
        
        response = self.snow_client(prompt, self.model, max_token)
        
        # Track output from main agent
        if self.main_agent_output_tracking_file:
            with open(self.main_agent_output_tracking_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== MAIN AGENT OUTPUT ===\n{response}")
        
        return response

    def text2emb(self, text):
        results = embed_model([text])
        return results[0]

    def search_memory(self, param, lib_emb):
        param_emb = self.text2emb(param)
        lib = [i[0] for i in lib_emb]
        lib_embs = [i[1] for i in lib_emb]
        dis = F.cosine_similarity(torch.tensor(lib_embs), torch.tensor(param_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        idx = range(len(dis))
        dis = list(zip(idx, dis))
        dis = sorted(dis, key=lambda x: x[1], reverse=True)
        data_sort = [lib[x[0]] for x in dis]
        return data_sort[0]

    def get_action(self, res, delimiter="=========================="):
        res = res.strip()
        if res.startswith(delimiter):
            res = res.split(delimiter)[1].strip()
        if delimiter in res:
            res = res.split(delimiter)[0].strip()
        # Use a more precise pattern that matches the entire action format
        answer_pattern = r"Action: \[(Search|PredictAnswer|SeekAdvice)\] \(([^)]*)\)"
        matches = re.findall(answer_pattern, res, re.DOTALL)
        if len(matches) < 1:
            return None, "", ""
        action_type, _ = matches[0]
        # get param with maximal matching
        param_start = res.find('(')
        if param_start == -1:
            return None, "", ""
        param_end = res.rfind(')')
        if param_end == -1 or param_end <= param_start:
            return None, "", ""
        param = res[param_start + 1:param_end].strip()
        # Extract prefix by finding the position of the action pattern
        prefix = f"Action: [{action_type}] ({param})"
        return action_type, param, prefix

    def step_iterate(self, data):
        lib = {i[0]: i[1] for i in data["context"]}
        lib_emb = [[i, self.text2emb(i)] for i in lib.keys()]
        supporting_facts = {}
        for i in data["supporting_facts"]:
            if i[0] not in lib or i[1] >= len(lib[i[0]]):
                continue
            if i[0] in supporting_facts:
                supporting_facts[i[0]].append(lib[i[0]][i[1]].strip())
            else:
                supporting_facts[i[0]] = [lib[i[0]][i[1]].strip()]
        supporting_facts_order = list(supporting_facts.keys())
        searched = set()
        
        # No proactive advice
        data["proactive_advice"] = None
        prompt_ = self.agent_prompt.format(data["question"])
        idx = 0
        questions_asked = 0
        free_advice = 2
        action_num = 0
        delimiter = "=========================="
        
        # Initialize steps list to store structured dialogue history
        data["steps"] = []
        
        # set seekadvice as False by default
        data["seekadvice"] = False
        
        def clean_res(res, delimiter):
            res = res.strip()
            if res.startswith(delimiter):
                res = res.split(delimiter)[1].strip()
            if delimiter in res:
                res = res.split(delimiter)[0].strip()
            return res
        
        while idx < 8:
            res = self.call_model(prompt_).strip('\n').strip('response:').strip(' ').strip('</s>')
            action, param, prefix = self.get_action(res, delimiter=delimiter)
            # For debugging purposes
            print(f"Action: {action}, Param: {param}, Prefix: {prefix}")
            print("---"*20)
            action_num += 1
        
            if action == "Search":
                try:
                    search_entity = self.search_memory(param, lib_emb)
                except:
                    break
                idx += 1
                #prefix += "[{}] ({})".format(action, param)
                # No summary from the first search -- provide all information
                suffix = "\nObservation: Search Result - {} (Full version)\n".format(search_entity)
                observe = "\n".join(lib[search_entity])
                lib = {k: v for k, v in lib.items() if k != search_entity}
                lib_emb = [i for i in lib_emb if i[0] != search_entity]
                prompt_ += " " + prefix + suffix + observe + f"\n{delimiter}\n"
                #print(prefix + suffix + observe + f"\n{delimiter}\n")
                data["agent_output"] = prompt_
                
                # Add search step to steps list
                data["steps"].append({
                    "step_type": "search",
                    "search_query": param,
                    "search_entity": search_entity,
                    "observation": observe,
                    "is_summary": search_entity not in searched
                })

            elif action == "SeekAdvice":
                # set seekadvice to True
                data["seekadvice"] = True
                if free_advice <= 0:
                    idx += 1
                else:
                    free_advice -= 1
                    
                self.cnt_reactive_advice += 1
                
                # Report current situation to advisor
                current_situation = f"I am trying to answer the question: {data['question']}\n"
                progress = delimiter.join(prompt_.split(delimiter)[1:]).strip()
                progress += res
                current_situation += f"Current progress: {progress}\n"
                
                # Start dialogue with advisor
                advisor_response = self.advisor.interact(current_situation)
                
                # Integrate advisor's response into the reasoning path
                #prefix += "[{}] ({})".format(action, param)
                suffix = "\nObservation: Expert Advice - "
                prompt_ += prefix + suffix + advisor_response['param'] + f"\n{delimiter}\n"
                
                # Add advice step to steps list
                data["steps"].append({
                    "step_type": "advisor_advice",
                    "advice": advisor_response['param']
                })
                
                # Add advisor response to data
                if "advisor_log" not in data:
                    data["advisor_log"] = []
                data["advisor_log"].append(advisor_response)
                
                # For debugging purposes
                #print(prefix + suffix + advisor_response['param'] + f"\n{delimiter}\n")
                #print("---"*20)
                
                data["agent_output"] = prompt_
            
            elif action == "PredictAnswer":
                data["resp_answer"] = param
                data["eval"] = data["answer"].strip().lower() == param.strip().lower()
                data["part_eval"] = (data["answer"].strip().lower() in param.strip().lower() or param.strip().lower() in data["answer"].strip().lower())
                claude_eval = self.snow_client(self.eval_prompt.format(question=data["question"], reference=data["answer"], response=param), "claude-3-5-sonnet", 500)
                if 'Yes' in claude_eval:
                    data["gpt_eval"] = True
                else:
                    data["gpt_eval"] = False
                #prefix += "[{}] ({})".format(action, param)
                prompt_ += " " + prefix
                prompt_ += f"\n{delimiter}\n"
                data["agent_output"] = prompt_
                
                # Add prediction step to steps list
                data["steps"].append({
                    "step_type": "prediction",
                    "predicted_answer": param,
                    "is_correct": data["eval"],
                    "is_partially_correct": data["part_eval"],
                    "gpt_evaluation": data["gpt_eval"]
                })
                break
        
        if "eval" not in data:
            data["eval"] = data["part_eval"] = data["gpt_eval"] = False
        return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HotpotQA Agent')
    parser.add_argument('--model', default="gpt-4")
    parser.add_argument('--advisor_model', default="claude-3-5-sonnet")
    parser.add_argument('--advisory_memory_path', default="agile/advisory_memory.json")
    parser.add_argument('--reflection', action="store_true", default=False)
    parser.add_argument('--seek_advice', action="store_true", default=False)
    parser.add_argument('--use_memory', action="store_true", default=False)
    parser.add_argument('--agent_prompt', default="prompt/hotpot_agent")
    parser.add_argument('--extra', default="1")
    parser.add_argument('--similarity_thred', type=float, default=0.46)
    parser.add_argument('--snowflake_connection_params', type=str, default="{}")
    parser.add_argument('--test_file', type=str, default="data/hotpotqa/test/data.jsonl")
    parser.add_argument('--output_file', type=str, default="results/hotpotqa/agile-hotpot-test.jsonl")
    parser.add_argument('--sample_limit', type=int, default=-1, help='Limit the number of samples to process')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--main_agent_input_tracking_file', type=str, default="token_tracking/main_agent_inputs.txt", help='File to track main agent inputs')
    parser.add_argument('--main_agent_output_tracking_file', type=str, default="token_tracking/main_agent_outputs.txt", help='File to track main agent outputs')
    parser.add_argument('--advisor_input_tracking_file', type=str, default="token_tracking/advisor_inputs.txt", help='File to track advisor inputs')
    parser.add_argument('--advisor_output_tracking_file', type=str, default="token_tracking/advisor_outputs.txt", help='File to track advisor outputs')
    args = parser.parse_args()
    
    # Fix note
    # 1) enforce maximal matching with find() and rfind() in get_action() [done]
    # 2) add rules to handle knowledge conflict -- assume question and docs are always correct [done]

    # Parse Snowflake connection parameters
    try:
        with open(args.snowflake_connection_params, 'r') as f:
            snowflake_connection_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.snowflake_connection_params} not found. Using empty connection parameters.")
        snowflake_connection_params = {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for snowflake_connection_params")
        snowflake_connection_params = {}
    
    # Create tracking directory and files
    tracking_dir = os.path.dirname(args.main_agent_input_tracking_file)
    if tracking_dir and not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
        print(f"Created tracking directory: {tracking_dir}")
    
    # Create tracking files if they don't exist
    tracking_files = [
        args.main_agent_input_tracking_file,
        args.main_agent_output_tracking_file,
        args.advisor_input_tracking_file,
        args.advisor_output_tracking_file
    ]
    
    for file_path in tracking_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Token tracking file: {file_path}\n")
                f.write(f"# Created at: {datetime.datetime.now()}\n\n")
            print(f"Created tracking file: {file_path}")
        
    with open(args.agent_prompt) as f:
        agent_prompt = f.read().strip()

    agent = HotpotAgent(
        model=args.model,
        reflection=args.reflection,
        seek_advice=args.seek_advice,
        use_memory=args.use_memory,
        agent_prompt=agent_prompt,
        similarity_thred=args.similarity_thred,
        snowflake_connection_params=snowflake_connection_params,
        advisor_model=args.advisor_model,
        advisory_memory_path=args.advisory_memory_path,
        main_agent_input_tracking_file=args.main_agent_input_tracking_file,
        main_agent_output_tracking_file=args.main_agent_output_tracking_file,
        advisor_input_tracking_file=args.advisor_input_tracking_file,
        advisor_output_tracking_file=args.advisor_output_tracking_file,
    )

    test_file = args.test_file
    with open(test_file) as f, open(args.output_file, "w") as f1:
        lines = f.readlines()
        if args.start_index > 0:
            lines = lines[args.start_index:]
        if args.sample_limit > 0:
            lines = lines[:args.sample_limit]
        for i, line in enumerate(lines):
            data = json.loads(line)
            data["agent_output"] = ""
            data = agent.step_iterate(data)
            f1.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # Print progress
            print(f"Progress percentage: {((i + 1) / len(lines)) * 100:.2f}%")