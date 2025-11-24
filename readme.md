# Implementation of SARA

This repository contains the implementation of SARA (Stepwise Advisory agent with Retrieval Augmentation), a system designed for question answering tasks on HotPotQA dataset. SARA uses an advisory memory system to improve its performance through reflection and learning from past experiences.

## Setup

### Environment Setup

1. Create a conda environment with Python 3.9:
   ```bash
   conda create -n sara python=3.9
   conda activate sara
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Snowflake API Configuration

This code requires Snowflake API access to run. You need to configure your Snowflake connection parameters:

1. Edit `snow_params.json` and fill in your Snowflake credentials:
   ```json
   {
       "account": "YOUR_ACCOUNT",
       "user": "YOUR_USERNAME",
       "password": "YOUR_PASSWORD",
       "role": "SYSADMIN",
       "database": "",
       "schema": "",
       "warehouse": ""
   }
   ```

2. Make sure you have the necessary Snowflake permissions and access to the required models (e.g., `llama3.1-70b`).

## Running SARA

### Using Pre-built Advisory Memory

To run SARA with the pre-built advisory memory, execute:

```bash
bash run_hotpotQA.sh
```

The script is configured to:
- Use the `llama3.1-70b` model
- Test on easy/medium/hard difficulty samples (configurable via `diff` variable)
- Load advisory memory from `build_advisory_memory/advisory_memory.json`
- Save results to `results/sara_70b_{difficulty}.jsonl`
- Enable reflection and memory usage
- Track token usage in `token_tracking/log_70b/{difficulty}/`

You can modify the script to change:
- `lm_name`: The language model to use
- `lm_codename`: Short name for the model (used in output paths)
- `diff`: Difficulty level (`easy`, `medium`, or `hard`)

### Evaluation

After running inference, you can evaluate the results using:

```bash
python evaluate/eval.py
```

The evaluation script:
- Loads results from `results/sara_{model}_{difficulty}.jsonl`
- Calculates F1 scores (SQuAD-style) for all predictions
- Reports overall F1 score and F1 score for cases where advice was requested
- Helps analyze the impact of the advisory memory system

Make sure to update the `lm_codename` and `diff` variables in `evaluate/eval.py` to match your run configuration.

## Building Advisory Memory

The `build_advisory_memory/` directory contains scripts and utilities for building the advisory memory system from training data. The advisory memory stores past experiences and advice that SARA can retrieve during inference.

### Process Overview

The advisory memory building process consists of several steps:

1. **`0_get_valid_ids.py`**: Extracts valid sample IDs from training data based on F1 score thresholds. This identifies cases where the model initially failed but could potentially benefit from advice.

2. **`1_1_filter_advice.py`**: Filters reactive advice cases from training data, keeping only those that correspond to valid IDs. This creates a curated set of advice-seeking scenarios.

3. **`1_2_analyze_context_reactive.py`**: Analyzes the context of reactive advice cases, extracting relevant situational information that will be used as keys for memory retrieval.

4. **`2_reflect_reactive.py`**: Performs reflection on reactive advice cases using the language model. This step generates reflective insights about why certain advice was helpful, what the situation was, and what the consequences were. The reflection process helps create more meaningful memory entries.

5. **`3_save_advisory_memory.py`**: Builds and saves the final advisory memory knowledge base. This script:
   - Processes reflected reactive advice records
   - Generates embeddings for memory keys (situational contexts)
   - Organizes the memory structure with keys, embeddings, and records
   - Saves the final `advisory_memory.json` file that can be loaded during inference

### Key Files

- **`models.py`**: Contains the `SnowClient` class for interacting with Snowflake API
- **`utils_build.py`**: Utility functions for building advisory memory, including embedding generation and similarity search
- **`snow_params.json`**: Snowflake connection parameters (same as root directory)
- **`advisory_memory.json`**: Final output file containing the built advisory memory

### Running the Build Process

To build advisory memory from scratch:

1. Ensure you have the required training data files in the appropriate locations
2. Configure `snow_params.json` with your Snowflake credentials
3. Run the scripts in sequence:
   ```bash
   cd build_advisory_memory
   python 0_get_valid_ids.py
   python 1_1_filter_advice.py
   python 1_2_analyze_context_reactive.py
   python 2_reflect_reactive.py
   python 3_save_advisory_memory.py
   ```

The final `advisory_memory.json` will be created in the `build_advisory_memory/` directory and can be used with the main SARA inference script.

## Project Structure

```
├── build_advisory_memory/     # Scripts for building advisory memory
├── data/                       # HotPotQA dataset files
├── evaluate/                  # Evaluation scripts
├── results/                    # Inference results
├── src/                        # Main source code
│   ├── hotpot_agent_sara.py   # Main SARA agent implementation
│   ├── models.py              # Model definitions
│   ├── utils_general.py       # General utilities
│   ├── utils_sara.py          # SARA-specific utilities
│   └── prompt/                # Prompt templates
├── token_tracking/            # Token usage logs
├── run_hotpotQA.sh            # Main execution script
├── snow_params.json           # Snowflake configuration
└── requirements.txt           # Python dependencies
```

## Notes

- The advisory memory system uses semantic similarity search to retrieve relevant past experiences during inference
- Token tracking files are generated during inference for analysis purposes
- The system supports different difficulty levels (easy, medium, hard) for evaluation
- Reflection is a key component that helps the agent learn from past mistakes and successes

