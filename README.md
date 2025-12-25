# LLM-Driven Preference Data Synthesis for Proactive Prediction of the User’s Next Utterance

This repository contains the official implementation of the data synthesis pipeline proposed in our paper:

> **LLM-Driven Preference Data Synthesis for Proactive Prediction of the User’s Next Utterance in Human–Machine Dialogue**

The code implements an LLM-driven framework for synthesizing **reasoning-aware preference data** tailored to the task of *proactive next utterance prediction* in multi-turn human–machine dialogues.

---

## Overview

Proactive prediction of a user’s next utterance requires models to anticipate user intent progression before the user explicitly expresses it. Existing approaches are limited by the scarcity of task-specific, reasoning-oriented training data.

This work addresses this challenge by introducing a **data synthesis paradigm** that:
- Explicitly models user intent evolution using **intent trees**;
- Decomposes prediction into **utterance category reasoning** and **dual-view insight reasoning**;
- Generates **preference-style data** with *chosen* and *rejected* reasoning trajectories;
- Automatically constructs hard negative samples via incorrect intent-path perturbation and reasoning revision.

The synthesized data can be directly used to train or align small-to-medium LLMs for proactive next utterance prediction.

---


## Repository Structure

```plaintext
proutt/
├── src/
│   ├── config.py          # Configuration and hyperparameters
│   ├── client.py          # Async LLM client and JSON-safe calls
│   ├── prompts.py         # All prompt templates
│   ├── utils.py           # Shared utility functions
│   ├── io_utils.py        # JSONL I/O helpers
│   └── synthesis.py       # Main data synthesis pipeline
│   └── postprocess.py     # Post-process synthesized_raw into SFT/DPO-style training formats
├── data/
│   ├── raw/               # Original input data (not included)
│   ├── synthesized_raw/   # Direct LLM-generated synthesis results before post-processing
│   └── processed/         # Final processed preference data used for training
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Release

The final processed preference datasets are publicly released on Hugging Face.  
Specifically, this repository provides links to the following datasets:

- **LMSYS-ProUtt-10K**: https://huggingface.co/datasets/jqwang/LMSYS_ProUtt_10K
- **CrossWOZ-ProUtt-5K**: https://huggingface.co/datasets/jqwang/CrossWOZ_ProUtt_5K

Both datasets are constructed using the proposed LLM-driven data synthesis pipeline and are ready for downstream training and evaluation in proactive next utterance prediction.


---

## Data Format

### Input Data

The input file (`input_file`) is a JSONL file where each line corresponds to one multi-turn dialogue sample, containing:
- a unique dialogue ID;
- a list of messages in OpenAI-style format (`role`, `content`).

Example:
```json
{
  "id": "sample_001",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
---
## Installation

### Requirements
- Python ≥ 3.9
- An OpenAI-compatible LLM API endpoint (e.g., Zhipu GLM, DeepSeek, Qwen)
Install dependencies:
```bash
pip install -r requirements.txt
```

## API Configuration
Before running the pipeline, set the API key as an environment variable.
Example (Zhipu / GLM):
```bash
export ZAI_API_KEY=<your_api_key>
```

The default configuration in src/synthesis.py is:
```python
base_url = "https://open.bigmodel.cn/api/paas/v4/"
model_id = "glm-4.6"
```
---
## Running the Data Synthesis Pipeline
### Step 1: Prepare Directories
```bash
mkdir -p data/raw data/synthesized_raw data/processed
```
Place the input file at:
```bash
data/raw/LMSYS.jsonl
```
### Step 2: Run
From the project root directory:
```bash
python -m src.synthesis
```
### Step 3: Output
The synthesized preference data will be written to:
```bash
data/synthesized_raw/LMSYS.jsonl
```

## Step 4: Post-process for Training
Convert the raw synthesized results into training-ready formats (SFT or preference/DPO):
- Preference (DPO-style):
```bash
python -m src.postprocess \
  --input data/synthesized_raw/LMSYS.jsonl \
  --output data/processed/LMSYS_dpo.json \
  --mode pref
```
- SFT:
```bash
python -m src.postprocess \
  --input data/synthesized_raw/LMSYS.jsonl \
  --output data/processed/LMSYS_sft.json \
  --mode sft
```

## Step 5: Final Output
The final processed datasets will be saved under:
```bash
data/processed/
```
---


## Citation
If you use this code, please cite:
```bibtex

```
