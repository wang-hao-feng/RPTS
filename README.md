# RPTS: Tree-Structured Reasoning Process Scoring for Faithful Multimodal Evaluation

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-%230099CC.svg)](https://aaai.org/aaai-conference/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.06899-b31b1b.svg)](https://arxiv.org/abs/2511.06899)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

📢 **Our work has been accepted as an Oral presentation at the 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026).** 

## 📖 Overview

Large Vision-Language Models (LVLMs) have advanced significantly in multimodal reasoning, yet existing benchmarks primarily evaluate final answers (e.g., multiple-choice), often ignoring the logical validity of the reasoning process. This leads to an inability to detect when models arrive at correct answers through flawed reasoning (“right answers for wrong reasons”).

To address this, we propose the **Reasoning Process Tree Score (RPTS)**, a novel metric that:
- Models the reasoning process as a **tree structure**, where leaf nodes are atomic visual/textual evidence and non-leaf nodes are inference steps.
- Dynamically weights each reasoning step based on its hierarchical position to compute a faithfulness score.
- Enables both **global reasoning assessment** and **precise error localization**.

To validate RPTS, we introduce **RPTS-Eval**, a benchmark containing 390 multimodal reasoning instances (374 images) with reliable visual-textual clues and annotated intermodal relationships.

## ✨ Key Features

- **RPTS Metric**: A tree-based scoring mechanism that evaluates the logical faithfulness of each step in a multimodal reasoning chain.
- **RPTS-Eval Benchmark**: A curated dataset for fine-grained evaluation of multimodal reasoning, featuring:
    - Atomic, reliable evidence as leaf nodes for tree construction.
    - Three defined intermodal relationship types (Guided, Adversarial, Independent).

## 🚀 Quick Start

### 1. Clone Repository

Clone this repository from GitHub:

```bash
git clone https://github.com/wang-hao-feng/RPTS.git
cd RPTS
```

### 2. Environment Setup

Clone the repository and create the conda environment:

```bash
git clone https://github.com/ShareGPT4Omni/ShareGPT4V.git
conda env create -f environment.yml
conda activate RPTS
```

### 3. Configuration

Set the required environment variables:

```bash
export HF_TOKEN="your-huggingface-token"           # For downloading datasets/models
export MODEL_PATH="your-path-for-storing-model-parameters"
export OPENAI_API_KEY="your-api-key"               # For GPT-4 based scoring/parsing
export RPTS_PATH="your-rpts-eval-path"            # Path to store the RPTS-Eval dataset
```

### 4. Download Data and Models

Download the RPTS-Eval dataset from [Hugging Face Datasets](https://huggingface.co/datasets/nimingshuaishi/RPTS-Eval) to the path specified in `RPTS_PATH`.

Download the required model parameters:

```bash
cd RPTS
python download_models.py
```

## 🧪 Evaluation Pipeline

The evaluation follows a multi-stage pipeline:

### Step 1: Run Inference
Generate reasoning outputs for the models on RPTS-Eval. Example for GPT-4o (English CoT):

```bash
mkdir -p results/cot_en
sh ./scripts/cot_en/gpt-4o.sh
```

Scripts for other models (GPT-4V, LLaVA-Next, etc.) are available in the `scripts/` directory.

### Step 2: Parse and Transform Reasoning
Reformat the model's free-text reasoning into a structured tree format (`[PREMISE] + [PREMISE] -> [CONCLUSION]`).

```bash
cd transform_reasoning
export LANGUAGE="en"  # or "zh"
export FILE_NAME="your_output_file.json"
python transform.py -l ${LANGUAGE} -f ${FILE_NAME} -ds ${RPTS_PATH}
```

### Step 3: Score Individual Reasoning Steps
Use an LLM-based scorer to evaluate the logical faithfulness of each parsed reasoning step.

```bash
cd evaluate
export LANGUAGE="en"
export FILE_NAME="transformed_file.json"
python score_reasoning.py -l ${LANGUAGE} -f ${FILE_NAME} -ds ${RPTS_PATH}
```

### Step 4: Calculate RPTS and Accuracy
Compute the final RPTS metric and overall accuracy, with optional filtering based on RPTS thresholds.

```bash
cd evaluate
export LANGUAGE="en"
export FILE_NAME="scored_file.json"
export LAMBDA=0.9   # Decay factor λ (controls weight decay across tree height)
export HF=1      # Focus height h_f (step with maximum weight)

# Basic calculation
python calculate.py -l ${LANGUAGE} -f ${FILE_NAME} -ds ${RPTS_PATH} --lambda_ ${LAMBDA} --hf ${HF}

# Calculate accuracy filtered by RPTS threshold (e.g., > 0.5)
python calculate.py -l ${LANGUAGE} -f ${FILE_NAME} -ds ${RPTS_PATH} --lambda_ ${LAMBDA} --hf ${HF} --filter --threshold 0.5
```

## 📊 RPTS Metric Details

The RPTS is calculated as a weighted average of scores $s_i$ for each reasoning step $i$:

$$RPTS = \frac{\sum w_i * s_i}{\sum w_i}$$

The weight $w_i$ for a step at tree height $h$ is:
$$w_i = \lambda ^ {|h_f - h|}$$

- $\lambda$: Decay factor controlling how weights diminish from the focus step.
- $h_f$: The focus height (step) receiving maximum weight.

By adjusting $\lambda$ and $h_f$, RPTS can emphasize different parts of the reasoning tree (e.g., root steps for evidence grounding or deeper steps for complex inference).

## 📄 License

This project is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
