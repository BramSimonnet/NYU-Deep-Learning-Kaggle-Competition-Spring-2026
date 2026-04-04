# NYU Deep Learning Midterm: Text-to-SVG Generation

## Team
- Bram Simonnet  
- Ryan Fleishman  

---

## Project Overview

This project addresses the task of generating valid Scalable Vector Graphics (SVG) from natural language prompts for the NYU Deep Learning Spring 2026 Kaggle competition.

Unlike standard text generation tasks, this problem requires strict adherence to structural constraints. Outputs must be valid XML, use only allowed SVG elements, respect complexity limits, and render correctly. Invalid SVGs receive a score of zero, making robustness a central challenge.

---

## Repository Structure

- training-book.ipynb  
  Fine-tunes a Qwen-based model with LoRA on filtered training data.

- inference-book.ipynb  
  Initial development inference notebook.

- inference-book-with-sim.ipynb  
  Final Kaggle submission notebook.

- requirements.txt  
  Python dependencies required to reproduce results.

---

## Methodology

### 1. Supervised Fine-Tuning

We fine-tune a Qwen2.5-Coder-1.5B-Instruct model (4-bit) using Low-Rank Adaptation (LoRA) on the competition dataset.

Each example is formatted as a chat-style interaction:
- System: instruction to generate compact, valid SVG only  
- User: natural language prompt  
- Assistant: target SVG  

We apply strict filtering to ensure all training SVGs satisfy the competition constraints (valid XML, allowed tags, path limits). This prevents the model from learning invalid patterns and improves robustness at inference time.

---

### 2. Inference Pipeline

Our final system uses a hybrid pipeline combining retrieval, generation, and post-processing.

#### Retrieval (Primary Mechanism)
- TF-IDF vectorization (unigrams + bigrams)
- cosine similarity between prompts
- nearest-neighbor SVG is always used (threshold = 0.0)

#### Generation
- LoRA fine-tuned model
- nucleus sampling (top-p = 0.9)
- temperature = 0.7
- max generation length = 1024 tokens

#### Post-processing
- SVG extraction from model output
- XML-based repair for malformed outputs
- normalization to 256x256 canvas
- enforcement of allowed tags and structural constraints

#### Fallback
- prompt-aware fallback SVG to guarantee valid output in failure cases

---

## Reproducibility

### Training Configuration
- Model: Qwen2.5-Coder-1.5B-Instruct (4-bit)
- LoRA rank: 16
- LoRA alpha: 16
- Max sequence length: 1024
- Learning rate: 2e-4
- Epochs: 1
- Batch size: 24
- Gradient accumulation steps: 8
- Optimizer: paged_adamw_8bit
- Scheduler: cosine
- Warmup ratio: 0.05
- Weight decay: 0.01
- Random seed: 42

### Environment
- GPU: Tesla T4 (Kaggle)
- Frameworks:
  - PyTorch
  - HuggingFace Transformers
  - TRL
  - Unsloth

---

## How to Run

### Install Dependencies
pip install -r requirements.txt

### Training
Run the training notebook:
training-book.ipynb

### Inference / Submission
Run the Kaggle submission notebook:
inference-book-with-sim.ipynb
note - in kaggle, link the weights as a dataset. Out of kaggle, you'll need to manually assign the path to your weights in the config cell

This will:
- load the fine-tuned model  
- generate SVG predictions for the test set  
- save submission.csv  

---

## Model Weights

Add link here:  
[MODEL WEIGHTS LINK]

---


## Key Design Insights

- Validity is more important than expressiveness due to the evaluation metric  
- Filtering invalid training data significantly improves robustness  
- Retrieval is highly effective for in-distribution prompts  
- Post-processing (repair + normalization) is critical for reducing invalid outputs  

---

## Acknowledgments

This project was completed as part of the NYU Deep Learning Spring 2026 course.

We use PyTorch, HuggingFace Transformers, TRL, and Unsloth for efficient fine-tuning and inference.
