# NYU Deep Learning Midterm: Text-to-SVG Generation

## Team
- Bram Simonnet
- Ryan [Last Name]

## Project goal
Generate valid SVG images from text prompts for the course Kaggle competition.

## Repository contents
- `notebooks/training.ipynb`: model training notebook
- `notebooks/kaggle_inference_submission.ipynb`: final Kaggle inference/submission notebook
- `src/`: reusable training, formatting, inference, and SVG utility code
- `report/`: final report
- `assets/`: qualitative examples

## Method
We fine-tune a Qwen-based language model with LoRA on the provided `train.csv` prompt-SVG pairs.
We format each example as a chat-style instruction:
- system: generate compact valid SVG only
- user: prompt
- assistant: SVG

We evaluate on a held-out validation split and generate final predictions for the Kaggle test set.

## Reproducibility
Main training settings:
- model: Qwen 2.5 Instruct 1.5B 4-bit
- LoRA rank: 16
- max sequence length: 2048
- epochs: 1
- batch size: 4 effective per step via gradient accumulation
- random seed: 42

## Model weights
Add final link here: [TODO]

## Kaggle notebook
Add final link here: [TODO]

## Notes
This repository uses only the provided competition training data for supervised fine-tuning.
