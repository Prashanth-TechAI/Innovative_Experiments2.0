# src/dataset.py

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from src.config import DATA_PATH, MAX_SEQ_LENGTH, TEACHER_MODEL_NAME

def get_tokenizer():
    # Use the teacher's tokenizer as our unified tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    return tokenizer

def preprocess_function(example):
    tokenizer = get_tokenizer()
    # Retrieve values using .get() to support different header capitalizations.
    instruction = example.get("instruction") or example.get("Instruction")
    response = example.get("response") or example.get("Response")
    
    if instruction is None or response is None:
        raise ValueError(f"Expected columns 'instruction' and 'response' but got {list(example.keys())}")
    
    # Build the prompt.
    prompt = "Instruction: " + instruction + "\nResponse:"
    # Add a space before the actual response.
    response_text = " " + response
    
    # Tokenize prompt and response (without automatically adding special tokens)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    
    # Combine prompt and response, then add the EOS token.
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    input_ids = input_ids[:MAX_SEQ_LENGTH]
    
    # Record the index where the response begins (for loss masking later)
    response_start = len(prompt_ids)
    return {"input_ids": input_ids, "response_start": response_start}

def load_and_preprocess_dataset():
    # Load CSV dataset using Hugging Face Datasets.
    dataset = load_dataset("csv", data_files={"train": DATA_PATH}, delimiter=",")["train"]
    # Select up to 100 examples (or all if fewer than 100).
    dataset = dataset.select(range(min(100, len(dataset))))
    # Apply the preprocessing function.
    tokenized_dataset = dataset.map(preprocess_function)
    return tokenized_dataset

def data_collator(features, tokenizer=None):
    """
    Pads input_ids to the same length for each batch and collates the response_start.
    """
    input_ids_list = [f["input_ids"] for f in features]
    response_starts = [f["response_start"] for f in features]
    
    if tokenizer is None:
        tokenizer = get_tokenizer()
    batch = tokenizer.pad({"input_ids": input_ids_list}, padding="longest", return_tensors="pt")
    batch["response_start"] = torch.tensor(response_starts, dtype=torch.long)
    return batch
