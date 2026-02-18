import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
import numpy as np

# ============
# Configuration
# ============
# Teacher (parent) model
TEACHER_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
# Student (small) model
STUDENT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
# CSV file containing Telugu instructions/responses.
DATA_PATH = "telugu_instruct.csv"  # adjust the path as needed

# Training hyperparameters
BATCH_SIZE = 4               # (small batch for trial; adjust as needed)
MAX_SEQ_LENGTH = 512         # maximum token length
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
TEMPERATURE = 2.0            # for distillation (softening the probability distributions)
ALPHA = 0.5                  # weight factor: combination of ground-truth loss & distillation loss

# ============
# Device Setup
# ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============
# 1. Load Tokenizer (we choose the teacher's tokenizer as our unified tokenizer)
# ============
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)
# If no EOS token exists, you might set it (for many causal models an EOS token is defined)
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"

# ============
# 2. Prepare the Dataset
# ============
# We assume the CSV has two columns: "instruction" and "response".
# We will build a prompt of the form:
#    "Instruction: <instruction>\nResponse:"  followed by the teacher’s target response.
# We also record the index where the response tokens start.
def preprocess_function(example):
    # Build the prompt (everything up to the response)
    prompt = "Instruction: " + example["instruction"] + "\nResponse:"
    # Add a space before the response to separate cleanly.
    response = " " + example["response"]

    # Tokenize prompt and response separately (without special tokens) so we can know the boundary.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # Combine prompt and response, and then add EOS token at the end.
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    # Truncate if too long.
    input_ids = input_ids[:MAX_SEQ_LENGTH]
    # Record where the response starts (this is our mask later)
    response_start = len(prompt_ids)
    # Build attention mask (1 for real tokens, 0 for padding; padding will be done later)
    return {"input_ids": input_ids, "response_start": response_start}

# Load the CSV dataset using Hugging Face Datasets.
# (If your CSV has a header with "instruction" and "response", this will work.)
raw_dataset = load_dataset("csv", data_files={"train": DATA_PATH}, delimiter="," )["train"]

# For a quick trial, select a subset (e.g. first 100 samples).
raw_dataset = raw_dataset.select(range(8))

# Apply preprocessing.
tokenized_dataset = raw_dataset.map(preprocess_function)

# ============
# Data Collator: Pads sequences to the same length and collates the response_start.
# ============
def data_collator(features):
    # features is a list of dictionaries with keys: "input_ids", "response_start"
    input_ids = [f["input_ids"] for f in features]
    response_starts = [f["response_start"] for f in features]

    # Pad the input_ids to the maximum length in the batch.
    batch = tokenizer.pad({"input_ids": input_ids}, padding="longest", return_tensors="pt")
    # Convert response_starts to a tensor (they are not padded; one per sample)
    batch["response_start"] = torch.tensor(response_starts, dtype=torch.long)
    return batch

# Create DataLoader.
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)

# ============
# 3. Load Teacher and Student Models
# ============
# We load them as causal language models.
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME).to(device)
student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_NAME).to(device)

# Freeze teacher’s parameters (we don’t update them during distillation).
for param in teacher.parameters():
    param.requires_grad = False

# ============
# 4. Define Optimizer & Scheduler
# ============
optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# ============
# 5. Training Loop: Distillation + Fine-Tuning (only over the "response" portion)
# ============
student.train()
teacher.eval()

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = 0.0
    for batch in train_dataloader:
        # Get input_ids, attention_mask, and response_start from the batch.
        input_ids = batch["input_ids"].to(device)         # shape: (B, L)
        attention_mask = batch["attention_mask"].to(device)   # shape: (B, L)
        response_starts = batch["response_start"].to(device)    # shape: (B)

        # Forward pass: Get logits from teacher and student.
        # We do not pass labels so that we obtain raw logits.
        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits  # (B, L, Vocab)

        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits      # (B, L, Vocab)

        # We now compute losses only for the tokens that are part of the "response".
        # Create a mask for each sequence: tokens before response_start are masked out (0), after are 1.
        batch_size, seq_length = input_ids.shape
        response_mask = torch.zeros((batch_size, seq_length), device=device)
        for i in range(batch_size):
            start = response_starts[i]
            # We want to include tokens from the response and the EOS token.
            response_mask[i, start:] = 1.0

        # ---- Compute Distillation Loss (KL Divergence) ----
        # Scale logits by temperature.
        teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
        student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
        # Compute token-wise KL divergence (without reduction yet)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # shape: (B, L)
        # Only consider tokens in the response region.
        kl_loss = (kl_loss * response_mask).sum() / response_mask.sum()
        kl_loss = kl_loss * (TEMPERATURE ** 2)  # scale loss as common in distillation

        # ---- Compute Hard (Ground-Truth) Loss ----
        # For causal LM, we shift the input_ids so that model predicts token t given tokens < t.
        # We use the same mask to compute loss only on response tokens.
        shift_logits = student_logits[:, :-1, :].contiguous()        # (B, L-1, V)
        shift_labels = input_ids[:, 1:].contiguous()                   # (B, L-1)
        shift_mask = response_mask[:, 1:]                              # (B, L-1)

        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1),
                                  reduction="none")
        ce_loss = ce_loss.view(batch_size, -1)
        ce_loss = (ce_loss * shift_mask).sum() / shift_mask.sum()

        # ---- Combined Loss ----
        loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

print("Training complete.")

# ============
# 6. (Optional) Evaluate: Generate a few responses from the fine-tuned student model.
# ============
student.eval()
print("\nSample Generation:")
sample_prompts = [
    "Instruction: తెలుగులో మీకు సహాయం కావాలంటే, దయచేసి అడగండి.\nResponse:",
    "Instruction: మీరు నన్ను ఎలా సహాయం చేయగలరు? \nResponse:"
]

for prompt in sample_prompts:
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate a response (using greedy decoding for simplicity)
    outputs = student.generate(**inputs, max_length=256, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Prompt ===")
    print(prompt)
    print("=== Generated Response ===")
    # Extract the part after "Response:" if needed.
    print(generated_text)
