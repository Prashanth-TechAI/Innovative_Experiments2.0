# src/train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from src.dataset import load_and_preprocess_dataset, data_collator, get_tokenizer
from src.model import load_models
from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, TEMPERATURE, ALPHA

def train_model(device):
    tokenizer = get_tokenizer()
    # Load and preprocess the dataset.
    dataset = load_and_preprocess_dataset()
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=lambda features: data_collator(features, tokenizer)
    )
    
    # Load teacher and student models.
    teacher, student = load_models(device)

    # Setup optimizer and learning rate scheduler.
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    student.train()
    teacher.eval()

    print("Starting training (distillation + fine-tuning)...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)          # (B, L)
            attention_mask = batch["attention_mask"].to(device)  # (B, L)
            response_starts = batch["response_start"].to(device) # (B)

            # Get teacher logits (without gradient computation).
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits  # (B, L, Vocab)

            # Get student logits.
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits  # (B, L, Vocab)

            # Create a mask to compute loss only on tokens corresponding to the response.
            batch_size, seq_length = input_ids.shape
            response_mask = torch.zeros((batch_size, seq_length), device=device)
            for i in range(batch_size):
                start = response_starts[i]
                response_mask[i, start:] = 1.0

            # ----- Distillation Loss (KL Divergence) -----
            teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
            student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # (B, L)
            kl_loss = (kl_loss * response_mask).sum() / response_mask.sum()
            kl_loss = kl_loss * (TEMPERATURE ** 2)

            # ----- Hard Loss (Cross-Entropy) -----
            # Shift logits and labels for causal LM.
            shift_logits = student_logits[:, :-1, :].contiguous()  # (B, L-1, Vocab)
            shift_labels = input_ids[:, 1:].contiguous()             # (B, L-1)
            shift_mask = response_mask[:, 1:]                        # (B, L-1)
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            )
            ce_loss = ce_loss.view(batch_size, -1)
            ce_loss = (ce_loss * shift_mask).sum() / shift_mask.sum()

            # Combined loss (weighted sum).
            loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("Training complete.")
    return student
