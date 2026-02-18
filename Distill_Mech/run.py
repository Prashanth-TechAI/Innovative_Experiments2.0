# run.py

import torch
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import TRAINED_MODEL_DIR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Stage 1 & 2: Distillation and fine-tuning.
    student = train_model(device)

    # Save the fine-tuned student model.
    student.save_pretrained(TRAINED_MODEL_DIR)
    print(f"Trained student model saved to: {TRAINED_MODEL_DIR}")

    # Stage 3: Evaluate / generate sample responses.
    evaluate_model(student, device)

if __name__ == "__main__":
    main()
