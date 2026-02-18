# src/config.py

TEACHER_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"  # Parent (teacher) model
STUDENT_MODEL_NAME = "/home/username/.llama/checkpoints/Llama3.2-1B"  # Local path to the student model

# Data
DATA_PATH = "data/telugu_instruct.csv"  # CSV file with Telugu instructions/responses

# Training hyperparameters
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
TEMPERATURE = 2.0   # For distillation: softening logits
ALPHA = 0.5         # Weight between Cross-Entropy loss and distillation (KL) loss

# Save path for the trained student model
TRAINED_MODEL_DIR = "trained_telugu_student_model"
