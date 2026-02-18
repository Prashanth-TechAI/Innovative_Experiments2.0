# src/model.py

from transformers import AutoModelForCausalLM
from src.config import TEACHER_MODEL_NAME, STUDENT_MODEL_NAME

def load_models(device):
    # Load the teacher model and move it to the device.
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME)
    teacher.to(device)
    # Freeze teacher weights.
    for param in teacher.parameters():
        param.requires_grad = False

    # Load the student model and move it to the device.
    student = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_NAME)
    student.to(device)
    return teacher, student
