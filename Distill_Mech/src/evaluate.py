# src/evaluate.py

from src.dataset import get_tokenizer
import torch

def evaluate_model(student, device):
    student.eval()
    tokenizer = get_tokenizer()

    print("\nSample Generation:")
    sample_prompts = [
        "Instruction: తెలుగులో మీకు సహాయం కావాలంటే, దయచేసి అడగండి.\nResponse:",
        "Instruction: మీరు నన్ను ఎలా సహాయం చేయగలరు?\nResponse:"
    ]
    for prompt in sample_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = student.generate(**inputs, max_length=256, do_sample=False)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n=== Prompt ===")
        print(prompt)
        print("=== Generated Response ===")
        # Optionally, extract text after "Response:" for clarity.
        if "Response:" in generated_text:
            generated_text = generated_text.split("Response:")[1].strip()
        print(generated_text)
