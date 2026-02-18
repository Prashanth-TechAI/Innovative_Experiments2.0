import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = ""

async def generate_images(prompt: str, num_images: int = 4):
    model = genai.GenerativeModel(model_name=MODEL_ID)

    # Remove 'response_modality' and use correct parameters
    response = model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            response_mime_type="image/png"
        )
    )

    images = []
    for part in response.parts:
        if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
            images.append(part.inline_data.data)

    return images[:num_images]