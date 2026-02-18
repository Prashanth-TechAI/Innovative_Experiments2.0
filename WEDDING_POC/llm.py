import os
import httpx
from data import items
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = ""
GROQ_MODEL = ""

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

async def generate_prompt_from_items(selected, user_prompt):
    item_lookup = {i["id"]: i["prompt"] for i in items}
    item_descriptions = "\n".join([f"{q} {item_lookup[i]}" for i, q in selected])

    system_msg = {
        "role": "system",
        "content": "You are a wedding decor assistant helping generate image prompts for Indian wedding stage design."
    }
    user_msg = {
        "role": "user",
        "content": (
            f"The user wants the design to look like this:\n"
            f"\"{user_prompt}\"\n\n"
            f"Here are the available items and their quantities:\n{item_descriptions}\n\n"
            f"Write a single, rich, visually detailed prompt for generating the design image."
        )
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GROQ_BASE_URL,
                headers=headers,
                json={
                    "model": GROQ_MODEL,
                    "messages": [system_msg, user_msg],
                    "temperature": 0.8
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                error_details = e.response.json()
                raise Exception(f"Bad request: {error_details.get('error', 'Unknown error')}")
            raise