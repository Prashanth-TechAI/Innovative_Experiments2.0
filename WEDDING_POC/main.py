from fastapi import FastAPI, Form, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from data import items
from llm import generate_prompt_from_items
from gemini import generate_images
import uvicorn
import os
import base64
import logging
from starlette.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def home(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request, "items": items})
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/generate", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def generate(
    request: Request,
    selected_ids: str = Form(None),
    counts: str = Form(None),
    user_prompt: str = Form(...),
    custom_images: list[UploadFile] = File(None)
):
    try:
        # Validate inputs
        if not user_prompt:
            raise HTTPException(status_code=400, detail="User prompt is required")

        selected = []
        if selected_ids and counts:
            ids = selected_ids.split(",")
            count_values = counts.split(",")

            if len(ids) != len(count_values):
                raise HTTPException(status_code=400, detail="Number of selected items and counts do not match")

            qty = []
            for value in count_values:
                if value.strip() == "":
                    qty.append(1)
                else:
                    try:
                        qty.append(int(value))
                    except ValueError:
                        raise HTTPException(status_code=400, detail="Invalid count value")

            selected = [(i, q) for i, q in zip(ids, qty)]

        # Handle Custom Images
        custom_image_descriptions = []
        if custom_images:
            for img in custom_images:
                contents = await img.read()
                encoded_img = base64.b64encode(contents).decode()
                custom_image_descriptions.append(f"Custom Image: {img.filename} - {encoded_img}")

        # Generate Prompt
        prompt = await generate_prompt_from_items(selected, user_prompt)

        # Include custom images in the prompt if available
        if custom_image_descriptions:
            prompt += "\n\nIncluding custom images:\n" + "\n".join(custom_image_descriptions)

        # Generate Images
        images = await generate_images(prompt)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "items": items,
            "images": images,
            "final_prompt": prompt
        })

    except HTTPException as e:
        logger.error(f"HTTPException in generate route: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error in generate route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)