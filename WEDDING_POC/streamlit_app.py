import streamlit as st
import os
from google import generativeai
from google.generativeai import types
from PIL import Image
import io

# Configure the Generative AI client
generativeai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = "gemini-2.0-flash-preview-image-generation"

def generate_design_from_images(uploaded_files, user_prompt):
    # Combine images and prompt into a list of contents for the API
    contents = [
        f"I want a wedding decoration design based on the following items: {user_prompt}"
    ]
    
    # Add uploaded images to the contents
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = buffered.getvalue()
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.InlineData(
                            data=image_data, mime_type="image/png"
                        )
                    )
                ],
            )
        )

    # Configure the generation request
    config = types.GenerateContentConfig(
        response_modalities=["Text", "Image"]
    )

    # Generate the content
    client = generativeai.get_client()
    response = client.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=config
    )

    # Extract and display the generated images
    generated_images = []
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            generated_images.append(part.inline_data.data)

    return generated_images

def main():
    st.title("Wedding Decoration Designer")
    
    # User prompt input
    user_prompt = st.text_input("Describe your desired wedding decoration theme:", "e.g., elegant pastel-themed Indian wedding stage")
    
    # File uploader for decoration item images
    uploaded_files = st.file_uploader("Upload images of decoration items", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if st.button("Generate Design"):
        if not user_prompt:
            st.error("Please provide a description of your desired theme.")
        elif not uploaded_files:
            st.error("Please upload at least one image of a decoration item.")
        else:
            try:
                with st.spinner("Generating your design..."):
                    generated_images = generate_design_from_images(uploaded_files, user_prompt)
                    
                st.success("Design generated successfully!")
                for img_data in generated_images:
                    st.image(io.BytesIO(img_data), use_column_width=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()