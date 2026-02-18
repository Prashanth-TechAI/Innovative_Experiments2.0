import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import PIL.Image
import io
import base64
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Gemini Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

def initialize_client():
    """Initialize the Gemini client with API key"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Please set your GEMINI_API_KEY in the .env file")
            return None
        
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return None

def generate_image(client, prompt, response_modalities=['Text', 'Image']):
    """Generate image using Gemini API"""
    try:
        MODEL_ID = "gemini-2.0-flash-preview-image-generation"
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=response_modalities
            )
        )
        return response
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def edit_image(client, text_prompt, image):
    """Edit image using Gemini API"""
    try:
        MODEL_ID = "gemini-2.0-flash-preview-image-generation"
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[text_prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        return response
    except Exception as e:
        st.error(f"Error editing image: {str(e)}")
        return None

def extract_image_from_response(response):
    """Extract image data from API response"""
    try:
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        return None
    except Exception as e:
        st.error(f"Error extracting image: {str(e)}")
        return None

def extract_text_from_response(response):
    """Extract text data from API response"""
    try:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                return part.text
        return None
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def save_image_locally(image_data, filename):
    """Save image data to local file"""
    try:
        # Create images directory if it doesn't exist
        Path("generated_images").mkdir(exist_ok=True)
        
        file_path = Path("generated_images") / filename
        file_path.write_bytes(image_data)
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def main():
    st.title("🎨 Gemini Image Generator")
    st.markdown("Generate and edit images using Google's Gemini AI")
    
    # Initialize client
    if st.session_state.client is None:
        with st.spinner("Initializing Gemini client..."):
            st.session_state.client = initialize_client()
    
    if st.session_state.client is None:
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    st.sidebar.info("Make sure your GEMINI_API_KEY is set in the .env file")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🖼️ Generate Image", "✏️ Edit Image"])
    
    with tab1:
        st.header("Generate New Image")
        
        # Text input for image generation
        prompt = st.text_area(
            "Enter your image description:",
            placeholder="A beautiful sunset over mountains with a lake in the foreground",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            generate_btn = st.button("🎨 Generate Image", type="primary")
        
        if generate_btn and prompt:
            with st.spinner("Generating image... This may take a few moments."):
                response = generate_image(st.session_state.client, prompt)
                
                if response:
                    # Extract and display image
                    image_data = extract_image_from_response(response)
                    text_response = extract_text_from_response(response)
                    
                    if image_data:
                        # Display the generated image
                        st.success("Image generated successfully!")
                        
                        # Convert bytes to PIL Image for display
                        image = PIL.Image.open(io.BytesIO(image_data))
                        st.image(image, caption=f"Generated: {prompt}")
                        
                        # Display any text response
                        if text_response:
                            st.write("**AI Response:**")
                            st.write(text_response)
                        
                        # Save image option
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("💾 Save Image"):
                                filename = f"generated_{len(st.session_state.generated_images)}.png"
                                saved_path = save_image_locally(image_data, filename)
                                if saved_path:
                                    st.success(f"Image saved as {saved_path}")
                                    st.session_state.generated_images.append({
                                        'prompt': prompt,
                                        'path': saved_path,
                                        'image': image
                                    })
                        
                        with col2:
                            # Download button
                            st.download_button(
                                label="⬇️ Download Image",
                                data=image_data,
                                file_name=f"gemini_generated_{len(st.session_state.generated_images)}.png",
                                mime="image/png"
                            )
    
    with tab2:
        st.header("Edit Existing Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image to edit:",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )
        
        # Or select from generated images
        if st.session_state.generated_images:
            st.write("**Or select from previously generated images:**")
            selected_image_idx = st.selectbox(
                "Choose an image:",
                range(len(st.session_state.generated_images)),
                format_func=lambda x: f"Image {x+1}: {st.session_state.generated_images[x]['prompt'][:50]}..."
            )
        
        # Edit prompt
        edit_prompt = st.text_area(
            "Describe how you want to edit the image:",
            placeholder="Change the color of the sky to purple and add some stars",
            height=100
        )
        
        edit_btn = st.button("✏️ Edit Image", type="primary")
        
        if edit_btn and edit_prompt:
            # Determine which image to use
            image_to_edit = None
            
            if uploaded_file is not None:
                image_to_edit = PIL.Image.open(uploaded_file)
                st.write("Using uploaded image:")
                st.image(image_to_edit, caption="Original Image", width=300)
            elif st.session_state.generated_images and 'selected_image_idx' in locals():
                image_to_edit = st.session_state.generated_images[selected_image_idx]['image']
                st.write("Using selected image:")
                st.image(image_to_edit, caption="Original Image", width=300)
            
            if image_to_edit:
                with st.spinner("Editing image... This may take a few moments."):
                    response = edit_image(st.session_state.client, edit_prompt, image_to_edit)
                    
                    if response:
                        # Extract and display edited image
                        image_data = extract_image_from_response(response)
                        text_response = extract_text_from_response(response)
                        
                        if image_data:
                            st.success("Image edited successfully!")
                            
                            # Display the edited image
                            edited_image = PIL.Image.open(io.BytesIO(image_data))
                            
                            # Show before and after
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Before:**")
                                st.image(image_to_edit, caption="Original")
                            with col2:
                                st.write("**After:**")
                                st.image(edited_image, caption="Edited")
                            
                            # Display any text response
                            if text_response:
                                st.write("**AI Response:**")
                                st.write(text_response)
                            
                            # Save and download options
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button("💾 Save Edited Image"):
                                    filename = f"edited_{len(st.session_state.generated_images)}.png"
                                    saved_path = save_image_locally(image_data, filename)
                                    if saved_path:
                                        st.success(f"Edited image saved as {saved_path}")
                            
                            with col2:
                                st.download_button(
                                    label="⬇️ Download Edited Image",
                                    data=image_data,
                                    file_name=f"gemini_edited_{len(st.session_state.generated_images)}.png",
                                    mime="image/png"
                                )
            else:
                st.warning("Please upload an image or select from generated images to edit.")
    
    # Display generated images history
    if st.session_state.generated_images:
        st.header("📸 Generated Images History")
        
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.generated_images):
            with cols[idx % 3]:
                st.image(img_data['image'], caption=f"{img_data['prompt'][:30]}...")
                st.caption(f"Saved at: {img_data['path']}")

if __name__ == "__main__":
    main()