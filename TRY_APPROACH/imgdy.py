import streamlit as st
import os
import pathlib
import tempfile
from PIL import Image
import io
import base64
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Image Generator & Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

def initialize_gemini_client(api_key):
    """Initialize Gemini client with API key"""
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        return client, types
    except ImportError:
        st.error("Google GenAI library not installed. Please install it using: pip install google-genai>=1.5.0")
        return None, None
    except Exception as e:
        st.error(f"Error initializing Gemini client: {str(e)}")
        return None, None

def generate_image(client, types, prompt, model_id="gemini-2.0-flash-preview-image-generation"):
    """Generate image using Gemini API"""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        
        # Extract image data
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data, part.inline_data.mime_type
        
        return None, None
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise e

def edit_image(client, types, prompt, uploaded_file, model_id="gemini-2.0-flash-preview-image-generation"):
    """Edit image using Gemini API"""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Convert uploaded file to PIL Image
        pil_image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (for compatibility)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            pil_image = pil_image.convert('RGB')
        
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        
        # Extract image data
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data, part.inline_data.mime_type
        
        return None, None
    except Exception as e:
        logger.error(f"Error editing image: {str(e)}")
        raise e

def save_image_to_temp(image_data, filename):
    """Save image data to temporary file"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AI Image Generator & Editor</h1>
        <p>Create and edit stunning images with Google's Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google AI Studio API key",
            placeholder="Enter your API key here..."
        )
        
        if api_key:
            st.session_state.api_key_set = True
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your Google API key to continue")
        
        st.markdown("---")
        
        # Model selection
        model_id = st.selectbox(
            "Model",
            [""],
            help="Select the Gemini model for image generation"
        )
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        <div class="feature-box">
            <h4>📝 Instructions</h4>
            <ul>
                <li>Get your API key from <a href="https://ai.google.dev/" target="_blank">Google AI Studio</a></li>
                <li>Enter detailed prompts for better results</li>
                <li>Use the Image Editor to modify existing images</li>
                <li>Download generated images using the download button</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if not st.session_state.api_key_set:
        st.markdown("""
        <div class="feature-box">
            <h3>🚀 Get Started</h3>
            <p>To begin generating and editing images, please:</p>
            <ol>
                <li>Obtain your API key from <a href="https://ai.google.dev/" target="_blank">Google AI Studio</a></li>
                <li>Enter the API key in the sidebar</li>
                <li>Start creating amazing images!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize client
    client, types = initialize_gemini_client(api_key)
    if not client:
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🎨 Generate Images", "✏️ Edit Images", "📁 Gallery"])
    
    with tab1:
        st.header("Generate New Images")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "Image Description",
                placeholder="Describe the image you want to generate...\nExample: A beautiful sunset over mountains with a lake in the foreground",
                height=100,
                help="Be specific and detailed for better results"
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                st.info("Note: Some content policies apply. Generating people may be restricted.")
                
                example_prompts = [
                    "A serene landscape with mountains and a lake",
                    "A futuristic city with flying cars",
                    "A magical forest with glowing mushrooms",
                    "A cozy coffee shop interior",
                    "Abstract art with vibrant colors"
                ]
                
                selected_example = st.selectbox(
                    "Example Prompts",
                    [""] + example_prompts,
                    help="Select an example prompt to get started"
                )
                
                if selected_example and st.button("Use Example"):
                    prompt = selected_example
                    st.rerun()
        
        with col2:
            st.markdown("### Preview")
            if prompt:
                st.info(f"Ready to generate: {prompt[:50]}...")
            else:
                st.info("Enter a prompt to preview")
        
        if st.button("🎨 Generate Image", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("Please enter a description for the image")
                return
            
            try:
                with st.spinner("Generating image... This may take a few moments"):
                    image_data, mime_type = generate_image(client, types, prompt, model_id)
                    
                    if image_data:
                        # Display generated image
                        st.success("✅ Image generated successfully!")
                        
                        # Convert bytes to displayable format
                        image = Image.open(io.BytesIO(image_data))
                        st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                        
                        # Save to session state
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state.generated_images.append({
                            'image': image_data,
                            'prompt': prompt,
                            'timestamp': timestamp,
                            'type': 'generated'
                        })
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Image",
                            data=image_data,
                            file_name=f"generated_image_{timestamp}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate image. Please try again with a different prompt.")
                        
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                logger.error(f"Generation error: {str(e)}")
    
    with tab2:
        st.header("Edit Existing Images")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Image to Edit",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Upload an image you want to edit"
            )
            
            if uploaded_file:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            edit_prompt = st.text_area(
                "Edit Instructions",
                placeholder="Describe how you want to modify the image...\nExample: Change the background to a sunset, add flowers, make it more colorful",
                height=100,
                help="Describe the changes you want to make"
            )
            
            # Example edit prompts
            with st.expander("💡 Edit Examples"):
                edit_examples = [
                    "Change the background to a beach scene",
                    "Add colorful flowers to the foreground",
                    "Make it look like a painting",
                    "Change the lighting to golden hour",
                    "Add snow falling"
                ]
                
                selected_edit = st.selectbox(
                    "Example Edit Prompts",
                    [""] + edit_examples
                )
                
                if selected_edit and st.button("Use Edit Example"):
                    edit_prompt = selected_edit
                    st.rerun()
        
        if st.button("✏️ Edit Image", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload an image to edit")
                return
            
            if not edit_prompt.strip():
                st.error("Please enter editing instructions")
                return
            
            try:
                with st.spinner("Editing image... This may take a few moments"):
                    # Pass the uploaded file directly instead of reading it
                    edited_data, mime_type = edit_image(client, types, edit_prompt, uploaded_file, model_id)
                    
                    if edited_data:
                        st.success("✅ Image edited successfully!")
                        
                        # Display edited image
                        edited_image = Image.open(io.BytesIO(edited_data))
                        
                        # Show before and after
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Before", use_container_width=True)
                        with col2:
                            st.image(edited_image, caption="After", use_container_width=True)
                        
                        # Save to session state
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state.generated_images.append({
                            'image': edited_data,
                            'prompt': edit_prompt,
                            'timestamp': timestamp,
                            'type': 'edited'
                        })
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Edited Image",
                            data=edited_data,
                            file_name=f"edited_image_{timestamp}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to edit image. Please try again with different instructions.")
                        
            except Exception as e:
                st.error(f"Error editing image: {str(e)}")
                logger.error(f"Edit error: {str(e)}")
    
    with tab3:
        st.header("Image Gallery")
        
        if not st.session_state.generated_images:
            st.info("No images generated yet. Create some images in the other tabs!")
            return
        
        # Display images in gallery
        cols = st.columns(3)
        
        for idx, img_data in enumerate(reversed(st.session_state.generated_images)):
            col_idx = idx % 3
            
            with cols[col_idx]:
                # Display image
                image = Image.open(io.BytesIO(img_data['image']))
                st.image(image, use_column_width=True)
                
                # Image info
                st.caption(f"**{img_data['type'].title()}**: {img_data['prompt'][:50]}...")
                st.caption(f"**Created**: {img_data['timestamp']}")
                
                # Download button
                st.download_button(
                    label="📥 Download",
                    data=img_data['image'],
                    file_name=f"{img_data['type']}_image_{img_data['timestamp']}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                    use_container_width=True
                )
        
        # Clear gallery button
        if st.button("🗑️ Clear Gallery", use_container_width=True):
            st.session_state.generated_images = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p><small>Remember to keep your API key secure and never share it publicly.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()