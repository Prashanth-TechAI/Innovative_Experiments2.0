import streamlit as st
import tempfile
import os
from groq import Groq
import time
import requests
import threading
import queue
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Real-Time STT Comparison",
    page_icon="🎤",
    layout="wide"
)

# Get API keys from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Check if API keys are loaded
if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
    st.error("❌ API keys not found! Please add GROQ_API_KEY and DEEPGRAM_API_KEY to your .env file")
    st.stop()

# Initialize session state
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {
        'turbo': {'text': '', 'time': 0, 'status': 'Ready'},
        'large': {'text': '', 'time': 0, 'status': 'Ready'},
        'deepgram': {'text': '', 'time': 0, 'status': 'Ready'}
    }
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None

def transcribe_groq_turbo(audio_bytes, result_queue):
    """Transcribe with Whisper Turbo in parallel"""
    try:
        start_time = time.time()
        result_queue.put(('turbo', 'status', 'Transcribing...'))
        
        client = Groq(api_key=GROQ_API_KEY)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_file_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="en"
                )
            duration = time.time() - start_time
            result_queue.put(('turbo', 'complete', transcription.text, duration))
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        result_queue.put(('turbo', 'error', str(e), 0))

def transcribe_groq_large(audio_bytes, result_queue):
    """Transcribe with Whisper Large in parallel"""
    try:
        start_time = time.time()
        result_queue.put(('large', 'status', 'Transcribing...'))
        
        client = Groq(api_key=GROQ_API_KEY)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_file_path, audio_file.read()),
                    model="whisper-large-v3",
                    language="en"
                )
            duration = time.time() - start_time
            result_queue.put(('large', 'complete', transcription.text, duration))
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        result_queue.put(('large', 'error', str(e), 0))

def transcribe_deepgram(audio_bytes, result_queue):
    """Transcribe with Deepgram in parallel"""
    try:
        start_time = time.time()
        result_queue.put(('deepgram', 'status', 'Transcribing...'))
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        params = {
            "model": "nova-2",
            "language": "en-US",
            "smart_format": "true"
        }
        
        response = requests.post(url, headers=headers, params=params, data=audio_bytes)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
            duration = time.time() - start_time
            result_queue.put(('deepgram', 'complete', transcript, duration))
        else:
            result_queue.put(('deepgram', 'error', f"{response.status_code}: {response.text}", 0))
    except Exception as e:
        result_queue.put(('deepgram', 'error', str(e), 0))

# Header
st.title("🎤 Real-Time STT Comparison")
st.markdown("**All 3 models process simultaneously - Compare speed and accuracy live!**")
st.markdown("---")

# Big centered microphone section
st.markdown("<h2 style='text-align: center;'>🎙️ SPEAK NOW</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Click the microphone below to start recording</h4>", unsafe_allow_html=True)

# Audio recorder centered
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="6x",
        key="audio_recorder"
    )

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>REAL-TIME PROCESSING OF 3 MODELS</h3>", unsafe_allow_html=True)
st.markdown("")

# Three columns for real-time display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='text-align: center;'>whisper-large-v3</h4>", unsafe_allow_html=True)
    turbo_status = st.empty()
    turbo_box = st.empty()
    turbo_time = st.empty()
    
    turbo_status.info(f"📊 Status: {st.session_state.transcriptions['turbo']['status']}")
    turbo_box.text_area(
        "response :",
        value=st.session_state.transcriptions['turbo']['text'],
        height=300,
        key="turbo_display",
        disabled=True
    )
    if st.session_state.transcriptions['turbo']['time'] > 0:
        turbo_time.success(f"⏱️ {st.session_state.transcriptions['turbo']['time']:.2f}s | {len(st.session_state.transcriptions['turbo']['text'])} chars")

with col2:
    st.markdown("<h4 style='text-align: center;'>whisper-large-v3-turbo</h4>", unsafe_allow_html=True)
    large_status = st.empty()
    large_box = st.empty()
    large_time = st.empty()
    
    large_status.info(f"📊 Status: {st.session_state.transcriptions['large']['status']}")
    large_box.text_area(
        "response :",
        value=st.session_state.transcriptions['large']['text'],
        height=300,
        key="large_display",
        disabled=True
    )
    if st.session_state.transcriptions['large']['time'] > 0:
        large_time.success(f"⏱️ {st.session_state.transcriptions['large']['time']:.2f}s | {len(st.session_state.transcriptions['large']['text'])} chars")

with col3:
    st.markdown("<h4 style='text-align: center;'>Deepgram Nova Model</h4>", unsafe_allow_html=True)
    deepgram_status = st.empty()
    deepgram_box = st.empty()
    deepgram_time = st.empty()
    
    deepgram_status.info(f"📊 Status: {st.session_state.transcriptions['deepgram']['status']}")
    deepgram_box.text_area(
        "response :",
        value=st.session_state.transcriptions['deepgram']['text'],
        height=300,
        key="deepgram_display",
        disabled=True
    )
    if st.session_state.transcriptions['deepgram']['time'] > 0:
        deepgram_time.success(f"⏱️ {st.session_state.transcriptions['deepgram']['time']:.2f}s | {len(st.session_state.transcriptions['deepgram']['text'])} chars")

# Process audio when recorded
if audio_bytes and audio_bytes != st.session_state.last_audio:
    st.session_state.last_audio = audio_bytes
    
    # Reset transcriptions
    st.session_state.transcriptions = {
        'turbo': {'text': '', 'time': 0, 'status': 'Processing...'},
        'large': {'text': '', 'time': 0, 'status': 'Processing...'},
        'deepgram': {'text': '', 'time': 0, 'status': 'Processing...'}
    }
    
    # Update status displays
    turbo_status.warning("🔄 Processing...")
    large_status.warning("🔄 Processing...")
    deepgram_status.warning("🔄 Processing...")
    
    # Create queue for results
    result_queue = queue.Queue()
    
    # Start all three transcriptions in parallel threads
    thread1 = threading.Thread(target=transcribe_groq_turbo, args=(audio_bytes, result_queue))
    thread2 = threading.Thread(target=transcribe_groq_large, args=(audio_bytes, result_queue))
    thread3 = threading.Thread(target=transcribe_deepgram, args=(audio_bytes, result_queue))
    
    thread1.start()
    thread2.start()
    thread3.start()
    
    # Monitor results in real-time
    completed = {'turbo': False, 'large': False, 'deepgram': False}
    
    while not all(completed.values()):
        try:
            result = result_queue.get(timeout=0.1)
            model, msg_type, *data = result
            
            if msg_type == 'status':
                st.session_state.transcriptions[model]['status'] = data[0]
                if model == 'turbo':
                    turbo_status.warning(f"🔄 {data[0]}")
                elif model == 'large':
                    large_status.warning(f"🔄 {data[0]}")
                else:
                    deepgram_status.warning(f"🔄 {data[0]}")
            
            elif msg_type == 'complete':
                text, duration = data
                st.session_state.transcriptions[model]['text'] = text
                st.session_state.transcriptions[model]['time'] = duration
                st.session_state.transcriptions[model]['status'] = 'Complete ✅'
                completed[model] = True
                
                # Update display immediately
                if model == 'turbo':
                    turbo_status.success("✅ Complete!")
                    turbo_box.text_area("response :", value=text, height=300, key=f"turbo_{time.time()}", disabled=True)
                    turbo_time.success(f"⏱️ {duration:.2f}s | {len(text)} chars")
                elif model == 'large':
                    large_status.success("✅ Complete!")
                    large_box.text_area("response :", value=text, height=300, key=f"large_{time.time()}", disabled=True)
                    large_time.success(f"⏱️ {duration:.2f}s | {len(text)} chars")
                else:
                    deepgram_status.success("✅ Complete!")
                    deepgram_box.text_area("response :", value=text, height=300, key=f"deepgram_{time.time()}", disabled=True)
                    deepgram_time.success(f"⏱️ {duration:.2f}s | {len(text)} chars")
            
            elif msg_type == 'error':
                error_msg = data[0]
                st.session_state.transcriptions[model]['text'] = f"❌ Error: {error_msg}"
                st.session_state.transcriptions[model]['status'] = 'Error ❌'
                completed[model] = True
                
                if model == 'turbo':
                    turbo_status.error("❌ Error")
                    turbo_box.text_area("response :", value=f"Error: {error_msg}", height=300, key=f"turbo_err_{time.time()}", disabled=True)
                elif model == 'large':
                    large_status.error("❌ Error")
                    large_box.text_area("response :", value=f"Error: {error_msg}", height=300, key=f"large_err_{time.time()}", disabled=True)
                else:
                    deepgram_status.error("❌ Error")
                    deepgram_box.text_area("response :", value=f"Error: {error_msg}", height=300, key=f"deepgram_err_{time.time()}", disabled=True)
            
        except queue.Empty:
            time.sleep(0.1)
    
    # Wait for all threads to complete
    thread1.join()
    thread2.join()
    thread3.join()
    
    # Show winner
    st.markdown("---")
    times = {
        'Whisper Turbo': st.session_state.transcriptions['turbo']['time'],
        'Whisper Large': st.session_state.transcriptions['large']['time'],
        'Deepgram Nova-2': st.session_state.transcriptions['deepgram']['time']
    }
    
    valid_times = {k: v for k, v in times.items() if v > 0}
    if valid_times:
        fastest = min(valid_times, key=valid_times.get)
        st.success(f"🏆 **FASTEST MODEL: {fastest}** ({valid_times[fastest]:.2f} seconds)")
    
    st.info("💡 **Compare the results above** to see which model gives the most accurate transcription!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<p>Real-Time STT Comparison Tool | All 3 models process simultaneously</p>"
    "<p style='font-size: 12px;'>Watch them race! See speed and accuracy differences live.</p>"
    "</div>",
    unsafe_allow_html=True
)