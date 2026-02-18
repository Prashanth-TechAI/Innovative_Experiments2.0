from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import subprocess
import threading
import time
import requests
import socket
import os
import signal
import psutil
from typing import Optional

app = FastAPI(
    title="AI Image Generator API",
    description="FastAPI wrapper for Streamlit AI Image Generator & Editor",
    version="1.0.0"
)

# Global variables to track Streamlit process
streamlit_process = None
streamlit_port = 8501
streamlit_url = f"http://localhost:{streamlit_port}"

def find_free_port():
    """Find a free port for Streamlit"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port: int):
    """Kill any process running on the specified port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                if connections:
                    for conn in connections:
                        if conn.laddr.port == port:
                            print(f"Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}")
                            proc.kill()
                            return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
    return False

def start_streamlit_app():
    """Start the Streamlit application"""
    global streamlit_process, streamlit_port, streamlit_url
    
    try:
        # Kill any existing process on the port
        if is_port_in_use(streamlit_port):
            print(f"Port {streamlit_port} is in use, attempting to free it...")
            kill_process_on_port(streamlit_port)
            time.sleep(2)
        
        # If port is still in use, find a new one
        if is_port_in_use(streamlit_port):
            streamlit_port = find_free_port()
            streamlit_url = f"http://localhost:{streamlit_port}"
            print(f"Using alternative port: {streamlit_port}")
        
        # Start Streamlit process
        cmd = [
            "streamlit", "run", "imgdy.py",
            "--server.port", str(streamlit_port),
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none"
        ]
        
        print(f"Starting Streamlit app with command: {' '.join(cmd)}")
        streamlit_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        # Wait for Streamlit to start
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(streamlit_url, timeout=2)
                if response.status_code == 200:
                    print(f"Streamlit app started successfully on {streamlit_url}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if streamlit_process.poll() is not None:
                stdout, stderr = streamlit_process.communicate()
                print(f"Streamlit process exited with code {streamlit_process.returncode}")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False
            
            time.sleep(1)
            print(f"Waiting for Streamlit to start... Attempt {attempt + 1}/{max_attempts}")
        
        print("Streamlit failed to start within timeout period")
        return False
        
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        return False

def stop_streamlit_app():
    """Stop the Streamlit application"""
    global streamlit_process
    
    if streamlit_process:
        try:
            if os.name != 'nt':  # Unix/Linux/macOS
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
            else:  # Windows
                streamlit_process.terminate()
            
            streamlit_process.wait(timeout=10)
            print("Streamlit app stopped successfully")
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGKILL)
            else:
                streamlit_process.kill()
            print("Streamlit app force killed")
        except Exception as e:
            print(f"Error stopping Streamlit: {e}")
        finally:
            streamlit_process = None

@app.get("/", summary="Root endpoint", description="Welcome message and API information")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Image Generator & Editor API",
        "description": "FastAPI wrapper for Streamlit AI Image Generator",
        "version": "1.0.0",
        "endpoints": {
            "launch_app": "/launch-streamlit-app",
            "stop_app": "/stop-streamlit-app",
            "app_status": "/app-status",
            "docs": "/docs"
        }
    }

@app.post("/launch-streamlit-app", 
          summary="Launch Streamlit App", 
          description="Starts the Streamlit AI Image Generator application and returns the access URL")
async def launch_streamlit_app():
    """
    Launch the Streamlit AI Image Generator & Editor application.
    
    This endpoint:
    1. Starts a Streamlit server running the image generator app
    2. Returns the URL where you can access the live application
    3. The app provides full image generation and editing capabilities
    
    Returns:
        - success: Boolean indicating if the app started successfully
        - url: Direct URL to access the Streamlit application
        - message: Status message
        - port: Port number where the app is running
    """
    global streamlit_process, streamlit_url
    
    try:
        # Check if imgdy.py exists
        if not os.path.exists("imgdy.py"):
            raise HTTPException(
                status_code=404, 
                detail="imgdy.py file not found. Please ensure the Streamlit app file is in the same directory."
            )
        
        # If already running, return existing URL
        if streamlit_process and streamlit_process.poll() is None:
            try:
                response = requests.get(streamlit_url, timeout=2)
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Streamlit app is already running",
                        "url": streamlit_url,
                        "port": streamlit_port,
                        "redirect_url": streamlit_url
                    }
            except:
                pass
        
        # Start the Streamlit app in a separate thread
        def start_app():
            start_streamlit_app()
        
        thread = threading.Thread(target=start_app)
        thread.daemon = True
        thread.start()
        
        # Wait a bit for the app to start
        time.sleep(3)
        
        # Check if the app started successfully
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                if streamlit_process and streamlit_process.poll() is None:
                    response = requests.get(streamlit_url, timeout=2)
                    if response.status_code == 200:
                        return {
                            "success": True,
                            "message": "Streamlit AI Image Generator & Editor launched successfully!",
                            "url": streamlit_url,
                            "port": streamlit_port,
                            "redirect_url": streamlit_url,
                            "instructions": [
                                "1. Click on the URL above or copy it to your browser",
                                "2. Enter your Google AI Studio API key in the sidebar",
                                "3. Start generating and editing images!",
                                "4. Use the tabs to switch between Generate, Edit, and Gallery"
                            ]
                        }
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(0.5)
        
        # If we get here, the app didn't start successfully
        if streamlit_process:
            stop_streamlit_app()
        
        raise HTTPException(
            status_code=500,
            detail="Failed to start Streamlit application. Please check if Streamlit is installed and imgdy.py is valid."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error launching Streamlit app: {str(e)}")

@app.post("/stop-streamlit-app",
          summary="Stop Streamlit App",
          description="Stops the running Streamlit application")
async def stop_streamlit_app_endpoint():
    """
    Stop the running Streamlit AI Image Generator application.
    
    Returns:
        - success: Boolean indicating if the app was stopped successfully
        - message: Status message
    """
    global streamlit_process
    
    try:
        if streamlit_process and streamlit_process.poll() is None:
            stop_streamlit_app()
            return {
                "success": True,
                "message": "Streamlit app stopped successfully"
            }
        else:
            return {
                "success": True,
                "message": "No Streamlit app was running"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping Streamlit app: {str(e)}")

@app.get("/app-status",
         summary="Check App Status",
         description="Check if the Streamlit application is currently running")
async def app_status():
    """
    Check the status of the Streamlit application.
    
    Returns:
        - running: Boolean indicating if the app is running
        - url: URL of the running app (if running)
        - port: Port number (if running)
        - message: Status message
    """
    global streamlit_process, streamlit_url
    
    try:
        if streamlit_process and streamlit_process.poll() is None:
            # Check if the app is actually responding
            try:
                response = requests.get(streamlit_url, timeout=2)
                if response.status_code == 200:
                    return {
                        "running": True,
                        "url": streamlit_url,
                        "port": streamlit_port,
                        "message": "Streamlit app is running and accessible"
                    }
            except requests.exceptions.RequestException:
                pass
        
        return {
            "running": False,
            "url": None,
            "port": None,
            "message": "Streamlit app is not running"
        }
        
    except Exception as e:
        return {
            "running": False,
            "url": None,
            "port": None,
            "message": f"Error checking app status: {str(e)}"
        }

@app.get("/redirect-to-app",
         summary="Redirect to App",
         description="Redirects directly to the running Streamlit application")
async def redirect_to_app():
    """
    Redirect directly to the Streamlit application if it's running.
    """
    global streamlit_process, streamlit_url
    
    if streamlit_process and streamlit_process.poll() is None:
        try:
            response = requests.get(streamlit_url, timeout=2)
            if response.status_code == 200:
                return RedirectResponse(url=streamlit_url)
        except requests.exceptions.RequestException:
            pass
    
    raise HTTPException(
        status_code=404,
        detail="Streamlit app is not running. Please use /launch-streamlit-app first."
    )

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when FastAPI shuts down"""
    print("Shutting down FastAPI server...")
    stop_streamlit_app()

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("Access the API documentation at: http://localhost:8000/docs")
    print("Use the /launch-streamlit-app endpoint to start the image generator")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)