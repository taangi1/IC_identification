# app.py
import cv2
import torch
from flask import Flask, render_template, Response
from ultralytics import YOLO
import os

app = Flask(__name__)

# Path to the latest trained model
MODEL_PATH = '160p500es-seg.pt'
WEBCAM_INDEX = 0  # Typically 0 for the default webcam

# --- Global Variables for Video Capture and Model ---
camera = None
model = None

def load_model():
    """Loads the YOLOv8 segmentation model."""
    global model
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please ensure '160p500es-seg.pt' is in the same directory as app.py")
            print("or update the MODEL_PATH variable with the correct path.")
            return False

        model = YOLO(MODEL_PATH)
        print(f"YOLOv8 segmentation model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return False

def initialize_camera():
    """Initializes the webcam."""
    global camera
    try:
        camera = cv2.VideoCapture(WEBCAM_INDEX)
        if not camera.isOpened():
            print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
            print("Please check if the webcam is connected and not in use by another application.")
            return False
        print(f"Webcam initialized at index {WEBCAM_INDEX}")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def generate_frames():
    """
    Generator function to capture frames from the webcam,
    perform YOLOv8 inference, and yield JPEG encoded frames.
    """
    global camera, model

    if not camera:
        if not initialize_camera():
            print("Camera not available. Cannot generate frames.")
            return

    if not model:
        if not load_model():
            print("Model not loaded. Cannot generate frames.")
            return

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera. Exiting video stream.")
            break
        else:
            try:
                # Perform YOLOv8 inference
                # The 'stream=True' argument makes it suitable for video streams,
                # processing frames as they arrive.
                results = model(frame, stream=True, verbose=False)

                try:
                    first_result = next(results)
                    annotated_frame = first_result.plot()
                except StopIteration:
                    # Handle the case where the generator is empty (no results)
                    print("Warning: Model returned no results for the frame.")
                    return

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')
            except Exception as e:
                print(f"Error during frame processing or inference: {e}")
                # If an error occurs, try to continue or break if persistent
                continue # Try to process the next frame

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams the video feed with YOLOv8 predictions."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Provides a simple status check for model and camera."""
    model_status = "Loaded" if model else "Not Loaded"
    camera_status = "Initialized" if camera and camera.isOpened() else "Not Initialized"
    return f"Model Status: {model_status}, Camera Status: {camera_status}"

@app.after_request
def add_header(response):
    """Adds headers to prevent caching of video feed."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    # Initialize camera and load model when the application starts
    initialize_camera()
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
