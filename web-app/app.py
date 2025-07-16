# app.py
import cv2
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import os
import threading
import time # For potential delays if needed

app = Flask(__name__)

# Latest model
MODEL_PATH = '160p500es-seg.pt'
WEBCAM_INDEX = 0  # Typically 0 for the default webcam

# --- Global Variables for Video Capture, Model, and Last Detection ---
camera = None
model = None

# Store the detection results of the last processed frame
# This dictionary will hold:
# - 'frame_shape': (height, width) of the original frame
# - 'boxes': List of detected bounding boxes (xyxy, conf, cls)
# - 'masks': List of detected segmentation masks (xy coordinates)
# - 'names': Dictionary mapping class IDs to class names
last_detection_results = {
    'frame_shape': (0, 0),
    'boxes': [],
    'masks': [],
    'names': {}
}
# Lock to ensure thread-safe access to last_detection_results
detection_lock = threading.Lock()

def load_model():
    """Loads the YOLOv8 segmentation model."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please ensure '160p500es-seg.pt' is in the same directory as app.py")
            print("or update the MODEL_PATH variable with the correct path.")
            return False

        # Load the YOLOv8 segmentation model
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
    It also updates the global last_detection_results.
    """
    global camera, model, last_detection_results, detection_lock

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
                results = model(frame, stream=True, verbose=False)

                # Prepare data to store for click detection
                current_boxes = []
                current_masks = []
                class_names = model.names # Get class names from the model

                # Iterate over results and draw predictions
                for r in results:
                    # Store bounding boxes and draw them
                    if r.boxes:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            current_boxes.append({
                                'xyxy': [x1, y1, x2, y2],
                                'conf': conf,
                                'cls': cls,
                                'name': class_names[cls]
                            })
                            annotated_frame = r.plot()
                    else:
                        annotated_frame = r.plot()

                # Update last_detection_results in a thread-safe manner
                with detection_lock:
                    last_detection_results['frame_shape'] = frame.shape[:2] # (height, width)
                    last_detection_results['boxes'] = current_boxes
                    last_detection_results['masks'] = current_masks
                    last_detection_results['names'] = class_names

                # Encode the frame in JPEG format for streaming
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error during frame processing or inference: {e}")
                # If an error occurs, try to continue to the next frame
                continue

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams the video feed with YOLOv8 predictions."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click_box', methods=['POST'])
def click_box():
    """
    Receives click coordinates from the frontend,
    and checks if they fall within any detected bounding box.
    """
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    click_x_canvas = data.get('x')
    click_y_canvas = data.get('y')
    canvas_width = data.get('canvasWidth')
    canvas_height = data.get('canvasHeight')

    if None in [click_x_canvas, click_y_canvas, canvas_width, canvas_height]:
        return jsonify({"status": "error", "message": "Missing click coordinates or canvas dimensions"}), 400

    print(f"Received click: ({click_x_canvas}, {click_y_canvas}) on canvas ({canvas_width}x{canvas_height})")

    with detection_lock:
        frame_height, frame_width = last_detection_results['frame_shape']
        boxes = last_detection_results['boxes']
        names = last_detection_results['names']

    if frame_width == 0 or frame_height == 0:
        return jsonify({"status": "info", "message": "No frame data available yet. Try again."})

    # Calculate scaling factors to map canvas coordinates to original frame coordinates
    # The canvasWidth and canvasHeight here refer to the *displayed* dimensions of the video element
    scale_x = frame_width / canvas_width
    scale_y = frame_height / canvas_height

    # Scale click coordinates to the original frame's resolution
    click_x_frame = int(click_x_canvas * scale_x)
    click_y_frame = int(click_y_canvas * scale_y)

    print(f"Scaled click: ({click_x_frame}, {click_y_frame}) on frame ({frame_width}x{frame_height})")

    clicked_objects = []
    for box_data in boxes:
        x1, y1, x2, y2 = box_data['xyxy']
        # Check if the scaled click coordinates are within the bounding box
        if x1 <= click_x_frame <= x2 and y1 <= click_y_frame <= y2:
            clicked_objects.append({
                "class_id": box_data['cls'],
                "class_name": box_data['name'],
                "confidence": box_data['conf'],
                "box_coordinates": box_data['xyxy']
            })

    if clicked_objects:
        print(f"Clicked on: {clicked_objects}")
        return jsonify({"status": "success", "clicked_objects": clicked_objects})
    else:
        print("No object clicked.")
        return jsonify({"status": "info", "message": "No object detected at the clicked location."})

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
    # Using threaded=True allows Flask to handle multiple requests concurrently,
    # which is necessary for the video stream and click events to work together.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
