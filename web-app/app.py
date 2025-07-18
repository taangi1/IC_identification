from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
from paddleocr import PaddleOCR
from ultralytics import YOLO
import os
import traceback

app = Flask(__name__)

MODEL_PATH = '160p500es-seg.pt'

# --- Global Variables for Video Capture, Models, and Last Detection ---
yolo_lock = threading.Lock()
yolo_model = None
ocr_lock = threading.Lock()
ocr_model = None # New: PaddleOCR model instance

# Store the detection results of the last processed frame
last_detection_results = {
    'frame': None, # New: Store the raw frame for OCR
    'frame_shape': (0, 0),
    'boxes': [],
    'masks': [],
    'names': {}
}
detection_lock = threading.Lock()

def load_yolo_model():
    """Loads the YOLOv8 segmentation model."""
    global yolo_model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: YOLOv8 model file not found at {MODEL_PATH}")
            print("Please ensure '160p500es-seg.pt' is in the same directory as app.py")
            return False
        yolo_model = YOLO(MODEL_PATH)
        print(f"YOLOv11 segmentation model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        return False

def load_ocr_model():
    """Loads the PaddleOCR model."""
    global ocr_model
    try:
        with ocr_lock:
            if ocr_model is None:
                # Initialize PaddleOCR. 'en' for English
                ocr_model = PaddleOCR(
                                    use_doc_orientation_classify=True, 
                                    use_doc_unwarping=False, 
                                    use_textline_orientation=True,
                                    lang='en')
                print("PaddleOCR model loaded.")
    except Exception as e:
        print(f"Error loading PaddleOCR model: {e}")
        print("Please ensure PaddlePaddle and PaddleOCR are correctly installed.")
        return False

def find_available_cameras():
    """
    Checks for available camera indices on the server.
    Tries to open cameras by index from 0 up to a limit (e.g., 10).
    Returns a list of valid integer indices.
    """
    available_indices = []
    # Check up to 10 potential camera indices.
    # This is a practical limit for most systems.
    for i in range(10):
        # On Windows, the DirectShow backend can be slow to initialize.
        # CAP_DSHOW can speed it up, but we'll use the default for cross-platform.
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # The camera at index `i` is available.
            available_indices.append(i)
            # It's crucial to release the camera after checking.
            cap.release()
    print(f"Server discovered camera indices: {available_indices}")
    return available_indices

def generate_frames(camera_id):
    """
    Generator function to capture frames from the specified camera,
    encode them as JPEG, and yield them for streaming.
    """

    global yolo_model, last_detection_results, detection_lock

    try:
        camera_index = int(camera_id)
    except (ValueError, TypeError):
        print(f"Error: Invalid camera_id '{camera_id}'. Must be an integer.")
        return
    
    if not yolo_model:
        if not load_yolo_model():
            print("Segmentation model not loaded. Cannot generate frames.")
            return

    video_capture = cv2.VideoCapture(camera_index)

    if not video_capture.isOpened():
        print(f"Error: Could not open video stream for camera index: {camera_index}")
        return

    print(f"Successfully opened camera index: {camera_index}")

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to read frame from camera. Exiting video stream.")
            break
        else:
            try:
                # Store a copy of the original frame *before* drawing detections
                original_frame_copy = frame.copy()

                # Perform YOLOv11 inference
                with yolo_lock:
                    results = yolo_model(frame, stream=True, verbose=False)

                current_boxes = []
                current_masks = []
                class_names = yolo_model.names

                for r in results:
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
                    last_detection_results['frame'] = original_frame_copy.copy() # Store the original frame
                    last_detection_results['frame_shape'] = frame.shape[:2]
                    last_detection_results['boxes'] = current_boxes
                    last_detection_results['masks'] = current_masks
                    last_detection_results['names'] = class_names

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error during frame processing or inference: {e}")
                continue

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/cameras')
def api_cameras():
    """
    API endpoint to get the list of camera indices available on the server.
    """
    indices = find_available_cameras()
    return jsonify(indices)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    camera_id = request.args.get('camera_id')
    if camera_id is None:
        return "Error: camera_id parameter is missing.", 400
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click_box', methods=['POST'])
def click_box():
    """
    Receives click coordinates from the frontend,
    checks if they fall within any detected bounding box,
    and performs OCR on the corresponding region.
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
        original_frame = last_detection_results['frame'].copy()
        frame_height, frame_width = last_detection_results['frame_shape']
        boxes = last_detection_results['boxes']

    if original_frame is None or frame_width == 0 or frame_height == 0:
        return jsonify({"status": "info", "message": "No frame data available yet. Try again."})

    scale_x = frame_width / canvas_width
    scale_y = frame_height / canvas_height

    click_x_frame = int(click_x_canvas * scale_x)
    click_y_frame = int(click_y_canvas * scale_y)

    print(f"Scaled click: ({click_x_frame}, {click_y_frame}) on frame ({frame_width}x{frame_height})")

    clicked_box_data = None
    for box_data in boxes:
        x1, y1, x2, y2 = box_data['xyxy']
        if x1 <= click_x_frame <= x2 and y1 <= click_y_frame <= y2:
            clicked_box_data = box_data
            break # Found the first box clicked, break

    if clicked_box_data:
        x1, y1, x2, y2 = clicked_box_data['xyxy']
        # Crop the region from the original frame
        cropped_region = original_frame[y1:y2, x1:x2]
        cropped_region_for_ocr = cropped_region.copy()

        if cropped_region.size == 0:
            return jsonify({"status": "error", "message": "Cropped region is empty."})

        try:
            # Perform OCR on the cropped region
            with ocr_lock:
                ocr_results = ocr_model.predict(cropped_region_for_ocr)

            extracted_text = []
            if ocr_results:
                for res in ocr_results:
                    if res: # Check if results are not empty
                        texts = res['rec_texts']
                        for line in texts:
                            if line:
                                extracted_text.append(line)

            if extracted_text:
                ocr_message = "Extracted Text:<br>" + "<br>".join(extracted_text)
                return jsonify({"status": "success", "ocr_result": ocr_message})
            else:
                return jsonify({"status": "info", "message": "No text detected in the clicked region."})

        except Exception as e:
            print(f"Error during OCR: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"Error performing OCR: {e}. Try again"})
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
    load_yolo_model()
    load_ocr_model()
    app.run(host='0.0.0.0', port=5000)
