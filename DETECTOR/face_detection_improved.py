import cv2
import numpy as np
import time
import os
from datetime import datetime
from flask import Flask, Response, render_template_string, send_file, jsonify
from flask_socketio import SocketIO, emit
import base64
import threading
from zeroconf import ServiceInfo, Zeroconf
import socket
import io
import traceback
import urllib.request
import shutil
import hashlib
import math

# Add at the top with other imports
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("pyngrok not installed, remote access will not be available")

# Create Flask and Socket.io app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Define absolute paths for model files
DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(DETECTOR_DIR, 'cascades')
FACE_DETECTOR_DIR = os.path.join(DETECTOR_DIR, 'face_detector')

# Define cascade file paths with absolute paths
CASCADE_FILES = {
    'eye': os.path.join(CASCADE_DIR, 'haarcascade_eye.xml'),
    'nose': os.path.join(CASCADE_DIR, 'haarcascade_mcs_nose.xml'),
    'mouth': os.path.join(CASCADE_DIR, 'haarcascade_smile.xml'),
    'left_ear': os.path.join(CASCADE_DIR, 'haarcascade_mcs_leftear.xml'),
    'right_ear': os.path.join(CASCADE_DIR, 'haarcascade_mcs_rightear.xml')
}

# Define DNN model files with absolute paths
DNN_FILES = {
    'config': os.path.join(FACE_DETECTOR_DIR, 'deploy.prototxt'),
    'model': os.path.join(FACE_DETECTOR_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
}

# Global variables for sharing data between threads
faces_data = []
current_frame_encoded = None
current_frame = None
stop_signal = False
last_capture_faces = []

# Global variables for face detection
face_detector = None
face_net = None
eye_cascade = None
nose_cascade = None
mouth_cascade = None
left_ear_cascade = None
right_ear_cascade = None
dnn_detectors = {}

# Create directory for storing captured faces
if not os.path.exists('detected_faces'):
    os.makedirs('detected_faces')

def get_local_ip():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def register_service():
    """Register the face detection service using Zeroconf"""
    local_ip = get_local_ip()
    port = 5000
    
    zeroconf = Zeroconf()
    service_info = ServiceInfo(
        "_http._tcp.local.",
        "FaceDetectionServer._http._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties={'path': '/'},
        server=f"face-detection-server.local."
    )
    
    print(f"Registering service on {local_ip}:{port}")
    zeroconf.register_service(service_info)
    return zeroconf, service_info

def init_feature_detectors():
    """Initialize feature detectors for facial features"""
    global eye_cascade, nose_cascade, smile_cascade, left_ear_cascade, right_ear_cascade
    
    success = True
    
    # Since we've removed Haar cascades completely, simplify this function
    print("Feature detection disabled - using only SSD ResNet for face detection")
    return success

class DNNFaceDetector:
    """Class to handle different DNN face detection models"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.net = None
        self.initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize the DNN model"""
        try:
            if self.model_name == 'ssd_resnet':
                model_path = os.path.join(FACE_DETECTOR_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
                config_path = os.path.join(FACE_DETECTOR_DIR, 'deploy.prototxt')
                
                if not os.path.exists(model_path) or not os.path.exists(config_path):
                    print(f"Model files not found for SSD ResNet: {model_path}")
                    return False
                
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self.size = (300, 300)
                self.scale = 1.0
                self.mean = [104.0, 177.0, 123.0]
                self.threshold = 0.5
                self.swapRB = True
            
            # Set computation preferences for better performance
            if self.net is not None:
                # Try to use OpenCL if available for better performance
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Safer target for Raspberry Pi
                
                print(f"Successfully initialized {self.model_name}")
                self.initialized = True
                return True
            else:
                print(f"Failed to initialize {self.model_name}")
                return False
                
        except Exception as e:
            print(f"Error initializing {self.model_name}: {e}")
            traceback.print_exc()
            return False
    
    def detect(self, frame):
        """Detect faces using the DNN model"""
        if not self.initialized or self.net is None:
            return []
        
        height, width = frame.shape[:2]
        faces = []
        
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                frame, self.scale, self.size,
                mean=self.mean,
                swapRB=self.swapRB
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Process detections for SSD ResNet
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.threshold:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Convert numpy integers to Python integers
                startX = int(startX)
                startY = int(startY)
                endX = int(endX)
                endY = int(endY)
                
                # Ensure coordinates are within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)
                
                # Skip invalid detections
                if startX >= endX or startY >= endY:
                    continue
                
                # Use standard Python types to avoid JSON serialization issues
                faces.append({
                    'x': int(startX),
                    'y': int(startY),
                    'width': int(endX - startX),
                    'height': int(endY - startY),
                    'confidence': float(confidence)
                })
            
            return faces
            
        except Exception as e:
            print(f"Error in {self.model_name} detection: {e}")
            traceback.print_exc()
            return []

def init_face_detectors():
    """Initialize face detectors - using only SSD ResNet model"""
    global face_detector, face_net, dnn_detectors
    
    success = True
    dnn_detectors = {}
    
    # Initialize only SSD ResNet detector
    try:
        detector = DNNFaceDetector('ssd_resnet')
        if detector.initialized:
            dnn_detectors['ssd_resnet'] = detector
            print("Successfully initialized ssd_resnet detector")
        else:
            print("Failed to initialize ssd_resnet detector - model may not be available")
            success = False
    except Exception as e:
        print(f"Error initializing ssd_resnet detector: {e}")
        success = False
    
    # Make sure detector is working
    if not dnn_detectors:
        print("ERROR: No working face detectors available!")
        success = False
    
    return success

def camera_processing():
    """Process camera feed and detect faces"""
    global faces_data, current_frame_encoded, current_frame, stop_signal
    
    print("Starting camera processing with enhanced face detection")
    
    # Initialize camera with fallback options
    camera_indexes = [0, 1, 2]  # Try multiple camera indexes including Raspberry Pi cam
    cap = None
    
    # For Raspberry Pi - check for Pi camera
    pi_camera = False
    picam2 = None
    try:
        # Check if running on Raspberry Pi
        if os.path.exists('/sys/firmware/devicetree/base/model'):
            with open('/sys/firmware/devicetree/base/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    print("Detected Raspberry Pi hardware")
                    # Try to import picamera2 (better Raspberry Pi camera support)
                    try:
                        # Only import picamera2 if we're on a Raspberry Pi
                        # This avoids the import error on non-Pi systems
                        import importlib
                        picamera2_module = importlib.import_module('picamera2')
                        Picamera2 = picamera2_module.Picamera2
                        
                        picam2 = Picamera2()
                        # Configure the camera for a reasonable resolution
                        picam2.configure(picam2.create_preview_configuration(
                            main={"size": (640, 480), "format": "RGB888"}))
                        picam2.start()
                        pi_camera = True
                        print("Successfully initialized Raspberry Pi camera using picamera2")
                    except ImportError:
                        print("picamera2 not available, trying standard OpenCV camera")
                    except Exception as e:
                        print(f"Error initializing Pi camera: {e}")
    except Exception as e:
        print(f"Error checking for Raspberry Pi: {e}")
    
    # If Pi camera initialization failed or not running on Pi, try OpenCV cameras
    if not pi_camera:
        for idx in camera_indexes:
            try:
                # First try with DirectShow (Windows)
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    # Try V4L2 (Linux/Raspberry Pi)
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if cap is None or not cap.isOpened():
                        # Try without specific backend
                        cap = cv2.VideoCapture(idx)
                        if cap is None or not cap.isOpened():
                            print(f"Failed to open camera with index {idx}")
                            continue
                    
                # Check if we can read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully initialized camera with index {idx}")
                    break
                else:
                    print(f"Could not read frame from camera {idx}")
                    cap.release()
                    cap = None
            except Exception as e:
                print(f"Error initializing camera {idx}: {e}")
                if cap is not None:
                    cap.release()
                    cap = None
    
    if cap is None and not pi_camera:
        print("Failed to initialize any camera")
        return
    
    # Set camera properties for better performance if using OpenCV
    if cap is not None:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance on Raspberry Pi
        except Exception as e:
            print(f"Warning: Could not set camera properties: {e}")
    
    face_detected = False
    last_notification_time = 0
    notification_cooldown = 5  # seconds
    last_frame_time = 0
    frame_interval = 0.2  # ~5 FPS - adjust based on device performance
    frame_count = 0
    last_detection_attempt = 0
    detection_interval = 0.25  # Only try detection every 4 frames to improve performance
    
    # For Raspberry Pi - slower processing to avoid overheating
    if pi_camera:
        detection_interval = 0.5  # Every 2 seconds
        frame_interval = 0.3     # ~3 FPS
    
    print("Enhanced face detection system started. Press 'q' to quit.")
    
    while not stop_signal:
        try:
            # Get frame based on camera type
            if pi_camera:
                # Get frame from picamera2
                frame = picam2.capture_array()
                # Convert from RGB to BGR (OpenCV format)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret = True
            else:
                # Get frame from OpenCV camera
                ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to grab frame, retrying...")
                time.sleep(0.5)
                continue
            
            # Store the current frame for streaming
            current_frame = frame.copy()
            frame_count += 1
            
            # Process frame for face detection at regular intervals
            current_time = time.time()
            
            # Limit how often we perform detection (expensive operation)
            do_detection = current_time - last_detection_attempt >= detection_interval
            
            # Always do streaming updates at the frame_interval rate
            if current_time - last_frame_time >= frame_interval:
                # Process frame for streaming
                output_frame = frame.copy()
                
                # Only run detection at specific intervals to improve performance
                if do_detection:
                    # Detect faces using our function with priority for ssd_resnet
                    detection_result = detect_faces(frame, {
                        'method': 'ssd_resnet',
                        'min_confidence': 0.5,
                        'min_face_size': 30
                    })
                    
                    # Update faces data
                    faces_data = detection_result['faces']
                    last_detection_attempt = current_time
                
                # Draw rectangles for detected faces
                for face in faces_data:
                    # Ensure all values are serializable
                    x = int(face['x'])
                    y = int(face['y'])
                    w = int(face['width'])
                    h = int(face['height'])
                    conf = float(face['confidence'])
                    method = str(face['method'])
                    
                    # Use different colors based on detection method
                    color = (0, 255, 0)  # Default green
                    if method == 'ssd_resnet':
                        color = (255, 0, 255)  # Purple for ssd_resnet
                    
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output_frame, f"{method[:4]} ({conf:.2f})", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw landmarks if available
                    if 'landmarks' in face:
                        for landmark in face['landmarks']:
                            # Ensure landmarks are serializable
                            lx = int(landmark[0])
                            ly = int(landmark[1])
                            cv2.circle(output_frame, (lx, ly), 2, (0, 0, 255), -1)
                
                # Convert frame for streaming
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, buffer = cv2.imencode('.jpg', output_frame, encode_params)
                current_frame_encoded = base64.b64encode(buffer).decode('utf-8')
                
                # Create a serializable copy of faces_data
                serializable_faces = []
                for face in faces_data:
                    serializable_face = {
                        'x': int(face['x']),
                        'y': int(face['y']),
                        'width': int(face['width']),
                        'height': int(face['height']),
                        'confidence': float(face['confidence']),
                        'method': str(face['method'])
                    }
                    # Only include landmarks if present
                    if 'landmarks' in face:
                        serializable_face['landmarks'] = [(int(l[0]), int(l[1])) for l in face['landmarks']]
                    serializable_faces.append(serializable_face)
                
                # Emit frame update with serializable data
                try:
                    socketio.emit('frame_update', {
                        'image': current_frame_encoded,
                        'faces': serializable_faces,
                        'timestamp': float(current_time)
                    })
                except Exception as e:
                    print(f"Error emitting frame update: {e}")
                    traceback.print_exc()
                
                last_frame_time = current_time
                
                # Handle face detection events
                if len(faces_data) > 0:
                    if not face_detected or (current_time - last_notification_time > notification_cooldown):
                        print(f"ALERT: {len(faces_data)} face(s) detected!")
                        last_notification_time = current_time
                        face_detected = True
                        
                        # Emit face detection event with serializable data
                        try:
                            socketio.emit('face_detected', {
                                'count': int(len(faces_data)),
                                'timestamp': float(current_time),
                                'image': current_frame_encoded,
                                'faces': serializable_faces
                            })
                        except Exception as e:
                            print(f"Error emitting face detection event: {e}")
                            traceback.print_exc()
                else:
                    face_detected = False
            
            # Display frame if not running headless
            try:
                # Add FPS counter
                fps = 1.0 / (current_time - (last_frame_time - frame_interval)) if frame_interval > 0 else 0
                cv2.putText(frame, f"Faces: {len(faces_data)} FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Face Detection System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_signal = True
                    break
            except Exception as e:
                # Likely running headless without display
                pass
                
        except Exception as e:
            print(f"Error in camera processing: {e}")
            traceback.print_exc()
            time.sleep(1)  # Wait before retrying
    
    # Cleanup
    if pi_camera:
        try:
            picam2.stop()
        except:
            pass
    elif cap is not None:
        cap.release()
    
    cv2.destroyAllWindows()

# Create a route for the video stream (enhanced for mobile compatibility)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Create a direct image endpoint for devices that don't support MJPEG
@app.route('/current_image.jpg')
def current_image():
    global current_frame
    if current_frame is None:
        # Return a blank image if no frame is available
        blank_image = np.zeros((480, 640, 3), np.uint8)
        _, img_encoded = cv2.imencode('.jpg', blank_image)
        return Response(img_encoded.tobytes(), mimetype='image/jpeg')
    
    # Return the current frame as a JPEG image
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded = cv2.imencode('.jpg', current_frame, encode_params)
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')

# Create a simple HTML page to display the video stream
@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html>
      <head>
        <title>Live Face Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            text-align: center; 
            background-color: #f0f0f0;
          }
          h1 { color: #333; }
          .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }
          .video-container {
            width: 100%;
            margin: 0 auto;
            position: relative;
          }
          img { 
            width: 100%;
            max-width: 100%; 
            border: 1px solid #ddd;
            border-radius: 4px;
          }
        </style>
        <script>
          // Refresh the image every second for browsers that don't support MJPEG
          function setupImageRefresh() {
            const img = document.getElementById('stream-img');
            const streamUrl = "{{ url_for('video_feed') }}";
            const imageUrl = "{{ url_for('current_image') }}";
            
            // Try to use MJPEG first
            img.src = streamUrl;
            
            // If MJPEG fails, fall back to refreshing JPG
            img.onerror = function() {
              console.log("MJPEG stream failed, falling back to refreshing JPEG");
              img.onerror = null;  // Prevent recurring errors
              img.src = imageUrl + "?t=" + new Date().getTime();
              
              // Refresh the image periodically
              setInterval(function() {
                if (img) {
                  img.src = imageUrl + "?t=" + new Date().getTime();
                }
              }, 200);  // Refresh 5 times per second
            };
          }
          
          window.onload = setupImageRefresh;
        </script>
      </head>
      <body>
        <div class="container">
          <h1>Live Face Detection Stream</h1>
          <div class="video-container">
            <img id="stream-img" src="{{ url_for('video_feed') }}" alt="Live Stream" />
          </div>
          <p>Connect to this stream from your mobile app</p>
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_template)

# Create a mobile-friendly page for embedding in WebView
@app.route('/mobile_stream')
def mobile_stream():
    html_template = '''
    <!DOCTYPE html>
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
          body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: #000;
          }
          .video-container {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
          }
          img {
            width: 100%;
            height: 100%;
            object-fit: contain;
          }
        </style>
        <script>
          // Refresh the image for browsers that don't support MJPEG
          function setupStream() {
            const img = document.getElementById('stream-img');
            const streamUrl = "{{ url_for('video_feed') }}";
            const imageUrl = "{{ url_for('current_image') }}";
            
            // Try MJPEG first
            img.src = streamUrl;
            
            // If MJPEG fails, fall back to refreshing JPG
            img.onerror = function() {
              console.log("MJPEG stream failed, falling back to refreshing JPEG");
              img.onerror = null;
              img.src = imageUrl + "?t=" + new Date().getTime();
              
              // Refresh the image periodically
              setInterval(function() {
                if (img) {
                  img.src = imageUrl + "?t=" + new Date().getTime();
                }
              }, 100);  // Refresh 10 times per second
            };
          }
          
          window.onload = setupStream;
        </script>
      </head>
      <body>
        <div class="video-container">
          <img id="stream-img" alt="Live Stream" />
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_template)

@socketio.on('request_stream')
def handle_stream_request():
    print("Client requested video stream")
    # Send the video stream URL to the client with the mobile-friendly endpoint
    emit('stream_acknowledged', {
        'status': 'streaming',
        'stream_url': f'http://{get_local_ip()}:5000/mobile_stream'
    })

@socketio.on('request_capture')
def handle_capture_request():
    global current_frame, last_capture_faces
    print("Client requested frame capture")
    
    if current_frame is not None:
        # Create a high-quality copy of the current frame
        capture_frame = current_frame.copy()
        
        # Clear the previous capture faces
        last_capture_faces = []
        
        # Detect faces using our function with both detection methods for better results
        try:
            # Try ssd_resnet first (more accurate but might not be available)
            result_ssd_resnet = detect_faces(capture_frame, {'method': 'ssd_resnet', 'min_confidence': 0.5})
            
            # Use the result with more faces, or default to ssd_resnet result
            if result_ssd_resnet['num_faces'] > 0:
                print(f"Using ssd_resnet detection with {result_ssd_resnet['num_faces']} faces")
                detect_result = result_ssd_resnet
                faces_data = result_ssd_resnet['faces']
            else:
                print(f"Using ssd_resnet detection with {result_ssd_resnet['num_faces']} faces")
                detect_result = result_ssd_resnet
                faces_data = result_ssd_resnet['faces']
            
            # last_capture_faces should already be updated by the detect_faces function
            # but update it again just to be sure
            if len(last_capture_faces) == 0 and len(faces_data) > 0:
                print("No face crops in last_capture_faces, but faces were detected. Recapturing...")
                # Try again with the method that worked better
                detect_faces(capture_frame, {'method': 'ssd_resnet', 'min_confidence': 0.5})
            
            # Draw rectangles for detected faces for visual feedback
            for face in faces_data:
                x, y, w, h = face['x'], face['y'], face['width'], face['height']
                conf = face['confidence']
                
                # Draw rectangle on the capture frame
                cv2.rectangle(capture_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(capture_frame, f"Face ({conf:.2f})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error detecting faces for capture: {e}")
            traceback.print_exc()
        
        # Encode with high quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', capture_frame, encode_params)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send back the high-quality image
        emit('capture_result', {
            'image': frame_base64,
            'timestamp': time.time(),
            'faces': faces_data if 'faces_data' in locals() else [],
            'face_count': len(last_capture_faces),
            'detection_method': detect_result['detection_method'] if 'detect_result' in locals() else 'none'
        })
    else:
        emit('capture_result', {
            'error': 'No frame available',
            'timestamp': time.time()
        })

@socketio.on('request_face_crops')
def handle_face_crops_request(data):
    global last_capture_faces
    
    print(f"Client requested face crops, {len(last_capture_faces)} available")
    
    if not last_capture_faces:
        emit('face_crops_result', {
            'error': 'No face crops available',
            'timestamp': time.time()
        })
        return
    
    timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Prepare face crops as base64 images
    face_results = []
    try:
        for i, face_data in enumerate(last_capture_faces):
            try:
                face_crop = face_data['crop']
                
                # Ensure the crop is valid
                if face_crop is None or face_crop.size == 0:
                    print(f"Invalid face crop for face {i+1}, skipping")
                    continue
                
                # Add padding around the face (20% on each side)
                h, w = face_crop.shape[:2]
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                
                # Add a border of black pixels for clearer separation
                face_crop_padded = cv2.copyMakeBorder(
                    face_crop, 
                    pad_h, pad_h, pad_w, pad_w, 
                    cv2.BORDER_CONSTANT, 
                    value=[0, 0, 0]
                )
                
                # Add debug visualization - green outline
                debug_crop = face_crop_padded.copy()
                cv2.rectangle(debug_crop, (0, 0), (debug_crop.shape[1]-1, debug_crop.shape[0]-1), (0, 255, 0), 2)
                
                # Add face index number to the image for reference
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(debug_crop, f"Face #{i+1}", (10, 25), font, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_crop, f"Conf: {face_data['confidence']:.2f}", (10, 50), font, 0.7, (0, 255, 0), 2)
                
                # Save to disk for debug purposes
                face_filename = f"detected_faces/face_{i+1}_conf_{face_data['confidence']:.2f}_{timestamp}.jpg"
                try:
                    cv2.imwrite(face_filename, debug_crop)
                    print(f"Face {i+1} saved as {face_filename}")
                except Exception as e:
                    print(f"Error saving face {i+1} to disk: {e}")
                
                # Encode for transmission - use higher quality for better results
                try:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    success, buffer = cv2.imencode('.jpg', debug_crop, encode_params)
                    
                    if not success:
                        print(f"Failed to encode face {i+1}, skipping")
                        continue
                        
                    crop_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Check if the encoded image is valid
                    if not crop_base64 or len(crop_base64) < 100:
                        print(f"Invalid base64 data for face {i+1}, length: {len(crop_base64) if crop_base64 else 0}")
                        continue
                    
                    print(f"Successfully encoded face {i+1}, base64 length: {len(crop_base64)}")
                    
                    face_results.append({
                        'index': i,
                        'x': face_data['x'],
                        'y': face_data['y'],
                        'width': face_data['width'],
                        'height': face_data['height'],
                        'confidence': face_data['confidence'],
                        'image': crop_base64
                    })
                except Exception as e:
                    print(f"Error encoding face {i+1}: {e}")
                    traceback.print_exc()
            except Exception as face_error:
                print(f"Error processing face {i+1}: {face_error}")
                traceback.print_exc()
        
        # Send all face crops to the client
        print(f"Sending {len(face_results)} face crops to client")
        emit('face_crops_result', {
            'faces': face_results,
            'timestamp': time.time(),
            'count': len(face_results)
        })
    except Exception as e:
        print(f"Error in handle_face_crops_request: {e}")
        traceback.print_exc()
        emit('face_crops_result', {
            'error': f'Error processing face crops: {str(e)}',
            'timestamp': time.time()
        })

@socketio.on('set_quality')
def handle_quality_change(data):
    global frame_interval
    quality = data.get('quality', 'medium')
    print(f"Stream quality changed to {quality}")
    
    if quality == 'low':
        frame_interval = 0.4  # ~2.5 FPS - extremely stable
    elif quality == 'medium':
        frame_interval = 0.25  # ~4 FPS - good balance
    else:  # high
        frame_interval = 0.18  # ~5.5 FPS - faster but may flicker on slower devices

@app.route('/api/status')
def api_status():
    """API endpoint to check server status"""
    global face_detector, face_net
    
    # Check for SSD ResNet detector
    detection_method = 'unknown'
    if 'ssd_resnet' in dnn_detectors:
        detection_method = 'SSD ResNet'
    else:
        detection_method = 'No working detector'
    
    # Return basic status information
    return jsonify({
        'status': 'online',
        'version': '1.0',
        'face_detector': detection_method,
        'timestamp': datetime.now().isoformat(),
        'server_ip': get_local_ip(),
        'endpoints': {
            'video_feed': '/video_feed',
            'current_image': '/current_image.jpg',
            'mobile_stream': '/mobile_stream',
            'test_detection': '/api/test_detection'
        }
    })

def perform_self_test():
    """Perform a self-test to verify the face detection system is working"""
    try:
        print("Performing face detection self-test...")
        
        # Check if we have at least one detector available
        if not dnn_detectors or 'ssd_resnet' not in dnn_detectors:
            print("❌ Self-test ERROR: No face detectors available")
            return False
        
        # Test with a simple blank image
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        test_img = cv2.rectangle(test_img, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Try to detect faces
        result = detect_faces(test_img)
        
        # Check if detection function runs without errors
        if result is None:
            print("❌ Self-test ERROR: detect_faces returned None")
            return False
            
        # Updated check for the new return format (dictionary)
        if 'faces' in result:
            print("✓ Self-test passed: detect_faces returned proper format")
            return True
        else:
            print("❌ Self-test ERROR: Missing 'faces' key in result")
            return False
            
    except Exception as e:
        print(f"❌ Self-test ERROR: {e}")
        traceback.print_exc()
        return False

@app.route('/api/test_detection')
def test_detection():
    """Test face detection with a realistic sample image"""
    try:
        # Create a more realistic test image with a face
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Fill with skin tone
        test_img[:, :] = [204, 172, 156]  # BGR skin tone
        
        # Draw face oval
        cv2.ellipse(test_img, (200, 200), (120, 160), 0, 0, 360, (226, 194, 178), -1)
        
        # Draw eyes
        # Left eye
        cv2.ellipse(test_img, (150, 160), (30, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(test_img, (150, 160), 10, (70, 50, 50), -1)
        
        # Right eye
        cv2.ellipse(test_img, (250, 160), (30, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(test_img, (250, 160), 10, (70, 50, 50), -1)
        
        # Draw eyebrows
        cv2.line(test_img, (120, 130), (180, 140), (70, 35, 35), 5)
        cv2.line(test_img, (220, 140), (280, 130), (70, 35, 35), 5)
        
        # Draw nose
        cv2.line(test_img, (200, 160), (190, 210), (178, 146, 129), 8)
        cv2.line(test_img, (190, 210), (210, 210), (178, 146, 129), 8)
        
        # Draw mouth
        cv2.ellipse(test_img, (200, 260), (60, 25), 0, 0, 180, (151, 104, 138), -1)
        
        # Try different detection methods to see which one works better
        result_ssd_resnet = detect_faces(test_img, {'method': 'ssd_resnet', 'min_confidence': 0.3})
        result_haar = detect_faces(test_img, {'method': 'haar'})
        
        # Use the result with more faces, or default to Haar cascade result
        result = result_ssd_resnet if result_ssd_resnet['num_faces'] >= result_haar['num_faces'] and result_ssd_resnet['num_faces'] > 0 else result_haar
        
        # Add the test image to the response
        _, buffer = cv2.imencode('.jpg', test_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'message': 'Face detection test completed',
            'test_image': f'data:image/jpeg;base64,{img_base64}',
            'detected_faces': result['num_faces'],
            'detection_method': result['detection_method'],
            'details': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Face detection test failed: {str(e)}',
            'error': traceback.format_exc()
        })

@socketio.on('request_single_face')
def handle_single_face_request(data):
    global current_frame, last_capture_faces
    print(f"Client requested single face capture for index {data.get('face_index')}")
    
    face_index = data.get('face_index')
    if face_index is None:
        emit('single_face_result', {
            'error': 'No face index provided',
            'timestamp': time.time()
        })
        return
    
    if current_frame is None:
        emit('single_face_result', {
            'error': 'No frame available',
            'timestamp': time.time()
        })
        return
    
    # Create a high-quality copy of the current frame
    capture_frame = current_frame.copy()
    
    # Detect faces using our function with both detection methods for better results
    try:
        # Try ssd_resnet first (more accurate but might not be available)
        result_ssd_resnet = detect_faces(capture_frame, {'method': 'ssd_resnet', 'min_confidence': 0.5})
        
        # Use the result with more faces, or default to ssd_resnet result
        if result_ssd_resnet['num_faces'] > 0:
            print(f"Using ssd_resnet detection with {result_ssd_resnet['num_faces']} faces")
            faces_data = result_ssd_resnet['faces']
        else:
            print(f"Using ssd_resnet detection with {result_ssd_resnet['num_faces']} faces")
            faces_data = result_ssd_resnet['faces']
        
        if not faces_data or face_index >= len(faces_data):
            emit('single_face_result', {
                'error': f'Face index {face_index} not found',
                'timestamp': time.time()
            })
            return
        
        # Get the selected face data
        face_data = faces_data[face_index]
        x, y, w, h = face_data['x'], face_data['y'], face_data['width'], face_data['height']
        conf = face_data['confidence']
        
        # Add padding around the face (20% on each side)
        expand = int(max(w, h) * 0.2)
        x_exp = max(0, x - expand)
        y_exp = max(0, y - expand)
        w_exp = min(capture_frame.shape[1] - x_exp, w + 2*expand)
        h_exp = min(capture_frame.shape[0] - y_exp, h + 2*expand)
        
        # Crop the face with padding
        face_crop = capture_frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        
        # Add debug visualization
        debug_crop = face_crop.copy()
        cv2.rectangle(debug_crop, (0, 0), (debug_crop.shape[1]-1, debug_crop.shape[0]-1), (0, 255, 0), 2)
        cv2.putText(debug_crop, f"Face #{face_index+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_crop, f"Conf: {conf:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save to disk for debug purposes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"detected_faces/face_{face_index+1}_conf_{conf:.2f}_{timestamp}.jpg"
        cv2.imwrite(face_filename, debug_crop)
        print(f"Face {face_index+1} saved as {face_filename}")
        
        # Encode for transmission
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, buffer = cv2.imencode('.jpg', debug_crop, encode_params)
        
        if not success:
            emit('single_face_result', {
                'error': f'Failed to encode face {face_index}',
                'timestamp': time.time()
            })
            return
        
        crop_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send the face crop back to the client
        emit('single_face_result', {
            'face': {
                'index': face_index,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': conf,
                'image': crop_base64
            },
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Error processing single face request: {e}")
        traceback.print_exc()
        emit('single_face_result', {
            'error': f'Error processing face: {str(e)}',
            'timestamp': time.time()
        })

def calculate_face_quality(face_roi):
    """
    Calculate a confidence/quality score for a face region detected by Haar cascade
    Higher score means better quality face detection
    """
    if face_roi is None or face_roi.size == 0:
        return 0.3  # Default mediocre confidence
    
    # Calculate face quality metrics
    try:
        # 1. Use variance of laplacian for blur detection
        gray = face_roi if len(face_roi.shape) == 2 else cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian) / 10000  # Normalize
        blur_score = min(max(blur_score, 0), 1)  # Clamp to [0,1]
        
        # 2. Calculate histogram distribution for exposure quality
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-5)  # Avoid division by zero
        hist_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        exposure_score = min(hist_entropy / 8, 1.0)  # Normalize, 8 is max entropy
        
        # 3. Face size relative to standard face size (penalty for very small faces)
        h, w = gray.shape[:2]
        size_score = min((h * w) / (100 * 100), 1.0)  # Normalize to 100x100 reference
        
        # 4. Face proportions (width/height ratio should be ~0.75-0.85 for ideal faces)
        ratio = w / h if h > 0 else 0
        proportion_score = 1.0 - min(abs(ratio - 0.8) * 2, 1.0)  # Closer to 0.8 is better
        
        # 5. Contrast measurement (higher contrast typically means better face detection)
        min_val, max_val = np.min(gray), np.max(gray)
        contrast = (max_val - min_val) / 255.0 if max_val > min_val else 0
        contrast_score = min(contrast * 1.5, 1.0)  # Boost contrast importance slightly
        
        # 6. Edge density (faces typically have good edge structure)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (gray.size + 1e-5)
        edge_score = min(edge_density * 15, 1.0)  # Scale appropriately
        
        # Combine all factors with weights
        final_score = (
            blur_score * 0.25 +           # Sharpness is very important
            exposure_score * 0.15 +       # Good exposure helps
            size_score * 0.20 +           # Larger faces are better
            proportion_score * 0.15 +     # Face shape matters
            contrast_score * 0.15 +       # Good contrast helps detection
            edge_score * 0.10             # Edge structure helps confirm it's a face
        )
        
        # Scale to match DNN confidence range (0.5-1.0)
        scaled_score = 0.5 + (final_score * 0.5)
        return float(scaled_score)
    
    except Exception as e:
        print(f"Error calculating face quality: {e}")
        return 0.5  # Default medium confidence

def detect_faces(frame, detect_params=None):
    """
    Detect faces using SSD ResNet model
    """
    if detect_params is None:
        detect_params = {}
    
    # Extract parameters with defaults
    min_confidence = detect_params.get('min_confidence', 0.5)
    min_face_size = detect_params.get('min_face_size', 30)  # Minimum size in pixels
    
    height, width = frame.shape[:2]
    face_data = []
    # Global declaration moved before any assignment
    global last_capture_faces
    face_crops = []  # Temporary local variable to store face crops
    
    # Make a copy for processing if needed
    process_frame = frame.copy()
    
    # For small frames, skip downscaling to improve detection of small faces
    if height < 300 or width < 300:
        # For small frames, slightly enhance contrast for better detection
        process_frame = cv2.convertScaleAbs(process_frame, alpha=1.1, beta=5)
    
    # Use SSD ResNet detector
    if 'ssd_resnet' in dnn_detectors:
        try:
            model_faces = dnn_detectors['ssd_resnet'].detect(process_frame)
            
            # Filter by confidence
            model_faces = [f for f in model_faces if f['confidence'] >= min_confidence]
            
            if model_faces:
                for face in model_faces:
                    face['method'] = 'ssd_resnet'
                    # Make sure all values are standard Python types, not NumPy types
                    face['x'] = int(face['x'])
                    face['y'] = int(face['y'])
                    face['width'] = int(face['width'])
                    face['height'] = int(face['height'])
                    face['confidence'] = float(face['confidence'])
                    face_data.append(face)
                    
                    # Create face crops
                    x, y, w, h = face['x'], face['y'], face['width'], face['height']
                    # Add padding for better face recognition (20%)
                    pad_w = int(w * 0.2)
                    pad_h = int(h * 0.2)
                    crop_x = max(0, x - pad_w)
                    crop_y = max(0, y - pad_h)
                    crop_w = min(width - crop_x, w + 2*pad_w)
                    crop_h = min(height - crop_y, h + 2*pad_h)
                    
                    if crop_w > 0 and crop_h > 0:
                        face_crop = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        face_crop_data = face.copy()
                        face_crop_data['crop'] = face_crop
                        face_crops.append(face_crop_data)
        except Exception as e:
            print(f"Error with ssd_resnet detector: {e}")
    
    # Perform Non-Maximum Suppression (NMS) to filter out overlapping detections
    if len(face_data) > 1:
        try:
            boxes = np.array([[f['x'], f['y'], f['x'] + f['width'], f['y'] + f['height']] for f in face_data])
            scores = np.array([f['confidence'] for f in face_data])
            
            # Use a generous threshold for IoU to keep unique faces
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
            
            if len(indices) > 0:
                # Filter face_data and face_crops
                face_data = [face_data[i] for i in indices.flatten()]
                face_crops = [face_crops[i] for i in indices.flatten() if i < len(face_crops)]
        except Exception as e:
            print(f"Error in NMS filtering: {e}")
    
    # Sort by confidence
    face_data.sort(key=lambda x: x['confidence'], reverse=True)
    face_crops.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Update the global last_capture_faces variable
    last_capture_faces = face_crops
    
    # All detections will be from ssd_resnet
    detection_method = 'ssd_resnet' if face_data else 'none'
    
    # Return as a dictionary instead of a tuple
    return {
        'faces': face_data, 
        'crops': face_crops,
        'method': detection_method
    }

def generate_frames():
    """Generate frames for video streaming"""
    global current_frame, faces_data
    while not stop_signal:
        if current_frame is not None:
            # Create a copy with face rectangles
            output_frame = current_frame.copy()
            
            # Draw rectangles for detected faces
            for face in faces_data:
                x, y, w, h = face['x'], face['y'], face['width'], face['height']
                conf = face['confidence']
                method = face['method']
                color = (0, 255, 0)  # Default green
                if method == 'ssd_resnet':
                    color = (255, 0, 255)  # Purple for ssd_resnet
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output_frame, f"{method} ({conf:.2f})", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, yield an empty frame
            time.sleep(0.1)

# Add this function near the get_local_ip function
def setup_ngrok(port=5000):
    """Set up ngrok tunnel to expose the local server"""
    if not NGROK_AVAILABLE:
        return None
        
    try:
        # Set auth token
        ngrok.set_auth_token("2jsIuWon6X9ZRwiBdHVKNG0pWp8_5qWA6iKMSirXGV9SG9SWQ")
        
        # Open a ngrok tunnel to the HTTP server
        public_url = ngrok.connect(port, "http")
        print(f"Public URL: {public_url}")
        
        # Log tunnel info for debugging
        tunnels = ngrok.get_tunnels()
        print(f"Active tunnels: {tunnels}")
        
        return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        traceback.print_exc()
        return None
    
if __name__ == '__main__':
    # Initialize face detectors
    if not init_face_detectors():
        print("Warning: Some face detectors failed to initialize")
    
    # Initialize feature detectors
    if not init_feature_detectors():
        print("Warning: Some feature detectors failed to initialize")
    
    # Perform self-test
    self_test_result = perform_self_test()
    if not self_test_result:
        print("Warning: Self-test failed, but starting server anyway. Face detection may not work properly.")
    
    # Start camera processing thread
    thread = threading.Thread(target=camera_processing)
    thread.daemon = True
    thread.start()
    
    # Register service for discovery
    zeroconf_instance, service_info = register_service()
    
    # Set up ngrok for public access (before starting the server)
    public_url = None
    if NGROK_AVAILABLE:
        public_url = setup_ngrok(5000)
    
    try:
        local_url = f"http://{get_local_ip()}:5000"
        print(f"Server running locally at {local_url}")
        if public_url:
            print(f"Public access URL: {public_url}")
            print(f"Share this URL to access the camera feed from anywhere")
        print("Press Ctrl+C to stop the server")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
    finally:
        print("Shutting down...")
        stop_signal = True
        # Clean up ngrok tunnel
        if NGROK_AVAILABLE:
            try:
                ngrok.kill()
            except:
                pass
        if zeroconf_instance:
            try:
                zeroconf_instance.unregister_service(service_info)
                zeroconf_instance.close()
            except Exception as e:
                print(f"Error during zeroconf cleanup: {e}")
        print("Server stopped")
