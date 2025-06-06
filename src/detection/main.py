import streamlit as st
import cv2
import numpy as np
import threading
from queue import Queue
import time
import torch
import torchvision.transforms as transforms
import yaml
import mediapipe as mp
from object_detection import detect_objects
from baggage_tracker import BaggageTracker

def load_calibration(yaml_path='camera_params.yaml'):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        return np.array(data['camera_matrix']), np.array(data['dist_coeff'])
    except FileNotFoundError:
        st.error("camera_params.yaml not found! Please ensure it exists in the project directory.")
        return None, None

camera_matrix, dist_coeff = load_calibration()
if camera_matrix is None or dist_coeff is None:
    st.stop()


try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    st.error(f"Failed to load MiDaS: {e}. Please check torch, torchvision, and timm installations.")
    st.stop()

midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.title("Security System with Calibration and Pose Estimation")
ip_address = st.text_input("Enter IP Webcam URL", "http://192.168.xx.xx/video")
start_button = st.button("Start Video Stream")
video_placeholder = st.empty()
alerts_placeholder = st.empty()
coords_placeholder = st.empty()
status_placeholder = st.empty()
debug_placeholder = st.empty()

frame_queue = Queue(maxsize=10)
tracker = BaggageTracker()
connection_status = {"connected": False, "message": "Not connected."}

def video_capture_thread(ip_address, queue, status):
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        try:
            cap = cv2.VideoCapture(ip_address)
            if not cap.isOpened():
                status["message"] = f"Could not connect to {ip_address}. Retry {retry_count + 1}/{max_retries}."
                retry_count += 1
                time.sleep(2)
                continue
            status["connected"] = True
            status["message"] = "Connected to IP Webcam!"
            while True:
                ret, frame = cap.read()
                if not ret:
                    status["connected"] = False
                    status["message"] = "Failed to retrieve frame. Connection lost."
                    break
                if not queue.full():
                    queue.put(frame)
                else:
                    # Clear queue to avoid backlog
                    while not queue.empty():
                        queue.get()
                    queue.put(frame)
                time.sleep(0.01)
            cap.release()
            break
        except Exception as e:
            status["message"] = f"Webcam error: {e}. Retry {retry_count + 1}/{max_retries}."
            retry_count += 1
            time.sleep(2)
    if retry_count >= max_retries:
        status["connected"] = False
        status["message"] = f"Failed to connect after {max_retries} retries. Check IP, Wi-Fi, and app."

def compute_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img_tensor = midas_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = midas(img_tensor)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return depth

def compute_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        return results.pose_landmarks
    return None

if start_button and ip_address:
    thread = threading.Thread(target=video_capture_thread, args=(ip_address, frame_queue, connection_status), daemon=True)
    thread.start()
    time.sleep(2)

    last_processed_time = 0
    PROCESS_INTERVAL = 1.0  

    last_frame = None
    while True:
        
        if connection_status["connected"]:
            status_placeholder.success(connection_status["message"])
        else:
            status_placeholder.error(connection_status["message"])

        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                status_placeholder.error("Received empty frame.")
                break
            frame = cv2.resize(frame, (640, 480))
            last_frame = frame.copy()
            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeff)

            current_time = time.time()
            if current_time - last_processed_time >= PROCESS_INTERVAL:
                depth_map = compute_depth(frame_undistorted)
                pose_landmarks = compute_pose(frame_undistorted)
                detections = detect_objects(frame_undistorted)
                # Debug detections
                debug_text = "Detections:\n" + "\n".join(
                    [f"Class: {det['class']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}" for det in detections]
                )
                debug_placeholder.text(debug_text)
                frame, alerts, coords = tracker.update(frame_undistorted, detections, depth_map, camera_matrix, dist_coeff, pose_landmarks)
                last_processed_time = current_time
            else:
                frame = last_frame  

            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            alerts_placeholder.text("Alerts:\n" + "\n".join(alerts))
            coords_text = "Coordinates (meters):\n" + "\n".join(
                [f"{k}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})" for k, v in coords.items()]
            )
            coords_placeholder.text(coords_text)
        else:
            if last_frame is not None:
                video_placeholder.image(last_frame, channels="BGR", use_column_width=True)
            time.sleep(0.05)
else:
    st.info("Enter IP Webcam URL and click 'Start Video Stream'.")


pose.close()