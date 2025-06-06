import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import threading
from queue import Queue
import time
import torch
import torchvision.transforms as transforms

model = YOLO("yolov8n.pt")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_calibration(yaml_path='camera_params.yaml'):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        return np.array(data['camera_matrix']), np.array(data['dist_coeff'])
    except FileNotFoundError:
        st.error("camera_params.yaml not found!")
        return None, None

camera_matrix, dist_coeff = load_calibration()
if camera_matrix is None or dist_coeff is None:
    st.stop()

st.title("Baggage Detection with MiDaS Depth")
ip_address = st.text_input("Enter IP Webcam URL", "http://192.168.1.100:8080/video")
start_button = st.button("Start Video Stream")
video_placeholder = st.empty()
coords_placeholder = st.empty()
status_placeholder = st.empty()
frame_queue = Queue(maxsize=10)
DEPTH_SCALE = 0.001

def video_capture_thread(ip_address, queue):
    cap = cv2.VideoCapture(ip_address)
    if not cap.isOpened():
        st.error(f"Could not connect to {ip_address}. Check URL and network.")
        return
    st.success("Connected to IP Webcam!")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to retrieve frame. Connection may have dropped.")
            break
        if not queue.full():
            queue.put(frame)
        else:
            time.sleep(0.01)
    cap.release()

def detect_baggage(frame):
    results = model(frame, verbose=False)
    baggage_coords = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            if label in ["suitcase", "backpack", "handbag"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                baggage_coords.append((center_x, center_y))
    return frame, baggage_coords

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

def compute_coordinates(pixel_coords, camera_matrix, depth_map):
    points = np.array([pixel_coords], dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeff, P=camera_matrix)
    u, v = undistorted[0, 0]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    z = depth_map[int(v), int(u)] * DEPTH_SCALE
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    return (x, y, z)

if start_button and ip_address:
    thread = threading.Thread(target=video_capture_thread, args=(ip_address, frame_queue), daemon=True)
    thread.start()
    time.sleep(1)
    last_processed_time = 0
    PROCESS_INTERVAL = 0.5
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                status_placeholder.error("Received empty frame.")
                break
            frame = cv2.resize(frame, (640, 480))
            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeff)
            current_time = time.time()
            if current_time - last_processed_time >= PROCESS_INTERVAL:
                depth_map = compute_depth(frame_undistorted)
                frame_undistorted, baggage_coords = detect_baggage(frame_undistorted)
                coords_text = "Baggage Coordinates (meters):\n"
                for i, (px, py) in enumerate(baggage_coords):
                    world_coords = compute_coordinates((px, py), camera_matrix, depth_map)
                    coords_text += f"Bag {i+1}: ({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f})\n"
                last_processed_time = current_time
            video_placeholder.image(frame_undistorted, channels="BGR", use_column_width=True)
            coords_placeholder.text(coords_text)
            status_placeholder.text("Processing frames...")
        else:
            status_placeholder.warning("Waiting for frames from IP Webcam...")
            time.sleep(0.1)
else:
    st.info("Enter IP Webcam URL and click 'Start Video Stream'.")
