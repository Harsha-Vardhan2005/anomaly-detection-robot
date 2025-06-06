#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os
import logging
import sys
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory for model and cascade files
BASE_DIR = '/home/jetauto/catkin_ws/src/bhai/scripts'

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load Haar cascade
face_cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
if not os.path.exists(face_cascade_path):
    logging.error(f"Haar cascade not found at {face_cascade_path}")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    logging.error("Failed to load Haar cascade")
    sys.exit(1)

# Load emotion model
try:
    json_file = open(os.path.join(BASE_DIR, 'emotion_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(os.path.join(BASE_DIR, 'emotion_model.h5'))
    logging.info("Loaded emotion model from disk")
except Exception as e:
    logging.error(f"Failed to load emotion model: {e}")
    sys.exit(1)

# Initialize camera
def initialize_camera(device=0, retries=3, delay=1):
    for attempt in range(retries):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)
            logging.info(f"Camera initialized: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, {cap.get(cv2.CAP_PROP_FPS)} FPS")
            time.sleep(delay)
            return cap
        logging.warning(f"Camera init failed (attempt {attempt+1})")
        cap.release()
        time.sleep(delay)
    logging.error("Could not open webcam")
    sys.exit(1)

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_content = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Emotion Detection Stream</title>
            </head>
            <body>
                <img src="/video" width="640" height="480">
            </body>
            </html>
            '''
            self.wfile.write(html_content.encode())
        elif self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            cap = initialize_camera()
            frame_count = 0
            frame_skip = 2
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning("Failed to read frame, retrying...")
                        cap.release()
                        cap = initialize_camera()
                        continue

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        # Still send frame to keep stream smooth
                        _, jpeg = cv2.imencode('.jpg', frame)
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()
                        continue

                    # Emotion detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        roi_gray = gray[y:y+h, x:x+w]
                        try:
                            cropped_img = cv2.resize(roi_gray, (48, 48))
                            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                            cropped_img = cropped_img / 255.0
                            emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
                            maxindex = int(np.argmax(emotion_prediction))
                            label = emotion_dict[maxindex]
                            confidence = emotion_prediction[0][maxindex] * 100
                            cv2.putText(
                                frame,
                                f"{label} ({confidence:.1f}%)",
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA
                            )
                        except:
                            continue

                    # Stream frame
                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()

            except Exception as e:
                logging.error(f"Streaming error: {e}")
            finally:
                cap.release()

def run(server_class=HTTPServer, handler_class=VideoStreamHandler, port=9000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()