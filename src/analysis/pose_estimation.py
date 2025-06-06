import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
loitering_time = {}

def detect_pose(frame, frame_count):
    global loitering_time

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width)
        y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height)
        key = (x // 50, y // 50)
        if key not in loitering_time:
            loitering_time[key] = frame_count
        else:
            if frame_count - loitering_time[key] > 70:
                cv2.putText(frame, "Loitering!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame
