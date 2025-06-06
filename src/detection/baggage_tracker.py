import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class BaggageTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)
        self.baggage_tracks = {}
        self.person_tracks = {}
        self.unattended_threshold = 10
        self.suspicious_threshold = 20

    def compute_coordinates(self, pixel_coords, depth_map, camera_matrix, dist_coeff):
        points = np.array([pixel_coords], dtype=np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeff, P=camera_matrix)
        u, v = undistorted[0, 0]
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        z = depth_map[int(v), int(u)] * 0.001
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        return (x, y, z)

    def is_facing_camera(self, landmarks, frame_shape):
        nose = landmarks.landmark[0]
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        h, w = frame_shape[:2]
        nose_x, nose_y = nose.x * w, nose.y * h
        shoulder_mid_x = ((left_shoulder.x + right_shoulder.x) * w) / 2
        return abs(nose_x - shoulder_mid_x) < w * 0.1

    def is_stationary(self, current_pos, prev_pos):
        if prev_pos is None:
            return True
        return np.linalg.norm(np.array(current_pos) - np.array(prev_pos)) < 20

    def update(self, frame, detections, depth_map, camera_matrix, dist_coeff, pose_landmarks):
        tracks = []
        alerts = []
        coords = {}

        for det in detections:
            if 'bbox' not in det or len(det['bbox']) != 4:
                continue
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class']
            bbox = ([x1, y1, x2 - x1, y2 - y1], conf, cls)
            tracks.append(bbox)

        tracked_objects = self.tracker.update_tracks(tracks, frame=frame)

        current_bags = {}
        people_nearby = False

        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = track.det_class
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            det_color = next((det['color'] for det in detections if det['class'] == label and 
                              det['bbox'][0] == x1 and det['bbox'][1] == y1), (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), det_color, 2)
            cv2.putText(frame, f"{label} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2)

            if label in ["suitcase", "backpack", "handbag"]:
                current_bags[track_id] = {
                    'bbox': ltrb,
                    'frame_count': self.baggage_tracks.get(track_id, {'frame_count': 0})['frame_count'] + 1,
                    'last_person_frame': self.baggage_tracks.get(track_id, {'last_person_frame': -self.unattended_threshold})['last_person_frame']
                }
            elif label in ["knife", "gun"]:
                alerts.append(f"Weapon detected: ID {track_id}")
                coords[f"Weapon {track_id}"] = self.compute_coordinates((center_x, center_y), depth_map, camera_matrix, dist_coeff)
                people_nearby = True
            elif label == "person":
                people_nearby = True
                prev_pos = self.person_tracks.get(track_id, {}).get('last_pos')
                current_pos = (center_x, center_y)
                frame_count = self.person_tracks.get(track_id, {'frame_count': 0})['frame_count'] + 1

                facing_camera = False
                if pose_landmarks:
                    facing_camera = self.is_facing_camera(pose_landmarks, frame.shape)
                
                stationary = self.is_stationary(current_pos, prev_pos)
                suspicious_count = self.person_tracks.get(track_id, {'suspicious_count': 0})['suspicious_count']
                
                if stationary and facing_camera:
                    suspicious_count += 1
                else:
                    suspicious_count = 0

                if suspicious_count > self.suspicious_threshold:
                    alerts.append(f"Suspicious Person detected: ID {track_id}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    cv2.putText(frame, f"Suspicious {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                self.person_tracks[track_id] = {
                    'last_pos': current_pos,
                    'frame_count': frame_count,
                    'suspicious_count': suspicious_count
                }

        for bag_id, bag_info in current_bags.items():
            if people_nearby:
                bag_info['last_person_frame'] = bag_info['frame_count']
            if bag_info['frame_count'] - bag_info['last_person_frame'] > self.unattended_threshold:
                x1, y1, x2, y2 = map(int, bag_info['bbox'])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                coords[f"Unattended Bag {bag_id}"] = self.compute_coordinates((center_x, center_y), depth_map, camera_matrix, dist_coeff)
                alerts.append(f"Unattended Bag detected: ID {bag_id}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f"Unattended {bag_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        self.baggage_tracks = current_bags
        return frame, alerts, coords