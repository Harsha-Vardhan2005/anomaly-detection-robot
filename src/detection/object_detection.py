from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8m.pt")

def detect_objects(frame):
    """
    Detect objects in the frame using YOLO.
    Returns the annotated frame and lists of detected bags, people, and weapons.
    """
    results = model(frame, verbose=False)
    bags, people, weapons = [], [], []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            # Assign colors based on object type
            if label in ["knife", "gun", "pistol"]:
                weapons.append((x1, y1, x2, y2, conf))
                color = (0, 0, 255)  # Red for weapons
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            elif label in ["backpack", "handbag", "suitcase"]:
                bags.append((x1, y1, x2, y2, conf))
                color = (0, 255, 255)  # Yellow for baggage
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            elif label == "person":
                people.append((x1, y1, x2, y2, conf))
                color = (0, 255, 0)  # Green for people
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

    return frame, bags, people, weapons