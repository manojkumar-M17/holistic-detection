from ultralytics import YOLO
import os
from config.config import YOLO_MODEL_PATH, PERSON_CLASS, FORBIDDEN_CLASSES, ENABLE_OBJECT_DETECTION

class StudentDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        """
        Initializes the YOLOv8 detector and multi-class object tracker.
        """
        self.model_path = model_path
        print(f"[YOLO] Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("[YOLO] Model loaded successfully.")

    def detect_and_track(self, frame):
        """
        Runs YOLOv8 tracking to track students and detect forbidden objects (phones, books, laptops).
        Returns a dict:
        {
            "students": [{'id': track_id, 'bbox': (x1, y1, x2, y2), 'conf': confidence}],
            "objects": [{'class_id': cid, 'label': label, 'bbox': (x1, y1, x2, y2), 'conf': confidence}]
        }
        """
        if frame is None:
            return {"students": [], "objects": []}

        # Target classes: Person (0), Laptop (63), Phone (67), Book (73)
        target_classes = [PERSON_CLASS]
        if ENABLE_OBJECT_DETECTION:
            target_classes.extend(list(FORBIDDEN_CLASSES.keys()))

        results = self.model.track(frame, persist=True, classes=target_classes, verbose=False)
        
        tracked_students = []
        detected_objects = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None:
                for i in range(len(boxes)):
                    xyxy = boxes[i].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_id = int(boxes[i].cls[0].item())
                    conf = float(boxes[i].conf[0].item())
                    
                    if cls_id == PERSON_CLASS:
                        track_id = int(boxes[i].id[0].item()) if boxes[i].id is not None else i
                        tracked_students.append({
                            "id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf
                        })
                    elif cls_id in FORBIDDEN_CLASSES:
                        label = FORBIDDEN_CLASSES[cls_id]
                        detected_objects.append({
                            "class_id": cls_id,
                            "label": label,
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf
                        })
                        
        return {
            "students": tracked_students,
            "objects": detected_objects
        }

def find_student_objects(student_bbox, detected_objects, margin=30):
    """
    Finds all detected forbidden objects located inside or immediately adjacent to a student's bounding box.
    """
    sx1, sy1, sx2, sy2 = student_bbox
    # Expand student box slightly by margin to catch phones held near body
    esx1 = sx1 - margin
    esy1 = sy1 - margin
    esx2 = sx2 + margin
    esy2 = sy2 + margin
    
    correlated_objects = []
    for obj in detected_objects:
        ox1, oy1, ox2, oy2 = obj["bbox"]
        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2
        
        # Check if object center is inside expanded student box
        if esx1 <= cx <= esx2 and esy1 <= cy <= esy2:
            correlated_objects.append(obj)
            
    return correlated_objects

