from ultralytics import YOLO
import cv2


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", min_confidence: float = 0.4):
        self.model = YOLO(model_name)
        self.min_confidence = min_confidence

    def close(self):
        """Release any resources if the backend exposes them."""
        return None

    def find_person(self, frame):
        results = self.model(frame)[0]
        person_boxes = []
        for det in results.boxes:
            if int(det.cls.item()) == 0 and det.conf.item() >= self.min_confidence:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                person_boxes.append((x1, y1, x2, y2, det.conf.item()))

        if not person_boxes:
            return None

        person_boxes.sort(key=lambda item: item[4], reverse=True)
        x1, y1, x2, y2, _ = person_boxes[0]
        return x1, y1, x2, y2
