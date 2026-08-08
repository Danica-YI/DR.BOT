import cv2


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {index}")
    return cap


def draw_overlay(frame, state: str, message: str, bbox=None, status=None) -> None:
    color = (0, 255, 0)
    if status == "medical":
        color = (0, 0, 255)
    elif status == "resource":
        color = (0, 255, 255)
    elif status == "ok":
        color = (0, 200, 0)
    elif status == "no":
        color = (0, 165, 255)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if message:
        cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if status:
        cv2.putText(frame, f"STATUS: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
