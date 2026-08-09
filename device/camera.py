import sys
import time
import cv2


def status_color(status):
    if status in ("medical", "no", "no_response"):
        return (0, 0, 255)
    if status in ("resource", "help", "result"):
        return (0, 165, 255)
    return (0, 200, 0)


def _candidate_backends():
    if sys.platform == "win32":
        return [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
            ("CAP_ANY", cv2.CAP_ANY),
        ]
    return [("CAP_ANY", cv2.CAP_ANY)]


def open_camera(index: int = 0) -> cv2.VideoCapture:
    errors = []
    for backend_name, backend in _candidate_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            errors.append(f"{backend_name}: open failed")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        time.sleep(1)
        readable = False
        for _ in range(10):
            ok, _frame = cap.read()
            if ok:
                readable = True
                break
        if readable:
            return cap

        errors.append(f"{backend_name}: opened but no readable frames")
        cap.release()

    attempted = ", ".join(name for name, _ in _candidate_backends())
    details = "; ".join(errors) if errors else "no backend attempts recorded"
    raise RuntimeError(
        f"Unable to open camera {index}. Attempted backends: {attempted}. Details: {details}"
    )


def draw_overlay(frame, state: str, message: str, bbox=None, status=None) -> None:
    color = status_color(status)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if message:
        cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if status:
        cv2.putText(frame, f"STATUS: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
