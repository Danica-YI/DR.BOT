#!/usr/bin/env python3
"""Device integration main runner."""

import argparse
import time

from device.camera import draw_overlay, open_camera
from device.controller import PersonDetector
from device.pose_classifier import PoseClassifier
from device.api_client import post_report


def parse_args():
    parser = argparse.ArgumentParser(description="Device integration runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Dashboard API base URL")
    parser.add_argument("--device-id", default="DR-01", help="Device ID to report")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--lat", type=float, default=-27.4698, help="Report latitude")
    parser.add_argument("--lon", type=float, default=153.0251, help="Report longitude")
    parser.add_argument("--no-display", action="store_true", help="Do not show the OpenCV preview window")
    parser.add_argument("--prompt-text", default="Please show your condition clearly.", help="Text prompt shown when asking for a gesture")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    return parser.parse_args()


def main():
    args = parse_args()

    camera = open_camera(args.camera_index)
    detector = PersonDetector(model_name=args.model)
    classifier = PoseClassifier()

    state = "SEARCHING"
    center_frames = 0
    ask_time = 0
    result_time = 0
    bbox = None
    status_text = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("ERROR: camera frame lost")
                break

            height, width = frame.shape[:2]
            message = ""

            if state == "SEARCHING":
                found = detector.find_person(frame)
                if found:
                    x1, y1, x2, y2 = found
                    bbox = (x1, y1, x2, y2)
                    person_center = (x1 + x2) / 2
                    frame_center = width / 2
                    delta = person_center - frame_center
                    if abs(delta) < width * 0.1:
                        center_frames += 1
                        message = "Person centered. Stabilizing..."
                        if center_frames >= 5:
                            state = "ASKING"
                            ask_time = time.time()
                            message = args.prompt_text
                            print("Person found and centered. Asking for gesture...")
                    else:
                        center_frames = 0
                        message = "Move camera left" if delta < 0 else "Move camera right"
                else:
                    bbox = None
                    center_frames = 0
                    message = "Searching for person..."

            elif state == "ASKING":
                message = args.prompt_text
                if time.time() - ask_time > 2.0:
                    state = "WAITING"
                    message = "Waiting for gesture/pose..."
                    print("Switching to WAITING state")

            elif state == "WAITING":
                status = classifier.classify(frame)
                if status:
                    success, response = post_report(args.base_url, args.device_id, status, args.lat, args.lon)
                    status_text = status
                    state = "SENT" if success else "ERROR"
                    result_time = time.time()
                    print(f"POST {status} -> {response}")
                    message = "Report submitted" if success else f"Error sending report: {response}"
                else:
                    message = "Pose seen but status unclear. Please repeat."

            elif state == "SENT":
                message = f"Sent {status_text}. Resetting in 3 seconds..."
                if time.time() - result_time > 3.0:
                    state = "SEARCHING"
                    status_text = None
                    bbox = None

            elif state == "ERROR":
                message = "Network error. Retry in 3 seconds..."
                if time.time() - result_time > 3.0:
                    state = "SEARCHING"
                    status_text = None
                    bbox = None

            draw_overlay(frame, state, message, bbox=bbox, status=status_text)

            if not args.no_display:
                import cv2
                cv2.imshow("Device Integration", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        if not args.no_display:
            import cv2
            cv2.destroyAllWindows()
        classifier.close()


if __name__ == "__main__":
    main()
