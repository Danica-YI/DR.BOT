# Triage Companion — Backend Dashboard

Simple Flask backend + live dashboard for the Triage Companion hackathon project (Brains & Bots, QUT).

## Setup

```bash
pip install flask opencv-python ultralytics mediapipe requests
python3 app.py
```

Then open http://localhost:5000

## Device integration

This repository includes a simple demo script for on-device integration using YOLO person detection and MediaPipe Pose.

Run the dashboard backend first, then start the device integration script:

```bash
python device.py
```

The script will:

- capture camera frames with OpenCV
- detect a person using YOLO
- center the view on the person
- prompt the person for a gesture/pose
- classify status using MediaPipe Pose landmarks
- POST the result to `/api/report`

## Offline voice-first triage

The device asks each question using offline text-to-speech and waits for a
spoken YES/NO answer first. If speech is unavailable, unrecognised, or times
out, it automatically falls back to multi-frame gesture recognition.

Install the optional offline audio dependencies:

```bash
python -m pip install pyttsx3 vosk sounddevice
```

Download and unpack a small English Vosk model locally, then run:

```bash
python device.py --vosk-model path/to/vosk-model-small-en-us-0.15
```

The model can also be selected with the `VOSK_MODEL_PATH` environment
variable. If audio dependencies, the microphone, or the model are missing,
the workflow continues using gestures. To intentionally test gesture-only
fallback, run `python device.py --no-voice`.

Default response windows are four seconds for voice and five seconds for a
stable gesture. Override them with `--voice-timeout` and `--gesture-timeout`.

## API

### `POST /api/report`
Receive a single triage report.

```json
{
  "device_id": "V003",
  "status": "medical",
  "timestamp": "2026-08-07T14:32:18+10:00",
  "location": {"lat": -27.4698, "lon": 153.0251}
}
```
`status` must be one of: `"No any response"`, `"Both needpip install flask opencv-python ultralytics mediapipe requests
pip install pyttsx3 vosk sounddevice"`, `"medical"`, `"resource"`

### `POST /api/reports/batch`
Receive multiple queued reports at once (used after a device reconnects following an offline period).

```json
{ "reports": [ {...}, {...} ] }
```
Each item has the same shape as `/api/report`.

### `GET /api/reports`
Returns all reports, most recent first. Used by the dashboard page.
