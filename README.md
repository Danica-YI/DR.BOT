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
`status` must be one of: `"ok"`, `"medical"`, `"resource"`

### `POST /api/reports/batch`
Receive multiple queued reports at once (used after a device reconnects following an offline period).

```json
{ "reports": [ {...}, {...} ] }
```
Each item has the same shape as `/api/report`.

### `GET /api/reports`
Returns all reports, most recent first. Used by the dashboard page.
