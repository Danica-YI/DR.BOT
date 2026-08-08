# Triage Companion — Backend Dashboard

Simple Flask backend + live dashboard for the Triage Companion hackathon project (Brains & Bots, QUT).

## Requirements

- Python 3.10+
- Windows / Linux / macOS with Python support
- A webcam for live device integration

## Dependencies

The project uses the following Python packages:

- `flask` — backend web server and API
- `requests` — robot/backend communication
- `opencv-python` — camera capture and preview
- `ultralytics` — YOLO person detection (device integration)
- `mediapipe` — pose classification (device integration)

Install all dependencies with:

```bash
python -m pip install flask requests opencv-python ultralytics mediapipe
```

If you only want to run the backend/dashboard, `flask` is the minimum required dependency.

## Run the backend dashboard

From the repository root:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## Run the mock robot pipeline

This project also includes a local robot package that runs a mock AI pipeline, offline queue, and backend sync logic.

From the repository root:

```bash
python -m robot.main
```

By default, the robot pipeline uses mock components:

- `robot/camera/camera.py` returns a placeholder frame
- `robot/perception/perception_client.py` returns mock detection results
- `robot/triage/interaction.py` returns a mock triage assessment

This is useful for validating the state machine and backend sync flow without a real camera or AI model.

### Mock robot environment variables

Use these environment variables to configure the robot runner:

- `BACKEND_URL` — backend address, e.g. `http://10.51.75.63:5000`
- `MOCK_CAMERA` — `true` / `false`
- `MOCK_PERCEPTION` — `true` / `false`
- `MOCK_TRIAGE` — `true` / `false`

Example:

```bash
# Windows PowerShell
$env:BACKEND_URL = "http://10.51.75.63:5000"
$env:MOCK_CAMERA = "true"
$env:MOCK_PERCEPTION = "true"
$env:MOCK_TRIAGE = "true"
python -m robot.main
```

## Run heartbeat-only testing

Use this mode when you want the laptop to behave like a robot device on the dashboard
without generating incident reports.

```bash
# Windows PowerShell
$env:BACKEND_URL = "http://10.51.75.63:5000"
python -m robot.heartbeat_only
```

The runner sends only `/api/heartbeat` updates at the interval defined by
`HEARTBEAT_INTERVAL_SECONDS` in [robot/config.py](robot/config.py).

## Run the live device integration

A second integration entrypoint exists in `device.py` and the `device/` package.

Start the dashboard backend first, then run:

```bash
python device.py
```

This script uses a real camera and the device AI stack to:

- capture webcam frames with OpenCV
- detect a person using YOLO
- ask the person for a gesture or pose
- classify the condition using pose landmarks
- POST a report to `/api/report`

### Device script options

```bash
python device.py --help
```

Useful options include:

- `--base-url` — backend API base URL (default `http://127.0.0.1:5000`)
- `--device-id` — device ID to report
- `--camera-index` — OpenCV camera index
- `--lat` and `--lon` — location coordinates
- `--no-display` — run without preview window

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

`status` must be one of: `"ok"`, `"medical"`, `"resource"`

### `POST /api/reports/batch`

Receive multiple queued reports at once (used after a device reconnects following an offline period).

```json
{ "reports": [ {...}, {...} ] }
```

Each item has the same shape as `/api/report`.

### `GET /api/reports`

Returns all reports, most recent first. Used by the dashboard page.
