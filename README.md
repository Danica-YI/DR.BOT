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

After detecting a person, the device asks the bilingual initial question and
waits up to five seconds for voice. A valid response locks English or Chinese
for the assessment. If there is no initial voice response, the device asks the
bilingual second prompt and monitors voice and MediaPipe gestures concurrently
for another five seconds. Speech selects voice mode, a gesture selects gesture
mode, and neither produces `NO_RESPONSE` for responder review without inferring
a medical diagnosis.

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

The initial and combined second response windows default to five seconds and
are both configured with `--voice-timeout`. `--gesture-timeout` controls gesture
answer windows after gesture mode has been selected.

### Optional one-time English/Chinese detection

Install `faster-whisper` and provide a multilingual model that is already
available locally:

```bash
python -m pip install faster-whisper
python device.py --vosk-model ./vosk-model-small-en-us-0.15 --whisper-model ./whisper-tiny --audio-device 1
```

The initial response is transcribed once with automatic language detection.
The selected `en` or `zh` language is then cached for the assessment. English
answers use lightweight Vosk after detection; Chinese answers use the already
loaded Whisper model with `language="zh"`, so language detection is not repeated
for every question. Runtime model downloads are disabled to preserve offline
operation; `--whisper-model` must resolve from a local directory or cache.

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
