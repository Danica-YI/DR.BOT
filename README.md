# Triage Companion — Backend Dashboard

Simple Flask backend + live dashboard for the Triage Companion hackathon project (Brains & Bots, QUT).

## Setup

```bash
pip install flask
python3 app.py
```

Then open http://localhost:5000

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
