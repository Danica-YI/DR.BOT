import datetime
import base64
import requests

from device.triage_flow import OfflineTriageStore


VALID_REPORT_STATUSES = {"no_response", "both", "medical", "resource"}


def _build_photo_data_url(photo):
    if photo is None:
        return None
    if isinstance(photo, str):
        photo = photo.strip()
        if not photo:
            return None
        return photo if photo.startswith("data:image/") else None
    if isinstance(photo, (bytes, bytearray)):
        encoded = base64.b64encode(photo).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    return None


def _normalize_report_status(assessment: dict) -> str:
    status = assessment.get("priority")
    if status in VALID_REPORT_STATUSES:
        return status
    if assessment.get("response_detected") is False and assessment.get("review_reason") in (
        "NO_RESPONSE",
        "NO_VOICE_OR_GESTURE_RESPONSE",
    ):
        return "no_response"
    if assessment.get("needs_supply"):
        return "resource"
    return "medical"


def post_report(base_url: str, device_id: str, status: str, lat: float, lon: float):
    """Post a single simple status report (used for quick/manual testing)."""
    payload = {
        "device_id": device_id,
        "status": status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "location": {"lat": lat, "lon": lon}
    }
    url = f"{base_url.rstrip('/')}/api/report"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def post_triage_assessment(base_url: str, device_id: str, assessment: dict, lat: float, lon: float):
    """
    Post a full triage assessment to the dashboard backend.

    The backend only understands the simple report shape:
    { device_id, status, timestamp, location }
    so we extract assessment["priority"] as the status value here.
    """
    payload = {
        "device_id": device_id,
        "status": _normalize_report_status(assessment),
        "timestamp": assessment.get("timestamp")
        or datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "location": {"lat": lat, "lon": lon},
    }
    photo = _build_photo_data_url(assessment.get("photo"))
    if photo:
        payload["photo"] = photo
    url = f"{base_url.rstrip('/')}/api/report"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        store = OfflineTriageStore()
        store.enqueue(assessment)
        return False, str(exc)


def flush_triage_queue(base_url: str, device_id: str, lat: float, lon: float):
    """
    Flush any locally cached assessments (saved while offline) to the
    backend's batch endpoint. Each queued assessment is converted to the
    same simple report shape used by post_triage_assessment.
    """
    store = OfflineTriageStore()
    pending = store.get_pending()
    if not pending:
        return True, {"queued": 0}

    reports = [
        {
            "device_id": device_id,
            "status": _normalize_report_status(item["payload"]),
            "timestamp": item["payload"].get("timestamp")
            or datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            "location": {"lat": lat, "lon": lon},
            **(
                {"photo": _build_photo_data_url(item["payload"].get("photo"))}
                if _build_photo_data_url(item["payload"].get("photo"))
                else {}
            ),
        }
        for item in pending
    ]
    payload = {"reports": reports}
    url = f"{base_url.rstrip('/')}/api/reports/batch"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        store.mark_synced([item["id"] for item in pending])
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def post_target_frame(
    base_url: str,
    device_id: str,
    image_bytes: bytes,
    lat: float,
    lon: float,
    metadata: dict | None = None,
):
    """Best-effort target frame upload compatibility wrapper.

    The current Flask backend does not expose a dedicated target-frame endpoint,
    so this function returns a structured failure instead of crashing callers.
    """
    del image_bytes
    del lat
    del lon
    del metadata
    return False, f"target frame upload endpoint is not configured for {base_url} ({device_id})"
