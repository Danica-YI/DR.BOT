import datetime
import requests

from device.triage_flow import OfflineTriageStore


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
        "status": assessment.get("priority"),
        "timestamp": assessment.get("timestamp")
        or datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "location": {"lat": lat, "lon": lon},
    }
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
            "status": item["payload"].get("priority"),
            "timestamp": item["payload"].get("timestamp")
            or datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            "location": {"lat": lat, "lon": lon},
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