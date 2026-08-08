import datetime
import requests

from device.triage_flow import OfflineTriageStore


def post_report(base_url: str, device_id: str, status: str, lat: float, lon: float):
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
    payload = {
        "device_id": device_id,
        "assessment": assessment,
        "timestamp": assessment.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "location": {"lat": lat, "lon": lon},
    }
    url = f"{base_url.rstrip('/')}/api/triage"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        store = OfflineTriageStore()
        store.enqueue(assessment)
        return False, str(exc)


def flush_triage_queue(base_url: str, device_id: str, lat: float, lon: float):
    store = OfflineTriageStore()
    pending = store.get_pending()
    if not pending:
        return True, {"queued": 0}

    payload = {
        "device_id": device_id,
        "assessments": [item["payload"] for item in pending],
        "location": {"lat": lat, "lon": lon},
    }
    url = f"{base_url.rstrip('/')}/api/triage/batch"
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        store.mark_synced([item["id"] for item in pending])
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)
