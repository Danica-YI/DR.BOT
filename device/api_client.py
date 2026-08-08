import datetime
import requests


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
