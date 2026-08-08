"""Payload adapter for backend data formats."""

from __future__ import annotations

import logging
from typing import Any

VALID_REPORT_STATUSES = {"ok", "medical", "resource"}


class PayloadAdapter:
    """Convert robot internal models into existing backend payloads."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def to_backend_report_payload(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Adapt an internal incident report to the backend /api/report payload."""
        self.logger.debug("Adapting incident report to backend payload.")

        robot_id = incident.get("robot_id")
        if not robot_id:
            raise ValueError("Incident is missing robot_id")

        timestamp = incident.get("timestamp")
        if not timestamp:
            raise ValueError("Incident is missing timestamp")

        person = incident.get("person", {})
        status = person.get("status")
        if status not in VALID_REPORT_STATUSES:
            if person.get("detected") is False:
                status = "ok"
                self.logger.info(
                    "Person not detected in incident; defaulting backend status to 'ok'."
                )
            else:
                raise ValueError(
                    "Incident person status must be one of: ok, medical, resource"
                )

        location = incident.get("location", {})
        latitude = location.get("latitude")
        longitude = location.get("longitude")

        backend_payload: dict[str, Any] = {
            "device_id": robot_id,
            "status": status,
            "timestamp": timestamp,
        }

        if latitude is not None or longitude is not None:
            backend_payload["location"] = {
                "lat": latitude,
                "lon": longitude,
            }

        return backend_payload

    def to_backend_heartbeat_payload(self, heartbeat: dict[str, Any]) -> dict[str, Any]:
        """Adapt an internal heartbeat/state payload to the backend /api/heartbeat format."""
        self.logger.debug("Adapting heartbeat payload for backend.")

        device_id = heartbeat.get("device_id")
        if not device_id:
            raise ValueError("Heartbeat payload is missing device_id")

        payload: dict[str, Any] = {
            "device_id": device_id,
            "lat": heartbeat.get("lat"),
            "lon": heartbeat.get("lon"),
            "battery": heartbeat.get("battery"),
            "status": heartbeat.get("status"),
        }
        return payload
