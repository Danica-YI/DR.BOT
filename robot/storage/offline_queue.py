"""Offline queue for robot data buffering."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class OfflineQueue:
    """Simple JSONL offline buffer for incident reports."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path or Path(__file__).resolve().parent.parent / "data" / "offline_incidents.jsonl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("OfflineQueue initialized at %s", self.storage_path)

    def add_incident(self, incident: dict[str, Any]) -> None:
        """Save a new incident locally before backend sync."""
        incident_id = incident.get("incident_id")
        if not incident_id:
            raise ValueError("Incident must contain an incident_id")

        records = self._load_records()
        if any(record.get("incident_id") == incident_id for record in records):
            self.logger.warning("Incident %s already exists in offline queue; skipping add.", incident_id)
            return

        entry = {
            "incident_id": incident_id,
            "synced": False,
            "payload": incident,
        }
        try:
            with self.storage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.logger.info("Added incident %s to offline queue.", incident_id)
        except OSError as exc:
            self.logger.error("Failed to write offline incident: %s", exc, exc_info=True)
            raise

    def get_unsynced_incidents(self) -> list[dict[str, Any]]:
        """Return all locally stored incidents that have not been synchronized."""
        records = self._load_records()
        return [record for record in records if not record.get("synced", False)]

    def mark_synced(self, incident_id: str) -> bool:
        """Mark a locally stored incident as synced after a backend success."""
        records = self._load_records()
        updated = False
        for record in records:
            if record.get("incident_id") == incident_id:
                record["synced"] = True
                updated = True
                break

        if not updated:
            self.logger.warning("Incident %s not found in offline queue.", incident_id)
            return False

        self._write_records(records)
        self.logger.info("Marked incident %s as synced.", incident_id)
        return True

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                        records.append(record)
                    except json.JSONDecodeError as exc:
                        self.logger.warning(
                            "Skipping corrupted line %s in offline queue: %s",
                            line_number,
                            exc,
                        )
        except OSError as exc:
            self.logger.error("Failed to read offline queue: %s", exc, exc_info=True)
            return []

        return records

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        temp_path = self.storage_path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            temp_path.replace(self.storage_path)
        except OSError as exc:
            self.logger.error("Failed to write offline queue file: %s", exc, exc_info=True)
            raise


if __name__ == "__main__":
    import logging
    from ..report import create_incident_report

    logging.basicConfig(level=logging.INFO)
    queue = OfflineQueue()
    incident = create_incident_report(latitude=-27.47, longitude=153.02)
    queue.add_incident(incident.to_dict())

    unsynced = queue.get_unsynced_incidents()
    print("Unsynced incidents:", len(unsynced))
    for record in unsynced:
        print(record["incident_id"], record["synced"])

    if unsynced:
        queue.mark_synced(unsynced[0]["incident_id"])
        print("After sync:", [r["synced"] for r in queue.get_unsynced_incidents()])
