"""Simple synchronization manager for offline incident storage."""

from __future__ import annotations

import logging
from typing import Any

from ..config import SYNC_RETRY_INTERVAL
from ..storage.offline_queue import OfflineQueue
from .api_client import ApiClient


class SyncManager:
    """Synchronize pending offline incidents with the backend."""

    def __init__(
        self,
        offline_queue: OfflineQueue | None = None,
        api_client: ApiClient | None = None,
        retry_interval: int | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.offline_queue = offline_queue or OfflineQueue()
        self.api_client = api_client or ApiClient()
        self.retry_interval = retry_interval or SYNC_RETRY_INTERVAL
        self.logger.info(
            "SyncManager initialized; retry interval=%s seconds.",
            self.retry_interval,
        )

    def sync_pending_incidents(self) -> list[dict[str, Any]]:
        """Attempt to synchronize all unsynced incidents.

        Returns a list of sync results including whether each incident was marked synced.
        """
        unsynced = self.offline_queue.get_unsynced_incidents()
        self.logger.info("Found %s unsynced incident(s) to synchronize.", len(unsynced))

        results: list[dict[str, Any]] = []
        for record in unsynced:
            incident_id = record.get("incident_id")
            incident = record.get("payload", {})
            self.logger.info("Attempting sync for incident %s", incident_id)
            result = self.api_client.send_report(incident)

            if result.get("success"):
                self.offline_queue.mark_synced(incident_id)
                self.logger.info("Incident %s synced successfully.", incident_id)
                results.append({"incident_id": incident_id, "status": "synced", "result": result})
            else:
                self.logger.warning(
                    "Incident %s remained unsynced: %s",
                    incident_id,
                    result.get("error") or result.get("message"),
                )
                results.append({"incident_id": incident_id, "status": "failed", "result": result})

        return results

    def has_pending_incidents(self) -> bool:
        """Return whether there are unsynced incidents pending."""
        return bool(self.offline_queue.get_unsynced_incidents())


if __name__ == "__main__":
    import logging
    from ..report import create_incident_report

    logging.basicConfig(level=logging.INFO)
    queue = OfflineQueue()
    incident = create_incident_report(latitude=-27.4698, longitude=153.0251)
    queue.add_incident(incident.to_dict())

    manager = SyncManager(offline_queue=queue, api_client=ApiClient())
    results = manager.sync_pending_incidents()
    print(results)
