import datetime
import json
import os
import sqlite3
from typing import Any


def build_triage_assessment(
    response_type: str,
    can_walk: bool,
    heavy_bleeding: bool = False,
    breathing_difficulty: bool = False,
    trapped: bool = False,
    severe_pain: bool = False,
    needs_supply: bool = False,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Create a structured triage assessment from a short interaction loop.

    priority is one of: "no_response", "both", "medical", "resource"
    (matches the dashboard's 4-status system).
    """
    reasons: list[str] = []

    if heavy_bleeding:
        reasons.append("heavy bleeding")
    if breathing_difficulty:
        reasons.append("breathing difficulty")
    if trapped:
        reasons.append("trapped or unable to self-evacuate")
    if not can_walk:
        reasons.append("unable to walk")
    if severe_pain:
        reasons.append("severe pain")
    if needs_supply:
        reasons.append("needs supplies")

    if response_type == "unresponsive":
        reasons.append("no response to simple prompt")
        priority = "no_response"
    else:
        reasons.append(
            "person responded to prompt" if response_type == "responding" else "response unclear"
        )
        needs_medical = (
            heavy_bleeding or breathing_difficulty or trapped or severe_pain or not can_walk
        )
        if needs_medical and needs_supply:
            priority = "both"
        elif needs_medical:
            priority = "medical"
        else:
            # Covers both "needs_supply only" and "no issues reported" —
            # everyone who triggered a report gets at least a resource
            # check-in, since a fully healthy person shouldn't have
            # reached this point in the flow.
            priority = "resource"

    if confidence is None:
        confidence = 0.75
        if response_type == "responding" and can_walk:
            confidence += 0.05
        if heavy_bleeding or breathing_difficulty or trapped or not can_walk:
            confidence += 0.1
        confidence = min(confidence, 0.98)

    return {
        "assessment_id": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "response_type": response_type,
        "can_walk": can_walk,
        "heavy_bleeding": heavy_bleeding,
        "breathing_difficulty": breathing_difficulty,
        "trapped": trapped,
        "severe_pain": severe_pain,
        "needs_supply": needs_supply,
        "priority": priority,
        "reasons": reasons,
        "confidence": round(confidence, 2),
    }


class OfflineTriageStore:
    """Persist triage assessments locally until the dashboard can receive them."""

    def __init__(self, db_path: str | None = None):
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "device_triage_queue.db")
        self.db_path = db_path or default_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triage_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                synced INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()

    def enqueue(self, assessment: dict[str, Any]) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "INSERT INTO triage_queue (created_at, payload, synced) VALUES (?, ?, ?)",
            (assessment.get("timestamp") or "", json.dumps(assessment), 0),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def get_pending(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, payload FROM triage_queue WHERE synced = 0 ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return [
            {"id": row[0], "payload": json.loads(row[1])}
            for row in rows
        ]

    def pending_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM triage_queue WHERE synced = 0").fetchone()[0]
        conn.close()
        return int(count)

    def mark_synced(self, row_ids: list[int]) -> None:
        if not row_ids:
            return
        placeholders = ", ".join("?" for _ in row_ids)
        conn = sqlite3.connect(self.db_path)
        conn.execute(f"UPDATE triage_queue SET synced = 1 WHERE id IN ({placeholders})", row_ids)
        conn.commit()
        conn.close()