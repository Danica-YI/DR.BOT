"""Internal incident report model for the robot."""

from __future__ import annotations

import dataclasses
import datetime
import uuid
from typing import Any

from .config import ROBOT_ID


@dataclasses.dataclass
class IncidentReport:
    """Internal representation of a detected incident."""

    robot_id: str
    incident_id: str
    timestamp: str
    type: str
    location: dict[str, float]
    person: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def create_incident_report(
    incident_type: str = "person_detected",
    latitude: float = 0.0,
    longitude: float = 0.0,
    person_detected: bool = True,
    status: str = "medical",
    conscious: bool = True,
    mobility: str = "limited",
    needs: list[str] | None = None,
) -> IncidentReport:
    """Create a new internal incident report.

    Args:
        incident_type: Type of incident detected.
        latitude: Reported latitude.
        longitude: Reported longitude.
        person_detected: Whether a person was detected.
        status: Internal status classification.
        conscious: Victim consciousness state.
        mobility: Victim mobility assessment.
        needs: List of needs discovered during triage.

    Returns:
        An IncidentReport instance with internal fields.
    """
    if needs is None:
        needs = [status]

    incident_id = f"INC-{uuid.uuid4().hex[:12]}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    report = IncidentReport(
        robot_id=ROBOT_ID,
        incident_id=incident_id,
        timestamp=timestamp,
        type=incident_type,
        location={"latitude": latitude, "longitude": longitude},
        person={
            "detected": person_detected,
            "status": status,
            "conscious": conscious,
            "mobility": mobility,
            "needs": needs,
        },
    )
    return report


if __name__ == "__main__":
    report = create_incident_report()
    print(report.to_dict())
