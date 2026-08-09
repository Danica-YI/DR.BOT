"""State machine for robot mission flow."""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Any

from ..camera import Camera
from ..config import BACKEND_URL, HEARTBEAT_INTERVAL_SECONDS, ROBOT_ID
from ..hardware.motors import MotorController
from ..location_provider import LocationProvider
from ..perception import PerceptionClient
from ..report import create_incident_report
from ..storage.offline_queue import OfflineQueue
from ..triage import TriageInteraction
from ..communication import ApiClient, SyncManager


class RobotState(Enum):
    SEARCHING = auto()
    PERSON_DETECTED = auto()
    STOPPED = auto()
    INTERACTING = auto()
    REPORTING = auto()
    OFFLINE = auto()
    ERROR = auto()
    RETURNING = auto()


class StateMachine:
    """Simple state machine for robot mission control."""

    _TRANSITIONS = {
        RobotState.SEARCHING: {RobotState.PERSON_DETECTED, RobotState.OFFLINE, RobotState.ERROR},
        RobotState.PERSON_DETECTED: {RobotState.STOPPED, RobotState.ERROR},
        RobotState.STOPPED: {RobotState.INTERACTING, RobotState.SEARCHING, RobotState.ERROR},
        RobotState.INTERACTING: {RobotState.REPORTING, RobotState.ERROR},
        RobotState.REPORTING: {RobotState.SEARCHING, RobotState.RETURNING, RobotState.ERROR},
        RobotState.RETURNING: {RobotState.SEARCHING, RobotState.OFFLINE, RobotState.ERROR},
        RobotState.OFFLINE: {RobotState.SEARCHING, RobotState.ERROR},
        RobotState.ERROR: {RobotState.OFFLINE},
    }

    def __init__(
        self,
        camera: Camera | None = None,
        perception: PerceptionClient | None = None,
        motors: MotorController | None = None,
        triage: TriageInteraction | None = None,
        offline_queue: OfflineQueue | None = None,
        api_client: ApiClient | None = None,
        sync_manager: SyncManager | None = None,
        initial_state: RobotState = RobotState.OFFLINE,
        heartbeat_interval: int | None = None,
        periodic_test_report_interval: int | None = None,
        static_location: dict[str, float] | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.camera = camera or Camera()
        self.perception = perception or PerceptionClient()
        self.motors = motors or MotorController()
        self.triage = triage or TriageInteraction()
        self.offline_queue = offline_queue or OfflineQueue()
        self.api_client = api_client or ApiClient(base_url=BACKEND_URL)
        self.sync_manager = sync_manager or SyncManager(
            offline_queue=self.offline_queue,
            api_client=self.api_client,
        )
        self.location_provider = LocationProvider()
        self.current_state = initial_state
        self.static_location = static_location
        self.default_location: dict[str, float] | None = static_location or self.location_provider.get_current_location()
        self.heartbeat_interval = heartbeat_interval or HEARTBEAT_INTERVAL_SECONDS
        self.program_startup_time = time.time()
        self.last_heartbeat_sent_at = self.program_startup_time - self.heartbeat_interval
        self.periodic_test_report_interval = periodic_test_report_interval
        self.last_periodic_test_report_time = time.time()
        self.auto_report_enabled = not self._is_mock_report_mode()
        self.logger.info("Initialized state machine in %s", self.current_state.name)
        if not self.auto_report_enabled:
            self.logger.info(
                "Mock report mode detected; automatic incident reporting is disabled "
                "while heartbeat remains active."
            )

    @staticmethod
    def _parse_state(state: str | RobotState) -> RobotState:
        if isinstance(state, RobotState):
            return state
        if isinstance(state, str):
            try:
                return RobotState[state]
            except KeyError as exc:
                raise ValueError(f"Unknown robot state: {state}") from exc
        raise ValueError(f"Unsupported state type: {type(state).__name__}")

    def can_transition(self, next_state: str | RobotState) -> bool:
        next_state_enum = self._parse_state(next_state)
        allowed = self._TRANSITIONS.get(self.current_state, set())
        return next_state_enum in allowed

    def transition(self, next_state: str | RobotState) -> bool:
        next_state_enum = self._parse_state(next_state)

        if next_state_enum == self.current_state:
            self.logger.debug("Already in state %s", self.current_state.name)
            return True

        if not self.can_transition(next_state_enum):
            self.logger.warning(
                "Invalid state transition attempted from %s to %s",
                self.current_state.name,
                next_state_enum.name,
            )
            return False

        previous_state = self.current_state
        self.current_state = next_state_enum
        self.logger.info(
            "State transition: %s -> %s",
            previous_state.name,
            self.current_state.name,
        )
        return True

    def force_error(self, message: str | None = None) -> None:
        if message:
            self.logger.error("Entering ERROR state: %s", message)
        else:
            self.logger.error("Entering ERROR state")
        self.current_state = RobotState.ERROR

    def reset_to_offline(self) -> None:
        self.logger.info("Resetting state machine to OFFLINE")
        self.current_state = RobotState.OFFLINE

    def run_once(self) -> None:
        """Run a single state machine cycle."""
        try:
            self._send_heartbeat_if_due()
            self._send_periodic_test_report_if_due()

            if self.current_state == RobotState.OFFLINE:
                self.transition(RobotState.SEARCHING)

            if self.current_state == RobotState.SEARCHING:
                self._search()
            elif self.current_state == RobotState.PERSON_DETECTED:
                self._handle_person_detected()
            elif self.current_state == RobotState.STOPPED:
                self._prepare_interaction()
            elif self.current_state == RobotState.INTERACTING:
                self._perform_interaction()
            elif self.current_state == RobotState.REPORTING:
                self._report()
            else:
                self.logger.debug("Unsupported current state %s; returning to SEARCHING.", self.current_state.name)
                self.transition(RobotState.SEARCHING)
        except Exception as exc:
            self.logger.error("Unhandled error in state machine cycle: %s", exc, exc_info=True)
            self._safe_stop()
            self.force_error(str(exc))

    def _search(self) -> None:
        self.logger.info("Searching for persons...")
        try:
            frame = self.camera.get_frame()
            person_detected = self.perception.detect_person(frame)
        except Exception as exc:
            self.logger.warning("Search failed: %s", exc)
            self._safe_stop()
            return

        if not person_detected:
            self.logger.debug("No person detected; remaining in SEARCHING.")
            return

        if not self.auto_report_enabled:
            self.logger.debug(
                "Mock person detection ignored because automatic report generation is disabled."
            )
            return

        if self.transition(RobotState.PERSON_DETECTED):
            self._safe_stop()
            if self.transition(RobotState.STOPPED):
                self.transition(RobotState.INTERACTING)
                self._perform_interaction()

    def _handle_person_detected(self) -> None:
        self.logger.info("Person detected; stopping robot.")
        self._safe_stop()
        self.transition(RobotState.STOPPED)

    def _prepare_interaction(self) -> None:
        self.logger.info("Preparing interaction after stop.")
        if self.transition(RobotState.INTERACTING):
            self._perform_interaction()

    def _perform_interaction(self) -> None:
        self.logger.info("Running triage interaction.")
        if not self.auto_report_enabled:
            self.logger.info("Skipping triage/report flow because automatic reporting is disabled.")
            self.transition(RobotState.SEARCHING)
            return

        try:
            triage_result = self.triage.run_triage()
        except Exception as exc:
            self.logger.warning("Triage failed: %s", exc)
            self._safe_stop()
            return

        self.transition(RobotState.REPORTING)
        incident = self._create_incident(triage_result)
        self.offline_queue.add_incident(incident.to_dict())
        self._reset_heartbeat_timer()
        self._report()

    def _create_incident(self, triage_result: dict[str, Any]) -> Any:
        self.logger.debug("Creating internal incident from triage result.")
        return create_incident_report(
            incident_type="person_detected",
            latitude=(self._get_current_location() or {}).get("latitude", 0.0),
            longitude=(self._get_current_location() or {}).get("longitude", 0.0),
            person_detected=True,
            status=triage_result.get("status", "medical"),
            conscious=triage_result.get("conscious", True),
            mobility=triage_result.get("mobility", "limited"),
            needs=triage_result.get("needs", [triage_result.get("status", "medical")]),
        )

    def _report(self) -> None:
        self.logger.info("Attempting backend synchronization for pending incidents.")
        try:
            results = self.sync_manager.sync_pending_incidents()
            self.logger.info("Sync results: %s", results)
        except Exception as exc:
            self.logger.warning("Sync attempt failed: %s", exc)

        self.transition(RobotState.SEARCHING)

    def _safe_stop(self) -> None:
        try:
            self.motors.stop_robot()
        except Exception as exc:
            self.logger.error("Failed to execute safe stop: %s", exc, exc_info=True)

    def _is_mock_report_mode(self) -> bool:
        """Return whether detection/triage are running in mock mode.

        In this mode heartbeat stays enabled, but mock detections must not create
        real incident reports automatically.
        """
        return any(
            getattr(component, "use_mock", False)
            for component in (self.camera, self.perception, self.triage)
        )

    def _should_send_heartbeat(self) -> bool:
        """Check if it is time to send a heartbeat."""
        elapsed = time.time() - self.last_heartbeat_sent_at
        return elapsed >= self.heartbeat_interval

    def _reset_heartbeat_timer(self) -> None:
        """Reset the heartbeat timer by updating program startup time."""
        self.last_heartbeat_sent_at = time.time()
        self.logger.info("Heartbeat timer reset after new incident report.")

    def _get_current_location(self) -> dict[str, float] | None:
        """Fetch and cache the best current location."""
        if self.static_location is not None:
            return self.static_location
        latest = self.location_provider.get_current_location()
        if latest is not None:
            self.default_location = latest
        return self.default_location

    def _send_periodic_test_report_if_due(self) -> None:
        """Emit a synthetic report on a fixed interval for dashboard testing."""
        if not self.periodic_test_report_interval:
            return

        elapsed = time.time() - self.last_periodic_test_report_time
        if elapsed < self.periodic_test_report_interval:
            return

        self.logger.info("Periodic test report interval reached; creating a synthetic incident.")
        location = self._get_current_location()
        incident = create_incident_report(
            incident_type="periodic_test_report",
            latitude=(location or {}).get("latitude", 0.0),
            longitude=(location or {}).get("longitude", 0.0),
            person_detected=True,
            status="medical",
            conscious=True,
            mobility="limited",
            needs=["medical"],
        )
        self.offline_queue.add_incident(incident.to_dict())
        self.last_periodic_test_report_time = time.time()
        self._reset_heartbeat_timer()
        self._report()

    def _send_heartbeat_if_due(self) -> None:
        """Send a periodic heartbeat status update if the interval has elapsed."""
        if not self._should_send_heartbeat():
            return

        location = self._get_current_location() or {}
        heartbeat = {
            "device_id": ROBOT_ID,
            "lat": location.get("latitude"),
            "lon": location.get("longitude"),
            "battery": 75,
            "status": self.current_state.name.lower(),
        }

        try:
            result = self.api_client.send_heartbeat(heartbeat)
            if result.get("success"):
                self.logger.info("Heartbeat sent successfully: %s", result)
                self._reset_heartbeat_timer()
            else:
                self.logger.warning("Heartbeat send failed: %s", result.get("error"))
        except Exception as exc:
            self.logger.warning("Heartbeat send exception: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    machine = StateMachine()
    machine.run_once()
