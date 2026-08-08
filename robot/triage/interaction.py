"""Triage interaction helpers."""

import logging
from typing import Any


class TriageInteraction:
    """Mock triage interaction interface for victim assessment."""

    def __init__(self, use_mock: bool = True) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = use_mock

        if self.use_mock:
            self.logger.info("Mock triage interaction active.")
        else:
            self.logger.warning(
                "Real triage interaction requested, but no real conversation model is integrated yet. "
                "Using mock responses until the AI/speech module is implemented."
            )
            self.use_mock = True

    def run_triage(self, context: Any | None = None) -> dict:
        """Run a mock victim triage interaction.

        Args:
            context: Optional contextual data from vision or previous state.

        Returns:
            Structured triage data suitable for report generation.
        """
        self.logger.debug("Running mock triage interaction.")
        return self._mock_response()

    def _mock_response(self) -> dict:
        return {
            "conscious": True,
            "mobility": "limited",
            "status": "medical",
            "needs": ["medical"],
            "notes": "Mock victim triage response. Replace with real interaction module later."
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    triage = TriageInteraction()
    result = triage.run_triage()
    print(result)
