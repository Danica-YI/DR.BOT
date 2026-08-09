"""Helpers for obtaining the robot's current location."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any


class LocationProvider:
    """Resolve the current device location with environment fallback."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def get_current_location(self) -> dict[str, float] | None:
        """Return the best available current location."""
        env_location = self._location_from_env()
        if env_location is not None:
            return env_location

        windows_location = self._location_from_windows()
        if windows_location is not None:
            return windows_location

        return None

    def _location_from_env(self) -> dict[str, float] | None:
        latitude = os.getenv("ROBOT_LATITUDE")
        longitude = os.getenv("ROBOT_LONGITUDE")
        if latitude is None or longitude is None:
            return None

        try:
            return {
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        except ValueError:
            self.logger.warning("Invalid ROBOT_LATITUDE/ROBOT_LONGITUDE values; ignoring env location.")
            return None

    def _location_from_windows(self) -> dict[str, float] | None:
        """Use Windows location services when available."""
        command = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Device
$watcher = New-Object System.Device.Location.GeoCoordinateWatcher
$null = $watcher.Start()
for ($i = 0; $i -lt 20; $i++) {
    $location = $watcher.Position.Location
    if ($location -and -not $location.IsUnknown) {
        @{
            latitude = $location.Latitude
            longitude = $location.Longitude
        } | ConvertTo-Json -Compress
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
exit 1
"""
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=12,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.logger.debug("Windows location lookup failed to run: %s", exc)
            return None

        if completed.returncode != 0 or not completed.stdout.strip():
            stderr = completed.stderr.strip()
            if stderr:
                self.logger.debug("Windows location lookup unavailable: %s", stderr)
            return None

        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
            return {
                "latitude": float(payload["latitude"]),
                "longitude": float(payload["longitude"]),
            }
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to parse Windows location output: %s", exc)
            return None
