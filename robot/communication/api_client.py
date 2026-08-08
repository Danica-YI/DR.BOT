"""API client for robot backend communication."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import requests
    from requests.exceptions import RequestException, Timeout
except ImportError:  # pragma: no cover
    requests = None
    RequestException = Exception
    Timeout = Exception

from ..config import BACKEND_URL
from .payload_adapter import PayloadAdapter


DEFAULT_TIMEOUT_SECONDS = 10


class ApiClient:
    """Backend API client for the robot."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.base_url = (base_url or BACKEND_URL).rstrip('/')
        self.timeout = timeout or DEFAULT_TIMEOUT_SECONDS
        self.adapter = PayloadAdapter()
        self.session = requests.Session() if requests else None
        self.logger.info("API client initialized for backend %s", self.base_url)

    def send_report(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Send an incident report to the existing backend /api/report endpoint."""
        payload = self.adapter.to_backend_report_payload(incident)
        return self._post('/api/report', payload)

    def send_heartbeat(self, heartbeat: dict[str, Any]) -> dict[str, Any]:
        """Send a heartbeat payload to the existing backend /api/heartbeat endpoint."""
        payload = self.adapter.to_backend_heartbeat_payload(heartbeat)
        return self._post('/api/heartbeat', payload)

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.debug("Sending POST %s with payload: %s", url, payload)

        headers = {'Content-Type': 'application/json'}
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        if requests:
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                return self._build_response(response.status_code, response.text, response)
            except Timeout as exc:
                self.logger.warning("Backend request timed out: %s", exc)
                return self._error_result("timeout", str(exc))
            except RequestException as exc:
                self.logger.warning("Backend request failed: %s", exc)
                return self._error_result("connection_error", str(exc))
            except Exception as exc:
                self.logger.error("Unexpected error during backend request: %s", exc, exc_info=True)
                return self._error_result("unexpected_error", str(exc))
        return self._urllib_post(url, data, headers)

    def _build_response(
        self,
        status_code: int,
        response_text: str,
        response: Any = None,
    ) -> dict[str, Any]:
        if 200 <= status_code < 300:
            self.logger.info("Backend request succeeded (%s)", status_code)
            body = self._parse_json(response_text)
            return {
                'success': True,
                'status_code': status_code,
                'response': body,
            }

        self.logger.warning(
            "Backend request returned non-2xx status %s: %s",
            status_code,
            response_text,
        )
        return {
            'success': False,
            'status_code': status_code,
            'error': response_text,
        }

    def _parse_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _error_result(self, error_type: str, message: str) -> dict[str, Any]:
        return {
            'success': False,
            'error': error_type,
            'message': message,
        }

    def _urllib_post(self, url: str, data: bytes, headers: dict[str, str]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode('utf-8')
                return self._build_response(response.status, body)
        except urllib.error.URLError as exc:
            self.logger.warning("Backend request failed (urllib): %s", exc)
            return self._error_result('connection_error', str(exc))
        except Exception as exc:
            self.logger.error("Unexpected urllib error during backend request: %s", exc, exc_info=True)
            return self._error_result('unexpected_error', str(exc))
