"""Skip camera detection and run the cloud voice triage flow directly."""

from __future__ import annotations

import argparse
import json
import logging

from .cloud_triage_voice import CloudTriageInterviewer
from .config import BACKEND_URL, ROBOT_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cloud multilingual voice triage immediately without camera detection."
    )
    parser.add_argument("--base-url", default=BACKEND_URL, help="Backend API base URL")
    parser.add_argument("--device-id", default=ROBOT_ID, help="Device ID to report")
    parser.add_argument("--lat", type=float, default=-27.4698, help="Report latitude")
    parser.add_argument("--lon", type=float, default=153.0251, help="Report longitude")
    parser.add_argument("--audio-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to record each spoken answer")
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Run the interview but do not POST the final assessment to the backend",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()

    interviewer = CloudTriageInterviewer(
        base_url=args.base_url,
        device_id=args.device_id,
        lat=args.lat,
        lon=args.lon,
        audio_device=args.audio_device,
        record_seconds=args.duration,
    )
    assessment, success, response = interviewer.run_once(submit_report=not args.no_submit)

    print(
        json.dumps(
            {
                "submitted": not args.no_submit,
                "success": success,
                "response": response,
                "assessment": assessment,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
