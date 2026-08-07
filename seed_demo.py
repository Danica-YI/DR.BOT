"""
Seed the dashboard with realistic demo data.
Run this script to quickly populate the map and table
with sample reports spread across Brisbane.

Usage:
    python seed_demo.py           # seed data (default: http://127.0.0.1:5000)
    python seed_demo.py --clear   # clear all data first, then seed
    python seed_demo.py --url http://10.88.63.39:5000   # custom server URL
"""

import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

# Parse args
if "--url" in sys.argv:
    idx = sys.argv.index("--url")
    if idx + 1 < len(sys.argv):
        BASE_URL = sys.argv[idx + 1]

DEMO_REPORTS = [
    {
        "device_id": "DR-01",
        "status": "medical",
        "timestamp": "2026-08-08T10:30:00+10:00",
        "location": {"lat": -27.4698, "lon": 153.0251}
    },
    {
        "device_id": "DR-02",
        "status": "resource",
        "timestamp": "2026-08-08T10:32:00+10:00",
        "location": {"lat": -27.4750, "lon": 153.0180}
    },
    {
        "device_id": "DR-03",
        "status": "medical",
        "timestamp": "2026-08-08T10:34:00+10:00",
        "location": {"lat": -27.4620, "lon": 153.0300}
    },
    {
        "device_id": "DR-01",
        "status": "ok",
        "timestamp": "2026-08-08T10:36:00+10:00",
        "location": {"lat": -27.4710, "lon": 153.0230}
    },
    {
        "device_id": "DR-04",
        "status": "medical",
        "timestamp": "2026-08-08T10:38:00+10:00",
        "location": {"lat": -27.4800, "lon": 153.0100}
    },
    {
        "device_id": "DR-02",
        "status": "resource",
        "timestamp": "2026-08-08T10:40:00+10:00",
        "location": {"lat": -27.4580, "lon": 153.0350}
    },
]

if "--clear" in sys.argv:
    print(f"Clearing all reports on {BASE_URL}...")
    r = requests.delete(f"{BASE_URL}/api/reports")
    print(f"  -> {r.json()}")

print(f"Seeding {len(DEMO_REPORTS)} reports to {BASE_URL}...")
for i, report in enumerate(DEMO_REPORTS):
    r = requests.post(f"{BASE_URL}/api/report", json=report)
    status_icon = {"medical": "🔴", "resource": "🟡", "ok": "🟢"}
    icon = status_icon.get(report["status"], "⚪")
    print(f"  {icon} {report['device_id']}  {report['status']:10s}  -> {r.json()}")

print("Done! Refresh the dashboard to see the data.")