"""Stable dashboard launcher without Flask debug reloader."""

import os

from app import app


if __name__ == "__main__":
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
