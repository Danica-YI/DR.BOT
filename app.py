from flask import Flask, request, jsonify, render_template
import sqlite3
import os

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'reports.db')


def init_db():
    """Create the reports table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            lat REAL,
            lon REAL,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def dashboard():
    """Serve the live dashboard page."""
    return render_template('index.html')


@app.route('/api/report', methods=['POST'])
def receive_report():
    """
    Receive a single triage report from a device.

    Expected JSON body:
    {
      "device_id": "V003",
      "status": "medical",      # "ok" | "medical" | "resource"
      "timestamp": "2026-08-07T14:32:18+10:00",
      "location": {"lat": -27.4698, "lon": 153.0251}
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Missing or invalid JSON body"}), 400

    required_fields = ['device_id', 'status', 'timestamp']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

    if data['status'] not in ('ok', 'medical', 'resource'):
        return jsonify({"success": False, "error": "status must be one of: ok, medical, resource"}), 400

    location = data.get('location') or {}
    lat = location.get('lat')
    lon = location.get('lon')

    conn = get_db()
    conn.execute(
        'INSERT INTO reports (device_id, status, timestamp, lat, lon) VALUES (?, ?, ?, ?, ?)',
        (data['device_id'], data['status'], data['timestamp'], lat, lon)
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True}), 200


@app.route('/api/reports/batch', methods=['POST'])
def receive_batch():
    """
    Receive multiple queued reports at once (used when a device reconnects
    after being offline and needs to flush its local cache).

    Expected JSON body:
    { "reports": [ {...}, {...}, ... ] }   # same shape as /api/report
    """
    data = request.get_json(silent=True)
    if not data or 'reports' not in data:
        return jsonify({"success": False, "error": "Missing 'reports' array"}), 400

    conn = get_db()
    inserted = 0
    for r in data['reports']:
        if not all(k in r for k in ('device_id', 'status', 'timestamp')):
            continue
        if r['status'] not in ('ok', 'medical', 'resource'):
            continue
        location = r.get('location') or {}
        conn.execute(
            'INSERT INTO reports (device_id, status, timestamp, lat, lon) VALUES (?, ?, ?, ?, ?)',
            (r['device_id'], r['status'], r['timestamp'], location.get('lat'), location.get('lon'))
        )
        inserted += 1
    conn.commit()
    conn.close()

    return jsonify({"success": True, "inserted": inserted}), 200


@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Return all reports, most recent first."""
    conn = get_db()
    rows = conn.execute('SELECT * FROM reports ORDER BY timestamp DESC').fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
