from flask import Flask, request, jsonify, render_template
import json
import sqlite3
import os

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'reports.db')


def ensure_column(conn, table_name, column_name, definition):
    columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


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
            photo TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            battery INTEGER,
            status TEXT,
            last_seen TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS triage_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            assessment_json TEXT NOT NULL,
            timestamp TEXT,
            lat REAL,
            lon REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    ensure_column(conn, 'reports', 'photo', 'TEXT')
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_photo_value(photo):
    if photo is None or not isinstance(photo, str):
        return None
    photo = photo.strip()
    if not photo:
        return None
    if photo.startswith('data:image/'):
        return photo
    return None


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
      "status": "medical",      # "no_response" | "both" | "medical" | "resource"
      "timestamp": "2026-08-07T14:32:18+10:00",
      "location": {"lat": -27.4698, "lon": 153.0251},
      "photo": "data:image/jpeg;base64,/9j/4AAQ..."   # optional
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Missing or invalid JSON body"}), 400

    required_fields = ['device_id', 'status', 'timestamp']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

    if data['status'] not in ('no_response', 'both', 'medical', 'resource'):
        return jsonify({"success": False, "error": "status must be one of: no_response, both, medical, resource"}), 400

    location = data.get('location') or {}
    lat = location.get('lat')
    lon = location.get('lon')
    photo = normalize_photo_value(data.get('photo'))

    conn = get_db()
    cursor = conn.execute(
        'INSERT INTO reports (device_id, status, timestamp, lat, lon, photo, resolved) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (data['device_id'], data['status'], data['timestamp'], lat, lon, photo, 0)
    )

    # Report also counts as heartbeat (device is alive)
    conn.execute('''
        INSERT INTO devices (device_id, lat, lon, last_seen)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(device_id) DO UPDATE SET
            lat = excluded.lat,
            lon = excluded.lon,
            last_seen = CURRENT_TIMESTAMP
    ''', (data['device_id'], lat, lon))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"success": True, "report_id": report_id}), 200


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
        if r['status'] not in ('no_response', 'both', 'medical', 'resource'):
            continue
        location = r.get('location') or {}
        photo = normalize_photo_value(r.get('photo'))
        conn.execute(
            'INSERT INTO reports (device_id, status, timestamp, lat, lon, photo, resolved) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (r['device_id'], r['status'], r['timestamp'], location.get('lat'), location.get('lon'), photo, 0)
        )
        inserted += 1
    conn.commit()
    conn.close()

    return jsonify({"success": True, "inserted": inserted}), 200


@app.route('/api/triage', methods=['POST'])
def receive_triage_assessment():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Missing or invalid JSON body"}), 400

    assessment = data.get('assessment') or {}
    if not assessment:
        return jsonify({"success": False, "error": "Missing assessment payload"}), 400

    location = data.get('location') or {}
    conn = get_db()
    conn.execute(
        'INSERT INTO triage_assessments (device_id, assessment_json, timestamp, lat, lon, resolved) VALUES (?, ?, ?, ?, ?, ?)',
        (
            data['device_id'],
            json.dumps(assessment),
            data.get('timestamp') or assessment.get('timestamp'),
            location.get('lat'),
            location.get('lon'),
            0,
        ),
    )
    conn.commit()
    conn.close()

    report_status = assessment.get('priority')
    if assessment.get('response_detected') is False and assessment.get('review_reason') in (
        'NO_RESPONSE', 'NO_VOICE_OR_GESTURE_RESPONSE'
    ):
        report_status = 'no_response'

    if report_status in ('medical', 'resource', 'no_response'):
        photo = normalize_photo_value(data.get('photo') or assessment.get('photo'))
        conn = get_db()
        conn.execute(
            'INSERT INTO reports (device_id, status, timestamp, lat, lon, photo, resolved) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                data['device_id'],
                report_status,
                assessment.get('timestamp') or data.get('timestamp'),
                location.get('lat'),
                location.get('lon'),
                photo,
                0,
            ),
        )
        conn.commit()
        conn.close()

    return jsonify({"success": True, "assessment_id": assessment.get('assessment_id')}), 200


@app.route('/api/triage/batch', methods=['POST'])
def receive_triage_batch():
    data = request.get_json(silent=True)
    if not data or 'assessments' not in data:
        return jsonify({"success": False, "error": "Missing 'assessments' array"}), 400

    inserted = 0
    conn = get_db()
    for assessment in data['assessments']:
        if not assessment:
            continue
        conn.execute(
            'INSERT INTO triage_assessments (device_id, assessment_json, timestamp, lat, lon, resolved) VALUES (?, ?, ?, ?, ?, ?)',
            (
                data['device_id'],
                json.dumps(assessment),
                assessment.get('timestamp'),
                data.get('location', {}).get('lat'),
                data.get('location', {}).get('lon'),
                0,
            ),
        )
        report_status = assessment.get('priority')
        if assessment.get('response_detected') is False and assessment.get('review_reason') in (
            'NO_RESPONSE', 'NO_VOICE_OR_GESTURE_RESPONSE'
        ):
            report_status = 'no_response'
        if report_status in ('medical', 'resource', 'no_response'):
            photo = normalize_photo_value(assessment.get('photo'))
            conn.execute(
                'INSERT INTO reports (device_id, status, timestamp, lat, lon, photo, resolved) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (
                    data['device_id'], report_status, assessment.get('timestamp'),
                    data.get('location', {}).get('lat'), data.get('location', {}).get('lon'), photo, 0,
                ),
            )
        inserted += 1
    conn.commit()
    conn.close()
    return jsonify({"success": True, "inserted": inserted}), 200


@app.route('/api/triage', methods=['GET'])
def get_triage_assessments():
    conn = get_db()
    rows = conn.execute('SELECT * FROM triage_assessments ORDER BY timestamp DESC').fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """
    Return reports, most recent first. Supports optional filtering:

      GET /api/reports                       -> all reports
      GET /api/reports?resolved=false         -> only unresolved
      GET /api/reports?resolved=true          -> only resolved
      GET /api/reports?status=medical         -> only a given status
      GET /api/reports?status=medical&resolved=false  -> combine filters
    """
    query = 'SELECT * FROM reports WHERE 1=1'
    params = []

    status = request.args.get('status')
    if status in ('no_response', 'both', 'medical', 'resource'):
        query += ' AND status = ?'
        params.append(status)

    resolved = request.args.get('resolved')
    if resolved is not None:
        query += ' AND resolved = ?'
        params.append(1 if resolved.lower() == 'true' else 0)

    query += ' ORDER BY timestamp DESC'

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route('/api/reports/<int:report_id>/resolve', methods=['PATCH'])
def resolve_report(report_id):
    """Mark a report as resolved (staff has handled this person)."""
    conn = get_db()
    cur = conn.execute('UPDATE reports SET resolved = 1 WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        return jsonify({"success": False, "error": "Report not found"}), 404
    return jsonify({"success": True}), 200


@app.route('/api/reports/<int:report_id>/unresolve', methods=['PATCH'])
def unresolve_report(report_id):
    """Undo a resolve mark, in case it was clicked by mistake."""
    conn = get_db()
    cur = conn.execute('UPDATE reports SET resolved = 0 WHERE id = ?', (report_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        return jsonify({"success": False, "error": "Report not found"}), 404
    return jsonify({"success": True}), 200


@app.route('/api/reports', methods=['DELETE'])
def clear_reports():
    """Wipe all reports. Useful for resetting before a demo run."""
    conn = get_db()
    conn.execute('DELETE FROM reports')
    conn.commit()
    conn.close()
    return jsonify({"success": True}), 200


# ── Device heartbeat ──────────────────────────────────

@app.route('/api/heartbeat', methods=['POST'])
def receive_heartbeat():
    """
    Receive a heartbeat from a device.

    Expected JSON body:
    {
      "device_id": "DR-01",
      "lat": -27.4698,
      "lon": 153.0251,
      "battery": 72,
      "status": "scanning"    // "scanning" | "locked" | "idle"
    }
    """
    data = request.get_json(silent=True)
    if not data or 'device_id' not in data:
        return jsonify({"success": False, "error": "Missing device_id"}), 400

    conn = get_db()
    conn.execute('''
        INSERT INTO devices (device_id, lat, lon, battery, status, last_seen)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(device_id) DO UPDATE SET
            lat = excluded.lat,
            lon = excluded.lon,
            battery = excluded.battery,
            status = excluded.status,
            last_seen = CURRENT_TIMESTAMP
    ''', (data['device_id'], data.get('lat'), data.get('lon'),
          data.get('battery'), data.get('status')))
    conn.commit()
    conn.close()
    return jsonify({"success": True}), 200


@app.route('/api/devices', methods=['GET'])
def get_devices():
    """
    Return all known devices with their last heartbeat info.
    Each device includes an 'online' field: true if last_seen
    is within the last 600 seconds, false otherwise. Also includes
    'seconds_ago': how many seconds since the last update.
    """
    conn = get_db()
    rows = conn.execute('''
        SELECT *,
            CASE WHEN (julianday('now') - julianday(last_seen)) * 86400 < 600
                 THEN 1 ELSE 0 END AS online,
            CAST((julianday('now') - julianday(last_seen)) * 86400 AS INTEGER) AS seconds_ago
        FROM devices
        ORDER BY device_id
    ''').fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
