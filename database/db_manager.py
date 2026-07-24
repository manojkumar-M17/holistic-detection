import sqlite3
import os
import csv
import io
from datetime import datetime
from config.config import DB_PATH

def init_db():
    """
    Initializes the SQLite database, creates the incidents table,
    and applies schema migrations for severity, category, risk_score, and status.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            activity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            screenshot_path TEXT NOT NULL,
            severity TEXT DEFAULT 'MEDIUM',
            category TEXT DEFAULT 'GENERAL',
            risk_score REAL DEFAULT 50.0,
            status TEXT DEFAULT 'UNREVIEWED'
        )
    ''')
    
    # Check existing columns to apply auto-migration if needed
    cursor.execute("PRAGMA table_info(incidents)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "severity" not in columns:
        cursor.execute("ALTER TABLE incidents ADD COLUMN severity TEXT DEFAULT 'MEDIUM'")
    if "category" not in columns:
        cursor.execute("ALTER TABLE incidents ADD COLUMN category TEXT DEFAULT 'GENERAL'")
    if "risk_score" not in columns:
        cursor.execute("ALTER TABLE incidents ADD COLUMN risk_score REAL DEFAULT 50.0")
    if "status" not in columns:
        cursor.execute("ALTER TABLE incidents ADD COLUMN status TEXT DEFAULT 'UNREVIEWED'")
        
    conn.commit()
    conn.close()

def log_incident(student_id, activity, screenshot_path, severity="MEDIUM", category="GENERAL", risk_score=50.0):
    """
    Logs a new suspicious incident into SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db_screenshot_path = os.path.join("screenshots", os.path.basename(screenshot_path)).replace("\\", "/")
    
    cursor.execute('''
        INSERT INTO incidents (student_id, activity, timestamp, screenshot_path, severity, category, risk_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'UNREVIEWED')
    ''', (student_id, activity, timestamp, db_screenshot_path, severity, category, risk_score))
    
    incident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[DB] Logged incident #{incident_id} [{severity}] for Student {student_id}: {activity} (Risk: {risk_score:.1f}%)")
    return incident_id

def update_incident_status(incident_id, status):
    """
    Updates the review status of an incident ('UNREVIEWED', 'CONFIRMED', 'DISMISSED').
    """
    valid_statuses = {"UNREVIEWED", "CONFIRMED", "DISMISSED"}
    if status not in valid_statuses:
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE incidents SET status = ? WHERE id = ?', (status, incident_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def get_all_incidents(student_id=None, category=None, severity=None, status=None):
    """
    Retrieves filtered logged incidents sorted by timestamp descending.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = 'SELECT * FROM incidents WHERE 1=1'
    params = []
    
    if student_id is not None and str(student_id).strip() != "":
        query += ' AND student_id = ?'
        params.append(student_id)
    if category is not None and category != "ALL":
        query += ' AND category = ?'
        params.append(category)
    if severity is not None and severity != "ALL":
        query += ' AND severity = ?'
        params.append(severity)
    if status is not None and status != "ALL":
        query += ' AND status = ?'
        params.append(status)
        
    query += ' ORDER BY timestamp DESC'
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    """
    Computes summary analytics for dashboard statistics and charts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM incidents')
    total_alerts = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT student_id) FROM incidents')
    unique_students_flagged = cursor.fetchone()[0]
    
    cursor.execute('SELECT severity, COUNT(*) FROM incidents GROUP BY severity')
    severity_counts = dict(cursor.fetchall())
    
    cursor.execute('SELECT category, COUNT(*) FROM incidents GROUP BY category')
    category_counts = dict(cursor.fetchall())
    
    cursor.execute('SELECT status, COUNT(*) FROM incidents GROUP BY status')
    status_counts = dict(cursor.fetchall())
    
    conn.close()
    return {
        "total_alerts": total_alerts,
        "unique_students": unique_students_flagged,
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "status_counts": status_counts
    }

def export_incidents_csv():
    """
    Generates a CSV string of all logged incidents for proctoring audit reports.
    """
    incidents = get_all_incidents()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Incident ID", "Student ID", "Activity", "Severity", "Category", "Risk Score (%)", "Status", "Timestamp", "Screenshot Path"])
    for inc in incidents:
        writer.writerow([
            inc["id"],
            inc["student_id"],
            inc["activity"],
            inc.get("severity", "MEDIUM"),
            inc.get("category", "GENERAL"),
            f"{inc.get('risk_score', 50.0):.1f}",
            inc.get("status", "UNREVIEWED"),
            inc["timestamp"],
            inc["screenshot_path"]
        ])
        
    return output.getvalue()

def clear_all_incidents():
    """
    Clears all incident records from database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM incidents')
    conn.commit()
    conn.close()

