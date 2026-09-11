"""
Database Manager
----------------
Handles SQLite database operations for the
AI Exam Hall Monitoring System.
"""

import sqlite3
import os
import csv
import io
from datetime import datetime

from config.config import DB_PATH


class DatabaseManager:
    """
    Handles all SQLite database operations.
    """

    def __init__(self, db_path=None):
        """
        Initialize DatabaseManager.

        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database.
        """

        self.db_path = os.path.abspath(
            db_path if db_path is not None else DB_PATH
        )

        self._ensure_database_directory()


    # ==========================================
    # DATABASE PATH HANDLING
    # ==========================================

    def _ensure_database_directory(self):
        """
        Create the database parent directory if needed.
        """

        db_directory = os.path.dirname(self.db_path)

        if db_directory:
            os.makedirs(
                db_directory,
                exist_ok=True
            )


    def _get_connection(self):
        """
        Create and return a SQLite connection.
        """

        self._ensure_database_directory()

        conn = sqlite3.connect(
            self.db_path
        )

        return conn


    # ==========================================
    # INITIALIZE DATABASE
    # ==========================================

    def init_db(self):
        """
        Create incidents table and apply migrations.
        """

        self._ensure_database_directory()

        conn = self._get_connection()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    student_id INTEGER NOT NULL,

                    activity TEXT NOT NULL,

                    timestamp TEXT NOT NULL,

                    screenshot_path TEXT,

                    severity TEXT DEFAULT 'MEDIUM',

                    category TEXT DEFAULT 'GENERAL',

                    risk_score REAL DEFAULT 50.0,

                    status TEXT DEFAULT 'UNREVIEWED'
                )
            """)

            # ----------------------------------
            # DATABASE MIGRATION SUPPORT
            # ----------------------------------

            cursor.execute(
                "PRAGMA table_info(incidents)"
            )

            columns = {
                column[1]
                for column in cursor.fetchall()
            }

            migrations = {

                "severity":
                    "ALTER TABLE incidents "
                    "ADD COLUMN severity TEXT DEFAULT 'MEDIUM'",

                "category":
                    "ALTER TABLE incidents "
                    "ADD COLUMN category TEXT DEFAULT 'GENERAL'",

                "risk_score":
                    "ALTER TABLE incidents "
                    "ADD COLUMN risk_score REAL DEFAULT 50.0",

                "status":
                    "ALTER TABLE incidents "
                    "ADD COLUMN status TEXT DEFAULT 'UNREVIEWED'"
            }

            for column, query in migrations.items():

                if column not in columns:

                    cursor.execute(query)

            conn.commit()

            return True

        finally:

            conn.close()


    # ==========================================
    # INSERT INCIDENT
    # ==========================================

    def insert_incident(
        self,
        incident
    ):
        """
        Insert a new incident into the database.
        """

        student_id = incident.get(
            "student_id"
        )

        event = incident.get(
            "event",
            incident.get(
                "activity",
                "UNKNOWN"
            )
        )

        timestamp = incident.get(
            "timestamp",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        screenshot_path = incident.get(
            "screenshot",
            incident.get(
                "screenshot_path"
            )
        )

        severity = incident.get(
            "severity",
            "MEDIUM"
        )

        risk_score = incident.get(
            "risk_score",
            50.0
        )

        category = incident.get(
            "category"
        )

        if category is None:
            category = self.get_category(
                event
            )

        conn = self._get_connection()

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO incidents (

                    student_id,
                    activity,
                    timestamp,
                    screenshot_path,
                    severity,
                    category,
                    risk_score,
                    status

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    event,
                    timestamp,
                    screenshot_path,
                    severity,
                    category,
                    risk_score,
                    "UNREVIEWED"
                )
            )

            incident_id = cursor.lastrowid

            conn.commit()

            return incident_id

        finally:

            conn.close()


    # ==========================================
    # CATEGORY DETECTION
    # ==========================================

    def get_category(
        self,
        event
    ):
        """
        Automatically determine incident category.
        """

        event = str(event).upper()


        if event in [

            "LOOK_LEFT",
            "LOOK_RIGHT",
            "LOOK_UP",
            "LOOK_DOWN",
            "LOOKING_AWAY"

        ]:

            return "GAZE"


        if event in [

            "CELL_PHONE",
            "PHONE",
            "BOOK",
            "LAPTOP"

        ]:

            return "FORBIDDEN_OBJECT"


        if event in [

            "STANDING",
            "LEANING",
            "LEAVING_SEAT"

        ]:

            return "POSTURE"


        if event in [

            "HAND_FACE",
            "HAND_RAISED"

        ]:

            return "HAND_BEHAVIOUR"


        if event in [

            "MULTIPLE_PERSON",
            "MULTIPLE_PEOPLE"

        ]:

            return "PERSON_DETECTION"


        if event in [

            "TALKING",
            "WHISPERING",
            "LOUD_NOISE"

        ]:

            return "AUDIO"


        return "GENERAL"


    # ==========================================
    # UPDATE INCIDENT STATUS
    # ==========================================

    def update_incident_status(
        self,
        incident_id,
        status
    ):
        """
        Update incident review status.
        """

        valid_statuses = {

            "UNREVIEWED",

            "CONFIRMED",

            "DISMISSED"
        }


        if status is None:

            return False


        status = str(status).upper()


        if status not in valid_statuses:

            return False


        conn = self._get_connection()

        try:

            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE incidents

                SET status = ?

                WHERE id = ?
                """,
                (
                    status,
                    incident_id
                )
            )

            affected = cursor.rowcount

            conn.commit()

            return affected > 0

        finally:

            conn.close()


    # ==========================================
    # GET ALL INCIDENTS
    # ==========================================

    def get_all_incidents(
        self,
        student_id=None,
        category=None,
        severity=None,
        status=None
    ):
        """
        Retrieve incidents with optional filters.
        """

        conn = self._get_connection()

        try:

            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()

            query = """
                SELECT *

                FROM incidents

                WHERE 1 = 1
            """

            params = []


            if student_id is not None:

                query += (
                    " AND student_id = ?"
                )

                params.append(
                    student_id
                )


            if (
                category is not None
                and str(category).upper() != "ALL"
            ):

                query += (
                    " AND category = ?"
                )

                params.append(
                    category
                )


            if (
                severity is not None
                and str(severity).upper() != "ALL"
            ):

                query += (
                    " AND severity = ?"
                )

                params.append(
                    severity
                )


            if (
                status is not None
                and str(status).upper() != "ALL"
            ):

                query += (
                    " AND status = ?"
                )

                params.append(
                    status
                )


            query += (
                " ORDER BY id DESC"
            )


            cursor.execute(
                query,
                params
            )

            rows = cursor.fetchall()

            return [

                dict(row)

                for row in rows
            ]

        finally:

            conn.close()


    # ==========================================
    # GET DATABASE STATISTICS
    # ==========================================

    def get_stats(self):
        """
        Return database statistics.
        """

        conn = self._get_connection()

        try:

            cursor = conn.cursor()


            # Total alerts

            cursor.execute(
                "SELECT COUNT(*) FROM incidents"
            )

            total_alerts = cursor.fetchone()[0]


            # Unique students

            cursor.execute(
                """
                SELECT COUNT(DISTINCT student_id)

                FROM incidents
                """
            )

            unique_students = cursor.fetchone()[0]


            # Severity counts

            cursor.execute(
                """
                SELECT severity, COUNT(*)

                FROM incidents

                GROUP BY severity
                """
            )

            severity_counts = dict(
                cursor.fetchall()
            )


            # Category counts

            cursor.execute(
                """
                SELECT category, COUNT(*)

                FROM incidents

                GROUP BY category
                """
            )

            category_counts = dict(
                cursor.fetchall()
            )


            # Status counts

            cursor.execute(
                """
                SELECT status, COUNT(*)

                FROM incidents

                GROUP BY status
                """
            )

            status_counts = dict(
                cursor.fetchall()
            )


            return {

                "total_alerts":
                    total_alerts,

                "unique_students":
                    unique_students,

                "severity_counts":
                    severity_counts,

                "category_counts":
                    category_counts,

                "status_counts":
                    status_counts
            }

        finally:

            conn.close()


    # ==========================================
    # EXPORT CSV
    # ==========================================

    def export_incidents_csv(self):
        """
        Export all incidents as CSV string.
        """

        incidents = self.get_all_incidents()

        output = io.StringIO()

        writer = csv.writer(
            output
        )


        writer.writerow([

            "Incident ID",

            "Student ID",

            "Activity",

            "Severity",

            "Category",

            "Risk Score",

            "Status",

            "Timestamp",

            "Screenshot Path"
        ])


        for incident in incidents:

            writer.writerow([

                incident.get("id"),

                incident.get("student_id"),

                incident.get("activity"),

                incident.get(
                    "severity",
                    "MEDIUM"
                ),

                incident.get(
                    "category",
                    "GENERAL"
                ),

                incident.get(
                    "risk_score",
                    50.0
                ),

                incident.get(
                    "status",
                    "UNREVIEWED"
                ),

                incident.get("timestamp"),

                incident.get(
                    "screenshot_path"
                )
            ])


        return output.getvalue()


    # ==========================================
    # CLEAR DATABASE
    # ==========================================

    def clear_all_incidents(self):
        """
        Delete all incidents.
        """

        conn = self._get_connection()

        try:

            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM incidents"
            )

            conn.commit()

            return True

        finally:

            conn.close()


# =================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =================================================

_default_manager = None


def get_manager():
    """
    Return database manager for the CURRENT DB_PATH.

    Important:
    Tests patch db_manager.DB_PATH dynamically.
    Therefore the manager must be recreated whenever
    DB_PATH changes.
    """

    global _default_manager

    current_path = os.path.abspath(
        DB_PATH
    )


    if (

        _default_manager is None

        or os.path.abspath(
            _default_manager.db_path
        ) != current_path

    ):

        _default_manager = DatabaseManager(
            current_path
        )


    return _default_manager


def init_db():
    """
    Initialize database.
    """

    return get_manager().init_db()


def log_incident(
    student_id,
    activity,
    screenshot_path=None,
    severity="MEDIUM",
    category=None,
    risk_score=50.0
):
    """
    Log a new incident.
    """

    incident = {

        "student_id":
            student_id,

        "event":
            activity,

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "screenshot":
            screenshot_path,

        "severity":
            severity,

        "risk_score":
            risk_score
    }


    if category is not None:

        incident[
            "category"
        ] = category


    return get_manager().insert_incident(
        incident
    )


def update_incident_status(
    incident_id,
    status
):
    """
    Update incident status.
    """

    return get_manager().update_incident_status(
        incident_id,
        status
    )


def get_all_incidents(
    student_id=None,
    category=None,
    severity=None,
    status=None
):
    """
    Get all incidents with optional filters.
    """

    return get_manager().get_all_incidents(
        student_id=student_id,
        category=category,
        severity=severity,
        status=status
    )


def get_stats():
    """
    Get database statistics.
    """

    return get_manager().get_stats()


def export_incidents_csv():
    """
    Export incidents as CSV.
    """

    return get_manager().export_incidents_csv()


def clear_all_incidents():
    """
    Clear all incidents.
    """

    return get_manager().clear_all_incidents()