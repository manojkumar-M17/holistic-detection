"""
Security & Sanitization Tests
-----------------------------
Tests verification of HTML escaping and XSS defense in reporting and export components.
"""

import html
import unittest
from unittest.mock import patch

from modules.report_generator import generate_proctoring_report


class TestSecuritySanitization(unittest.TestCase):

    @patch("modules.report_generator.get_all_incidents")
    @patch("modules.report_generator.get_stats")
    def test_xss_payload_in_exam_and_candidate_names_is_escaped(self, mock_stats, mock_incidents):
        mock_stats.return_value = {
            "total_alerts": 1,
            "severity_counts": {"HIGH": 1},
            "category_counts": {"PHONE_USAGE": 1}
        }
        mock_incidents.return_value = [
            {
                "id": 1,
                "student_id": 101,
                "severity": "HIGH",
                "category": "<script>alert('xss_cat')</script>",
                "activity": "<img src=x onerror=alert('xss_act')>",
                "risk_score": 85.5,
                "timestamp": "2026-09-13 <script>evil()</script>"
            }
        ]

        malicious_exam = '<script>alert("exam")</script>'
        malicious_candidate = '<svg onload=alert("user")>'

        _, report_html = generate_proctoring_report(malicious_exam, malicious_candidate)

        # Raw malicious tags must NOT appear unescaped in the HTML
        self.assertNotIn('<script>alert("exam")</script>', report_html)
        self.assertNotIn('<svg onload=alert("user")>', report_html)
        self.assertNotIn("<script>alert('xss_cat')</script>", report_html)
        self.assertNotIn("<img src=x onerror=alert('xss_act')>", report_html)
        self.assertNotIn("<script>evil()</script>", report_html)

        # Escaped representations must be present
        self.assertIn(html.escape(malicious_exam), report_html)
        self.assertIn(html.escape(malicious_candidate), report_html)
        self.assertIn(html.escape("<script>alert('xss_cat')</script>"), report_html)
        self.assertIn(html.escape("<img src=x onerror=alert('xss_act')>"), report_html)


if __name__ == "__main__":
    unittest.main()

