import os
import time
import html
from database.db_manager import get_all_incidents, get_stats
import config.config as cfg

def generate_proctoring_report(exam_name=cfg.DEFAULT_EXAM_NAME, candidate_name=cfg.DEFAULT_CANDIDATE_NAME):
    """
    Generates a comprehensive HTML/Printable PDF Proctoring Audit Report
    summarizing total incidents, integrity score, severity metrics, and incident log details.
    Saved to the reports directory.
    """
    safe_exam_name = html.escape(str(exam_name))
    safe_candidate_name = html.escape(str(candidate_name))

    incidents = get_all_incidents()
    stats = get_stats()
    
    total_incidents = stats["total_alerts"]
    severity_counts = stats["severity_counts"]
    category_counts = stats["category_counts"]
    
    crit_count = severity_counts.get("CRITICAL", 0)
    high_count = severity_counts.get("HIGH", 0)
    med_count = severity_counts.get("MEDIUM", 0) 
    low_count = severity_counts.get("LOW", 0)
    
    # Calculate Integrity Score (100% minus weighted incident penalties)
    integrity_score = max(0, 100 - (crit_count * 25 + high_count * 15 + med_count * 5 + low_count * 2))
    
    status_badge = "PASSED" if integrity_score >= 80 else ("NEEDS REVIEW" if integrity_score >= 50 else "HIGH RISK / FAILED")
    badge_color = "#10b981" if integrity_score >= 80 else ("#f59e0b" if integrity_score >= 50 else "#ef4444")
    
    incidents_rows_html = ""
    for inc in incidents:
        raw_sev = str(inc.get("severity", "MEDIUM")).upper()
        sev = html.escape(raw_sev)
        s_color = "#ef4444" if raw_sev == "CRITICAL" else ("#f97316" if raw_sev == "HIGH" else ("#f59e0b" if raw_sev == "MEDIUM" else "#06b6d4"))
        
        inc_id = html.escape(str(inc.get('id', '')))
        student_id = html.escape(str(inc.get('student_id', '')))
        category = html.escape(str(inc.get('category', 'GENERAL')))
        activity = html.escape(str(inc.get('activity', '')))
        timestamp = html.escape(str(inc.get('timestamp', '')))
        risk_score_val = float(inc.get('risk_score', 0))

        incidents_rows_html += f"""
        <tr>
            <td>#{inc_id}</td>
            <td>Student {student_id}</td>
            <td><span style="background:{s_color}22; color:{s_color}; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px;">{sev}</span></td>
            <td>{category}</td>
            <td>{activity}</td>
            <td>{risk_score_val:.1f}%</td>
            <td>{timestamp}</td>
        </tr>
        """

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Proctoring Audit Report - {safe_exam_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 20px; margin-bottom: 24px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #38bdf8; margin: 0; }}
        .meta {{ color: #94a3b8; font-size: 14px; margin-top: 4px; }}
        .badge {{ background: {badge_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
        .stat-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }}
        .stat-val {{ font-size: 28px; font-weight: bold; color: #f8fafc; margin-top: 4px; }}
        .stat-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        tr:hover {{ background: #334155; }}
        .footer {{ margin-top: 40px; text-align: center; color: #64748b; font-size: 12px; border-top: 1px solid #334155; padding-top: 16px; }}
        @media print {{
            body {{ background: white; color: black; padding: 0; }}
            .container {{ background: white; box-shadow: none; padding: 0; color: black; }}
            .stat-card {{ background: #f8fafc; border-color: #e2e8f0; }}
            .stat-val {{ color: black; }}
            th {{ background: #f1f5f9; color: #475569; }}
            td {{ border-color: #e2e8f0; color: black; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 class="title">AI Proctoring Audit Report</h1>
                <div class="meta">Exam: <strong>{safe_exam_name}</strong> | Candidate: <strong>{safe_candidate_name}</strong></div>
                <div class="meta">Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="badge">{status_badge} ({integrity_score}%)</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-lbl">Integrity Score</div>
                <div class="stat-val" style="color:{badge_color};">{integrity_score}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-lbl">Total Incidents</div>
                <div class="stat-val">{total_incidents}</div>
            </div>
            <div class="stat-card">
                <div class="stat-lbl">Critical / High</div>
                <div class="stat-val" style="color:#ef4444;">{crit_count + high_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-lbl">Medium / Low</div>
                <div class="stat-val" style="color:#f59e0b;">{med_count + low_count}</div>
            </div>
        </div>

        <h3 style="color:#38bdf8; margin-bottom: 8px;">Detailed Incident Log</h3>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Candidate</th>
                    <th>Severity</th>
                    <th>Category</th>
                    <th>Activity / Trigger</th>
                    <th>Risk Score</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                {incidents_rows_html if incidents_rows_html else '<tr><td colspan="7" style="text-align:center; color:#94a3b8;">No suspicious incidents logged for this session.</td></tr>'}
            </tbody>
        </table>

        <div class="footer">
            Holistic AI Exam Surveillance & Proctoring System &bull; Confidential & Automated Report
        </div>
    </div>
</body>
</html>
"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"Proctoring_Report_{timestamp}.html"
    report_filepath = os.path.join(cfg.REPORTS_DIR, report_filename)

    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(report_html)
        
    print(f"[REPORT] Generated proctoring report: {report_filepath}")
    return report_filepath, report_html
