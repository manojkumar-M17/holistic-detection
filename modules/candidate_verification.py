"""
Candidate Verification (correctly named)
This file provides `CandidateVerifier` exported under the expected module name.
"""

from typing import Dict
import re


class CandidateVerifier:

	def __init__(self):
		self.verified_students: Dict[str, bool] = {}

	def validate_student_id(self, student_id):
		if student_id is None:
			return False

		student_id = str(student_id).strip()

		if not student_id:
			return False

		pattern = r"^[A-Za-z0-9_-]{3,20}$"
		return bool(re.match(pattern, student_id))

	def check_face_presence(self, face_count):
		if face_count == 0:
			return {"status": "FAILED", "message": "No face detected"}

		if face_count > 1:
			return {"status": "WARNING", "message": "Multiple faces detected"}

		return {"status": "PASSED", "message": "Single face detected"}

	def verify_candidate(self, student_id, face_count):
		if not self.validate_student_id(student_id):
			return {
				"verified": False,
				"student_id": student_id,
				"status": "FAILED",
				"message": "Invalid student ID",
			}

		face_result = self.check_face_presence(face_count)

		if face_result["status"] != "PASSED":
			return {
				"verified": False,
				"student_id": student_id,
				"status": face_result["status"],
				"message": face_result["message"],
			}

		self.verified_students[student_id] = True

		return {
			"verified": True,
			"student_id": student_id,
			"status": "VERIFIED",
			"message": "Candidate ready for monitoring",
		}

	def is_verified(self, student_id):
		return self.verified_students.get(student_id, False)

	def remove_candidate(self, student_id):
		if student_id in self.verified_students:
			del self.verified_students[student_id]

	def clear(self):
		self.verified_students.clear()

__all__ = ["CandidateVerifier"]
