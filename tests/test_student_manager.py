from modules.student_manager import StudentManager


manager =  StudentManager()


# Register Student 1
student = manager.register_student(
    student_id=1,
    bbox=(100, 100, 300, 400),
    confidence=0.92
)

print("\nStudent Registered:")
print(student)


# Update Student
manager.update_student(
    student_id=1,
    bbox=(120, 110, 320, 410),
    confidence=0.95
)


# Update activity
manager.update_activity(
    student_id=1,
    activity="LOOK_LEFT"
)


# Update risk
manager.update_risk(
    student_id=1,
    risk_score=35,
    risk_level="LOW"
)


print("\nUpdated Student:")
print(manager.get_student(1))


# Register another student
manager.register_student(
    student_id=2,
    bbox=(400, 100, 600, 400),
    confidence=0.89
)


manager.update_activity(
    student_id=2,
    activity="CELL_PHONE"
)


manager.update_risk(
    student_id=2,
    risk_score=90,
    risk_level="CRITICAL"
)


print("\nAll Students:")

for student in manager.get_all_students():

    print(student)


print("\nDashboard Data:")

print(manager.get_dashboard_data())