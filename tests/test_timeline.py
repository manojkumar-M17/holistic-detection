from modules.timeline import TimelineManager


timeline = TimelineManager()


# Student 1 activities

timeline.add_event(
    student_id=1,
    activity="NORMAL",
    risk_score=0,
    risk_level="NORMAL"
)

timeline.add_event(
    student_id=1,
    activity="LOOK_LEFT",
    risk_score=20,
    risk_level="LOW"
)

timeline.add_event(
    student_id=1,
    activity="HAND_FACE",
    risk_score=45,
    risk_level="MEDIUM"
)

timeline.add_event(
    student_id=1,
    activity="CELL_PHONE",
    risk_score=95,
    risk_level="CRITICAL"
)


# Student 2

timeline.add_event(
    student_id=2,
    activity="NORMAL",
    risk_score=0,
    risk_level="NORMAL"
)

timeline.add_event(
    student_id=2,
    activity="LOOK_RIGHT",
    risk_score=25,
    risk_level="LOW"
)


print("\n==============================")
print("STUDENT 1 TIMELINE")
print("==============================")

for event in timeline.get_student_timeline(1):

    print(
        event.time_string,
        "|",
        event.activity,
        "| Risk:",
        event.risk_score,
        "| Level:",
        event.risk_level
    )


print("\n==============================")
print("STUDENT 2 TIMELINE")
print("==============================")

for event in timeline.get_student_timeline(2):

    print(
        event.time_string,
        "|",
        event.activity,
        "| Risk:",
        event.risk_score,
        "| Level:",
        event.risk_level
    )


print("\n==============================")
print("EVENT COUNT")
print("==============================")

print(
    "Student 1:",
    timeline.get_event_count(1)
)

print(
    "Student 2:",
    timeline.get_event_count(2)
)