from modules.risk_engine import RiskEngine


def print_student(title, student):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Student ID    : {student.student_id}")
    print(f"Risk Score    : {student.risk_score}")
    print(f"Warning Level : {student.warning_level}")
    print(f"Last Activity : {student.last_activity}")
    print(f"Violations    : {student.violations}")


def main():

    print("\n")
    print("=" * 60)
    print("RISK ENGINE INTEGRATION TEST")
    print("=" * 60)

    engine = RiskEngine()

    student_id = 1


    # TEST 1
    student = engine.update(
        student_id,
        "LOOK_RIGHT"
    )

    print_student(
        "TEST 1: LOOK RIGHT",
        student
    )


    # TEST 2
    student = engine.update(
        student_id,
        "HAND_FACE"
    )

    print_student(
        "TEST 2: HAND NEAR FACE",
        student
    )


    # TEST 3
    student = engine.update(
        student_id,
        "STANDING"
    )

    print_student(
        "TEST 3: STANDING",
        student
    )


    # TEST 4
    student = engine.update(
        student_id,
        "CELL_PHONE"
    )

    print_student(
        "TEST 4: CELL PHONE",
        student
    )


    # TEST 5
    student = engine.update(
        student_id,
        "NORMAL"
    )

    print_student(
        "TEST 5: NORMAL BEHAVIOUR",
        student
    )


    # TEST 6
    student = engine.update(
        student_id,
        "BOOK_DETECTED"
    )

    print_student(
        "TEST 6: BOOK DETECTED",
        student
    )


    # TEST 7
    student = engine.update(
        student_id,
        "STUDENT_ABSENT"
    )

    print_student(
        "TEST 7: STUDENT ABSENT",
        student
    )


    print("\n" + "=" * 60)
    print("RISK ENGINE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()