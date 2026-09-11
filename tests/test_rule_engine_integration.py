from modules.risk_engine import RiskEngine
from modules.rule_engine import RuleEngine


def print_test(title, events):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("Generated Events:")

    for event in events:
        print(f" - {event}")


def main():

    print("\n")
    print("=" * 60)
    print("RULE ENGINE INTEGRATION TEST")
    print("=" * 60)

    # Create engines
    risk_engine = RiskEngine()

    rule_engine = RuleEngine(
        risk_engine
    )

    student_id = 1


    # ==========================================
    # TEST 1: LOOKING RIGHT
    # ==========================================

    behaviour_1 = {

        "gaze_direction": "LOOKING_RIGHT",

        "gaze_event": "LOOKING_RIGHT",

        "is_looking_away": True,

        "is_standing": False,

        "hand_near_face": False,

        "has_forbidden_object": False,

        "is_absent": False,

        "shoulder_tilt": 5.0
    }

    events = rule_engine.evaluate(
        student_id,
        behaviour_1
    )

    print_test(
        "TEST 1: LOOKING RIGHT",
        events
    )


    # ==========================================
    # TEST 2: CELL PHONE
    # ==========================================

    behaviour_2 = {

        "gaze_direction": "FORWARD",

        "gaze_event": None,

        "is_looking_away": False,

        "has_forbidden_object": True,

        "forbidden_objects": [
            "Cell Phone"
        ],

        "is_standing": False,

        "hand_near_face": False,

        "is_absent": False,

        "shoulder_tilt": 0.0
    }

    events = rule_engine.evaluate(
        student_id,
        behaviour_2
    )

    print_test(
        "TEST 2: CELL PHONE",
        events
    )


    # ==========================================
    # TEST 3: MULTIPLE BEHAVIOURS
    # ==========================================

    behaviour_3 = {

        "gaze_direction": "LOOKING_DOWN",

        "gaze_event": "LOOKING_DOWN",

        "is_looking_away": True,

        "is_standing": True,

        "hand_near_face": True,

        "has_forbidden_object": True,

        "forbidden_objects": [
            "Book"
        ],

        "shoulder_tilt": 30.0,

        "is_absent": False
    }

    events = rule_engine.evaluate(
        student_id,
        behaviour_3
    )

    print_test(
        "TEST 3: MULTIPLE SUSPICIOUS EVENTS",
        events
    )


    # ==========================================
    # TEST 4: NORMAL
    # ==========================================

    behaviour_4 = {

        "gaze_direction": "FORWARD",

        "gaze_event": None,

        "is_looking_away": False,

        "is_standing": False,

        "hand_near_face": False,

        "has_forbidden_object": False,

        "is_absent": False,

        "shoulder_tilt": 5.0
    }

    events = rule_engine.evaluate(
        student_id,
        behaviour_4
    )

    print_test(
        "TEST 4: NORMAL BEHAVIOUR",
        events
    )


    print("\n" + "=" * 60)
    print("RULE ENGINE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()