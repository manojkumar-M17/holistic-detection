from modules.gaze_tracker import GazeTracker
import time

def main():
    print("=" * 50)
    print("GAZE TRACKER TEST")
    print("=" * 50)

    gaze_tracker = GazeTracker(
        smoothing_window=5,
        min_duration=1.0,
        cooldown=2.0
    )

    # Test 1: Forward
    print("\n[Test 1] FORWARD")

    direction = gaze_tracker.get_gaze_direction(
        yaw=5,
        pitch=3
    )

    print("Direction:", direction)

    # Test 2: Looking Right
    print("\n[Test 2] LOOKING RIGHT")

    direction = gaze_tracker.get_gaze_direction(
        yaw=30,
        pitch=2
    )

    print("Direction:", direction)

    # Test 3: Looking Left
    print("\n[Test 3] LOOKING LEFT")

    direction = gaze_tracker.get_gaze_direction(
        yaw=-30,
        pitch=2
    )

    print("Direction:", direction)

    # Test 4: Looking Down
    print("\n[Test 4] LOOKING DOWN")

    direction = gaze_tracker.get_gaze_direction(
        yaw=2,
        pitch=25
    )

    print("Direction:", direction)

    # Test 5: Looking Up
    print("\n[Test 5] LOOKING UP")

    direction = gaze_tracker.get_gaze_direction(
        yaw=2,
        pitch=-25
    )

    print("Direction:", direction)

    # Test 6: Looking Away
    print("\n[Test 6] LOOKING AWAY")

    result = gaze_tracker.is_looking_away(
        yaw=30,
        pitch=5,
        yaw_threshold=20,
        pitch_threshold=15
    )

    print("Looking Away:", result)

    print("\n" + "=" * 50)
    print("TEST COMPLETED")
    print("=" * 50)

    # Test 7: Stable Gaze Event
    print("\n[Test 7] STABLE GAZE EVENT")

    print("Waiting for stable LOOKING_DOWN event...")

    for i in range(15):

        event = gaze_tracker.get_stable_gaze_event(
            yaw=5,
            pitch=25,
            confidence=1.0
        )

        print(f"Frame {i + 1}: Event = {event}")

        time.sleep(0.1)

if __name__ == "__main__":
    main()