"""
Demo Mode Tests
---------------
Tests verifying synthetic demo camera generation, CLI flag parsing,
and deterministic proctoring loop simulation without hardware devices.
"""

import unittest
import numpy as np
import threading
import time

from modules.camera import CameraManager
from dashboard.app import SharedState
import main


class TestDemoMode(unittest.TestCase):

    def test_synthetic_demo_camera_frames(self):
        """Verify CameraManager in demo mode yields valid BGR frames."""
        cam = CameraManager(source="demo", width=640, height=480)
        self.assertTrue(getattr(cam, "is_synthetic", False))

        ret, frame = cam.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

        processed_bgr, processed_rgb = cam.preprocess_frame(frame)
        self.assertEqual(processed_bgr.shape, (480, 640, 3))
        self.assertEqual(processed_rgb.shape, (480, 640, 3))

        cam.release()
        self.assertFalse(getattr(cam, "is_synthetic", False))

    def test_demo_processing_loop_execution(self):
        """Verify processing_loop in is_demo=True runs and updates SharedState."""
        SharedState.current_frame = None
        # Reset shutdown event
        main.shutdown_event.clear()

        # Run loop in background thread for a brief period
        demo_thread = threading.Thread(
            target=main.processing_loop,
            args=(True,),
            name="Test_Demo_Thread",
            daemon=True
        )
        demo_thread.start()

        # Wait dynamically until demo loop processes at least one frame
        start_wait = time.time()
        while SharedState.current_frame is None and (time.time() - start_wait < 10.0):
            time.sleep(0.05)

        # Trigger shutdown
        main.shutdown_event.set()
        demo_thread.join(timeout=5.0)

        # Verify SharedState was populated by the demo simulation
        self.assertIsNotNone(SharedState.current_frame)
        self.assertEqual(SharedState.current_frame.shape, (480, 640, 3))
        self.assertIn("rms", SharedState.audio_metrics)


if __name__ == "__main__":
    unittest.main()

