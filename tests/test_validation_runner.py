import os
import tempfile
import unittest
from unittest.mock import patch

from validation.runner import ValidationInputError, ValidationRunner


class TestValidationRunner(unittest.TestCase):

    def test_missing_video_is_rejected(self):
        with self.assertRaisesRegex(ValidationInputError, "does not exist"):
            ValidationRunner("missing-session.mp4").run()

    @patch("validation.runner.cv2.VideoCapture")
    def test_zero_frame_video_is_rejected(self, mock_capture_class):
        capture = mock_capture_class.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = 30.0
        capture.read.return_value = (False, None)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "empty.mp4")
            open(video_path, "wb").close()
            with self.assertRaisesRegex(ValidationInputError, "no readable frames"):
                ValidationRunner(video_path, temp_dir).run()


if __name__ == "__main__":
    unittest.main()
