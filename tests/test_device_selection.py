import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_detection(self):
        for count, expected in ((0, "cpu"), (1, "cuda")):
            with self.subTest(count=count), patch.object(
                main.ctranslate2, "get_cuda_device_count", return_value=count
            ):
                self.assertEqual(main.detect_device(), expected)

    def test_explicit_device_does_not_use_auto_detection(self):
        for device, compute_type in (("cpu", "int8"), ("cuda", "float16")):
            with self.subTest(device=device), patch.object(
                sys, "argv", ["main.py", main.__file__, "--language", "en", "--device", device]
            ), patch.object(main, "update_last_dir"), patch.object(
                main, "detect_device", side_effect=AssertionError("Unexpected auto detection")
            ), patch.object(
                main, "load_model_once", side_effect=RuntimeError("Stop before transcription")
            ) as load, contextlib.redirect_stdout(io.StringIO()):
                main.main()
                self.assertEqual(load.call_args.args[1:3], (device, compute_type))

    def test_launcher_device_takes_precedence(self):
        with patch.object(
            sys, "argv", ["main.py", main.__file__, "--language", "en", "--device", "cuda", "--device", "cpu"]
        ), patch.object(main, "update_last_dir"), patch.object(
            main, "load_model_once", side_effect=RuntimeError("Stop before transcription")
        ) as load, contextlib.redirect_stdout(io.StringIO()):
            main.main()
            self.assertEqual(load.call_args.args[1], "cpu")


if __name__ == "__main__":
    unittest.main()
