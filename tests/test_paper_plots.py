import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "plots" / "run_paper_plots.py"

class PaperPlotsTests(unittest.TestCase):
    def test_list_from_another_working_directory(self):
        result = subprocess.run([sys.executable, str(RUNNER), "--list"], cwd=ROOT.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("history", result.stdout)
        self.assertIn("gaussianity_methods", result.stdout)

    def test_missing_input_fails_without_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = subprocess.run([sys.executable, str(RUNNER), "--only", "performance",
                                     "--results-dir", str(base / "missing"),
                                     "--output-dir", str(base / "figures")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("spectra/lorenz96.json", result.stderr)
            self.assertFalse(any(p.is_file() for p in base.rglob("*")))

    def test_forecast_includes_initial_reconstruction_at_zero(self):
        spec = importlib.util.spec_from_file_location("plot_paths_test", ROOT / "plots" / "_plot_paths.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(list(module.forecast_steps(4)), [0, 1, 2, 3])

    def test_output_equal_to_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(RUNNER), "--only", "performance",
                                     "--results-dir", directory, "--output-dir", directory],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("separate figure directory", result.stderr.lower())

if __name__ == "__main__":
    unittest.main()
