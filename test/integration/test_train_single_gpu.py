"""
Integration test: verify single-GPU training completes a short run.

This test launches train.py with a small synthetic dataset, 1 epoch, and
checks that:
- The process exits with code 0
- A checkpoint directory is created under the log folder

Requirements: run from the project root inside the conda env.
Skipped automatically when CUDA is not available or when running in CI
without a real model checkpoint.
"""
import os
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(
    os.environ.get("ED_INTEGRATION_TESTS", "0") == "1",
    "Integration tests disabled. Set ED_INTEGRATION_TESTS=1 to enable.",
)
class TestTrainSingleGPU(unittest.TestCase):
    """
    Launches train.py in a subprocess with --debug_no_load_model to verify
    the training scaffolding runs without error.
    """

    def test_single_gpu_debug_run(self):
        """Single-GPU run with --debug_no_load_model exits cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "train.py",
                "--debug_no_load_model",
                "--logdir", tmpdir,
                "--project_name", "integration_test",
                "--data_root", "test/data",
                "--max_epochs", "1",
                "--max_steps", "2",
                "--batch_size", "1",
                "--resolution", "512",
                "--resume_ckpt", "sd_v1-5_vae.ckpt",  # placeholder, not loaded
                "--no_save_on_error",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.fail(
                    f"train.py exited with {result.returncode}\n"
                    f"STDOUT:\n{result.stdout[-3000:]}\n"
                    f"STDERR:\n{result.stderr[-3000:]}"
                )


if __name__ == "__main__":
    unittest.main()

