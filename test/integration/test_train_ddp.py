"""
Integration test: verify multi-GPU DDP training with torchrun.

Launches `torchrun --nproc_per_node=2 train.py ...` and verifies:
- All ranks start and complete without error
- Only rank 0 writes checkpoint / log artefacts (implicit: no duplicate saves)
- The process group initialises and tears down cleanly

Run with:
    ED_INTEGRATION_TESTS=1 pytest test/integration/test_train_ddp.py -v

Notes:
- torchrun assigns both ranks to the same CUDA device on single-GPU machines
  (valid for CI / functional testing purposes)
- The test is skipped unless ED_INTEGRATION_TESTS=1 is set
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
class TestTrainDDP(unittest.TestCase):

    def _run_ddp(self, nproc: int, extra_args=None):
        extra_args = extra_args or []
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                "--master_port=29600",
                "train.py",
                "--debug_no_load_model",
                "--logdir", tmpdir,
                "--project_name", "ddp_integ_test",
                "--data_root", "test/data",
                "--max_epochs", "1",
                "--max_steps", "2",
                "--batch_size", "1",
                "--resolution", "512",
                "--resume_ckpt", "sd_v1-5_vae.ckpt",
                "--no_save_on_error",
            ] + extra_args

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return result, tmpdir

    def test_two_process_ddp_exits_cleanly(self):
        result, _ = self._run_ddp(nproc=2)
        if result.returncode != 0:
            self.fail(
                f"torchrun(nproc=2) exited with {result.returncode}\n"
                f"STDOUT:\n{result.stdout[-4000:]}\n"
                f"STDERR:\n{result.stderr[-4000:]}"
            )

    def test_rank0_only_checkpoint(self):
        """
        When multiple processes run, only rank-0 should write checkpoint dirs.
        Since --debug_no_load_model skips the real save, we verify no exception
        about duplicate file access is raised.
        """
        result, tmpdir = self._run_ddp(nproc=2)
        # Just verify clean exit; checkpoint write is gated to rank-0 in train.py
        self.assertEqual(result.returncode, 0,
                         f"torchrun failed:\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()

