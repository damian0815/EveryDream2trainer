"""
Unit tests for model.training_model.find_last_checkpoint() and find_last_lora().

These tests exercise the directory-walking logic only; no model is loaded.
"""
import os
import tempfile
import unittest

from model.training_model import find_last_checkpoint, find_last_lora


class TestFindLastCheckpoint(unittest.TestCase):
    def _make_ckpt(self, root, name, marker):
        path = os.path.join(root, name)
        os.makedirs(path)
        marker_path = os.path.join(path, marker)
        with open(marker_path, "w") as f:
            f.write("test")
        return path

    def _touch(self, path, mtime):
        os.utime(path, (mtime, mtime))

    def test_default_markers_find_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            older = self._make_ckpt(root, "ckpt-older", "model_index.json")
            newer = self._make_ckpt(root, "ckpt-newer", "model_index.json")
            self._touch(older, 1000)
            self._touch(newer, 2000)
            self.assertEqual(find_last_checkpoint(root), newer)

    def test_find_last_lora_returns_newest_adapter_dir(self):
        with tempfile.TemporaryDirectory() as root:
            older = self._make_ckpt(root, "lora-older", "adapter_config.json")
            newer = self._make_ckpt(root, "lora-newer", "adapter_config.json")
            self._touch(older, 1000)
            self._touch(newer, 2000)
            self.assertEqual(find_last_lora(root), newer)

    def test_default_markers_ignore_lora_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            lora_dir = self._make_ckpt(root, "lora-only", "adapter_config.json")
            self._touch(lora_dir, 2000)
            with self.assertRaises(ValueError):
                find_last_checkpoint(root)

    def test_find_last_lora_ignores_model_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = self._make_ckpt(root, "ckpt-only", "model_index.json")
            self._touch(model_dir, 2000)
            with self.assertRaises(ValueError):
                find_last_lora(root)

    def test_raises_when_no_marker_found(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "empty"))
            with self.assertRaises(ValueError):
                find_last_checkpoint(root)

    def test_errored_dir_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            errored = self._make_ckpt(root, "ckpt-errored", "adapter_config.json")
            self._touch(errored, 2000)
            with self.assertRaises(ValueError):
                find_last_lora(root)


if __name__ == "__main__":
    unittest.main()
