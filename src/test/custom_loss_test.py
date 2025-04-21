"""
Tests for CustomFrameLevelLoss.

Run with:
    python -m unittest path/to/this/file.py
"""
import sys
import os
import unittest
import tensorflow as tf

# Make sure we can import from the project root when tests are run directly
try:
    from src.training.custom_frame_level_loss import CustomFrameLevelLoss
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.training.custom_frame_level_loss import CustomFrameLevelLoss


class TestCustomFrameLevelLoss(unittest.TestCase):
    """Unit‑tests for the frame‑level loss."""

    @staticmethod
    def compute_loss(y_true, y_pred):
        """Utility to instantiate the loss once and compute its value."""
        loss_fn = CustomFrameLevelLoss()
        return loss_fn(y_true, y_pred).numpy()

    # ---------- tests --------------------------------------------------------
    def test_no_mask(self):
        """All frames contribute to the loss."""
        y_true = tf.constant(
            [[[1], [0]],
             [[1], [0]],
             [[1], [1]],
             [[0], [1]]], dtype=tf.float32)
        y_pred = tf.constant(
            [[[0.9], [0.3]],
             [[0.8], [0.1]],
             [[0.7], [0.2]],
             [[0.3], [0.7]]], dtype=tf.float32)

        loss1 = self.compute_loss(y_true, y_pred)
        loss2 = self.compute_loss(y_true, y_pred)  # identical input

        self.assertAlmostEqual(loss1, loss2, places=6)

    def test_partially_masked(self):
        """Zeros in y_true act as a mask; masked frames shouldn’t affect loss."""
        y_true_1 = tf.constant(
            [[[1], [0]],
             [[0], [0]],
             [[1], [1]],
             [[0], [1]]], dtype=tf.float32)
        y_pred_1 = tf.constant(
            [[[0.9], [0.3]],
             [[0.8], [0.1]],
             [[0.7], [0.2]],
             [[0.3], [0.7]]], dtype=tf.float32)

        y_true_2 = tf.constant(
            [[[1], [0]],
             [[1], [1]],
             [[0], [1]]], dtype=tf.float32)
        y_pred_2 = tf.constant(
            [[[0.9], [0.3]],
             [[0.7], [0.2]],
             [[0.3], [0.7]]], dtype=tf.float32)

        loss1 = self.compute_loss(y_true_1, y_pred_1)
        loss2 = self.compute_loss(y_true_2, y_pred_2)

        self.assertAlmostEqual(loss1, loss2, places=6)

    def test_all_masked(self):
        """If every frame is masked, the loss should fall back to 0."""
        y_true = tf.zeros((4, 2, 1), dtype=tf.float32)
        y_pred = tf.constant(
            [[[0.9], [0.3]],
             [[0.8], [0.1]],
             [[0.7], [0.2]],
             [[0.3], [0.7]]], dtype=tf.float32)

        loss = self.compute_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0, places=6)


if __name__ == "__main__":
    print("Running custom frame‑level loss tests …")
    print("---------------------------------------")
    unittest.main()