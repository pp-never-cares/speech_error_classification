"""
custom_frame_level_loss.py

Custom frame-level loss function for the model training process.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



@tf.keras.utils.register_keras_serializable()
class CustomFrameLevelLoss(tf.keras.losses.Loss):
    """
    Custom frame-level loss function that only considers frames within event-containing utterances.

    Methods:
    - __init__: Initializes the loss function.
    - call: Computes the frame-level loss.
    - get_config: Returns the configuration of the loss function.
    - from_config: Creates a loss function from the configuration
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.SUM, name='custom_frame_level_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred_frame: tf.Tensor) -> tf.Tensor:
        """
        Custom frame-level loss function that only considers frames within event-containing utterances.

        Args:
        - y_true (tf.Tensor): The true labels.
        - y_pred_frame (tf.Tensor): The predicted labels at the frame level.

        Returns:
        - loss (tf.Tensor): The computed frame-level loss.
        """
        # small_loss = tf.constant(1e-7, dtype=tf.float32)
        # epsilon = K.epsilon()

        # Compute y_true_utt to check if an utterance contains an event or not
        y_true_utt = tf.cast(tf.reduce_any(
            tf.equal(y_true, 1), axis=-1), tf.int32)

        # Create a mask for the utterances where y_true_utt is 1 (indicating event presence)
        mask = tf.equal(y_true_utt, 1)
        mask_expanded = tf.expand_dims(mask, axis=-1)

        # Mask the y_true and y_pred_frame to exclude non-event utterances
        y_true_masked = tf.boolean_mask(y_true, mask_expanded)
        y_pred_frame_masked = tf.boolean_mask(y_pred_frame, mask_expanded)

        # Compute the loss
        tf.debugging.assert_all_finite(y_pred_frame_masked, "pred before BC not finite")
        # loss = tf.cond(
        #     tf.equal(tf.size(y_true_masked), 0),
        #     lambda: tf.constant(0.0, dtype=tf.float32),
        #     lambda: tf.reduce_mean(
        #         keras.losses.binary_crossentropy(
        #             y_true_masked, y_pred_frame_masked)
        #     )
        # )
        loss = tf.cond(
            tf.equal(tf.size(y_true_masked), 0),
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    y_true_masked, tf.clip_by_value(
                                                 y_pred_frame_masked,
                                                 1e-7, 1. - 1e-7))
            )
        )
        tf.debugging.assert_all_finite(loss, "loss not finite")
        return loss

    def get_config(self):
        base_config = super().get_config()
        return base_config
