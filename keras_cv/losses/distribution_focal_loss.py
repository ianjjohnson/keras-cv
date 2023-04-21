# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_cv")
class DistributionFocalLoss(keras.losses.Loss):
    """Implements Distribution Focal loss

    Distribution Focal loss is a modified cross-entropy designed to perform well
    with object detectors which make left / top / right / bottom predictions
    with respect to an anchor point (e.g. YOLOV8).

    Args:
        alpha: a float value between 0 and 1 representing a weighting factor
            used to deal with class imbalance. Positive classes and negative
            classes have alpha and (1 - alpha) as their weighting factors
            respectively. Defaults to 0.25.
        gamma: a positive float value representing the tunable focusing
            parameter, defaults to 2.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, `y_pred` is assumed to encode a probability distribution.
            Default to `False`.
        label_smoothing: Float in `[0, 1]`. If higher than 0 then smooth the
            labels by squeezing them towards `0.5`, i.e., using
            `1. - 0.5 * label_smoothing` for the target class and
            `0.5 * label_smoothing` for the non-target class.

    References:
        - [Generalized Focal Loss paper](https://arxiv.org/pdf/2006.04388v1.pdf)

    Usage with the `compile()` API:
    ```python
    model.compile(loss=keras_cv.losses.DistributionFocalLoss())
    ```
    """

    def call(self, y_true, y_pred):
        """Computes the DistributionFocalLoss for a batch of boxes.

        For object detection, predictions and targets should be in left / top /
        right / bottom format.
        """
        target_left = tf.cast(y_true, tf.float32)  # target left
        target_right = target_left + 1  # target right
        weight_left = target_right - y_true  # weight left
        weight_right = 1 - weight_left  # weight right

        left_loss = tf.reshape(
            keras.losses.sparse_categorical_crossentropy(
                tf.reshape(target_left, (-1,)), y_pred
            )
            * weight_left,
            y_true.shape,
        )
        right_loss = tf.reshape(
            keras.losses.sparse_categorical_crossentropy(
                tf.reshape(target_right, (-1,)), y_pred
            )
            * weight_right,
            y_true.shape,
        )

        return tf.reduce_mean(left_loss + right_loss, axis=-1, keepdims=True)
