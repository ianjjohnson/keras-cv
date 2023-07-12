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

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models import DeepLabV3Plus
from keras_cv.models import ResNet18V2Backbone
from keras_cv.tests.test_case import TestCase


class DeepLabV3PlusTest(TestCase):
    def test_deeplab_v3_plus_construction(self):
        backbone = ResNet18V2Backbone(input_shape=[512, 512, 3])
        model = DeepLabV3Plus(backbone=backbone, num_classes=1)
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    @pytest.mark.large
    def test_deeplab_v3_plus_call(self):
        backbone = ResNet18V2Backbone(input_shape=[512, 512, 3])
        model = DeepLabV3Plus(backbone=backbone, num_classes=1)
        images = np.random.uniform(size=(2, 512, 512, 3))
        _ = model(images)
        _ = model.predict(images)

    def test_trainable_variable_count(self):
        target_size = [512, 512]
        images = np.ones(shape=[1] + target_size + [3])

        backbone = ResNet18V2Backbone(input_shape=target_size + [3])
        model = DeepLabV3Plus(backbone=backbone, num_classes=1)

        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        outputs = model(images)

        self.assertEqual(len(model.trainable_variables), 83)
        # Output shape
        self.assertEqual(outputs.shape, tuple([1] + target_size + [1]))

    @pytest.mark.large
    def test_weights_change(self):
        target_size = [512, 512, 3]

        images = np.ones([1] + target_size)
        labels = np.zeros([1] + target_size)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.repeat(2)
        ds = ds.batch(2)

        backbone = ResNet18V2Backbone(input_shape=target_size)
        model = DeepLabV3Plus(backbone=backbone, num_classes=3)

        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        original_weights = model.get_weights()
        model.fit(ds, epochs=1)
        updated_weights = model.get_weights()

        for w1, w2 in zip(original_weights, updated_weights):
            self.assertNotAllClose(w1, w2)
            self.assertFalse(ops.any(ops.isnan(w2)))

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        target_size = [512, 512, 3]

        backbone = ResNet18V2Backbone(input_shape=target_size)
        model = DeepLabV3Plus(backbone=backbone, num_classes=1)

        input_batch = np.ones(shape=[2] + target_size)
        model_output = model(input_batch)

        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DeepLabV3Plus)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)
