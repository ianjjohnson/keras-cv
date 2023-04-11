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

import copy

import pytest
import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv.models.object_detection.yolo_v8.yolo_v8_detector_presets import (
    yolo_v8_detector_presets,
)


class YOLOV8DetectorTest(tf.test.TestCase):
    def test_serialization(self):
        model = keras_cv.models.YOLOV8Detector(
            num_classes=20,
            bounding_box_format="xywh",
            fpn_depth=1,
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                "yolov8_n_backbone"
            ),
        )
        serialized_1 = keras.utils.serialize_keras_object(model)
        restored = keras.utils.deserialize_keras_object(
            copy.deepcopy(serialized_1)
        )
        serialized_2 = keras.utils.serialize_keras_object(restored)
        self.assertEqual(serialized_1, serialized_2)


@pytest.mark.large
class YOLOV8DetectorSmokeTest(tf.test.TestCase):
    def test_preset_with_forward_pass(self):
        model = keras_cv.models.YOLOV8Detector.from_preset(
            "yolov8_n_coco",
            bounding_box_format="xywh",
        )

        image = tf.ones((1, 512, 512, 3))
        encoded_predictions = model(image)

        self.assertAllClose(
            encoded_predictions["boxes"][0, 0:5, 0],
            [7.677088, 9.361263, 9.6241255, 9.454895, 8.99809],
        )
        self.assertAllClose(
            encoded_predictions["classes"][0, 0:5, 0],
            [
                1.6198268e-05,
                2.1332640e-05,
                1.5359996e-05,
                1.4784595e-05,
                1.2798749e-05,
            ],
        )


@pytest.mark.extra_large
class YOLOV8DetectorPresetFullTest(tf.test.TestCase):
    """
    Test the full enumeration of our presets.
    This every presets for YOLOV8Detector and is only run manually.
    Run with:
    `pytest keras_cv/models/object_detection/yolo_v8/yolo_v8_detector_test.py --run_extra_large`
    """  # noqa: E501

    def test_load_yolov8_detector(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in yolo_v8_detector_presets:
            model = keras_cv.models.YOLOV8Detector.from_preset(
                preset, bounding_box_format="xywh"
            )
            model(input_data)