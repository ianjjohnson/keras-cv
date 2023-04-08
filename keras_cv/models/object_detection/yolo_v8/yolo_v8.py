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

import tensorflow as tf
from keras import layers
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.object_detection import predict_utils
from keras_cv.models.object_detection.__internal__ import unpack_input
from keras_cv.models.object_detection.yolo_v8.compat_anchor_generation import (
    get_anchors,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_label_encoder import (
    YOLOV8LabelEncoder,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import conv_bn
from keras_cv.models.object_detection.yolo_v8.yolo_v8_layers import (
    csp_with_2_conv,
)
from keras_cv.models.object_detection.yolo_v8.yolo_v8_presets import (
    yolo_v8_presets,
)
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.train import get_feature_extractor

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]


def path_aggregation_fpn(features, depth=3, name=None):
    # yolov8
    # 9: p5 1024 ---+----------------------+-> 21: out2 1024
    #               v [up 1024 -> concat]  ^ [down 512 -> concat]
    # 6: p4 512 --> 12: p4p5 512 --------> 18: out1 512
    #               v [up 512 -> concat]   ^ [down 256 -> concat]
    # 4: p3 256 --> 15: p3p4p5 256 --------+--> 15: out0 128
    # features: [p3, p4, p5]
    channel_axis = -1
    upsamples = [features[-1]]
    # upsamples: [p5], features[:-1][::-1]: [p4, p3] -> [p5, p4p5, p3p4p5]
    for id, feature in enumerate(features[:-1][::-1]):
        size = tf.shape(feature)[1:-1]
        nn = tf.image.resize(upsamples[-1], size, method="nearest")
        nn = tf.concat([nn, feature], axis=channel_axis)

        out_channel = feature.shape[channel_axis]
        nn = csp_with_2_conv(
            nn,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=f"{name}_p{len(features) + 1 - id}",
        )
        upsamples.append(nn)

    downsamples = [upsamples[-1]]
    # downsamples: [p3p4p5], upsamples[:-1][::-1]: [p4p5, p5] -> [p3p4p5, p3p4p5 + p4p5, p3p4p5 + p4p5 + p5]
    for id, ii in enumerate(upsamples[:-1][::-1]):
        cur_name = f"{name}_c3n{id + 3}"
        nn = conv_bn(
            downsamples[-1],
            downsamples[-1].shape[channel_axis],
            kernel_size=3,
            strides=2,
            activation="swish",
            name=f"{cur_name}_down",
        )
        nn = tf.concat([nn, ii], axis=channel_axis)

        out_channel = ii.shape[channel_axis]
        nn = csp_with_2_conv(
            nn,
            channels=out_channel,
            depth=depth,
            shortcut=False,
            activation="swish",
            name=cur_name,
        )
        downsamples.append(nn)
    return downsamples


def yolov8_head(
    inputs,
    num_classes=80,
    bbox_len=64,
    name="yolov8_head",
):
    outputs = []
    reg_channels = max(16, bbox_len, inputs[0].shape[-1] // 4)
    cls_channels = max(num_classes, inputs[0].shape[-1])
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"

        reg_nn = conv_bn(
            feature,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_1",
        )
        reg_nn = conv_bn(
            reg_nn,
            reg_channels,
            3,
            activation="swish",
            name=f"{cur_name}_reg_2",
        )
        reg_out = layers.Conv2D(
            filters=bbox_len,
            kernel_size=1,
            name=f"{cur_name}_reg_3_conv",
        )(reg_nn)

        cls_nn = conv_bn(
            feature,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_1",
        )
        cls_nn = conv_bn(
            cls_nn,
            cls_channels,
            3,
            activation="swish",
            name=f"{cur_name}_cls_2",
        )
        cls_out = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_cls_3_conv",
        )(cls_nn)
        cls_out = layers.Activation("sigmoid", name=f"{cur_name}_classifier")(
            cls_out
        )

        out = tf.concat([reg_out, cls_out], axis=-1)
        out = layers.Reshape(
            [-1, out.shape[-1]], name=f"{cur_name}_output_reshape"
        )(out)
        outputs.append(out)

    outputs = tf.concat(outputs, axis=1)
    return outputs


def decode_regression_to_boxes(preds, regression_max=16):
    preds_bbox = tf.reshape(preds, (-1, preds.shape[1], 4, regression_max))
    preds_bbox = tf.nn.softmax(preds_bbox, axis=-1) * tf.range(
        regression_max, dtype="float32"
    )
    return tf.reduce_sum(preds_bbox, axis=-1)


def decode_boxes(preds, anchors):
    # Boxes expected to be in encoded format
    preds_top_left, preds_bottom_right = tf.split(preds, [2, 2], axis=-1)

    # Converts rel_yxyx anchors to rel_center_yxhw
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    pred_sum = preds_bottom_right + preds_top_left
    pred_hw_half = (preds_bottom_right - preds_top_left) / 2

    bboxes_center = pred_hw_half * anchors_hw + anchors_center
    bboxes_hw = pred_sum * anchors_hw

    # Preds in rel_yxyx
    preds_top_left = bboxes_center - 0.5 * bboxes_hw
    pred_bottom_right = preds_top_left + bboxes_hw

    # Returns results in rel_yxyx
    return tf.concat([preds_top_left, pred_bottom_right], axis=-1)


@keras.utils.register_keras_serializable(package="keras_cv")
class YOLOV8(Task):
    def __init__(
        self,
        bounding_box_format,
        backbone,
        fpn_depth,
        num_classes,
        anchor_generator=None,
        label_encoder=None,
        prediction_decoder=None,
        **kwargs,
    ):
        extractor_levels = [2, 3, 4]
        if 5 in backbone.pyramid_level_inputs.keys():
            extractor_levels.append(5)
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )

        images = layers.Input(feature_extractor.input_shape[1:])
        features = list(feature_extractor(images).values())

        # Apply the FPN
        fpn_features = path_aggregation_fpn(
            features, depth=fpn_depth, name="pa_fpn"
        )

        outputs = yolov8_head(
            fpn_features,
            num_classes,
            64,  # bbox_len
        )
        outputs = layers.Activation(
            "linear", dtype="float32", name="outputs_fp32"
        )(outputs)
        boxes, scores = outputs[:, :, :64], outputs[:, :, 64:]

        # Hack to make metrics pretty.
        classes = keras.layers.Concatenate(axis=1, name="class")([scores])
        boxes = keras.layers.Concatenate(axis=1, name="box")([boxes])

        outputs = {"boxes": boxes, "classes": classes}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        if anchor_generator is None:
            # Anchors for pre-trained
            # Gross hack for an anchor generator (for now)
            anchor_generator = lambda image_shape: {0: get_anchors(image_shape)}
            # This anchor generator generates rel_yxyx anchors
            anchor_generator.bounding_box_format = "rel_yxyx"
            self.anchor_generator = anchor_generator
        else:
            self.anchor_generator = anchor_generator

        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            bounding_box_format=bounding_box_format,
            anchor_generator=self.anchor_generator,
        )
        self.bounding_box_format = bounding_box_format
        self.prediction_decoder = (
            prediction_decoder
            or keras_cv.layers.MultiClassNonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                confidence_threshold=0.3,
                iou_threshold=0.5,
            )
        )
        self.backbone = backbone
        self.num_classes = num_classes

    def compile(
        self,
        box_loss=None,
        classification_loss=None,
        metrics=None,
        **kwargs,
    ):
        self.box_loss = box_loss or keras_cv.losses.SmoothL1Loss(
            l1_cutoff=1.0, reduction=keras.losses.Reduction.SUM
        )
        self.classification_loss = (
            classification_loss
            or keras_cv.losses.FocalLoss(
                from_logits=False, reduction=keras.losses.Reduction.SUM
            )
        )

        losses = {
            "boxes": self.box_loss,
            "classes": self.classification_loss,
        }

        self._has_user_metrics = metrics is not None and len(metrics) != 0
        self._user_metrics = metrics
        super().compile(loss=losses, **kwargs)

    def train_step(self, data):
        x, y = unpack_input(data)

        # Boxes are now delta-encoded in center_yxhw
        boxes, classes = self.label_encoder(x, y)

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            box_pred, cls_pred = outputs["boxes"], outputs["classes"]
            total_loss = self.compute_loss(
                x, box_pred, cls_pred, boxes, classes
            )

        # Training specific code
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if not self._has_user_metrics:
            return super().compute_metrics(x, {}, {}, sample_weight={})

        y_pred = self.decode_predictions(outputs, x)
        return self.compute_metrics(x, y, y_pred, sample_weight=None)

    def test_step(self, data):
        x, y = unpack_input(data)
        boxes, classes = self.label_encoder(x, y)
        boxes = bounding_box.convert_format(
            boxes,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )

        outputs = self(x, training=False)
        box_pred, cls_pred = outputs["boxes"], outputs["classes"]
        _ = self.compute_loss(x, box_pred, cls_pred, boxes, classes)

        if not self._has_user_metrics:
            return super().compute_metrics(x, {}, {}, sample_weight={})
        y_pred = self.decode_predictions(outputs, x)
        return self.compute_metrics(x, y, y_pred, sample_weight=None)

    def compute_loss(self, x, box_pred, cls_pred, boxes, classes):
        cls_labels = tf.one_hot(
            tf.cast(classes, dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32,
        )

        box_pred = decode_regression_to_boxes(box_pred, 64 // 4)

        positive_mask = tf.cast(tf.greater(classes, -1.0), dtype=tf.float32)
        normalizer = tf.maximum(tf.reduce_sum(positive_mask), 1)
        cls_weights = tf.cast(
            tf.math.not_equal(classes, -2.0), dtype=tf.float32
        )
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer

        y_true = {
            "boxes": boxes,
            "classes": cls_labels,
        }
        y_pred = {
            "boxes": box_pred,
            "classes": cls_pred,
        }
        sample_weights = {
            "boxes": box_weights,
            "classes": cls_weights,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def decode_predictions(
        self,
        pred,
        images,
    ):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes, 64 // 4)

        anchors = self.anchor_generator(image_shape=(640, 640, 3))
        anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
        anchors = bounding_box.convert_format(
            anchors,
            source=self.anchor_generator.bounding_box_format,
            target="rel_yxyx",
            images=images[0],
        )

        decoded_boxes = decode_boxes(boxes, anchors)
        decoded_boxes = bounding_box.convert_format(
            decoded_boxes,
            source="rel_yxyx",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(decoded_boxes, scores)

    def make_predict_function(self, force=False):
        return predict_utils.make_predict_function(self, force=force)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **yolo_v8_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **yolo_v8_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible backbones."""
        return copy.deepcopy(backbone_presets)
