# Copyright 2022 The KerasCV Authors
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
from absl.testing import parameterized
from keras import losses

from keras_cv.models import unet


class UNetTest(tf.test.TestCase, parameterized.TestCase):
    def test_diffusion_unet(self):
        diffusion_model = unet.UNet(
            include_rescaling=True,
            block_widths=[32, 64, 96],
            include_skip_connections=True,
            include_input_noise=True,
            input_shape=(64, 64, 3),
        )
        diffusion_model.compile(loss=losses.MeanAbsoluteError())

        images = tf.random.uniform((5, 64, 64, 3))
        noise = tf.random.uniform((5, 1, 1, 1))
        targets = tf.random.uniform((5, 64, 64, 3))

        diffusion_model.fit([images, noise], targets)

    def test_superresolution_unet(self):
        superresolution_model = unet.UNet(
            include_rescaling=True,
            bottleneck_depth=8,
            bottleneck_width=64,
            up_block_widths=[64, 64],
            input_shape=(16, 16, 3),
        )
        superresolution_model.compile(loss=losses.MeanAbsoluteError())

        images = tf.random.uniform((5, 16, 16, 3))
        targets = tf.random.uniform((5, 64, 64, 3))

        superresolution_model.fit(images, targets)

    def test_segmentation_unet(self):
        classes = 10

        segmentation_model = unet.UNet(
            include_rescaling=True,
            block_widths=[32, 64, 96],
            output_channels=classes,
            input_shape=(64, 64, 3),
            output_activation="softmax",
        )
        segmentation_model.compile(loss=losses.CategoricalCrossentropy())

        images = tf.random.uniform((5, 64, 64, 3))
        targets = tf.random.uniform((5, 64, 64, 10))

        segmentation_model.fit(images, targets)

    def test_block_widths_precludes_other_block_configuration(self):
        with self.assertRaises(ValueError):
            _ = unet.UNet(
                include_rescaling=True,
                block_widths=[32, 64],
                up_block_widths=[64, 64],
            )

        with self.assertRaises(ValueError):
            _ = unet.UNet(
                include_rescaling=True,
                block_widths=[32, 64],
                down_block_widths=[64, 64],
            )

        with self.assertRaises(ValueError):
            _ = unet.UNet(
                include_rescaling=True,
                block_widths=[32, 64],
                bottleneck_width=64,
            )

    def test_skip_connections_supported_iff_using_block_widths(self):
        with self.assertRaises(ValueError):
            _ = unet.UNet(
                include_rescaling=True,
                bottleneck_depth=8,
                bottleneck_width=64,
                up_block_widths=[64, 64],
                include_skip_connections=True,
            )

        _ = unet.UNet(
            include_rescaling=True, block_widths=[32, 64], include_skip_connections=True
        )

    def test_bottleneck_width_required_for_custom_up_and_down_blocks(self):
        with self.assertRaises(ValueError):
            _ = unet.UNet(
                include_rescaling=True,
                up_block_widths=[64, 64],
            )

        _ = unet.UNet(
            include_rescaling=True,
            bottleneck_width=64,
            up_block_widths=[64, 64],
        )


if __name__ == "__main__":
    tf.test.main()
