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

"""UNet models for KerasCV.

Reference:
  - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
"""

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def sinusoidal_embedding(
    x, embedding_min_frequency=1.0, embedding_max_frequency=1000.0, embedding_dims=32
):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, block_scale_factor):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=block_scale_factor)(x)
        return x

    return apply


def UpBlock(width, block_depth, block_scale_factor, include_skip_connections):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=block_scale_factor, interpolation="bilinear")(x)
        for _ in range(block_depth):
            if include_skip_connections:
                x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def UNet(
    include_rescaling,
    block_widths=None,
    down_block_widths=None,
    up_block_widths=None,
    block_depth=2,
    bottleneck_width=None,
    bottleneck_depth=None,
    block_scale_factor=2,
    input_shape=(224, 224, 3),
    output_channels=3,
    include_skip_connections=False,
    include_input_noise=False,
    weights=None,
    name="Unet",
    output_activation=None,
):

    if include_skip_connections and block_widths is None:
        raise ValueError(
            "`include_skip_connections` can only be used with a symmetrical UNet using `block_widths`."
        )

    if block_widths is None and down_block_widths is None and up_block_widths is None:
        raise ValueError(
            "Either `block_widths` or one of `down_block_widths` and `up_block_widths` must be specified."
        )

    if block_widths is None and bottleneck_width is None:
        raise ValueError(
            "`bottleneck_width` must be specified when using `down_block_widths` and/or `up_block_widths`"
        )

    if down_block_widths and up_block_widths and bottleneck_width is None:
        raise ValueError(
            "`bottleneck_width` must be specified when `up_block_widths` and `down_block_widths` are specified."
        )

    if block_widths:
        down_block_widths = block_widths[:-1]
        bottleneck_width = block_widths[-1]
        up_block_widths = reversed(block_widths[:-1])

    if bottleneck_depth is None:
        bottleneck_depth = block_depth

    inputs = layers.Input(input_shape)
    x = inputs

    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    if include_input_noise:
        noise_variances = keras.Input(shape=(1, 1, 1))

        noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
        noise_embedding = layers.UpSampling2D(
            size=input_shape[:2], interpolation="nearest"
        )(noise_embedding)

        x = layers.Conv2D(down_block_widths[0], kernel_size=1)(x)
        x = layers.Concatenate()([x, noise_embedding])

    skip_connections = []
    if down_block_widths:
        for width in down_block_widths:
            x = DownBlock(width, block_depth, block_scale_factor)([x, skip_connections])

    for _ in range(bottleneck_depth):
        x = ResidualBlock(bottleneck_width)(x)

    if up_block_widths:
        for width in up_block_widths:
            x = UpBlock(
                width, block_depth, block_scale_factor, include_skip_connections
            )([x, skip_connections])

    x = layers.Conv2D(
        output_channels,
        kernel_size=1,
        kernel_initializer="zeros",
        activation=output_activation,
    )(x)

    if include_input_noise:
        model = keras.Model([inputs, noise_variances], x, name=name)
    else:
        model = keras.Model(inputs, x, name=name)

    if weights is not None:
        model.load_weights(weights)

    return model
