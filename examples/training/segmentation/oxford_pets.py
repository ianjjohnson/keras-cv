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
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from matplotlib import pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from keras_cv import models

flags.DEFINE_string(
    "backup_path", None, "Directory which will be used for training backups."
)
flags.DEFINE_string(
    "weights_path", None, "Directory which will be used to store weight checkpoints."
)
flags.DEFINE_string(
    "tensorboard_path", None, "Directory which will be used to store tensorboard logs."
)
flags.DEFINE_integer("batch_size", 64, "Batch size for training and evaluation.")
flags.DEFINE_boolean(
    "use_xla", True, "Whether or not to use XLA (jit_compile) for training."
)
flags.DEFINE_float(
    "initial_learning_rate",
    0.0005,
    "Initial learning rate which will reduce on plateau.",
)


FLAGS = flags.FLAGS
FLAGS(sys.argv)

CLASSES = 2
IMAGE_SIZE = (128, 128)
EPOCHS = 20


@tf.function
def preprocess(record):
    resizing = layers.Resizing(
        width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], crop_to_aspect_ratio=False
    )

    image = record["image"]
    mask = record["segmentation_mask"]

    # Offset mask so that 0 is no pet, 1 is pet (instead of using 1 and 2)
    mask -= 1
    mask = tf.one_hot(tf.squeeze(mask, axis=-1), CLASSES)

    image = resizing(image)
    mask = resizing(mask)

    return image, mask


dataset = tfds.load("oxford_iiit_pet:3.*.*")

train_ds = (
    dataset["train"]
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FLAGS.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    dataset["test"]
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FLAGS.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# TODO(ianstenbit): Add data augmentation once supported by KerasCV layers

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = models.UNet(
        include_rescaling=True,
        block_widths=[32, 64, 96],
        output_channels=CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        output_activation="softmax",
    )

optimizer = optimizers.Adam(
    learning_rate=FLAGS.initial_learning_rate, global_clipnorm=10
)
loss_fn = losses.CategoricalCrossentropy()

with strategy.scope():
    training_metrics = [metrics.CategoricalAccuracy()]

callbacks = [
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, min_delta=0.001, min_lr=0.0001
    ),
    callbacks.EarlyStopping(patience=20),
    callbacks.BackupAndRestore(FLAGS.backup_path),
    callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
]

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=training_metrics,
    jit_compile=FLAGS.use_xla,
)

model.fit(
    train_ds,
    batch_size=FLAGS.batch_size,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=test_ds,
)


def visualize_predictions(dataset, title):
    plt.figure(figsize=(10, 10)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images, masks = samples
        plt.subplot(9, 3, 3 * i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
        plt.subplot(9, 3, 3 * i + 2)
        plt.imshow(tf.math.argmax(masks[0], axis=-1).numpy().astype("uint8"))
        plt.axis("off")
        plt.subplot(9, 3, 3 * i + 3)
        plt.imshow(
            tf.math.argmax(model(tf.expand_dims(images[0], axis=0)), axis=-1)
            .numpy()[0]
            .astype("uint8")
        )
        plt.axis("off")
    plt.show()


visualize_predictions(train_ds, "Example predictions")
