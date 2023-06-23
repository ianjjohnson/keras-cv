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
from keras_cv.backend.config import multi_backend

if multi_backend():
    import keras_core  # noqa: F403, F401
    from keras_core.backend import convert_to_numpy  # noqa: F403, F401
    from keras_core.backend import vectorized_map  # noqa: F403, F401
    from keras_core.ops import *  # noqa: F403, F401
    from keras_core.ops import arange as keras_core_arange  # noqa
    from keras_core.utils.image_utils import smart_resize  # noqa: F403, F401

    def arange(start, stop=None, step=1, dtype=None):
        if keras_core.backend.backend() == "tensorflow":
            # tfnp doesn't allow start, stop, and step to be symbolic
            # tensors. So, directly call `tf.range` instead.
            import tensorflow as tf  # noqa: F403, F401

            return tf.range(start, stop, step, dtype=dtype)
        return keras_core_arange(start, stop, step, dtype=dtype)

else:
    from keras_cv.backend.tf_ops import *  # noqa: F403, F401
