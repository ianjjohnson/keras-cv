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
    from keras_core.backend import vectorized_map  # noqa: F403, F401
    from keras_core.operations import *  # noqa: F403, F401
else:
    from keras_core.backend.tensorflow import *  # noqa: F403, F401
    from keras_core.backend.tensorflow.core import *  # noqa: F403, F401
    from keras_core.backend.tensorflow.math import *  # noqa: F403, F401
    from keras_core.backend.tensorflow.nn import *  # noqa: F403, F401
    from keras_core.backend.tensorflow.numpy import *  # noqa: F403, F401

    # Some TF APIs where the numpy API doesn't support raggeds that we need
    from tensorflow import concat as concatenate  # noqa: F403, F401
    from tensorflow import reduce_max as max  # noqa: F403, F401
    from tensorflow import reshape  # noqa: F403, F401
    from tensorflow import split  # noqa: F403, F401
