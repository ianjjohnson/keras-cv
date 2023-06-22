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

from keras_cv import backend
from keras_cv.backend import ops
from keras_cv.backend import tf_ops

_ORIGINAL_OPS = copy.copy(backend.ops.__dict__)
_ORIGINAL_SUPPORTS_RAGGED = backend.supports_ragged


class TFDataScope:
    def __enter__(self):
        for k, v in ops.__dict__.items():
            if k in tf_ops.__dict__:
                setattr(ops, k, getattr(tf_ops, k))
        backend.supports_ragged = lambda: True

    def __exit__(self, exc_type, exc_value, exc_tb):
        for k, v in ops.__dict__.items():
            setattr(ops, k, _ORIGINAL_OPS[k])
        backend.supports_ragged = _ORIGINAL_SUPPORTS_RAGGED
