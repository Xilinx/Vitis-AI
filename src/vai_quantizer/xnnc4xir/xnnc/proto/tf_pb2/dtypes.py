"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

TF_TO_NP = {
    "DT_HALF": np.float16,
    "DT_FLOAT": np.float32,
    "DT_DOUBLE": np.float64,
    "DT_INT32": np.int32,
    "DT_UINT8": np.uint8,
    "DT_UINT16": np.uint16,
    "DT_UINT32": np.uint32,
    "DT_UINT64": np.uint64,
    "DT_INT16": np.int16,
    "DT_INT8": np.int8,
    # NOTE(touts): For strings we use np.object as it supports variable length
    # strings.
    "DT_STRING": np.object,
    "DT_COMPLEX64": np.complex64,
    "DT_COMPLEX128": np.complex128,
    "DT_INT64": np.int64,
    "DT_BOOL": np.bool,
    # Ref types
    "DT_HALF_REF": np.float16,
    "DT_FLOAT_REF": np.float32,
    "DT_DOUBLE_REF": np.float64,
    "DT_INT32_REF": np.int32,
    "DT_UINT32_REF": np.uint32,
    "DT_UINT8_REF": np.uint8,
    "DT_UINT16_REF": np.uint16,
    "DT_INT16_REF": np.int16,
    "DT_INT8_REF": np.int8,
    "DT_STRING_REF": np.object,
    "DT_COMPLEX64_REF": np.complex64,
    "DT_COMPLEX128_REF": np.complex128,
    "DT_INT64_REF": np.int64,
    "DT_UINT64_REF": np.uint64,
    "DT_BOOL_REF": np.bool,
}
