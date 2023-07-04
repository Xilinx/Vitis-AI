
# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import ctypes

libvart_util = ctypes.CDLL('libvart-trace.so')
f_tracepoint = libvart_util.tracepoint_py_func
f_tracepoint.argtypes = [ctypes.c_int, ctypes.c_char_p]
f_tracepoint.restype = None


def vai_tracepoint(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        fname = func.__name__
        fname_c_char_p = ctypes.c_char_p(bytes(fname, 'ascii'))

        f_tracepoint(1, fname_c_char_p)
        result = func(*args, **kw)
        f_tracepoint(0, fname_c_char_p)

        return result
    return wrapper
