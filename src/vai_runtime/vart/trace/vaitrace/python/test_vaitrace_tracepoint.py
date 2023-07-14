
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

from vaitrace_py import vai_tracepoint
import os

"""
export VAI_TRACE_ENABLE=1
export VAI_TRACE_TS=XRT
"""
os.environ.setdefault("VAI_TRACE_ENABLE", "1")
os.environ.setdefault("VAI_TRACE_TS", "XRT")
os.environ.setdefault("VAI_TRACE_DIR", "./")


@vai_tracepoint
def hello_vaitrace(a, b):
    return a+b


@vai_tracepoint
def okey_dokey(a, b):
    return a*b


for i in range(100):
    if i % 2 == 0:
        hello_vaitrace(i*5, 555)
    else:
        okey_dokey(i, i*3)
