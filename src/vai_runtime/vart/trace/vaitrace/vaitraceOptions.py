
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


def merge(a: dict, b: dict):
    if hasattr(a, "keys") and hasattr(b, "keys"):
        for kb in b.keys():
            if kb in a.keys():
                merge(a[kb], b[kb])
            else:
                a.update(b)


def get(opt, key: str):
    t = opt.copy()
    for k in key.split('.'):
        t = t.get(k, {})
    return t


def set(opt, key: str, value):
    for k in key.split('.')[0:-1]:
        if not k in opt.keys():
            opt[k] = {}
        opt = opt[k]
        print(key.index(k), "### key:", k, " v:", opt)

    opt.update({key[-1]: value})
