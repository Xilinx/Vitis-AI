#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import gzip as compress
import os
import sys
def main():
    cache_file = os.path.realpath(sys.argv[1])
    dir = sys.argv[2]
    with compress.open(cache_file, 'rb') as f:
        for (k, v) in pickle.load(f).items():
            file = os.path.join(dir, k[0:2], k[2:])
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "wb") as s:
                s.write(v)


if __name__ == '__main__':
    main()
