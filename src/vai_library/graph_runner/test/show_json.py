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

import json
import sys


def main():
    json_file_name = sys.argv[1]
    c = 0
    with open(json_file_name) as json_file:
        meta_data = json.load(json_file)
        for (k, v) in meta_data.items():
            if not v['meta']['pass'] is True:
                reason = v['meta'].get("reason", "unknown")
                print(str(c) + "." + k + " fail because " + reason)
                c = c + 1
            else:
                pass  # print(k)


if __name__ == '__main__':
    main()
