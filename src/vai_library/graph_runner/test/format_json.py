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


import sys
import json


def main():
    json_file_name = sys.argv[1]
    print("read from " + sys.argv[1])
    with open(json_file_name) as json_file:
        meta_data = json.load(json_file)
    print("write to " + sys.argv[1])
    with open(sys.argv[1], "w") as f:
        f.write(json.dumps(meta_data, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
