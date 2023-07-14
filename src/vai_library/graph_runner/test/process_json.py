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
import os


def load_json(json_file_name):
    with open(json_file_name) as json_file:
        try:
            meta_data = json.load(json_file)
            return meta_data
        except json.decoder.JSONDecodeError as err:
            print("cannot load json from " +
                  json_file_name + " error:" + str(err))


def save_json(json_obj, filename):
    with open(filename, "w") as f:
        f.write(json.dumps(json_obj, indent=4, sort_keys=True))


def process(process, input_json_file, output_json_file):
    json = load_json(input_json_file)
    model = os.environ.get('MODEL', None)
    all_models = []
    if model:
        all_models = [model]
        json[model]['meta']['skip'] = False
    else:
        all_models = [k for k in json]
        from_model = os.environ.get('FROM_MODEL', None)
        if from_model:
            all_models = [k for k in all_models if k >= from_model]
    for k in all_models:
        # print("========= start to processing %s ....." % (k,))
        process(k, json[k])
        # print("========= processing: %s pass= %s" %
        # (k, json[k]['meta']['pass']))
    save_json(json, output_json_file)
