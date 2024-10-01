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
import os
import xir
import simple_graph
import process_json

error_counter = []


def main():
    json_file_name = sys.argv[1]
    process_json.process(run, json_file_name, json_file_name)
    sys.exit(1 if error_counter else 0)


def run(name, v):
    if v['meta'].get('skip', False) is True:
        return
    xmodel_file_name = os.path.join(
        v['meta']['path'], v['meta']['xmodel'])
    root = os.environ.get(
        'MODEL_ZOO_ROOT', '/scratch/models/xilinx_model_zoo_u50_1.3.0_amd64')
    path = os.path.join(root, "." + v["meta"]["path"])
    xmodel = v["meta"]["xmodel"]
    g = xir.Graph.deserialize(os.path.join(path, xmodel))
    batch = 1
    for (tensor_name, tensor_ref_file) in v["meta"]["init_tensors"].items():
        simple_graph.set_reference_input(g, tensor_name, tensor_ref_file)
        batch = len(tensor_ref_file)
    simple_graph.add_check_point(g, v["meta"]["dump_tensors_ref"], batch)
    simple_graph.normal_setting_for_graph(g)
    print("RUNNING %s : %s" % (name,  xmodel_file_name))
    errors = simple_graph.run_graph(g)
    for k in errors:
        for (b, actual_md5sum, expected_md5sum) in errors[k]:
            b = b % len(v['meta']['dump_tensors_ref'][k])
            v['meta']['dump_tensors_ref'][k][b]['md5sum_graph'] = actual_md5sum
    v['meta']['pass'] = True if not errors else False
    result = "SUCCESS" if v['meta']['pass'] is True else "FAIL"
    print("DONE(%s) %s : %s" % (result, name,  xmodel_file_name))
    if not v['meta']['pass'] is True:
        error_counter.append(name)
        v['meta']['reason'] = v['meta'].get(
            'reason', 'autoregressuion test failed.')
        v['meta']['skip'] = True


if __name__ == '__main__':
    main()
