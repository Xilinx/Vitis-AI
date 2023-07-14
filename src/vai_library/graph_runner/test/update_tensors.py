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
import process_json
import xir_extra_ops

error_counter = []


def main():
    json_file_name = sys.argv[1]
    process_json.process(run, json_file_name, json_file_name)
    sys.exit(1 if error_counter else 0)


def run(name, v):
    print("CHECKING %s" % (name,))
    root = os.environ.get(
        'MODEL_ZOO_ROOT', '/scratch/models/xilinx_model_zoo_u50_1.3.0_amd64')
    path = os.path.join(root, "." + v["meta"]["path"])
    xmodel = v["meta"]["xmodel"]
    realpath = os.path.join(path, xmodel)
    if not os.path.isfile(realpath):
        v['meta']['skip'] = True
        return
    g = xir.Graph.deserialize(realpath)
    batch = 1
    for (tensor_name, tensor_ref_file) in v["meta"]["init_tensors"].items():
        batch = len(tensor_ref_file)
    add_check_point(g, v["meta"]["dump_tensors_ref"], batch)


def default_file_entries(tensor, batch):
    return [{
        "file": xir_extra_ops.remove_xfix(tensor.name) + ".bin",
        "md5sum": '0'*32,
        "size": tensor.get_data_size()
    } for i in range(batch)]


def add_check_point(g, dump_tensors_ref, batch):
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        if sg.has_attr('device') and sg.get_attr('device') == 'USER':
            continue
        for tensor in sg.get_output_tensors():
            tensor_name = xir_extra_ops.remove_xfix(tensor.name)
            file_entries = dump_tensors_ref.get(
                tensor_name, default_file_entries(tensor, batch))
            dump_tensors_ref[tensor_name] = file_entries


if __name__ == '__main__':
    main()
