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

import xir
import logging
import xir_extra_ops
import sys
import os
import json
import pickle
import gzip as compress

_ORIGIN = {
    "caffe": "decent-caffe",
    "tensorflow": "decent-tensorflow",
    "tensorflow2": "decent-tensorflow",
}


def main():
    xmodel_path = os.path.realpath(sys.argv[1])
    root_path = os.path.dirname(xmodel_path)
    results = generate(xmodel_path, root_path)
    with open("vai-1.3-generated.json", "w") as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))
    print("total %s" % (len(results),))


def generate(xmodel_file, root_path):
    xmodel_file = os.path.realpath(xmodel_file)
    g = xir.Graph.deserialize(xmodel_file)
    modelfilename, extension = os.path.splitext(xmodel_file)
    folder_path, model_name = os.path.split(modelfilename)
    path = os.path.dirname(os.path.join("/", os.path.relpath(xmodel_file, root_path)))
    type = "UNKNOWN"
    if not g.has_attr("origin"):
        type = "pytorch"
        json_title = model_name + "-" + "pytorch"
    else:
        type = _ORIGIN[g.get_attr("origin")]
        json_title = model_name + "-" + g.get_attr("origin")
    logging.warning("processing %s" % (xmodel_file,))
    return {
        json_title: {
            "meta": {
                "type": type,
                "path": path,
                "xmodel": model_name + ".xmodel",
                "init_tensors": generate_input_tensors(g),
                "dump_tensors_ref": generate_output_tensors(g, type),
            }
        }
    }


def generate_input_tensors(g):
    name_and_sizes = [
        (op.get_output_tensor().name, op.get_output_tensor().get_data_size())
        for op in g.get_head_ops()
        if op.get_type() == "data-fix" or op.get_type() == "data"
    ]
    return {
        xir_extra_ops.remove_xfix(name): guess_and_maybe_copy_golden(name, size)
        for (name, size) in name_and_sizes
    }


def guess_and_maybe_copy_golden(name, size):
    tfname = xir_extra_ops.remove_xfix(name)
    filename = tfname.replace("/", "_") + ".bin"
    files = []
    file = os.path.join("golden", "fake", filename)
    md5sum = "0" * 32
    files.append({"file": file, "md5sum": md5sum, "size": size})
    return files


def generate_output_tensors(g, type):
    return {
        xir_extra_ops.remove_xfix(tensor.name): guess_and_maybe_copy_golden(
            tensor.name, tensor.get_data_size()
        )
        for sg in g.get_root_subgraph().toposort_child_subgraph()
        if sg.has_attr("device") and sg.get_attr("device") != "USER"
        for tensor0 in sg.get_output_tensors()
        for tensor in [
            workaround_bug_in_tf_maybe_remove_last_fixneuron(g, tensor0, type)
        ]
    }


def workaround_bug_in_tf_maybe_remove_last_fixneuron(g, tensor, type):
    op = tensor.producer
    if type != "pytorch" and op.get_type() == "fix":
        return op.get_input_op("input", 0).get_output_tensor()
    return tensor


if __name__ == "__main__":
    main()
