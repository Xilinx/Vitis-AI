
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
import hashlib
import itertools as it
import pickle
import gzip as compress
_ORIGIN = {
    "caffe": "decent-caffe",
    "tensorflow": "decent-tensorflow",
    "tensorflow2": "decent-tensorflow",
}
_DATA = {}


def main():
    from_path = os.path.realpath(sys.argv[1])
    results = {
        k: v
        for (k, v) in it.islice(
            ((k, v)
             for dirName, subdirList, fileList in os.walk(from_path)
             for fname in fileList
             if fname.endswith(".xmodel")
             for (k, v) in [generate(os.path.join(dirName, fname),  from_path)]
             ), 0, None)
    }
    with compress.open('golden.cache', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(_DATA, f, pickle.HIGHEST_PROTOCOL)
    with open("vai-1.3-generated.json", "w") as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))
    print("total %s" % (len(results),))


def generate(xmodel_file, root_path):
    xmodel_file = os.path.realpath(xmodel_file)
    g = xir.Graph.deserialize(xmodel_file)
    model_name = os.path.basename(os.path.dirname(xmodel_file))
    path = os.path.dirname(os.path.join(
        '/', os.path.relpath(xmodel_file, root_path)))
    type = "UNKNOWN"
    if not g.has_attr('origin'):
        type = "pytorch"
    else:
        type = _ORIGIN[g.get_attr('origin')]
    logging.warning('processing %s' % (xmodel_file, ))
    golden_result_dirs = guess_golden(g.get_attr("files_md5sum"), g.get_name())
    if not golden_result_dirs:
        logging.warning("cannot find %s" % (model_name,))
        # return (None, None)
    return (
        model_name,  {
            "meta": {
                "type": type,
                "path": path,
                "xmodel": model_name + ".xmodel",
                "init_tensors":
                generate_input_tensors(g, golden_result_dirs, path),
                "dump_tensors_ref":
                generate_output_tensors(g, golden_result_dirs, path),
            }
        }
    )


def generate_input_tensors(g, gdirs, path):
    name_and_sizes = [(op.get_output_tensor().name,
                       op.get_output_tensor().get_data_size())
                      for op in g.get_head_ops() if op.get_type() == "data-fix" or op.get_type() == "data"
                      ]
    return {
        xir_extra_ops.remove_xfix(name):
        guess_and_maybe_copy_golden(name, gdirs, path, size)
        for (name, size) in name_and_sizes
    }


def guess_and_maybe_copy_golden(name, gdirs, path, size):
    tfname = xir_extra_ops.remove_xfix(name)
    filename = tfname.replace('/', '_') + ".bin"
    files = []
    # logging.warning("gdirs = %s"%(gdirs,))
    for gdir in gdirs:
        basename = os.path.basename(gdir)
        dump0 = os.path.join(gdir,  filename)
        dump1 = dump0.replace("_fixed.bin", ".bin")
        dump2 = dump0.replace("_fix_.bin", "_fix.bin")
        dump3 = dump0.replace("_.bin", ".bin")
        path_i = os.path.join(path, "golden", basename)
        file = os.path.join("golden", basename, filename)
        md5sum = '0' * 32
        for dump in [dump0, dump1, dump2, dump3]:
            if os.path.isfile(dump):
                md5sum = md5(dump)
                file = dump
        files.append({
            'file': file,
            'md5sum': md5sum,
            'size': size
        })
    return files


def generate_output_tensors(g, gdirs, path):
    return {
        xir_extra_ops.remove_xfix(tensor.name):
        guess_and_maybe_copy_golden(
            tensor.name, gdirs, path, tensor.get_data_size())
        for sg in g.get_root_subgraph().toposort_child_subgraph()
        if sg.has_attr('device') and sg.get_attr('device') != 'USER'
        for tensor0 in sg.get_output_tensors()
        for tensor in [workaround_bug_in_tf_maybe_remove_last_fixneuron(g, tensor0)]
    }


def workaround_bug_in_tf_maybe_remove_last_fixneuron(g, tensor):
    op = tensor.producer
    if op.get_type() == 'fix':
        return op.get_input_op('input', 0).get_output_tensor()
    return tensor


def create_empty_file(filename, size):
    logging.warning("create empty file (%s: %s)" % (filename, size))
    with open(filename, "wb") as out:
        out.seek(size - 1)
        out.write(b'\0')


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        buf = f.read()
        hash_md5.update(buf)
    _DATA[hash_md5.hexdigest()] = buf
    return hash_md5.hexdigest()


def guess_golden(files_md5sum, graph_name):
    files = []
    for (k, v) in files_md5sum.items():
        dirname = os.path.dirname(k)
        golden = [
            dir
            for candidate in [
                os.path.join("..", "dump_results", "dump_results_0"),
                os.path.join("..", "dump_results", "dump_results_1"),
                os.path.join("..", "dump_results", "dump_results_2"),
                os.path.join("..", "dump_results", "dump_results_3"),
                os.path.join("..", "dump_results", "dump_results_4"),
                os.path.join("dump", "dump_results_0"),
                os.path.join("dump", "dump_results_1"),
                os.path.join("dump", "dump_results_2"),
                os.path.join("dump", "dump_results_3"),
                os.path.join("dump", "dump_results_4"),
                "dump_results_0",
                "dump_results_1",
                "dump_results_2",
                "dump_results_3",
                "dump_results_4",
                os.path.join("..", "dump_gpu"),
                os.path.join("deploy_check_data_int", graph_name)
            ]
            for dir in [os.path.join(dirname, candidate)]
            if os.path.isdir(os.path.join(dir))
        ]
        if golden:
            return golden
        files.append(k)
    logging.warning("cannot find golden for %s" % (files,))
    return ["fake"]


if __name__ == '__main__':
    main()
