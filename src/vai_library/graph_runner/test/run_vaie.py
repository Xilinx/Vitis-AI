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
import json
import shlex
import hashlib

MODE = os.environ.get('MODE', "run")


def main():
    json_file_name = sys.argv[1]
    errors = {}
    with open(json_file_name) as json_file:
        try:
            meta_data = json.load(json_file)
            for (k, v) in meta_data.items():
                r = run(k, v)
                if not r:
                    errors[k] = r

        except json.decoder.JSONDecodeError as err:
            print("cannot load json from " + json_file_name + " error:" +
                  str(err))
            sys.exit(2)
    with open(json_file_name, "w") as f:
        s = (json.dumps(meta_data, indent=4, sort_keys=True))
        f.write(s)
    if errors:
        for (k, v) in errors.items():
            print("%s failed." % (meta_data[k]["meta"]["xmodel"], ))
        sys.exit(1)
    return 0


def realpath(file):
    md5sum = file['md5sum']
    root = os.environ.get(
        'GOLDEN_CACHE',
        os.path.join('/', 'scratch', 'models', 'cache', 'golden'))
    rel = os.path.join(root, md5sum[0:2], md5sum[2:])
    return rel


def tensor_file_name(s):
    ret = s.replace("/", "_")
    return ret


def build_input(d, log_path, batch):
    return ";".join([("%s %s" % (k, realpath(v[batch])))
                     for (k, v) in d.items()])


def build_output(d, log_path, batch):
    return ";".join([
        ("%s %s" % (k, os.path.join(log_path,
                                    tensor_file_name(k) + ".bin"))) for k in d
    ])


def build_log_path(model, batch):
    ret = os.path.join("/", "tmp", os.environ.get("USER"), "vaie.log", model,
                       MODE, "batch_" + str(batch))
    return ret


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        buf = f.read()
        hash_md5.update(buf)
    return hash_md5.hexdigest()


def compare_output(d, log_path, batch):
    ret = True
    for k in d:
        v = d[k][batch]
        expected_md5sum = v['md5sum']
        actual_md5sum = md5(
            os.path.join(log_path,
                         tensor_file_name(k) + ".bin"))
        if expected_md5sum != actual_md5sum:
            print("expect %s actual %s: %s is not correct!" %
                  (expected_md5sum, actual_md5sum, k))
            ret = False
        else:
            print("expect %s actual %s: %s is  correct!" %
                  (expected_md5sum, actual_md5sum, k))
        if not v.get('freeze', False):
            v['md5sum'] = actual_md5sum
        v['md5sum_' + MODE] = actual_md5sum
    return ret


def run(name, v):
    model = os.environ.get('MODEL', None)
    xmodel = v["meta"]["xmodel"]
    if model and (model != xmodel and model != name):
        return True
    from_model = os.environ.get('FROM_MODEL', None)
    if from_model and name < from_model:
        print("skip " + name)
        return True
    skip = v["meta"].get("skip", False)
    if skip:
        print("skip %s" % (str(v["meta"]["xmodel"])))
    root = os.environ.get('MODEL_ZOO_ROOT',
                          '/scratch/models/xilinx_model_zoo_u50_1.3.0_amd64')
    path = os.path.join(root, "." + v["meta"]["path"])
    print("running " + xmodel + " : " + path + "/" + xmodel)
    num_of_batch = len(v["meta"]["init_tensors"][next(
        iter(v["meta"]["init_tensors"]))])
    ret = True
    print("num_of_batch     = " + str(num_of_batch))
    for batch in range(num_of_batch):
        log_path = build_log_path(name, batch)
        cmd = ["vaie-run"]
        cmd.append("-i")
        cmd.append(os.path.join(path, xmodel))
        cmd.append("--init")
        cmd.append(build_input(v["meta"]["init_tensors"], log_path, batch))
        cmd.append("--dump")
        cmd.append(build_output(v["meta"]["dump_tensors_ref"], log_path,
                                batch))
        cmd.append("--target")
        cmd.append(MODE)
        cmd.append("--log-path")
        cmd.append(log_path)
        cmd.append("--deploy")
        cmd.append("release")
        cmd.append("--disable-debug")
        cmd_str = ' '.join([shlex.quote(c) for c in cmd])
        print(cmd_str)
        with open('vaie.cmd', 'w') as f:
            print(cmd_str, file=f)
        import subprocess
        status = subprocess.run(cmd)
        if status.returncode == 0:
            ret = compare_output(v["meta"]["dump_tensors_ref"], log_path,
                                 batch) and ret
        else:
            print("cannot run vaie")
            ret = False
    return ret


if __name__ == '__main__':
    main()
