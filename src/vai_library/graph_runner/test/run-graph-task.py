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
import hashlib
import xir
import vart
import numpy as np
import hot_patch_xmodel


def md5(np_array):
    hash_md5 = hashlib.md5()
    hash_md5.update(np_array)
    return hash_md5.hexdigest()


g = xir.Graph.deserialize('/workspace/yolov4-tiny.xmodel')
the_root = g.get_root_subgraph()
the_root.get_name()
hot_patch_xmodel.hot_patch(the_root)

graph_runner = vart.RunnerExt.create_runner(the_root, "run")
inputs = graph_runner.get_inputs()
outputs = graph_runner.get_outputs()
with open('/scratch/models/cache/golden/74/32192dbe8b0cacdf99c2112732324b',
          'rb') as f:
    f.readinto(inputs[0])
print(md5(inputs[0]))
job = graph_runner.execute_async(inputs, outputs)
graph_runner.wait(job)
print(md5(outputs[0]))
print(md5(outputs[1]))
