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
import xir_extra_ops

list0 = [
    "unknown", "jing", "hu", "jin", "yu", "ji", "jin", "meng", "liao", "ji",
    "hei", "su", "zhe", "wan", "min", "gan", "lu", "yu", "e", "xiang", "yue",
    "gui", "qiong", "chuan", "gui", "yun", "zang", "shan", "gan", "qing",
    "ning", "xin"
]

list1 = [
    "unknown", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

list2 = [
    "unknown", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C",
    "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]


def jit(graph):
    graph.set_attr("need_preprocess", True)
    graph.set_attr("mean", [128.0, 128.0, 128.0])
    graph.set_attr("scale", [1.0, 1.0, 1.0])
    graph.set_attr("is_rgb_input", False)
    list_map = [list0, list1, list2, list2, list2, list2, list2]
    for i in range(7):
        op = graph.get_op("prob" + str(i + 1))
        op1 = graph.create_op("my_topk" + str(i + 1),
                              "topk",
                              attrs={"K": 1},
                              input_ops={"input": [op]},
                              subgraph=graph.get_leaf_subgraph(op))
        op1.get_output_tensor().set_attr("labels", list_map[i])

    xir_extra_ops.set_postprocessor(
        graph, "libxmodel_postprocessor_plate_number.so.3",
        {"input": ["my_topk" + str(i + 1) for i in range(7)]})
