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
import os
import xir_extra_ops


def jit(graph):
    graph.set_attr("xmodel_preprocessor", "libxmodel_preprocessor_efficientnet.so.3")
    graph.set_attr("need_preprocess", True)
    graph.set_attr("mean", [127.0, 127.0, 127.0])
    graph.set_attr("scale", [0.0078125, 0.0078125, 0.0078125])
    graph.set_attr("is_rgb_input", True)
    labels_list = (open(os.path.join(graph.get_attr("__dir__"), "word_list.txt"),
                        "r").read().splitlines())

    labels_list_1001 = ["background,"]
    labels_list_1001.extend(labels_list)
    graph.set_attr("labels", labels_list_1001)
    xir_extra_ops.set_postprocessor(graph, "libxmodel_postprocessor_classification.so.3",
                                    {"input": ["my_topk"]})
    conf_op = graph.get_op("logits_fix_")
    graph.create_op("aquant_logits_softmax",
                    "softmax",
                    attrs={"axis": -1},
                    input_ops={"input": [conf_op]},
                    subgraph=graph.get_leaf_subgraph(conf_op))
    graph.create_op(
        "my_topk",
        "topk",
        attrs={"K": 5},
        input_ops={"input": [graph.get_op("aquant_logits_softmax")]},
        subgraph=graph.get_leaf_subgraph(graph.get_op("aquant_logits_softmax")),
    )
    #graph.save_as_image(os.path.join(graph.get_attr("__dir__"), graph.get_attr("__basename__") + ".jit.svg"), "svg")
    #graph.serialize(os.path.join(graph.get_attr("__dir__"), graph.get_attr("__basename__") + ".jit.xmodel"))
    print(graph.get_name())
