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
    graph.set_attr("need_preprocess", True)
    graph.set_attr("mean", [128.0, 128.0, 128.0])
    graph.set_attr("scale", [1.0, 1.0, 1.0])
    graph.set_attr("is_rgb_input", False)
    conf_op = graph.get_op("pixel-conv-tiled_fixed_")
    graph.create_op(
        "pixel-conv-tiled_fixed_softmax",
        "softmax",
        attrs={"axis": -1},
        input_ops={"input": [conf_op]},
        subgraph=graph.get_leaf_subgraph(conf_op),
    )
    graph.set_attr("det_threshold", 0.9)
    graph.set_attr("nms_threshold", 0.3)
    xir_extra_ops.set_postprocessor(
        graph,
        "libxmodel_postprocessor_densebox.so.3",
        {
            "bbox": ["bb-output-tiled_fixed_"],
            "conf": ["pixel-conv-tiled_fixed_softmax"],
        },
    )
