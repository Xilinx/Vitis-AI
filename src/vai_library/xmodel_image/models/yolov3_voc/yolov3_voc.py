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


def jit(graph):
    graph.set_attr("need_preprocess", True)
    graph.set_attr("mean", [0.0, 0.0, 0.0])
    graph.set_attr("scale", [0.00390625, 0.00390625, 0.00390625])
    graph.set_attr("is_rgb_input", True)
    graph.set_attr("num_classes", 20)
    graph.set_attr("anchorCnt", 3)
    graph.set_attr("conf_threshold", 0.3)
    graph.set_attr("nms_threshold", 0.45)
    graph.set_attr("biases", [
        10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0,
        119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0
    ])

    xir_extra_ops.set_postprocessor(
        graph,
        "libxmodel_postprocessor_yolov3.so.3",
        {
            "input": [
                "layer81-conv_fixed_",
                "layer93-conv_fixed_",
                "layer105-conv_fixed_",
            ]
        },
    )
