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
    graph.set_attr("mean", [104.0, 117.0, 123.0])
    graph.set_attr("scale", [1.0, 1.0, 1.0])
    graph.set_attr("is_rgb_input", False)
    graph.set_attr("color1", [
        128, 232, 70, 156, 153, 153, 30, 0, 35, 152, 180, 60, 0, 142, 70, 100,
        100, 230, 32, 178
    ])
    graph.set_attr("color2", [
        64, 35, 70, 102, 153, 153, 170, 220, 142, 251, 130, 20, 0, 0, 0, 60,
        80, 0, 11, 43
    ])
    graph.set_attr("color3", [
        128, 244, 70, 102, 190, 153, 250, 220, 107, 152, 70, 220, 255, 0, 0, 0,
        0, 0, 119, 255
    ])
    xir_extra_ops.set_postprocessor(
        graph, "libxmodel_postprocessor_segmentation.so.3",
        {"input": ["pred_up_fixed_"]})
