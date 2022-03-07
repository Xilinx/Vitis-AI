#
# Copyright 2019 Xilinx Inc.
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
    graph.set_attr("mean", [127.5, 127.5, 127.5])
    graph.set_attr("scale", [0.0078431, 0.0078431, 0.0078431])
    graph.set_attr("is_rgb_input", False)
    xir_extra_ops.set_postprocessor(
        graph,
        "libxmodel_postprocessor_face_quality.so.2",
        {"quality": ["Addmm_4_fixed_"], "5pt": ["Addmm_2_fixed_"]},
    )
