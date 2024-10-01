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
    graph.set_attr("mean", [105.0, 117.0, 123.0])
    graph.set_attr("scale", [1.0, 1.0, 1.0])
    graph.set_attr("is_rgb_input", False)
    graph.set_attr("ratio", 8)
    graph.set_attr("ipm_left", 5.0)
    graph.set_attr("ipm_right", 75.0)
    graph.set_attr("ipm_top", 23.75)
    graph.set_attr("ipm_bottom", 50.0)
    graph.set_attr("ipm_interpolation", 0.0)
    graph.set_attr("ipm_vp_portion", 0.0)
    graph.set_attr("focal_length_x", 61.8)
    graph.set_attr("focal_length_y", 68.8)
    graph.set_attr("optical_center_x", 40.0)
    graph.set_attr("optical_center_y", 30.0)
    graph.set_attr("camera_height", 2179.8)
    graph.set_attr("pitch", 14.0)
    graph.set_attr("yaw", 0.0)

    xir_extra_ops.set_postprocessor(
        graph,
        "libxmodel_postprocessor_lane_detect.so.3",
        {"input": ["type-tile_fixed_"]},
    )
