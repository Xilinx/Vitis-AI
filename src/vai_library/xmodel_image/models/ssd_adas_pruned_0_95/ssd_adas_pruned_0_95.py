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
import json

prior_box_param1 = {
    "layer_width": 60,
    "layer_height": 45,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [21.0],
    "max_sizes": [45.0],
    "aspect_ratios": [2.0],
    "offset": 0.5,
    "step_width": 8.0,
    "step_height": 8.0,
    "flip": True,
    "clip": False
}

prior_box_param2 = {
    "layer_width": 30,
    "layer_height": 23,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [45.0],
    "max_sizes": [99.0],
    "aspect_ratios": [2.0, 3.0],
    "offset": 0.5,
    "step_width": 16.0,
    "step_height": 16.0,
    "flip": True,
    "clip": False
}

prior_box_param3 = {
    "layer_width": 15,
    "layer_height": 12,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [99.0],
    "max_sizes": [153.0],
    "aspect_ratios": [2.0, 3.0],
    "offset": 0.5,
    "step_width": 32.0,
    "step_height": 32.0,
    "flip": True,
    "clip": False
}

prior_box_param4 = {
    "layer_width": 8,
    "layer_height": 6,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [153.0],
    "max_sizes": [207.0],
    "aspect_ratios": [2.0, 3.0],
    "offset": 0.5,
    "step_width": 64.0,
    "step_height": 64.0,
    "flip": True,
    "clip": False
}

prior_box_param5 = {
    "layer_width": 6,
    "layer_height": 4,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [207.0],
    "max_sizes": [261.0],
    "aspect_ratios": [2.0],
    "offset": 0.5,
    "step_width": 100.0,
    "step_height": 100.0,
    "flip": True,
    "clip": False
}

prior_box_param6 = {
    "layer_width": 4,
    "layer_height": 2,
    "variances": [0.1, 0.1, 0.2, 0.2],
    "min_sizes": [261.0],
    "max_sizes": [315.0],
    "aspect_ratios": [2.0],
    "offset": 0.5,
    "step_width": 300.0,
    "step_height": 300.0,
    "flip": True,
    "clip": False
}


def jit(graph):
    graph.set_attr("need_preprocess", True)
    graph.set_attr("mean", [104.0, 117.0, 123.0])
    graph.set_attr("scale", [1.0, 1.0, 1.0])
    graph.set_attr("is_rgb_input", False)
    graph.set_attr("num_classes", 4)
    graph.set_attr("nms_threshold", 0.4)
    graph.set_attr("conf_threshold", [0.0, 0.6, 0.4, 0.3])
    graph.set_attr("keep_top_k", 200)
    graph.set_attr("top_k", 400)
    prior_box = {
        "prior_box_param": [
            prior_box_param1, prior_box_param2, prior_box_param3, prior_box_param4,
            prior_box_param5, prior_box_param6
        ]
    }
    json_prior_box_param = json.dumps(prior_box)
    #    print(json_prior_box_param)
    graph.set_attr("prior_box_param", json_prior_box_param)

    xir_extra_ops.set_postprocessor(graph, "libxmodel_postprocessor_ssd.so.3", {
        "bbox": ["mbox_loc_fixed_"],
        "conf": ["mbox_conf_flatten"]
    })
