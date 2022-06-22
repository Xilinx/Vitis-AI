/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>
#include <vector>
#include <unordered_map>

static std::unordered_map<std::string, int> dpu_batch = {
  {"u50lv_v3e", 5},
  {"u200_u250", 4},
  {"zcu_102_104", 1}
};

static std::unordered_map<std::string,
 std::pair<std::string, std::string>> tf_resnet_v1_50 {
  {"u50lv_v3e", {"resnet50", "graph_zoo/graph_tf_resnet_v1_50_u50lv_v3e.json"}},
  {"u200_u250" , {"resnet50", "graph_zoo/graph_tf_resnet_v1_50_u200_u250.json"}}
};

static std::unordered_map<std::string,
 std::pair<std::string, std::string>> cf_resnet50 {
  {"zcu_102_104", {"resnet50", "graph_zoo/graph_resnet50_zcu102_zcu104.json"}}
};

static std::unordered_map<std::string,
 std::pair<std::string, std::string>> cf_densebox_320_320 {
  {"u50lv_v3e", {"facedetect", "graph_zoo/graph_facedetect_u50lv_v3e.json"}},
  {"u200_u250", {"facedetect", "graph_zoo/graph_facedetect_u200_u250.json"}}
};
