/*
 * Copyright 2019 xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <vector>

namespace vitis {
namespace ai {
namespace Segmentation3DPost {


std::vector<int> post_prec(std::vector<float> proj_range,  std::vector<float> proj_argmax, std::vector<float> unproj_range, std::vector<float> px, std::vector<float> py); 



} //namespace
}
}
