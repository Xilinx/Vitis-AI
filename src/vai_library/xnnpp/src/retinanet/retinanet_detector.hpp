/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "vitis/ai/nnpp/retinanet.hpp"

namespace vitis {
namespace ai {

class RetinaNetDetector {
public:
    RetinaNetDetector(float nms_thres, float conf_thres);

    std::vector<size_t> detect(const std::vector<float>& scores, const std::vector<std::vector<float>>& boxes, const std::vector<size_t>& labels);
private:
    float nms_threshold_;
    float confidence_threshold_;
};

std::unique_ptr<RetinaNetDetector> CreateRetinaNetUniform(float nms_thres, float conf_thres);

}  // namespace ai
}  // namespace vitis
