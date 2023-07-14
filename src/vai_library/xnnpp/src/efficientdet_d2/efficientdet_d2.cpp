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

#include <algorithm>
//#include <cassert>
#include <cmath>
#include <vitis/ai/nnpp/efficientdet_d2.hpp>
#include "./postprocess.hpp"

namespace vitis {
namespace ai {

EfficientDetD2PostProcess::EfficientDetD2PostProcess(){};
EfficientDetD2PostProcess::~EfficientDetD2PostProcess(){};

std::unique_ptr<EfficientDetD2PostProcess> EfficientDetD2PostProcess::create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<EfficientDetD2PostProcess>(
      new EfficientDetD2Post(input_tensors, output_tensors, config));
}
}  // namespace ai
}  // namespace vitis
