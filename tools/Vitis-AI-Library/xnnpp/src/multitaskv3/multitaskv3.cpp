/*
 * Copyright 2019 Xilinx Inc.
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

#include "vitis/ai/nnpp/multitaskv3.hpp"

#include "./multitaskv3_imp.hpp"

namespace vitis {
namespace ai {

MultiTaskv3PostProcess::MultiTaskv3PostProcess() {}
MultiTaskv3PostProcess::~MultiTaskv3PostProcess() {}

std::unique_ptr<MultiTaskv3PostProcess> MultiTaskv3PostProcess::create(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<MultiTaskv3PostProcess>(
      new MultiTaskv3PostProcessImp(config, input_tensors, output_tensors));
  // return nullptr;
}

}  // namespace ai
}  // namespace vitis
