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

#include "vitis/ai/nnpp/multitask.hpp"

#include "./multitask_imp.hpp"

namespace vitis {
namespace ai {

MultiTaskPostProcess::MultiTaskPostProcess() {}
MultiTaskPostProcess::~MultiTaskPostProcess() {}

std::unique_ptr<MultiTaskPostProcess> MultiTaskPostProcess::create(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<MultiTaskPostProcess>(
      new MultiTaskPostProcessImp(config, input_tensors, output_tensors));
  // return nullptr;
}

}  // namespace ai
}  // namespace vitis
