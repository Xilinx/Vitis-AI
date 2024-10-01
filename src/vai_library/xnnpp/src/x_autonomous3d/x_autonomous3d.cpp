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
#include "./x_autonomous3d_post.hpp"
#include "vitis/ai/nnpp/x_autonomous3d.hpp"

namespace vitis {
namespace ai {

X_Autonomous3DPostProcess::X_Autonomous3DPostProcess(){};
X_Autonomous3DPostProcess::~X_Autonomous3DPostProcess(){};

std::unique_ptr<X_Autonomous3DPostProcess> X_Autonomous3DPostProcess::create(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  return std::unique_ptr<X_Autonomous3DPostProcess>(
      new X_Autonomous3DPost(input_tensors, output_tensors, config));
}

}  // namespace ai
}  // namespace vitis
