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

#include <array>
#include <vector>
#include "vitis/ai/nnpp/x_autonomous3d.hpp"
//#include <map>

namespace vitis {
namespace ai {

class X_Autonomous3DPost : public vitis::ai::X_Autonomous3DPostProcess {
 public:
  X_Autonomous3DPost(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  virtual ~X_Autonomous3DPost();

  virtual std::vector<X_Autonomous3DResult> process(size_t batch_size) override;

 private:
  X_Autonomous3DResult process_debug_float(size_t batch_index);
  X_Autonomous3DResult process_internal(size_t batch_index);
  X_Autonomous3DResult process_internal_debug(size_t index);
  X_Autonomous3DResult process_internal_simple(size_t batch_index);

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  std::map<std::string, vitis::ai::library::OutputTensor> output_tensor_map_;
  std::vector<float> iou_quality_cal_result_;
  std::vector<float> scores_;
};

}  // namespace ai
}  // namespace vitis
