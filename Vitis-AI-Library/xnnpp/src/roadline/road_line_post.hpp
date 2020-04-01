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
#pragma once
#include "predict.hpp"
#include "vitis/ai/nnpp/lanedetect.hpp"

namespace vitis {
namespace ai {

class RoadLinePost : public vitis::ai::RoadLinePostProcess {
 public:
  RoadLinePost(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  virtual ~RoadLinePost();

  virtual RoadLineResult road_line_post_process(int inWidth,
                                                int inHeight, unsigned int idx) override;

  virtual std::vector<RoadLineResult> road_line_post_process(const std::vector<int>& inWidth,
                                                const std::vector<int>& inHeight) override;

 private:
  std::unique_ptr<vitis::nnpp::roadline::IpmInfo> ipminfo_;
  std::unique_ptr<vitis::nnpp::roadline::Predict> predict_;
  const vitis::ai::proto::DpuModelParam config_;
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  std::vector<vitis::ai::library::OutputTensor> output_tensors_;
};
}  // namespace ai
}  // namespace vitis
