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
#include "./ssd_detector.hpp"
#include "vitis/ai/nnpp/refinedet.hpp"

namespace vitis {
namespace ai {

class RefineDetPost : public vitis::ai::RefineDetPostProcess {
 public:
  RefineDetPost(
      const std::vector<vitis::ai::library::InputTensor>&
          input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>&
          output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  virtual ~RefineDetPost();

  virtual RefineDetResult refine_det_post_process(unsigned int idx) override;
  virtual std::vector<RefineDetResult> refine_det_post_process() override;

 private:
  int num_classes_;
  std::vector<std::shared_ptr<std::vector<float>>> priors_;
  // DetectorImp
  std::unique_ptr<vitis::nnpp::refinedet::SSDdetector> detector_;
  const std::vector<vitis::ai::library::InputTensor>
      input_tensors_;
  std::vector<vitis::ai::library::OutputTensor>
      output_tensors_;
};
}  // namespace ai
}  // namespace vitis
