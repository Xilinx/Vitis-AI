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
#include <vector>
#include <vitis/ai/nnpp/efficientdet_d2.hpp>
#include "./anchor.hpp"

namespace vitis {
namespace ai {

class EfficientDetD2Post : public vitis::ai::EfficientDetD2PostProcess {
 public:
  EfficientDetD2Post(
      const std::vector<vitis::ai::library::InputTensor>& input_tensors,
      const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  virtual ~EfficientDetD2Post();

  virtual std::vector<EfficientDetD2Result> postprocess(
      size_t batch_size, const std::vector<int>& swidths,
      const std::vector<int>& sheights,
      const std::vector<float>& image_scales) override;
  EfficientDetD2Result postprocess_kernel(size_t batch_idx, int swidth,
                                          int sheight, float image_scale);

 private:
  int num_classes_;
  int min_level_;
  int max_level_;
  float score_thresh_;
  float nms_thresh_;
  int pre_nms_num_;
  int max_output_num_;
  std::shared_ptr<efficientdet_d2::Anchor> anchor_;
  std::map<int, vitis::ai::library::OutputTensor> bbox_output_layers_;
  std::map<int, vitis::ai::library::OutputTensor> cls_output_layers_;
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
};

}  // namespace ai
}  // namespace vitis

