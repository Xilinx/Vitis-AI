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

#include "vitis/ai/nnpp/pointpillars_nuscenes.hpp"
#include "./anchor.hpp"
#include <functional>

using namespace vitis::ai::pointpillars_nus;
namespace vitis {
namespace ai {

class PointPillarsNuscenesPost: public vitis::ai::PointPillarsNuscenesPostProcess {
 public:
  PointPillarsNuscenesPost(const std::vector<vitis::ai::library::InputTensor>& input_tensors,
          const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
          const vitis::ai::proto::DpuModelParam& config);
  virtual ~PointPillarsNuscenesPost();

  virtual std::vector<PointPillarsNuscenesResult> postprocess(size_t batch_size) override;
  PointPillarsNuscenesResult postprocess_internal(unsigned int idx);
  PointPillarsNuscenesResult postprocess_internal_simple(unsigned int idx);

 private:
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  Anchors anchors_;
  uint32_t output_score_index_;
  uint32_t output_bbox_index_;
  uint32_t output_dir_index_;
  int num_classes_;
  int nms_pre_;
  float nms_thresh_;
  int max_num_;
  float score_thresh_;
  int bbox_code_size_;
  //uint32_t input_width_;
  //uint32_t input_height_;


  //std::unique_ptr<pointpillars_nus::PointPillarsNuscenesDetector> detector_;
  //std::unique_ptr<char *> detector_;
};

}  // namespace ai
}  // namespace vitis
