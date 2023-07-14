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
#include "./ssd_detector.hpp"
#include "vitis/ai/nnpp/multitask.hpp"

// namespace vitis {
// namespace ai {
// namespace multitask {
// class SSDdetector;
// }
// } // namespace ai
// } // namespace vitis

namespace vitis {
namespace ai {

class MultiTaskPostProcessImp : public vitis::ai::MultiTaskPostProcess {
 public:
  MultiTaskPostProcessImp(
      const vitis::ai::proto::DpuModelParam& config,
      const std::vector<std::vector<vitis::ai::library::InputTensor>>&
          input_tensors,
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors);
  virtual ~MultiTaskPostProcessImp();

  std::vector<VehicleResult> process_det(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t batch_idx);
  cv::Mat process_seg(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t batch_idx);
  cv::Mat process_seg_visualization(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t batch_idx);

  virtual std::vector<MultiTaskResult> post_process_seg(
      size_t batch_size) override;

  virtual std::vector<MultiTaskResult> post_process_seg_visualization(
      size_t batch_size) override;

 private:
  int num_detection_classes_;
  int num_segmention_classes_;
  std::vector<std::vector<vitis::ai::multitask::SSDOutputInfo>> all_loc_infos_;
  std::vector<std::vector<vitis::ai::multitask::SSDOutputInfo>> all_conf_infos_;
  std::vector<float> softmax_result;
  std::unique_ptr<vitis::ai::multitask::SSDdetector> detector_;
  const std::vector<std::vector<vitis::ai::library::InputTensor>>
      input_tensors_;
  std::vector<std::vector<vitis::ai::library::OutputTensor>> output_tensors_;
  std::string scolor1_;
  std::string scolor2_;
  std::string scolor3_;
  std::vector<uint8_t> color_map_;
};

}  // namespace ai
}  // namespace vitis
