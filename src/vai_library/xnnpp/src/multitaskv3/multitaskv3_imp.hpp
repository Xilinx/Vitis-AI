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
#include "vitis/ai/nnpp/multitaskv3.hpp"

// namespace vitis {
// namespace ai {
// namespace multitask {
// class SSDdetector;
// }
// } // namespace ai
// } // namespace vitis

namespace vitis {
namespace ai {

class MultiTaskv3PostProcessImp : public vitis::ai::MultiTaskv3PostProcess {
 public:
  MultiTaskv3PostProcessImp(
      const vitis::ai::proto::DpuModelParam& config,
      const std::vector<std::vector<vitis::ai::library::InputTensor>>&
          input_tensors,
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors);
  virtual ~MultiTaskv3PostProcessImp();

  std::vector<Vehiclev3Result> process_det(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t batch_idx);
  cv::Mat process_seg(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t tensor_ind, size_t batch_idx);
  cv::Mat process_seg_visualization(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t tensor_ind, size_t batch_idx);
  cv::Mat process_seg_visualization_c(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t tensor_ind, size_t batch_idx);
  cv::Mat process_depth(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t tensor_ind, size_t batch_idx);

  cv::Mat process_depth_ori(
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      size_t tensor_ind, size_t batch_idx);
  virtual std::vector<MultiTaskv3Result> post_process(
      size_t batch_size) override;

  virtual std::vector<MultiTaskv3Result> post_process_visualization(
      size_t batch_size) override;

 private:
  int num_detection_classes_;
  int num_segmention_classes_;
  std::vector<std::vector<vitis::ai::multitaskv3::SSDOutputInfo>>
      all_loc_infos_;
  std::vector<std::vector<vitis::ai::multitaskv3::SSDOutputInfo>>
      all_conf_infos_;
  std::vector<std::vector<vitis::ai::multitaskv3::SSDOutputInfo>>
      all_centerness_infos_;
  std::vector<float> conf_result;
  std::vector<float> centerness_result;
  std::unique_ptr<vitis::ai::multitaskv3::SSDdetector> detector_;
  const std::vector<std::vector<vitis::ai::library::InputTensor>>
      input_tensors_;
  std::vector<std::vector<vitis::ai::library::OutputTensor>> output_tensors_;
  std::string scolor1_;
  std::string scolor2_;
  std::string scolor3_;
  std::vector<uint8_t> color_c1;
  std::vector<uint8_t> color_c2;
  std::vector<uint8_t> color_c3;
  std::vector<uint8_t> color_map_;
};

}  // namespace ai
}  // namespace vitis
