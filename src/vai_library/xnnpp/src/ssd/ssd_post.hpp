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

#include "vitis/ai/nnpp/ssd.hpp"
//#include <map>
#include "./ssd_detector.hpp"

namespace vitis {
namespace ai {

class SSDPost : public vitis::ai::SSDPostProcess {
 public:
  SSDPost(const std::vector<vitis::ai::library::InputTensor>& input_tensors,
          const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
          const vitis::ai::proto::DpuModelParam& config);
  virtual ~SSDPost();

  virtual std::vector<SSDResult> ssd_post_process(size_t batch_size) override;
  SSDResult ssd_post_process_internal_uniform(unsigned int idx);
  // SSDResult ssd_post_process_internal();
  // SSDResult ssd_mlperf_post_process();
  // std::vector<SSDResult> post_processing_arm(const cv::Mat &input_img);

 private:
  size_t num_classes_;
  bool is_tf_;
  bool is_mlperf_;
  std::set<int> bbox_layer_indexes_;
  std::set<int> conf_layer_indexes_;
  std::vector<vitis::ai::dpssd::SSDOutputInfo> output_layer_infos_;
  // std::vector<int> bbox_layer_indexes_;
  // std::map<uint32_t, vitis::ai::dpssd::SSDOutputInfo> bbox_layer_infos_;
  // std::set<int> bbox_layer_indexes_;
  // Prior Box
  std::vector<std::shared_ptr<std::vector<float>>> priors_;
  // DetectorImp
  std::unique_ptr<vitis::ai::dpssd::SSDdetector> detector_;
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
};

}  // namespace ai
}  // namespace vitis
