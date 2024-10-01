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

#include "vitis/ai/nnpp/retinaface.hpp"
#include <map>
#include <functional>
#include "./retinaface_detector.hpp"

using namespace vitis::ai::retinaface;
namespace vitis {
namespace ai {

class RetinaFacePost : public vitis::ai::RetinaFacePostProcess {
 public:
  RetinaFacePost(const std::vector<vitis::ai::library::InputTensor>& input_tensors,
          const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
          const vitis::ai::proto::DpuModelParam& config);
  virtual ~RetinaFacePost();

  virtual std::vector<RetinaFaceResult> retinaface_post_process(size_t batch_size) override;
  RetinaFaceResult retinaface_post_process_internal(unsigned int idx);

 private:
  //std::map<int32_t, StrideLayers, std::greater<int32_t>> stride_layers_; // key = stride
  StrideLayersMap stride_layers_; // key = stride

  std::vector<RetinaFaceOutputInfo> conf_layer_infos_;
  std::vector<float> softmax_data_;
  // Prior Box
  //std::vector<std::shared_ptr<std::vector<float>>> priors_;
  std::vector<std::vector<float>> priors_;
  const std::vector<vitis::ai::library::InputTensor> input_tensors_;
  const std::vector<vitis::ai::library::OutputTensor> output_tensors_;
  float nms_thresh_;
  float det_thresh_; 
  std::unique_ptr<retinaface::RetinaFaceDetector> detector_;
};

}  // namespace ai
}  // namespace vitis
