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

#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <string>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <xir/attrs/attrs.hpp>

#include "vitis/ai/fusion_cnn.hpp"

namespace vitis {
namespace ai {

class FusionCNNImp : public FusionCNN {
 public:
  explicit FusionCNNImp(const std::string& model_name, bool need_preprocess);
  explicit FusionCNNImp(const std::string& model_name, xir::Attrs* attrs,
                        bool need_preprocess);
  virtual ~FusionCNNImp();

  virtual vitis::ai::fusion_cnn::DetectResult run(
      const vitis::ai::fusion_cnn::DetectResult& detect_result_2d,
      vitis::ai::fusion_cnn::DetectResult& detect_result_3d,
      const vitis::ai::fusion_cnn::FusionParam& fusion_param) override;

  virtual std::vector<fusion_cnn::DetectResult> run(
      const std::vector<fusion_cnn::DetectResult>& batch_detect_result_2d,
      std::vector<fusion_cnn::DetectResult>& batch_detect_result_3d,
      const std::vector<fusion_cnn::FusionParam>& batch_fusion_params) override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual size_t get_input_batch() const override;

 private:
  std::tuple<V1I, unsigned int> preprocess(
      const vitis::ai::fusion_cnn::DetectResult& detect_result_2d,
      vitis::ai::fusion_cnn::DetectResult& detect_result_3d,
      const vitis::ai::fusion_cnn::FusionParam& fusion_param, int batch_idx);
  void postprocess(const V1I& indexes, const unsigned int count,
                   vitis::ai::fusion_cnn::DetectResult& result_3d,
                   int batch_idx);

  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_;
};

}  // namespace ai
}  // namespace vitis
