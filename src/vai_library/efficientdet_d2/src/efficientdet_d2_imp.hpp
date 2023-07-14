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

#include <vitis/ai/efficientdet_d2.hpp>

using std::shared_ptr;
using std::vector;

namespace vitis {

namespace ai {

class EfficientDetD2Imp : public EfficientDetD2 {
 public:
  EfficientDetD2Imp(const std::string& model_name, bool need_preprocess = true);
  EfficientDetD2Imp(const std::string& model_name, xir::Attrs* attrs,
                    bool need_preprocess = true);
  virtual ~EfficientDetD2Imp();

  virtual EfficientDetD2Result run(const cv::Mat& img) override;
  virtual std::vector<EfficientDetD2Result> run(
      const std::vector<cv::Mat>& img) override;

 private:
  std::vector<float> preprocess(const std::vector<cv::Mat>& batch_image_bgr,
                                size_t batch_size);
  std::unique_ptr<EfficientDetD2PostProcess> processor_;
};
}  // namespace ai
}  // namespace vitis

