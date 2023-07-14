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
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>

#include "vitis/ai/facelandmark.hpp"
using std::vector;

namespace vitis {
namespace ai {

/// \brief class for evaluate the quality of a face
///
class FaceLandmarkImp : public vitis::ai::TConfigurableDpuTask<FaceLandmark> {
 public:
  /// Destructor
  explicit FaceLandmarkImp(const std::string &model_name, bool need_preprocess);
  explicit FaceLandmarkImp(const std::string &model_name, xir::Attrs *attrs,  bool need_preprocess);
  virtual ~FaceLandmarkImp();

  /// Set an image and get running results of the network
  /// @param input_image
  virtual FaceLandmarkResult run(const cv::Mat &input_image) override;

  /// Set an image list and get running results of the network
  /// @param input_image
  virtual std::vector<FaceLandmarkResult> run(
      const std::vector<cv::Mat> &input_image) override;
};
/*!@} */
}  // namespace ai
}  // namespace vitis
