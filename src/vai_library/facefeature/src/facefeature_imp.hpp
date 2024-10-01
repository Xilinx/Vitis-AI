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
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/facefeature.hpp>

namespace vitis {
namespace ai {

/// \brief class for extract feature of a face
///
class FaceFeatureImp : public vitis::ai::TConfigurableDpuTask<FaceFeature> {
 public:
  explicit FaceFeatureImp(const std::string &model_name, bool need_preprocess);
  explicit FaceFeatureImp(const std::string &model_name, xir::Attrs *attrs, bool need_preprocess);
  /// Destructor
  virtual ~FaceFeatureImp();

  /// Set an image and get running results of the network
  /// @param input_image
  virtual FaceFeatureFloatResult run(const cv::Mat &input_image) override;

  /// Set an image and get running fixed results of the
  /// network
  /// @param input_image
  virtual FaceFeatureFixedResult run_fixed(const cv::Mat &input_image) override;
  virtual std::vector<FaceFeatureFloatResult> run(
      const std::vector<cv::Mat> &input_images) override;
  virtual std::vector<FaceFeatureFixedResult> run_fixed(
      const std::vector<cv::Mat> &input_images) override;

 private:
  void run_internal(const cv::Mat &input_image);
  void run_internal(const std::vector<cv::Mat> &input_image);
};
/*!@} */
}  // namespace ai
}  // namespace vitis
