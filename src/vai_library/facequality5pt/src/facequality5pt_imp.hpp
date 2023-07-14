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
#include "vitis/ai/facequality5pt.hpp"

namespace vitis{
namespace ai{

/// \brief class for evaluate the quality of a face 
///
class FaceQuality5ptImp : public vitis::ai::TConfigurableDpuTask<FaceQuality5pt> {
public:
  /// Constructor
  explicit FaceQuality5ptImp(const std::string &model_name, bool need_process = true);

  /// Destructor
  virtual ~FaceQuality5ptImp();

  virtual FaceQuality5pt::Mode getMode() override;
  virtual void setMode(FaceQuality5pt::Mode mode) override;

  /// Set a face and get the quality and 5 points of the face
  /// @param img  A face expanded to 1.2 times of detected counterpart and resized as input width * height
  virtual FaceQuality5ptResult run(const cv::Mat &img) override;
  //virtual FaceQuality5ptResult run_original(const cv::Mat &img) override;
  /// Set a image list and get quality and 5 points of every face
  virtual std::vector<FaceQuality5ptResult> run(const std::vector<cv::Mat> &images) override;
  //virtual std::vector<FaceQuality5ptResult> run_original(const std::vector<cv::Mat> &images) override;

private:
  Mode mode_;
};

}
}
