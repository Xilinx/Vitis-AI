/*
 * Copyright 2019 Xilinx Inc.
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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/retinaface.hpp>
using std::tuple;
using std::vector;

namespace vitis {
namespace ai {

class RetinaFaceImp : public vitis::ai::TConfigurableDpuTask<RetinaFace> {
 public:
  RetinaFaceImp(const std::string &model_name, bool need_preprocess);
  RetinaFaceImp(const std::string &model_name, xir::Attrs *attrs, bool need_preprocess);

  /// Destructor
  virtual ~RetinaFaceImp();
  /// Set an image and get positions, scores and landmarks of faces in the image
  virtual RetinaFaceResult run(const cv::Mat &img) override;

  /// Set an image list and get positions,scores and landmarks of faces in the image
  virtual std::vector<RetinaFaceResult> run(
      const std::vector<cv::Mat> &img) override;

 private:
  std::unique_ptr<RetinaFacePostProcess> processor_;
};

}  // namespace ai
}  // namespace vitis
