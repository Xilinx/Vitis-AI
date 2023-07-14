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
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/facedetect.hpp>
using std::tuple;
using std::vector;

namespace vitis {
namespace ai {
class DetectImp : public FaceDetect {
 public:
  DetectImp(const std::string& model_name, bool need_preprocess);
  DetectImp(const std::string& model_name, xir::Attrs* attrs,
            bool need_preprocess);

  /// Destructor
  virtual ~DetectImp();
  /// Set an image and get positions and scores of faces in the image
  virtual FaceDetectResult run(const cv::Mat& img) override;

  /// Set an image list and get positions and scores of faces in the image
  virtual std::vector<FaceDetectResult> run(
      const std::vector<cv::Mat>& img) override;
  /// Get detect threshold
  virtual float getThreshold() const override;
  /// Set detect threshold
  virtual void setThreshold(float threshold) override;
  virtual std::vector<FaceDetectResult> run(
      const std::vector<vart::xrt_bo_t>& input_bos) override;

 private:
  float det_threshold_;
  mutable std::mutex mtx_threshold_;
};

}  // namespace ai
}  // namespace vitis
