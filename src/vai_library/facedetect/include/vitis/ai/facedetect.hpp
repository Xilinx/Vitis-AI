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
/*
 * Filename: facedetect.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/facedetect.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting the position of faces in the input image
 (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is a vector of position and score for faces in the input image.
 *
 * Sample code:
 * @code
   auto image = cv::imread("sample_facedetect.jpg");
   auto network = vitis::ai::FaceDetect::create(
                  "densebox_640_360",
                  true);
   auto result = network->run(image);
   for (const auto &r : result.rects) {
      auto score = r.score;
      auto x = r.x * image.cols;
      auto y = r.y * image.rows;
      auto width = r.width * image.cols;
      auto height = r.height * image.rows;
   }
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_facedetect_result.jpg "result image"
 width=\textwidth
 *
 */
class FaceDetect : public ConfigurableDpuTaskBase {
 public:
  /**
 * @brief Factory function to get instance of derived classes of class
 FaceDetect
 *
 * @param model_name Model name
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of FaceDetect class.
 */
  static std::unique_ptr<FaceDetect> create(const std::string& model_name,
                                            bool need_preprocess = true);
  /**
 * @brief Factory function to get instance of derived classes of class
 FaceDetect
 *
 * @param model_name Model name
 * @param attrs Xir attributes
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of FaceDetect class.
 */

  static std::unique_ptr<FaceDetect> create(const std::string& model_name,
                                            xir::Attrs* attrs,
                                            bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit FaceDetect(const std::string& model_name, bool need_preprocess);
  explicit FaceDetect(const std::string& model_name, xir::Attrs* attrs,
                      bool need_preprocess);
  FaceDetect(const FaceDetect&) = delete;
  FaceDetect& operator=(const FaceDetect&) = delete;

 public:
  virtual ~FaceDetect();
  /**
   * @endcond
   */

  /**
   * @brief Function to get detect threshold.
   * @return The detect threshold. The value ranges from 0 to 1.0f.
   */
  virtual float getThreshold() const = 0;

  /**
   * @brief Function of update detect threshold.
   * @note The detection results will filter by detect threshold (score >=
   * threshold).
   * @param threshold The detect threshold. The value ranges from 0 to 1.0f.
   */
  virtual void setThreshold(float threshold) = 0;

  /**
   * @brief Function to get running result of the facedetect network.
   *
   * @param img Input Data ,input image (cv::Mat) need to be resized to
   *InputWidth and InputHeight required by the network.
   *
   * @return The detection result of the face detect network, filtered by score
   *>= det_threshold
   *
   */
  virtual FaceDetectResult run(const cv::Mat& img) = 0;

  /**
   * @brief Function to get running results of the facedetect neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of FaceDetectResult.
   *
   */
  virtual std::vector<FaceDetectResult> run(
      const std::vector<cv::Mat>& imgs) = 0;

  /**
   * @brief Function to get running results of the facedetect neural network in
   * batch mode , used to receive user's xrt_bo to support zero copy.
   *
   * @param input_bos The vector of vart::xrt_bo_t.
   *
   * @return The vector of FaceDetectResult.
   *
   */
  virtual std::vector<FaceDetectResult> run(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
