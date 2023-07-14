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
 * Filename: retinaface.hpp
 *
 * Description:
 * This network is used to getting position, score and landmark of faces in the
 * input image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf"
 * for more details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/retinaface.hpp>

#include "vitis/ai/configurable_dpu_task.hpp"
namespace xir {
class Attrs;
};

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting the position,score and landmark of faces in
 the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is a vector of position and score for faces in the input image.
 *
 * Sample code:
 * @code
   auto image = cv::imread("sample_retinaface.jpg");
   auto network = vitis::ai::RetinaFace::create(
                  "retinaface",
                  true);
   auto result = network->run(image);
   for (auto i = 0u; i < result.bboxes.size(); ++i) {
      auto score = result.bboxes[i].score;
      auto x = result.bboxes[i].x * image.cols;
      auto y = result.bboxes[i].y * image.rows;
      auto width = result.bboxes[i].width * image.cols;
      auto height = result.bboxes[i].height * image.rows;
      auto landmark = results.landmarks[i];
      for (auto j = 0; j < 5; ++j) {
        auto px = landmark[j].first * image.cols;
        auto py = landmark[j].second * image.rows;
      }
   }
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_retinaface_result.jpg "result image"
 width=\textwidth
 *
 */
class RetinaFace : public ConfigurableDpuTaskBase {
 public:
  /**
 * @brief Factory function to get an instance of derived classes of class
 RetinaFace
 *
 * @param model_name Model name
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of RetinaFace class.
 */
  static std::unique_ptr<RetinaFace> create(const std::string& model_name,
                                            bool need_preprocess = true);
  /**
 * @brief Factory function to get an instance of derived classes of class
 RetinaFace
 *
 * @param model_name Model name
 * @param attrs Xir attributes
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of RetinaFace class.
 */

  static std::unique_ptr<RetinaFace> create(const std::string& model_name,
                                            xir::Attrs* attrs,
                                            bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit RetinaFace(const std::string& model_name, bool need_preprocess);
  explicit RetinaFace(const std::string& model_name, xir::Attrs* attrs,
                      bool need_preprocess);
  RetinaFace(const RetinaFace&) = delete;
  RetinaFace& operator=(const RetinaFace&) = delete;

 public:
  virtual ~RetinaFace();
  /**
   * @endcond
   */

  /**
   * @brief Function to get running result of the retinaface network.
   *
   * @param img Input Data ,input image (cv::Mat) need to be resized to
   *InputWidth and InputHeight required by the network.
   *
   * @return The detection result of the face detect network , filtered by score
   *>= det_threshold
   *
   */
  virtual RetinaFaceResult run(const cv::Mat& img) = 0;

  /**
   * @brief Function to get running results of the retinaface neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of RetinaFaceResult.
   *
   */
  virtual std::vector<RetinaFaceResult> run(
      const std::vector<cv::Mat>& imgs) = 0;

  /**
   * @brief Function to get running results of the retina neural network in
   * batch mode , used to receive user's xrt_bo to support zero copy.
   *
   * @param input_bos The vector of vart::xrt_bo_t.
   *
   * @return The vector of RetinaFacesResult.
   *
   */
  virtual std::vector<RetinaFaceResult> run(
      const std::vector<vart::xrt_bo_t>& input_bos) = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
