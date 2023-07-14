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
 * Filename: posedetect.hpp
 *
 * Description:
 * This network is used to detecting a pose from a input image.
 *
 * Please refer to document "Xilinx_AI_User_Guide.pdf" for more details of these
 * APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/posedetect.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting a pose from an input image (cv::Mat).
 * @note Support detect a single pose.
 *
 * Input an image (cv::Mat).
 *
 * Output is a struct of PoseDetectResult, include 14 point.
 *
 * Sample code:
 * @code
   auto det = vitis::ai::PoseDetect::create("sp_net");
   auto image = cv::imread("sample_posedetect.jpg");
   auto results = det->run(image);
   for(auto result: results.pose14pt) {
       std::cout << result << std::endl;
   }
   @endcode

 * Display of the posedetect model results:
 * @image latex images/sample_posedetect_result.jpg "pose detect image" width=80px
 *
 */

class PoseDetect {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PoseDetect.
   *
   * @param model_name Model name .
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   * @return An instance of PoseDetect class.
   *
   */
  static std::unique_ptr<PoseDetect> create(const std::string &model_name,
                                            bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit PoseDetect();
  PoseDetect(const PoseDetect &) = delete;
  virtual ~PoseDetect();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the PoseDetect network (input image
   * columns).
   *
   * @return InputWidth of the PoseDetect network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the PoseDetect network (input image
   *rows).
   *
   *@return InputHeight of the PoseDetect network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
  /**
   * @brief Function to get running results of the posedetect neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return PoseDetectResult.
   *
   */
  virtual PoseDetectResult run(const cv::Mat &image) = 0;
  /**
   * @brief Function to get running results of the posedetect neural network in
   * batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of PoseDetectResult.
   *
   */
  virtual std::vector<PoseDetectResult> run(
      const std::vector<cv::Mat> &images) = 0;
};
}  // namespace ai
}  // namespace vitis
