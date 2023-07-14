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
 * Filename: platedetect.hpp
 *
 * Description:
 * This network is used to getting position and score of plate in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/platedetect.hpp>
namespace xir {
  class Attrs;
};
namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting the position of plate in a vehicle image
 (cv::Mat).
 *
 * Input is a vehicle image (cv::Mat).
 *
 * Output is position and score of plate in the input image.
 *
 * Sample code:
 * @code
   cv::Mat image = cv::imread("car.jpg");
   auto network = vitis::ai::PlateDetect::create(true);
   auto r = network->run(image);
   auto score = r.box.score.
   auto x = r.box.x * image.cols;
   auto y = r.box.y * image.rows;
   auto witdh = r.box.width * image.cols;
   auto height = r.box.height * image.rows;
   @endcode
 *
 */
class PlateDetect {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * platedetect.
   *
   * @param model_name the model name of the created model
   *
   * @param need_mean_scale_process Normalize with mean/scale or not, true by
   * default.
   *
   * @return An instance of the PlateDetect class.
   */
  static std::unique_ptr<PlateDetect> create(const std::string &model_name,
                                             bool need_mean_scale_process = true);
  /**
   * @brief Factory function to get an instance of derived classes of class
   * platedetect.
   *
   * @param model_name the model name of the created model
   *
   * @param attrs xir::Attrs pointer points to the provided attributes 
   *
   * @param need_mean_scale_process Normalize with mean/scale or not, true by
   * default.
   *
   * @return An instance of the PlateDetect class.
   */
  static std::unique_ptr<PlateDetect> create(const std::string &model_name,
                                             xir::Attrs *attrs,
                                             bool need_mean_scale_process = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit PlateDetect();
  PlateDetect(const PlateDetect &) = delete;
  PlateDetect &operator=(const PlateDetect &) = delete;

 public:
  virtual ~PlateDetect();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function to get InputWidth of the platedetect network (input image
   * columns).
   *
   * @return InputWidth of the platedetect network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the platedetect network (input image
   *rows).
   *
   *@return InputHeight of the platedetect network.
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
   * @brief Function of get running result of the platedetect network.
   *
   * @param image Input data of input image (cv::Mat) of detected counterpart
   * and resized as inputwidth an outputheight.
   *
   * @return Plate position and plate score.
   */
  virtual PlateDetectResult run(const cv::Mat &image) = 0;
  /**
   * @brief Function to get running results of the platedetect neural network in
   * batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of PLateDetectResult.
   *
   */
  virtual std::vector<PlateDetectResult> run(
      const std::vector<cv::Mat> &images) = 0;
};
}  // namespace ai
}  // namespace vitis
