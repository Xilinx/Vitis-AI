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
 * Filename: platenum.hpp
 * Description:
 * This network is used to recognized plate number from input image
 * Please refer to doc ument "xilinx_XXXX_user_guide.pdf" for more details of
 * these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/platenum.hpp>

namespace xir {
  class Attrs;
};
namespace vitis {
namespace ai {

/**
 * @brief Base class for recognizing plate from an image (cv::Mat).
 *
 * Input is a plate image (cv::Mat).
 *
 * Output is the number and color of plate in the input image.
 *
 * @note
    Only China plate
    Only edge platform supported
   @endnote
 *
 * sample code:
 * @code
   cv::Mat image = cv::imread("plate.jpg");
   auto network = vitis::ai::PlateNum::create(true);
   auto r = network->run(image);
   auto plate_number = r.plate_number;
   auto plate_color = r.plate_color;
   @endcode
 *
 */
class PlateNum {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PlateNum.
   *
   * @param model_name the model name of the created model
   *
   * @param need_mean_scale_process normalize with mean/scale or not, true by
   * default.
   *
   * @return An instance of PlateNum class.
   */
  static std::unique_ptr<PlateNum> create(const std::string &model_name,
                                          bool need_mean_scale_process = true);
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PlateNum.
   *
   * @param model_name the model name of the created model
   *
   * @param attrs xir::Attrs pointer points to the provided attributes 
   * 
   * @param need_mean_scale_process normalize with mean/scale or not, true by
   * default.
   *
   * @return An instance of PlateNum class.
   */
  static std::unique_ptr<PlateNum> create(const std::string &model_name,
                                          xir::Attrs *attrs,
                                          bool need_mean_scale_process = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit PlateNum();
  PlateNum(const PlateNum &) = delete;
  PlateNum &operator=(const PlateNum &) = delete;

 public:
  virtual ~PlateNum();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function to get InputWidth of the platenum network (input image
   * columns).
   *
   * @return InputWidth of the platenum network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the platenum network (input image
   *rows).
   *
   *@return InputHeight of the platenum network.
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
   * @brief Function of get running result of platenum network
   *
   * @param img Input data of input image (cv::Mat)
   * and resized as InputWidth and IntputHeight.
   *
   * @return The plate number and plate color.
   */
  virtual PlateNumResult run(const cv::Mat &img) = 0;
  /**
   * @brief Function to get running results of the platenum neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of PLateNumResult.
   *
   */
  virtual std::vector<PlateNumResult> run(const std::vector<cv::Mat> &imgs) = 0;
};
}  // namespace ai
}  // namespace vitis
