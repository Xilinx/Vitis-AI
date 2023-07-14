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
 * Filename: Monodepth2.hpp
 *
 * Description:
 * This network is used to classify the object from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

struct Monodepth2Result{
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// cv::Mat of returned pic
  cv::Mat mat;
};

/**
 * @brief Base class for Monodepth2 (production segmentation)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of segmentation results, named Monodepth2Result.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_monodepth2.jpg");
   auto Monodepth2 = vitis::ai::Monodepth2::create("monodepth2_pt",true);
   auto result = Monodepth2->run(img);
   // result is structure holding the mat.
   std::cout << result.mat.cols <<"\n";
   @endcode
 *
 */
class Monodepth2 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Monodepth2.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of Monodepth2 class.
   *
   */
  static std::unique_ptr<Monodepth2> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Monodepth2();
  Monodepth2(const Monodepth2 &) = delete;

 public:
  virtual ~Monodepth2();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the Monodepth2 network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return Monodepth2Result.
   *
   */
  virtual vitis::ai::Monodepth2Result run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the Monodepth2 network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of Monodepth2Result.
   *
   */
  virtual std::vector<vitis::ai::Monodepth2Result> run(
      const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the Monodepth2 network (input image columns).
   *
   * @return InputWidth of the Monodepth2 network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the Monodepth2 network (input image rows).
   *
   *@return InputHeight of the Monodepth2 network.
   */

  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
