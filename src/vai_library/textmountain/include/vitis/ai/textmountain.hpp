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
 * Filename: TextMountain.hpp
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
#include "vitis/ai/nnpp/textmountain.hpp"

namespace vitis {
namespace ai {

/**
 * @brief Base class for TextMountain (text detection)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of classification results, named TextMountainResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_TextMountain.jpg");
   auto TextMountain = vitis::ai::TextMountain::create("textmountain_pt",true);
   auto result = TextMountain->run(img);
   // result is structure holding the text information
   std::cout << result.res.size() <<"\n";
   @endcode
 *
 */
class TextMountain {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * TextMountain.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of TextMountain class.
   *
   */
  static std::unique_ptr<TextMountain> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit TextMountain();
  TextMountain(const TextMountain &) = delete;

 public:
  virtual ~TextMountain();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the TextMountain network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return TextMountainResult.
   *
   */
  virtual vitis::ai::TextMountainResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the TextMountain network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of TextMountainResult.
   *
   */
  virtual std::vector<vitis::ai::TextMountainResult> run(
      const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the TextMountain network (input image columns).
   *
   * @return InputWidth of the TextMountain network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the TextMountain network (input image rows).
   *
   *@return InputHeight of the TextMountain network.
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
