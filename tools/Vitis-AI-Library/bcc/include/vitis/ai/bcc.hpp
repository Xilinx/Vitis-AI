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
/*
 * Filename: BCC.hpp
 *
 * Description:
 * This network is used to get count of crowd from a input image.
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

struct BCCResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Count of crowd.
  int count;
};

/**
 * @brief Base class for BCC (Bayesian crowd counting)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named BCCResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_BCC.jpg");
   auto BCC = vitis::ai::BCC::create("bcc_pt",true);
   auto result = BCC->run(img);
   std::cout << result.count << "\n";
   @endcode
 *
 */
class BCC {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * BCC.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of BCC class.
   *
   */
  static std::unique_ptr<BCC> create(const std::string& model_name,
                                     bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit BCC();
  BCC(const BCC&) = delete;

 public:
  virtual ~BCC();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the BCC neural network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return BCCResult.
   *
   */
  virtual vitis::ai::BCCResult run(const cv::Mat& img) = 0;

  /**
   * @brief Function to get running results of the BCC neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of BCCResult.
   *
   */
  virtual std::vector<vitis::ai::BCCResult> run(
      const std::vector<cv::Mat>& imgs) = 0;

  /**
   * @brief Function to get InputWidth of the BCC network (input image columns).
   *
   * @return InputWidth of the BCC network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the BCC network (input image rows).
   *
   *@return InputHeight of the BCC network.
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
