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
 * Filename: PMG.hpp
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

struct PMGResult{
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// index num of the class.
  int classidx;
};

/**
 * @brief Base class for PMG (production recognication)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of classification results, named PMGResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_PMG.jpg");
   auto PMG = vitis::ai::PMG::create("pmg_pt",true);
   auto result = PMG->run(img);
   // result is structure holding the classindex .
   std::cout << result.classidx <<"\n";
   @endcode
 *
 */
class PMG {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * PMG.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of PMG class.
   *
   */
  static std::unique_ptr<PMG> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit PMG();
  PMG(const PMG &) = delete;

 public:
  virtual ~PMG();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the PMG network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return PMGResult.
   *
   */
  virtual vitis::ai::PMGResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the PMG network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of PMGResult.
   *
   */
  virtual std::vector<vitis::ai::PMGResult> run(
      const std::vector<cv::Mat> &imgs) = 0;

  /**
   * @brief Function to get InputWidth of the PMG network (input image columns).
   *
   * @return InputWidth of the PMG network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the PMG network (input image rows).
   *
   *@return InputHeight of the PMG network.
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
