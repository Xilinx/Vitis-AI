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
 * Filename: Cflownet.hpp
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

struct CflownetResult{
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// (128x128)
  std::vector<float> data;
};

/**
 * @brief Base class for Cflownet (production recognication)
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detected results, named CflownetResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_Cflownet.jpg");
   auto Cflownet = vitis::ai::Cflownet::create("bosch_fcnsemsegt",true);
   auto result = Cflownet->run(img);
   std::cout << result.width <<"\n";
   @endcode
 *
 */
class Cflownet {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Cflownet.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of Cflownet class.
   *
   */
  static std::unique_ptr<Cflownet> create(
      const std::string &model_name, bool need_preprocess = false);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Cflownet();
  Cflownet(const Cflownet &) = delete;

 public:
  virtual ~Cflownet();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the Cflownet network.
   *
   * @param p const float pointer points to input data buffer.
   *
   * @return CflownetResult.
   *
   */
  virtual vitis::ai::CflownetResult run(const float* p) = 0;

  /**
   * @brief Function to get running results of the Cflownet network in
   * batch mode.
   *
   * @param ps const vector of float pointer points to input data buffer.
   * The size of input images need equal to or less than
   * batch size obtained by get_input_batch.
   *
   * @return The vector of CflownetResult.
   *
   */
  virtual std::vector<vitis::ai::CflownetResult> run( const std::vector<const float*> ps) = 0;

  /**
   * @brief Function to get InputWidth of the Cflownet network (input image columns).
   *
   * @return InputWidth of the Cflownet network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the Cflownet network (input image rows).
   *
   *@return InputHeight of the Cflownet network.
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
