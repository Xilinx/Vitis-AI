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
 * Filename: cifar10classification.hpp
 *
 * Description:
 * This network is used to classificy object from a input image for CIFAR10 dataset.
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

/**
 * @brief Base class for classificy object from input image
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named Cifar10ClassificationResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_cifar10classification.jpg");
   auto cifar10classification =
   vitis::ai::Cifar10Classification::create("CIFAR10-Classification-with-TensorFlow",true); 
   auto result = cifar10classification->run(img); 
   // please check test samples for detail usage.
   @endcode
 *
 */

struct Cifar10ClassificationResult{
  // Weight of input image.
  int width;
  // Height of input image.
  int height;
  // class idx.
  int classIdx;
};

class Cifar10Classification {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Cifar10Classification.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of Cifar10Classification class.
   *
   */
  static std::unique_ptr<Cifar10Classification> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Cifar10Classification();
  Cifar10Classification(const Cifar10Classification &) = delete;

 public:
  virtual ~Cifar10Classification();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the Cifar10Classification neuron network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return Cifar10ClassificationResult.
   *
   */
  virtual vitis::ai::Cifar10ClassificationResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the Cifar10Classification neuron network in
   * batch mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of Cifar10ClassificationResult.
   *
   */
  virtual std::vector<vitis::ai::Cifar10ClassificationResult> run(
      const std::vector<cv::Mat> &img) = 0;

  /**
   * @brief Function to get InputWidth of the Cifar10Classification network (input image cols).
   *
   * @return InputWidth of the Cifar10Classification network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeigth of the Cifar10Classification network (input image rows).
   *
   *@return InputHeight of the Cifar10Classification network.
   */

  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
