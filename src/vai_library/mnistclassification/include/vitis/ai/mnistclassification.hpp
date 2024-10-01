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
 * Filename: mnistclassification.hpp
 *
 * Description:
 * This network is used to classify the number from a input image.
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
 * @brief Base class for classification for Mnist dataset .
 *
 * Input is an image (cv:Mat) which must be cv::IMREAD_GRAYSCALE style
 *
 * Output is a struct of detection results, named MnistClassificationResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_mnistclassification.jpg", cv::IMREAD_GRAYSCALE);
   auto mnistclassification =
   vitis::ai::MnistClassification::create("MNIST-Classification-TensorFlow",true); 
   auto results = mnistclassification->run(img); 
   // please check test samples for detail usage.
   @endcode
 *
 */

struct MnistClassificationResult{
  // Weight of input image.
  int width;
  // Height of input image.
  int height;
  // class idx. 0--9
  int classIdx;
};

class MnistClassification {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MnistClassification.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of MnistClassification class.
   *
   */
  static std::unique_ptr<MnistClassification> create(
      const std::string &model_name, bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MnistClassification();
  MnistClassification(const MnistClassification &) = delete;

 public:
  virtual ~MnistClassification();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the MnistClassification neural network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return MnistClassificationResult.
   *
   */
  virtual vitis::ai::MnistClassificationResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function to get running results of the MnistClassification neural network in
   * batch mode.
   *
   * @param images Input data of input images (vector<cv::Mat>).The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MnistClassificationResult.
   *
   */
  virtual std::vector<vitis::ai::MnistClassificationResult> run(
      const std::vector<cv::Mat> &img) = 0;

  /**
   * @brief Function to get InputWidth of the MnistClassification network (input image cols).
   *
   * @return InputWidth of the MnistClassification network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the MnistClassification network (input image rows).
   *
   *@return InputHeight of the MnistClassification network.
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
