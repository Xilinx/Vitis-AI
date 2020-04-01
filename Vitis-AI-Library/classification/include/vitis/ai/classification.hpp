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
 * Filename: classification.hpp
 *
 * Description:
 * This network is used to getting classification in the input image
 * Please refer to document "Xilinx_AI_SDK_User_guide.pdf" for more details of
 * these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/nnpp/classification.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting objects in the input image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is index and score of objects in the input image.
 *
 * Sample code:
 * @code
   auto image = cv::imread("sample_classification.jpg");
   auto network = vitis::ai::Classification::create(
                  "resnet_50",
                  true);
   auto result = network->run(image);
   for (const auto &r : result.scores) {
      auto score = r.score;
      auto index = network->lookup(r.index);
   }
   @endcode
 *
 */

class Classification {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Classification.
   *
   * @param model_name Model name.
   * @param need_preprocess Normalize with mean/scale or not,
   *default value is true.
   * @return An instance of Classification class.
   *
   */
  static std::unique_ptr<Classification> create(const std::string& model_name,
                                                bool need_preprocess = true);

  /**
   * @brief Get the classification corresponding by index
   * @param index The network result
   * @return Classification description, if index < 0, return empty string
   */
  static const char* lookup(int index);

 public:
   /**
   * @cond NOCOMMENTS
   */
  explicit Classification();
  Classification(const Classification&) = delete;
  virtual ~Classification();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running results of the classification neuron
   * network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return ClassificationResult.
   *
   */
  virtual vitis::ai::ClassificationResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the classification neuron
   * network in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>). The size of 
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of ClassificationResult.
   *
   */
  virtual std::vector<vitis::ai::ClassificationResult> run(
      const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get InputWidth of the classification network (input
   *image cols).
   *
   * @return InputWidth of the classification network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeigth of the classification network (input
   *image rows).
   *
   *@return InputHeight of the classification network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};

}  // namespace ai
}  // namespace vitis
