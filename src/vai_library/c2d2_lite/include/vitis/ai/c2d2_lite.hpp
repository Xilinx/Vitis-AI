/*
 * Copyright 2019 xilinx Inc.
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
 * Filename: C2D2_lite.hpp
 *
 * Description:
 * This network is used to detecting objects from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
namespace vitis {
namespace ai {

/**
 *@brief Base class for detecting objects in the input image(cv::Mat).
 *Input is an image(cv::Mat).
 *Output is the position of the objects in the input image.
 *Sample code:
 *@code
  std::vector<cv::Mat> images;
  for (auto name : image_names) {
    images.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
  }
  auto model = vitis::ai::C2D2_lite::create(C2D2_lite_0_pt, C2D2_lite_1_pt);
  auto result = model->run(images);
  std::cout << result;
  @endcode
 *
 */
class C2D2_lite {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * C2D2_lite.
   * @param model_name0 Model0 name
   * @param model_name1 Model1 name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of C2D2_lite class.
   *
   */
  static std::unique_ptr<C2D2_lite> create(const std::string& model_name0,
                                           const std::string& model_name1,
                                           bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit C2D2_lite();
  C2D2_lite(const C2D2_lite&) = delete;
  virtual ~C2D2_lite();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the C2D2_lite neural network.
   *
   * @param image Input data of input image (std::vector<cv::Mat>).
   *
   * @return A float data.
   *
   */
  virtual float run(const std::vector<cv::Mat>& image) = 0;
  /**
   * @brief Function to get running result of the C2D2_lite neural network
   * in batch mode.
   *
   * @param images Input data of input images
   * (std::vector<std::vector<cv::Mat>>). The size of input images equals batch
   * size obtained by get_input_batch.
   *
   * @return The vector of float data.
   *
   */
  virtual std::vector<float> run(
      const std::vector<std::vector<cv::Mat>>& images) = 0;

  /**
   * @brief Function to get InputWidth of the C2D2_lite network (input image
   * columns).
   *
   * @return InputWidth of the C2D2_lite network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the C2D2_lite network (input image
   *rows).
   *
   *@return InputHeight of the C2D2_lite network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   * time.
   * @note Different DPU core the batch size may be different. This depends on
   * the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
