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
 * Filename:segmentation.hpp
 *
 * Description:
 * Covid19Segmentation
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more
 *details of these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/segmentation.hpp>

namespace vitis {
namespace ai {

/// Declaration Covid19Segmentation Network
/// Branch positive detection: label0-negative, label1-positive
/// Branch Infected area detection: label0-negative, label1-consolidation, label2-GGO

/**
 * @brief Base class for Covid19Segmentation.
 *
 * Input is an image (cv:Mat).
 *
 * Output is result of running the Covid19Segmentation network.
 *
 *
 */

class Covid19Segmentation {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Covid19Segmentation.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Covid19Segmentation class.
   *
   */
  static std::unique_ptr<Covid19Segmentation> create(const std::string& model_name,
                                              bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Covid19Segmentation();
  Covid19Segmentation(const Covid19Segmentation&) = delete;

 public:
  virtual ~Covid19Segmentation();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the covid19segmentation network (input image
   * columns).
   *
   * @return InputWidth of the covid19segmentation network.
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the covid19segmentation network (input image
   * rows).
   *
   * @return InputHeight of the covid19segmentation network.
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
   * @brief Function to get running result of the covid19segmentation network.
   *
   * @note The type of CV_8UC1 of the covid19segmentation result.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return Covid19segmentation output data.
   *
   */
  virtual Covid19SegmentationResult run_8UC1(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the covid19segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC1 of the covid19segmentation result.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of Covid19segmentationResult.
   *
   */
  virtual std::vector<Covid19SegmentationResult> run_8UC1(
      const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get running result of the covid19segmentation network.
   *
   * @note The type of CV_8UC3 of the covid19segmentation result.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return Covid19segmentation image and shape.
   *
   */
  virtual Covid19SegmentationResult run_8UC3(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the covid19segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC3 of the Result's covid19segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of Covid19SegmentationResult.
   *
   */
  virtual std::vector<Covid19SegmentationResult> run_8UC3(
      const std::vector<cv::Mat>& images) = 0;
};

/**
 * @brief The Class of Covid19Segmentation8UC1, this class run function return a
 cv::Mat with the type is cv_8UC1
 *
 */
class Covid19Segmentation8UC1 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Covid19Segmentation8UC1.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Covid19Segmentation8UC1 class.
   *
   */
  static std::unique_ptr<Covid19Segmentation8UC1> create(const std::string& model_name,
                                                  bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Covid19Segmentation8UC1(std::unique_ptr<Covid19Segmentation> covid19segmentation);
  Covid19Segmentation8UC1(const Covid19Segmentation8UC1&) = delete;

 public:
  ~Covid19Segmentation8UC1();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the covid19segmentation network (input image
   *columns).
   *
   * @return InputWidth of the covid19segmentation network.
   */
  int getInputWidth() const;
  /**
   * @brief Function to get InputHeight of the covid19segmentation network (input image
   *rows).
   *
   * @return InputHeight of the covid19segmentation network.
   */
  int getInputHeight() const;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  size_t get_input_batch() const;

  /**
   *@brief Function to get running result of the covid19segmentation network.
   *@note The result cv::Mat of the type is CV_8UC1.
   *@param image  Input data of the image (cv::Mat)
   *@return The result of covid19segmentation network.
   */
  Covid19SegmentationResult run(const cv::Mat& image);
  /**
   * @brief Function to get running results of the covid19segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC1 of the Result's covid19segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of Covid19SegmentationResult.
   *
   */
  std::vector<Covid19SegmentationResult> run(const std::vector<cv::Mat>& images);
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<Covid19Segmentation> covid19segmentation_;
  /**
   * @endcond
   */
};

/**
 * @brief The Class of Covid19Segmentation8UC3, this class run function return a
 cv::Mat with the type is cv_8UC3
 *
 */
class Covid19Segmentation8UC3 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Covid19Segmentation8UC3.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Covid19Segmentation8UC3 class.
   *
   */
  static std::unique_ptr<Covid19Segmentation8UC3> create(const std::string& model_name,
                                                  bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Covid19Segmentation8UC3(std::unique_ptr<Covid19Segmentation> covid19segmentation);
  Covid19Segmentation8UC3(const Covid19Segmentation8UC3&) = delete;

 public:
  ~Covid19Segmentation8UC3();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the covid19segmentation network (input image
   *columns).
   *
   * @return InputWidth of the covid19segmentation network.
   */
  int getInputWidth() const;
  /**
   * @brief Function to get InputWidth of the covid19segmentation network (input
   *image
   *rows).
   *
   * @return InputWidth of the covid19segmentation network.
   */
  int getInputHeight() const;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  size_t get_input_batch() const;

  /**
   *@brief Function to get running result of the covid19segmentation network.
   *@note The result cv::Mat of the type is CV_8UC3.
   *@param image  Input data of the image (cv::Mat)
   *@return Covid19SegmentationResult The result of covid19segmentation network.
   */
  Covid19SegmentationResult run(const cv::Mat& image);
  /**
   * @brief Function to get running results of the covid19segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC3 of the Result's covid19segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of Covid19SegmentationResult.
   *
   */
  std::vector<Covid19SegmentationResult> run(const std::vector<cv::Mat>& images);
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<Covid19Segmentation> covid19segmentation_;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
