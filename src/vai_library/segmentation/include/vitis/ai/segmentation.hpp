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
 * Segmentation for ADAS
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

/// Declaration Segmentation Network
/// number of segmentation classes
/// label 0 name: "unlabeled"
/// label 1 name: "ego vehicle"
/// label 2 name: "rectification border"
/// label 3 name: "out of roi"
/// label 4 name: "static"
/// label 5 name: "dynamic"
/// label 6 name: "ground"
/// label 7 name: "road"
/// label 8 name: "sidewalk"
/// label 9 name: "parking"
/// label 10 name: "rail track"
/// label 11 name: "building"
/// label 12 name: "wall"
/// label 13 name: "fence"
/// label 14 name: "guard rail"
/// label 15 name: "bridge"
/// label 16 name: "tunnel"
/// label 17 name: "pole"

/**
 * @brief Base class for Segmentation.
 *
 * Input is an image (cv:Mat).
 *
 * Output is result of running the Segmentation network.
 *
 * Sample code :
   @code
    auto det =vitis::ai::Segmentation::create("fpn", true);

    auto img= cv::imread("sample_segmentation.jpg");
    int width = det->getInputWidth();
    int height = det->getInputHeight();
    cv::Mat image;
    cv::resize(img, image, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    auto result = det->run_8UC1(image);
    for (auto y = 0; y < result.segmentation.rows; y++) {
      for (auto x = 0; x < result.segmentation.cols; x++) {
            result.segmentation.at<uchar>(y,x) *= 10;
        }
    }
    cv::imwrite("segres.jpg",result.segmentation);

    auto resultshow = det->run_8UC3(image);
    resize(resultshow.segmentation, resultshow.segmentation,
 cv::Size(resultshow.cols * 2, resultshow.rows * 2));
    cv::imwrite("sample_segmentation_result.jpg",resultshow.segmentation);
   @endcode
 *
 * @image latex images/sample_segmentation_result.jpg "segmentation Visualization Result Image" width=\textwidth
 *
 */

class Segmentation {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Segmentation.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Segmentation class.
   *
   */
  static std::unique_ptr<Segmentation> create(const std::string& model_name,
                                              bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Segmentation();
  Segmentation(const Segmentation&) = delete;

 public:
  virtual ~Segmentation();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the segmentation network (input image
   * columns).
   *
   * @return InputWidth of the segmentation network.
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the segmentation network (input image
   * rows).
   *
   * @return InputHeight of the segmentation network.
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
   * @brief Function to get running result of the segmentation network.
   *
   * @note The type of CV_8UC1 of the segmentation result.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return A result that includes segmentation output data.
   *
   */
  virtual SegmentationResult run_8UC1(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC1 of the segmentation result.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of SegmentationResult.
   *
   */
  virtual std::vector<SegmentationResult> run_8UC1(
      const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get running result of the segmentation network.
   *
   * @note The type of CV_8UC3 of the segmentation result.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return A result that include segmentation image and shape;.
   *
   */
  virtual SegmentationResult run_8UC3(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC3 of the segmentation result.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of SegmentationResult.
   *
   */
  virtual std::vector<SegmentationResult> run_8UC3(
      const std::vector<cv::Mat>& images) = 0;
};

/**
 * @brief The Class of Segmentation8UC1. This class run function run(const cv::Mat& image) return a
 cv::Mat with the type is cv_8UC1.
 *Sample code :
   @code
    auto det =
 vitis::ai::Segmentation8UC1::create(vitis::ai::SEGMENTATION_FPN);
   auto img = cv::imread("sample_segmentation.jpg");
    int width = det->getInputWidth();
    int height = det->getInputHeight();
    cv::Mat image;
    cv::resize(img, image, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    auto result = det->run(image);
    for (auto y = 0; y < result.segmentation.rows; y++) {
      for (auto x = 0; x < result.segmentation.cols; x++) {
            result.segmentation.at<uchar>(y,x) *= 10;
        }
    }
    cv::imwrite("segres.jpg",result.segmentation);
   @endcode
 *
 */
class Segmentation8UC1 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Segmentation8UC1.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Segmentation8UC1 class.
   *
   */
  static std::unique_ptr<Segmentation8UC1> create(const std::string& model_name,
                                                  bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Segmentation8UC1(std::unique_ptr<Segmentation> segmentation);
  Segmentation8UC1(const Segmentation8UC1&) = delete;

 public:
  ~Segmentation8UC1();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the segmentation network (input image
   *columns).
   *
   * @return InputWidth of the segmentation network.
   */
  int getInputWidth() const;
  /**
   * @brief Function to get InputHeight of the segmentation network (input image
   *rows).
   *
   * @return InputHeight of the segmentation network.
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
   *@brief Function to get running result of the segmentation network.
   *@note The result cv::Mat of the type is CV_8UC1.
   *@param image  Input data of the image (cv::Mat)
   *@return A Struct of SegmentationResult ,the result of segmentation network.
   */
  SegmentationResult run(const cv::Mat& image);
  /**
   * @brief Function to get running results of the segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC1 of the Result's segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of SegmentationResult.
   *
   */
  std::vector<SegmentationResult> run(const std::vector<cv::Mat>& images);
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<Segmentation> segmentation_;
  /**
   * @endcond
   */
};

/**
 * @brief The Class of Segmentation8UC3, this class run function run(const cv::Mat& image) return a
 cv::Mat with the type is cv_8UC3
 *  Sample code :
   @code
    auto det =
 vitis::ai::Segmentation8UC3::create(vitis::ai::SEGMENTATION_FPN);
   auto img = cv::imread("sample_segmentation.jpg");

    int width = det->getInputWidth();
    int height = det->getInputHeight();
    cv::Mat image;
    cv::resize(img, image, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    auto result = det->run(image);
    cv::imwrite("segres.jpg",result.segmentation);
   @endcode
 *
 */
class Segmentation8UC3 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Segmentation8UC3.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Segmentation8UC3 class.
   *
   */
  static std::unique_ptr<Segmentation8UC3> create(const std::string& model_name,
                                                  bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Segmentation8UC3(std::unique_ptr<Segmentation> segmentation);
  Segmentation8UC3(const Segmentation8UC3&) = delete;

 public:
  ~Segmentation8UC3();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the segmentation network (input image
   *columns).
   *
   * @return InputWidth of the segmentation network.
   */
  int getInputWidth() const;
  /**
   * @brief Function to get InputWidth of the segmentation network (input
   *image rows).
   *
   * @return InputWidth of the segmentation network.
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
   *@brief Function to get running result of the segmentation network.
   *@note The result cv::Mat of the type is CV_8UC3.
   *@param image  Input data of the image (cv::Mat)
   *@return SegmentationResult, the result of the segmentation network.
   */
  SegmentationResult run(const cv::Mat& image);
  /**
   * @brief Function to get running results of the segmentation neural network
   * in batch mode.
   *
   * @note The type of CV_8UC3 of the segmentation result.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of SegmentationResult.
   *
   */
  std::vector<SegmentationResult> run(const std::vector<cv::Mat>& images);
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<Segmentation> segmentation_;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
