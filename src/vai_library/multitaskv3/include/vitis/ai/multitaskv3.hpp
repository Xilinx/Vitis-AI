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
 * Filename: multitaskv3.hpp
 *
 * Description:
 * This module implement MultiTaskv3 Network for ADAS, include detection,
 *segmentation, lane, drivable area, depth estimation;
 *
 * Please refer to document "vitis_XILINX_AI_SDK_user_guide.pdf" for more
 *details of these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/multitaskv3.hpp>

namespace vitis {
namespace ai {

#if 0
 class and corresponding name
 label: 0 name: "car"
 label: 1 name: "sign"
 label: 2 name: "person"
#endif

/**
 * @brief Base class for ADAS MuiltTask from an image (cv::Mat).
 *
 * Input an image (cv::Mat).
 *
 * Output is a struct of MultiTaskv3Result including segmentation results, 
 detection results and vehicle towards;
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTaskv3::create("multi_task");
    auto image = cv::imread("sample_multitaskv3.jpg");
    auto result = det->run_8UC3(image);
    cv::imwrite("sample_multitaskv3_result.jpg",result.segmentation);
    cv::imwrite("sample_multitaskv3_result.jpg",result.depth);
   @endcode
 *
 *
 * Display of the model results:
 * @image latex images/sample_multitaskv3_result.jpg "result image" width=\textwidth
 *
 */
class MultiTaskv3 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Multitaskv3.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Multitaskv3 class.
   *
   */
  static std::unique_ptr<MultiTaskv3> create(const std::string& model_name,
                                           bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTaskv3();
  MultiTaskv3(const MultiTaskv3&) = delete;

 public:
  virtual ~MultiTaskv3();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitaskv3 network (input image
   *columns).
   *
   * @return InputWidth of the multitaskv3 network.
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the multitaskv3 network (input image
   *rows).
   *
   * @return InputHeight of the multitaskv3 network.
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
   * @brief Function of get running result from the MultiTaskv3 network.
   * @note The type is CV_8UC1 of the MultiTaskv3Result.segmentation and all cv::Mat output.
   * @param image Input image
   * @return The struct of MultiTaskv3Result
   */
  virtual MultiTaskv3Result run_8UC1(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the MultiTaskv3 neural network in
   * batch mode.
   * @note The type is CV_8UC1 of all cv::Mat output.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskv3Result.
   *
   */
  virtual std::vector<MultiTaskv3Result> run_8UC1(
      const std::vector<cv::Mat>& images) = 0;

  /**
   * @brief Function to get running result from the MultiTaskv3 network.
   * @note The type is CV_8UC3 of all cv::Mat result except depth estimation.
   *@param image Input image;
   * @return The struct of MultiTaskv3Result
   */
  virtual MultiTaskv3Result run_8UC3(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the MultiTaskv3 neural network in
   * batch mode.
   * @note The type is CV_8UC3 of all cv::Mat result except depth estimation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskv3Result.
   *
   */
  virtual std::vector<MultiTaskv3Result> run_8UC3(
      const std::vector<cv::Mat>& images) = 0;
};

/**
 * @brief Base class for ADAS MuiltTask8UC1 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskv3Result including segmentation results, detection
 results and vehicle towards; The result cv::Mat type is CV_8UC1
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTaskv38UC1::create(vitis::ai::MULTITASKv3);
    auto image = cv::imread("sample_multitaskv3.jpg");
    auto result = det->run(image);
    cv::imwrite("res.jpg",result.segmentation);
   @endcode
  */
class MultiTaskv38UC1 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MultiTaskv38UC1.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of MultiTaskv38UC1 class.
   *
   */
  static std::unique_ptr<MultiTaskv38UC1> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTaskv38UC1>(
        new MultiTaskv38UC1(MultiTaskv3::create(model_name, need_preprocess)));
  }

  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTaskv38UC1(std::unique_ptr<MultiTaskv3> multitaskv3)
      : multitaskv3_{std::move(multitaskv3)} {}
  MultiTaskv38UC1(const MultiTaskv38UC1&) = delete;

 public:
  virtual ~MultiTaskv38UC1() {}
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitaskv3 network (input image
   *columns).
   *
   * @return InputWidth of the multitaskv3 network.
   */
  virtual int getInputWidth() const { return multitaskv3_->getInputWidth(); }
  /**
   * @brief Function to get InputHeight of the multitaskv3 network (input image
   *rows).
   *
   * @return InputHeight of the multitaskv3 network.
   */
  virtual int getInputHeight() const { return multitaskv3_->getInputHeight(); }

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   *
   * @note Different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const {
    return multitaskv3_->get_input_batch();
  }

  /**
   * @brief Function of get running result from the MultiTaskv3 network.
   * @note The type is CV_8UC1 of the MultiTaskv3Result.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskv3Result
   */
  virtual MultiTaskv3Result run(const cv::Mat& image) {
    return multitaskv3_->run_8UC1(image);
  }
  /**
   * @brief Function to get running results of the MultiTaskv3 neural network in
   * batch mode.
   * @note The type is CV_8UC1 of the MultiTaskv3Result.segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskv3Result.
   *
   */
  virtual std::vector<MultiTaskv3Result> run(const std::vector<cv::Mat>& images) {
    return multitaskv3_->run_8UC1(images);
  }
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<MultiTaskv3> multitaskv3_;
  /**
   * @endcond
   */
};

/**
 * @brief Base class for ADAS MuiltTask8UC3 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskv3Result including segmentation results, detection
 results and vehicle orientation; The result cv::Mat type is CV_8UC3(except depth estimation)
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTaskv38UC3::create(vitis::ai::MULITASK);
    auto image = cv::imread("sample_multitaskv3.jpg");
    auto result = det->run(image);
    cv::imwrite("res.jpg",result.segmentation);
   @endcode
  */
class MultiTaskv38UC3 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MultiTaskv38UC3.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of MultiTaskv38UC3 class.
   *
   */
  static std::unique_ptr<MultiTaskv38UC3> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTaskv38UC3>(
        new MultiTaskv38UC3(MultiTaskv3::create(model_name, need_preprocess)));
  }
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTaskv38UC3(std::unique_ptr<MultiTaskv3> multitaskv3)
      : multitaskv3_{std::move(multitaskv3)} {}
  MultiTaskv38UC3(const MultiTaskv38UC3&) = delete;

 public:
  virtual ~MultiTaskv38UC3() {}
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitaskv3 network (input image
   *columns).
   *
   * @return InputWidth of the multitaskv3 network.
   */
  virtual int getInputWidth() const { return multitaskv3_->getInputWidth(); }
  /**
   * @brief Function to get InputHeight of the multitaskv3 network (input image
   *rows).
   *
   * @return InputHeight of the multitaskv3 network.
   */
  virtual int getInputHeight() const { return multitaskv3_->getInputHeight(); }
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const {
    return multitaskv3_->get_input_batch();
  }

  /**
   * @brief Function of get running result from the MultiTaskv3 network.
   * @note The type is CV_8UC3 of the MultiTaskv3Result.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskv3Result
   */
  virtual MultiTaskv3Result run(const cv::Mat& image) {
    return multitaskv3_->run_8UC3(image);
  }
  /**
   * @brief Function to get running results of the MultiTaskv3 neural network in
   * batch mode.
   * @note The type is CV_8UC3 of the MultiTaskv3Result.segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskv3Result.
   *
   */
  virtual std::vector<MultiTaskv3Result> run(const std::vector<cv::Mat>& images) {
    return multitaskv3_->run_8UC3(images);
  }
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<MultiTaskv3> multitaskv3_;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
