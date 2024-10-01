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
 * Filename: multitask.hpp
 *
 * Description:
 * This module implement MultiTask Network for ADAS, include detection,
 *segmentation and car towards angle;
 *
 * Please refer to document "vitis_XILINX_AI_SDK_user_guide.pdf" for more
 *details of these APIs.
 */

#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/multitask.hpp>

namespace vitis {
namespace ai {

#if 0
 class and corresponding name
 label: 0 name: "background"
 label: 1 name: "person"
 label: 2 name: "car"
 label: 3 name: "truck"
 label: 4 name: "bus"
 label: 5 name: "bike"
 label: 6 name: "sign"
 label: 7 name: "light"
#endif

/**
 * @brief Base class for ADAS MuiltTask from an image (cv::Mat).
 *
 * Input an image (cv::Mat).
 *
 * Output is a struct of MultiTaskResult includes segmentation results, 
 detection results and vehicle towards;
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTask::create("multi_task");
    auto image = cv::imread("sample_multitask.jpg");
    auto result = det->run_8UC3(image);
    cv::imwrite("sample_multitask_result.jpg",result.segmentation);
   @endcode
 *
 *
 * Display of the model results:
 * @image latex images/sample_multitask_result.jpg "result image" width=\textwidth
 *
 */
class MultiTask {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Multitask.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Multitask class.
   *
   */
  static std::unique_ptr<MultiTask> create(const std::string& model_name,
                                           bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTask();
  MultiTask(const MultiTask&) = delete;

 public:
  virtual ~MultiTask();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *columns).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const = 0;
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note For different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const = 0;

  /**
   * @brief Function to get running result from the MultiTask network.
   * @note The type is CV_8UC1 of the MultiTaskResult.segmentation.
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run_8UC1(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the MultiTask neural network in
   * batch mode.
   * @note The type is CV_8UC1 of the MultiTaskResult.segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskResult.
   *
   */
  virtual std::vector<MultiTaskResult> run_8UC1(
      const std::vector<cv::Mat>& images) = 0;

  /**
   * @brief Function to get running result from the MultiTask network.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *@param image Input image;
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run_8UC3(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the MultiTask neural network in
   * batch mode.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskResult.
   *
   */
  virtual std::vector<MultiTaskResult> run_8UC3(
      const std::vector<cv::Mat>& images) = 0;
};

/**
 * @brief Base class for ADAS MuiltTask8UC1 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskResult includes segmentation results, detection
 results and vehicle towards; The result cv::Mat type is CV_8UC1
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTask8UC1::create(vitis::ai::MULTITASK);
    auto image = cv::imread("sample_multitask.jpg");
    auto result = det->run(image);
    cv::imwrite("res.jpg",result.segmentation);
   @endcode
  */
class MultiTask8UC1 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MultiTask8UC1.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of MultiTask8UC1 class.
   *
   */
  static std::unique_ptr<MultiTask8UC1> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTask8UC1>(
        new MultiTask8UC1(MultiTask::create(model_name, need_preprocess)));
  }

  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTask8UC1(std::unique_ptr<MultiTask> multitask)
      : multitask_{std::move(multitask)} {}
  MultiTask8UC1(const MultiTask8UC1&) = delete;

 public:
  virtual ~MultiTask8UC1() {}
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *columns).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const { return multitask_->getInputWidth(); }
  /**
   * @brief Function to get InputHeight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const { return multitask_->getInputHeight(); }

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   *
   * @note For different DPU core the batch size may be differnt. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const {
    return multitask_->get_input_batch();
  }

  /**
   * @brief Function to get running result from the MultiTask network.
   * @note The type is CV_8UC1 of the MultiTaskResult.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run(const cv::Mat& image) {
    return multitask_->run_8UC1(image);
  }
  /**
   * @brief Function to get running results of the MultiTask neural network in
   * batch mode.
   * @note The type of the MultiTaskResult.segmentation is CV_8UC1 .
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskResult.
   *
   */
  virtual std::vector<MultiTaskResult> run(const std::vector<cv::Mat>& images) {
    return multitask_->run_8UC1(images);
  }
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<MultiTask> multitask_;
  /**
   * @endcond
   */
};

/**
 * @brief Base class for ADAS MuiltTask8UC3 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskResult includes segmentation results, detection
 results and vehicle orientation; The result cv::Mat type is CV_8UC3
 *
 * Sample code:
 * @code
    auto det = vitis::ai::MultiTask8UC3::create(vitis::ai::MULITASK);
    auto image = cv::imread("sample_multitask.jpg");
    auto result = det->run(image);
    cv::imwrite("res.jpg",result.segmentation);
   @endcode
  */
class MultiTask8UC3 {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MultiTask8UC3.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of MultiTask8UC3 class.
   *
   */
  static std::unique_ptr<MultiTask8UC3> create(const std::string& model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTask8UC3>(
        new MultiTask8UC3(MultiTask::create(model_name, need_preprocess)));
  }
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTask8UC3(std::unique_ptr<MultiTask> multitask)
      : multitask_{std::move(multitask)} {}
  MultiTask8UC3(const MultiTask8UC3&) = delete;

 public:
  virtual ~MultiTask8UC3() {}
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *columns).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const { return multitask_->getInputWidth(); }
  /**
   * @brief Function to get InputHeight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const { return multitask_->getInputHeight(); }
  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note For different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   * @return Batch size.
   */
  virtual size_t get_input_batch() const {
    return multitask_->get_input_batch();
  }

  /**
   * @brief Function to get running result from the MultiTask network.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run(const cv::Mat& image) {
    return multitask_->run_8UC3(image);
  }
  /**
   * @brief Function to get running results of the MultiTask neural network in
   * batch mode.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MultiTaskResult.
   *
   */
  virtual std::vector<MultiTaskResult> run(const std::vector<cv::Mat>& images) {
    return multitask_->run_8UC3(images);
  }
  /**
   * @cond NOCOMMENTS
   */
 private:
  std::unique_ptr<MultiTask> multitask_;
  /**
   * @endcond
   */
};

}  // namespace ai
}  // namespace vitis
