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
 * Filename: multitask.hpp
 *
 * Description:
 * This module implement MultiTask Network for ADAS, include detection,
 * segmentation and car towards angle;
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <xilinx/ai/nnpp/multitask.hpp>

namespace xilinx {
namespace ai {
/**
 *@brief Multitask Network Type , declaration multitask Network
 */
/// number of classes
/// label: 0 name: "background"
/// label: 1 name: "person"
/// label: 2 name: "car"
/// label: 3 name: "truck"
/// label: 4 name: "bus"
/// label: 5 name: "bike"
/// label: 6 name: "sign"
/// label: 7 name: "light"

/**
 * @brief Base class for ADAS MuiltTask from an image (cv::Mat).
 *
 * Input an image (cv::Mat).
 *
 * Output is a struct of MultiTaskResult include segmentation results, detection
 detection results and vehicle towards;
 *
 * Sample code:
 * @code
    auto det = xilinx::ai::MultiTask::create("multi_task");
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
  static std::unique_ptr<MultiTask> create(const std::string &model_name,
                                           bool need_preprocess = true);

protected:
  explicit MultiTask();
  MultiTask(const MultiTask &) = delete;

public:
  virtual ~MultiTask();

public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *cols).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function of get running result from the MultiTask network.
   * @note The type is CV_8UC1 of the MultiTaskResult.segmentation.
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run_8UC1(const cv::Mat &image) = 0;

  /**
   * @brief Function of get running result from the MultiTask network.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *@param image Input image;
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run_8UC3(const cv::Mat &image) = 0;
};

/**
 * @brief Base class for ADAS MuiltTask8UC1 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskResult include segmentation results, detection
 results and vehicle towards; The result cv::Mat type is CV_8UC1
 *
 * Sample code:
 * @code
    auto det = xilinx::ai::MultiTask8UC1::create(xilinx::ai::MULTITASK);
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
  static std::unique_ptr<MultiTask8UC1> create(const std::string &model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTask8UC1>(
        new MultiTask8UC1(MultiTask::create(model_name, need_preprocess)));
  }

protected:
  explicit MultiTask8UC1(std::unique_ptr<MultiTask> multitask)
      : multitask_{std::move(multitask)} {}
  MultiTask8UC1(const MultiTask8UC1 &) = delete;

public:
  virtual ~MultiTask8UC1() {}

public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *cols).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const { return multitask_->getInputWidth(); }
  /**
   * @brief Function to get InputHight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const { return multitask_->getInputHeight(); }

  /**
   * @brief Function of get running result from the MultiTask network.
   * @note The type is CV_8UC1 of the MultiTaskResult.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run(const cv::Mat &image) {
    return multitask_->run_8UC1(image);
  }

private:
  std::unique_ptr<MultiTask> multitask_;
};

/**
 * @brief Base class for ADAS MuiltTask8UC3 from an image (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is struct MultiTaskResult include segmentation results, detection
 results and vehicle towards; The result cv::Mat type is CV_8UC3
 *
 * Sample code:
 * @code
    auto det = xilinx::ai::MultiTask8UC3::create(xilinx::ai::MULITASK);
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
  static std::unique_ptr<MultiTask8UC3> create(const std::string &model_name,
                                               bool need_preprocess = true) {
    return std::unique_ptr<MultiTask8UC3>(
        new MultiTask8UC3(MultiTask::create(model_name, need_preprocess)));
  }

protected:
  explicit MultiTask8UC3(std::unique_ptr<MultiTask> multitask)
      : multitask_{std::move(multitask)} {}
  MultiTask8UC3(const MultiTask8UC3 &) = delete;

public:
  virtual ~MultiTask8UC3() {}

public:
  /**
   * @brief Function to get InputWidth of the multitask network (input image
   *cols).
   *
   * @return InputWidth of the multitask network.
   */
  virtual int getInputWidth() const { return multitask_->getInputWidth(); }
  /**
   * @brief Function to get InputHight of the multitask network (input image
   *rows).
   *
   * @return InputHeight of the multitask network.
   */
  virtual int getInputHeight() const { return multitask_->getInputHeight(); }

  /**
   * @brief Function of get running result from the MultiTask network.
   * @note The type is CV_8UC3 of the MultiTaskResult.segmentation.
   *
   * @param image Input image
   * @return The struct of MultiTaskResult
   */
  virtual MultiTaskResult run(const cv::Mat &image) {
    return multitask_->run_8UC3(image);
  }

private:
  std::unique_ptr<MultiTask> multitask_;
};

} // namespace ai
} // namespace xilinx
