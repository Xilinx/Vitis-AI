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

 * Filename: hourglass.hpp
 *
 * Description:
 * This network is used to detecting poses from a input image.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details.
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/hourglass.hpp>

namespace vitis {
namespace ai {

/// Hourglass model, input size is 256x256.

/**
 * @brief Base class for detecting poses of people.
 *
 * Input is an image (cv:Mat).
 *
 * Output is HourglassResult.
 *
 * Sample code:
   @code
  auto image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "cannot load " << argv[2] << std::endl;
    abort();
  }
  auto det = vitis::ai::Hourglass::create(argv[1]);
  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 6},  {3, 6},  {3, 4}, {4, 5},
                                 {6, 7},   {7, 8},  {8, 9}, {7, 12},
                                 {12, 11}, {11, 10}, {7, 13}, {13, 14}, {14, 15}};

  auto results = det->run(image.clone());
  for (size_t i = 0; i < results.poses.size(); ++i) {
    cout<< results.poses[i].point<<endl;
    if (results.poses[i].type == 1) {
      cv::circle(image, results.poses[i].point, 5, cv::Scalar(0, 255, 0),
                 -1);
    }
  }
  for (size_t i = 0; i < limbSeq.size(); ++i) {
    Result a = results.poses[limbSeq[i][0]];
    Result b = results.poses[limbSeq[i][1]];
    if (a.type == 1 && b.type == 1) {
      cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
    }
  }
   @endcode
 *
 * Display of the hourglass model results:
 * @image latex images/sample_hourglass_result.jpg "hourglass result image" width=400px
 */

class Hourglass {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Hourglass.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is true.
   * @return An instance of Hourglass class.
   *
   */
  static std::unique_ptr<Hourglass> create(const std::string &model_name,
                                          bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit Hourglass();
  Hourglass(const Hourglass &) = delete;
  virtual ~Hourglass();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the hourglass neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return HourglassResult.
   *
   */
  virtual HourglassResult run(const cv::Mat &image) = 0;
  /**
   * @brief Function to get running results of the hourglass neural
   * network in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of HourglassResult.
   *
   */
  virtual std::vector<HourglassResult> run(
      const std::vector<cv::Mat> &images) = 0;
  /**
   * @brief Function to get InputWidth of the hourglass network (input image
   * columns).
   *
   * @return InputWidth of the hourglass network
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the hourglass network (input image
   * rows).
   *
   * @return InputHeight of the hourglass network.
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
};
}  // namespace ai
}  // namespace vitis
