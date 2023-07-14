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

 * Filename: movenet.hpp
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

namespace vitis {
namespace ai {

/// Movenet model, input size is 192x192.

/**
 * @brief Base class for detecting poses of people.
 *
 * Input is an image (cv:Mat).
 *
 * Output is MovenetResult.
 *
 * Sample code:
   @code
   auto image = cv::imread(argv[2]);
   if (image.empty()) {
     std::cerr << "cannot load " << argv[2] << std::endl;
     abort();
   }
   auto det = vitis::ai::Movenet::create(argv[1]);
   vector<vector<int>> limbSeq = {{0, 1}, {0, 2},{0, 3},{0, 4},{0, 5},{0, 6},
                                 {5, 7},  {7, 9},  {6, 8}, {8, 10},
                                  {5, 11},   {6, 12},  {11, 13}, {13, 15},
                                  {12, 14}, {14, 16}};

   auto results = det->run(image.clone());
   for (size_t i = 0; i < results.poses.size(); ++i) {
     cout<< results.poses[i]<<endl;
     if (results.poses[i].y >0 && results.poses[i].x > 0) {
       cv::putText(image, to_string(i),results.poses[i],
       cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 1, 0);
       cv::circle(image, results.poses[i], 5, cv::Scalar(0, 255, 0),
                  -1);
     }
   }
   for (size_t i = 0; i < limbSeq.size(); ++i) {
     auto a = results.poses[limbSeq[i][0]];
     auto b = results.poses[limbSeq[i][1]];
     if (a.x >0  && b.x > 0) {
       cv::line(image, a, b, cv::Scalar(255, 0, 0), 3, 4);
     }
   }

   @endcode
 *
 * Display of the movenet model results:
 * @image latex images/sample_movenet_result.jpg "movenet result image"
 width=400px
 */

struct MovenetResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// A vector of pose, pose is represented by a vector of Point.
  /// Joint points are arranged in order
  /// 0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
  /// 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8:
  /// 'right_elbow', 9: 'left_wrist', 10 : 'right_wrist', 11: 'left_hip', 12:
  /// 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16:
  /// 'right_ankle']
  std::vector<cv::Point2f> poses;
};

class Movenet {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Movenet.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default value is
   * true.
   * @return An instance of Movenet class.
   *
   */
  static std::unique_ptr<Movenet> create(const std::string& model_name,
                                         bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit Movenet();
  Movenet(const Movenet&) = delete;
  virtual ~Movenet();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the movenet neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return MovenetResult.
   *
   */
  virtual MovenetResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the movenet neural
   * network in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>). The size
   * of input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MovenetResult.
   *
   */
  virtual std::vector<MovenetResult> run(
      const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get InputWidth of the movenet network (input image
   * columns).
   *
   * @return InputWidth of the movenet network
   */
  virtual int getInputWidth() const = 0;
  /**
   * @brief Function to get InputHeight of the movenet network (input image
   * rows).
   *
   * @return InputHeight of the movenet network.
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
