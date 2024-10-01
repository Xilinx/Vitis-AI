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

 * Filename: openpose.hpp
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
#include <vitis/ai/nnpp/openpose.hpp>

namespace vitis {
namespace ai {

/// openpose model, input size is 368x368.

/**
 * @brief Base class for detecting poses of people.
 *
 * Input is an image (cv:Mat).
 *
 * Output is a OpenPoseResult.
 *
 * Sample code :
   @code
  auto image = cv::imread("sample_openpose.jpg");
  if (image.empty()) {
    std::cerr << "cannot load image" << std::endl;
    abort();
  }
  auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  vector<vector<int>> limbSeq = {{0,1}, {1,2}, {2,3}, {3,4}, {1,5}, {5,6},
 {6,7}, {1,8}, \ {8,9}, {9,10}, {1,11}, {11,12}, {12,13}}; float scale_x =
 float(image.cols) / float(width); float scale_y = float(image.rows) /
 float(height); auto results = det->run(image); for(size_t k = 1; k <
 results.poses.size(); ++k){ for(size_t i = 0; i < results.poses[k].size();
 ++i){ if(results.poses[k][i].type == 1){ results.poses[k][i].point.x *=
 scale_x; results.poses[k][i].point.y *= scale_y; cv::circle(image,
 results.poses[k][i].point, 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    for(size_t i = 0; i < limbSeq.size(); ++i){
        Result a = results.poses[k][limbSeq[i][0]];
        Result b = results.poses[k][limbSeq[i][1]];
        if(a.type == 1 && b.type == 1){
            cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
        }
    }
  }
   @endcode
 *
 * Display of the openpose model results:
 * @image latex images/sample_openpose_result.jpg "openpose result image"
 width=\textwidth
 */

class OpenPose {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * OpenPose.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not, default
   * value is true.
   * @return An instance of OpenPose class.
   *
   */
  static std::unique_ptr<OpenPose> create(const std::string& model_name,
                                          bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 public:
  explicit OpenPose();
  OpenPose(const OpenPose&) = delete;
  virtual ~OpenPose();
  /**
   * @endcond
   */
 public:
  /**
   * @brief Function to get running result of the openpose neural network.
   *
   * @param image Input data of input image (cv::Mat).
   *
   * @return OpenPoseResult.
   *
   */
  virtual OpenPoseResult run(const cv::Mat& image) = 0;
  /**
   * @brief Function to get running results of the openpose neural
   * network in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>). The size
   * of input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of OpenPoseResult.
   *
   */
  virtual std::vector<OpenPoseResult> run(
      const std::vector<cv::Mat>& images) = 0;
  /**
   * @brief Function to get InputWidth of the openpose network (input image
   *columns).
   *
   * @return InputWidth of the openpose network
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the openpose network (input image
   *rows).
   *
   *@return InputHeight of the openpose network.
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
