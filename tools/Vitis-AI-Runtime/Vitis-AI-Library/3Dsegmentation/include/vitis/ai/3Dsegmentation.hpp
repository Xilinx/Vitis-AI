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
 * Filename: 3Dsegmentation.hpp
 *
 * Description:
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
namespace vitis {
namespace ai {

/**
 * @brief Base class for detecting position of plate in a vehicle image
 (cv::Mat).
 *
   @endcode
 *
 */

struct Segmentation3DResult {
  int width;
  int height;
  std::vector<int> array;
};

class Segmentation3D {
 public:
  /**
   * @brief Factory function to get a instance of derived classes of class
   * 3Dsegmentation.
   *
   * @param need_mean_scale_process Normalize with mean/scale or not, true by
   * default.
   *
   * @returen A instance of the PlaterDatect class.
   */
  static std::unique_ptr<Segmentation3D> create(const std::string &model_name,
                                             bool need_preprocess = false);

 protected:
  explicit Segmentation3D();
  Segmentation3D(const Segmentation3D &) = delete;
  Segmentation3D &operator=(const Segmentation3D &) = delete;

 public:
  virtual ~Segmentation3D();

 public:
  /**
   * @brief Function to get InputWidth of the platedetect network (input image
   * cols).
   *
   * @return InputWidth of the platedetect network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the platedetect network (input image
   *rows).
   *
   *@return InputHeight of the platedetect network.
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

  /**
   * @brief Function of get running result of the platedetect network.
   *
   * @param img Input data of input image (cv::Mat) of detected counterpart
   * and resized as inputwidth an outputheight.
   *
   * @return plate position and plate score.
   */
  virtual Segmentation3DResult run(std::vector<std::vector<float>>& array) = 0;
  virtual std::vector<Segmentation3DResult> run(std::vector<std::vector<std::vector<float>>>& arrays) = 0;
};
}  // namespace ai
}  // namespace vitis
