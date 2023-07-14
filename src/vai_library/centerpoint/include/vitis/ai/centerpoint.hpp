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
 * Filename: centerpoint.hpp
 *
 * Description:
 * This network is used to detecting objects from a input points data.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/centerpoint.hpp>

namespace vitis {
namespace ai {


class CenterPoint{
 public:
  /**
 * @brief Factory function to get an instance of derived classes of class
 CenterPoint
 *
 value is true.
 * @return An instance of CenterPoint class.
 */
  static std::unique_ptr<CenterPoint> create(const std::string &model_name_0,
                                                      const std::string &model_name_1);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit CenterPoint();
  CenterPoint(const CenterPoint &) = delete;
  CenterPoint &operator=(const CenterPoint &) = delete;

 public:
  virtual ~CenterPoint();
  /**
   * @endcond
   */
  /**
   * @brief Function to get InputWidth of the centerpoint network (input image
   * columns).
   *
   * @return InputWidth of the centerpoint network
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the centerpoint network (input image
   *rows).
   *
   *@return InputHeight of the centerpoint network.
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
   * @brief Function to get result of the centerpoint network.
   *
   * @param input Input data of float vector.
   *
   * @return vector of CenterPointResult.
   *
   */
  virtual std::vector<CenterPointResult> run(const std::vector<float> &input) = 0;

  /**
   * @brief Function to get result of the centerpoint network in batch mode.
   *
   * @param inputs vector of Input data of float vector.
   *
   * @return vector of vector of CenterPointResult.
   *
   */
  virtual std::vector<std::vector<CenterPointResult>> run(
      const std::vector<std::vector<float>> &inputs) = 0;
};


}  // namespace ai
}  // namespace vitis
