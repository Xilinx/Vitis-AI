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
 * Filename: 3Dsegmentation.hpp
 *
 * Description:
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>

using V1F=std::vector<float>;
using V2F=std::vector<V1F>;
using V3F=std::vector<V2F>;
using V1I=std::vector<int>;
using V2I=std::vector<V1I>;

namespace vitis {
namespace ai {

/**
 * @brief Base class for segmentation 3D object data in the vector<float> mode.
 *
   @endcode
 *
 */
struct Segmentation3DResult {
  /// Width of the network model.
  int width;
  /// Height of the network model.
  int height;
  /// Input 3D object data.
  std::vector<int> array;
};

class Segmentation3D {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * 3Dsegmentation.
   *
   * @param need_mean_scale_process Normalize with mean/scale or not, true by
   * default.
   *
   * @return An instance of the 3Dsegmentation class.
   */
  static std::unique_ptr<Segmentation3D> create(const std::string &model_name,
                                             bool need_mean_scale_process = false);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Segmentation3D();
  Segmentation3D(const Segmentation3D &) = delete;
  Segmentation3D &operator=(const Segmentation3D &) = delete;

 public:
  virtual ~Segmentation3D();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function to get InputWidth of the 3D segmentation network. 
   *
   * @return InputWidth of the 3D segmentation network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the 3D segmentation network.
   *
   *@return InputHeight of the 3D segmentation network.
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
   * @brief Function of get running result of the 3D segmentation network.
   *
   * @param array Input data of 3D object data in vector<float> mode.
   *
   * @return Segmentation3DResult.
   */
  virtual Segmentation3DResult run(std::vector<std::vector<float>>& array) = 0;
  /**
   * @brief Function of get running result of the 3D segmentation network in batch mode.
   *
   * @param arrays  A vector of Input data of 3D object data in vector<float> mode.
   *
   * @return A vector of Segmentation3DResult.
   */
  virtual std::vector<Segmentation3DResult> run(std::vector<std::vector<std::vector<float>>>& arrays) = 0;
};
}  // namespace ai
}  // namespace vitis
