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
 * Filename: Unet2D.hpp
 *
 * Description:
 * This network is used to get count of crowd from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

struct Unet2DResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// 4 channels out data. Format: HWC
  std::vector<float> data; 
};

/**
 * @brief Base class for Unet2D
 *
 * Input is an 4 channel binary data: NxNx4
 *
 * Output is a struct of segmentation results, named Unet2DResult.
 *
 * Sample code:
   @code
   std::vector<float> vf = get_binary_data();
   auto Unet2D = vitis::ai::Unet2D::create("unet2d_tf");
   auto result = Unet2D->run(vf.data(), vf.size());
   std::cout << result.data.size() << "\n";
   @endcode
 *
 */
class Unet2D {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * Unet2D.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of Unet2D class.
   *
   */
  static std::unique_ptr<Unet2D> create(const std::string& model_name,
                                     bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit Unet2D();
  Unet2D(const Unet2D&) = delete;

 public:
  virtual ~Unet2D();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the Unet2D neural network.
   *
   * @param img pointer to Input data (binary data of 4 channels).
   * @param len length of Input data.
   *
   * @return Unet2DResult.
   *
   */
  virtual vitis::ai::Unet2DResult run(float* img, int len) = 0;
  /**
   * @brief Function of get result of the Unet2D neural network.
   *
   * @param img vector holding the Input data (binary data of 4 channels).
   *
   * @return Unet2DResult.
   *
   */
  virtual vitis::ai::Unet2DResult run(const std::vector<float>& img) = 0;
  /**
   * @brief Function to get running results of the Unet2D neural network in
   * batch mode.
   *
   * @param imgs vector of Input data of input images (vector<float*>).
   * The size of input images need equal to or less than batch size 
   * obtained by get_input_batch. 
   * If it is greater than batch, the extra part is ignored.
   *
   * @param len length of Input data: all input data should be same size.
   *
   * @return The vector of Unet2DResult.
   *
   */
  virtual std::vector<vitis::ai::Unet2DResult> run( const std::vector<float*>& imgs, int len) = 0;
  /**
   * @brief Function to get running results of the Unet2D neural network in
   * batch mode.
   *
   * @param imgs Input data of input images (vector<vector<float>>).
   * The size of input images need equal to or less than batch size
   * obtained by get_input_batch.
   * If it is greater than batch, the extra part is ignored.
   *
   * @return The vector of Unet2DResult.
   *
   */
  virtual std::vector<vitis::ai::Unet2DResult> run( const std::vector<std::vector<float>>& imgs) = 0;

  /**
   * @brief Function to get InputWidth of the Unet2D network (input image columns).
   *
   * @return InputWidth of the Unet2D network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the Unet2D network (input image rows).
   *
   *@return InputHeight of the Unet2D network.
   */

  virtual int getInputHeight() const = 0;

  /**
   * @brief Function to get the number of images processed by the DPU at one
   *time.
   * @note Different DPU core the batch size may be different. This depends on
   *the IP used.
   *
   *@return Batch size.
   */
  virtual size_t get_input_batch() const = 0;
};
}  // namespace ai
}  // namespace vitis
