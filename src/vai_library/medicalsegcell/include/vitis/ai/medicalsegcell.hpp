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
 * Filename: medicalsegcell.hpp
 *
 * Description:
 * This network is used to segment objects from a input image.
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

/**
 * @struct MedicalSegcellResult
 * @brief Struct of the result returned by the segmentation neural network.
 */
struct MedicalSegcellResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Segmentation result in cv::Mat mode.
  cv::Mat segmentation;
};

/**
 * @brief Base class for segmenting nuclei from images of cells.
 *
 * Input is an image (cv:Mat).
 *
 * Output is a struct of detection results, named MedicalSegcellResult.
 *
 * Sample code :
   @code
   Mat img = cv::imread("sample_medicalsegcell.jpg");
   auto medicalsegcell =
   vitis::ai::MedicalSegcell::create("medical_seg_cell_tf2",true);
   auto results = medicalsegcell->run(img);
   // results is structure holding cv::Mat.
   // please check test samples for detail usage.
   @endcode
 *
 */
class MedicalSegcell {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   * MedicalSegcell.
   *
   * @param model_name Model name
   * @param need_preprocess Normalize with mean/scale or not,
   * default value is true.
   * @return An instance of MedicalSegcell class.
   *
   */
  static std::unique_ptr<MedicalSegcell> create(const std::string& model_name,
                                                bool need_preprocess = true);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MedicalSegcell();
  MedicalSegcell(const MedicalSegcell&) = delete;

 public:
  virtual ~MedicalSegcell();
  /**
   * @endcond
   */

 public:
  /**
   * @brief Function of get result of the MedicalSegcell neural network.
   *
   * @param img Input data of input image (cv::Mat).
   *
   * @return MedicalSegcellResult.
   *
   */
  virtual vitis::ai::MedicalSegcellResult run(const cv::Mat& img) = 0;

  /**
   * @brief Function to get running results of the MedicalSegcell neural network
   * in batch mode.
   *
   * @param imgs Input data of input images (vector<cv::Mat>).The size of
   * input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of MedicalSegcellResult.
   *
   */
  virtual std::vector<vitis::ai::MedicalSegcellResult> run(
      const std::vector<cv::Mat>& imgs) = 0;

  /**
   * @brief Function to get InputWidth of the MedicalSegcell network (input
   * image columns).
   *
   * @return InputWidth of the MedicalSegcell network.
   */
  virtual int getInputWidth() const = 0;
  /**
   *@brief Function to get InputHeight of the MedicalSegcell network (input
   *image rows).
   *
   *@return InputHeight of the MedicalSegcell network.
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
