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
 * Filename: facequality5pt.hpp
 *
 * Description:
 * This network is used to getting quality and five key points of a face
 * Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vitis/ai/nnpp/facequality5pt.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for evaluating the quality and five key points coordinate
 of a face image (cv::Mat).
 *
 * Input is a face image (cv::Mat).
 *
 * Output is the quality and five key points coordinate of a face in the input
 image.
 *
 *
 * Sample code :
 * @code
   cv:Mat image = cv::imread("sample_facequality5pt.jpg");
   auto network =
         vitis::ai::FaceQuality5pt::create("face-quality", true);
   auto result = network->run(image);
   auto quality = result.score;
   auto points = result.points;
   for(int i = 0; i< 5 ; ++i){
       auto x = points[i].frist  * image.cols;
       auto y = points[j].second * image.rows;
   }
   @endcode
 * @note Default mode is day, if day night switch network is used and the
 background of the input image is night, please use API setMode
 * @code
 * network->setMode(vitis::ai::FaceQuality5pt::Mode::NIGHT);
 * @endcode
 *
 * Display of the FaceQuality5pt model results:
 * @image latex images/sample_facequality5pt_result.jpg "result image"
 width=\textwidth
 *
 */
class FaceQuality5pt {
 public:
  /**
   * @brief Scene of sending image.
   */
  enum class Mode {
    /// Use DAY when the background of the image is daytime.
    DAY,
    /// Use NIGHT when the background of the image is night.
    NIGHT
  };

  /**
   * @brief Factory function to get an instance of derived classes of class
   *FaceQuality5pt.
   *
   * @param model_name Model name
   * @param need_preprocess  Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of FaceQuality5pt class.
   *
   */
  static std::unique_ptr<FaceQuality5pt> create(const std::string& model_name,
                                                bool need_preprocess = true);

 protected:
  /**
   * @cond NOCOMMENTS
   */
  explicit FaceQuality5pt();
  FaceQuality5pt(const FaceQuality5pt& other) = delete;
  FaceQuality5pt& operator=(const FaceQuality5pt&) = delete;
  /**
   * @endcond
   */
 public:
  /**
   * @cond NOCOMMENTS
   */
  virtual ~FaceQuality5pt();
  /**
   * @endcond
   */
  /**
   * @brief Function to get InputWidth of the facequality5pt network (input
   * image columns).
   *
   * @return InputWidth of the facequality5pt network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeight of the facequality5pt network (input
   *image rows).
   *
   *@return InputHeight of facequality5pt network.
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
   * @brief Function to get Mode.
   */
  virtual Mode getMode() = 0;

  /**
   * @brief Function to set Mode.
   * @param mode Type::Mode
   * @return mode
   */
  virtual void setMode(Mode mode) = 0;

  /**
   * @brief Function of get running result of the facequality5pt network.
   *
   * @param img Input data of input image (cv::Mat) of detected counterpart
   * and resized to InputWidth and InputHeight required by the network.
   *
   * @return The result of the facequality5pt network.
   */

  virtual FaceQuality5ptResult run(const cv::Mat& img) = 0;

  // virtual FaceQuality5ptResult run_original(const cv::Mat &img) = 0;

  /**
   * @brief Function of get running results of the facequality5pt network in
   * batch mode.
   *
   * @param images Input data of input images (std::vector<cv::Mat>). The size
   * of input images equals batch size obtained by get_input_batch. The input
   * images need to be resized to InputWidth and InputHeight required by the
   * network.
   *
   * @return The vector of the FaceQuality5ptResult.
   */

  virtual std::vector<FaceQuality5ptResult> run(
      const std::vector<cv::Mat>& images) = 0;
  // virtual std::vector<FaceQuality5ptResult> run_original(const
  // std::vector<cv::Mat> &images) = 0;
};
/*!@} */
}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
