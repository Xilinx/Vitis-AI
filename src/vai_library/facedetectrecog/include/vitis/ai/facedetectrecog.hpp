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
 * Filename: facedetectrecog.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/facedetect.hpp>
#include <vitis/ai/facelandmark.hpp>
#include <vitis/ai/facefeature.hpp>

namespace xir {
  class Attrs;
};

namespace vitis {
namespace ai {

struct FaceDetectRecogFloatResult {
  ///width of a input image.
  int width;
  /// height of a input image.
  int height;
  std::vector<FaceDetectResult::BoundingBox> rects; 
  using vector_t = std::array<float, 512>;
  std::vector<vector_t> features; 
};

struct FaceDetectRecogFixedResult {
  ///width of a input image.
  int width;
  /// height of a input image.
  int height;
  std::vector<FaceDetectResult::BoundingBox> rects; 
  float feature_scale;
  using vector_t = std::array<int8_t, 512>;
  std::vector<vector_t> features; 
};

/**
 * @brief Base class for detecting the position of faces in the input image
 (cv::Mat).
 *
 * Input is an image (cv::Mat).
 *
 * Output is a vector of position and score for faces in the input image.
 *
 * Sample code:
 * @code
   auto image = cv::imread("sample_facedetectrecog.jpg");
   auto network = vitis::ai::FaceDetectRecog::create(
                  "densebox_640_360",
                  "face_landmark",
                  "facerec_resnet20",
                  true);
   auto result = network->run(image);
   auto det_rects = result.rects;
   for (const auto &r : det_rects) {
      auto score = r.score;
      auto x = r.x * image.cols;
      auto y = r.y * image.rows;
      auto width = r.width * image.cols;
      auto height = r.height * image.rows;
   }
   auto recog_result = result.features;
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_facedetectrecog_result.jpg "result image" width=\textwidth
 *
 */
class FaceDetectRecog {
 public:
  /**
 * @brief Factory function to get instance of derived classes of class
 FaceDetectRecog
 *
 * @param detect_model_name Face detect model name
 * @param landmark_model_name Face landmark model name
 * @param feature_model_name Face feature model name
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of FaceDetectRecog class.
 */
  static std::unique_ptr<FaceDetectRecog> create(const std::string &detect_model_name,
                                                 const std::string &landmark_model_name,
                                                 const std::string &feature_model_name,
                                                 bool need_preprocess = true);

  /**
 * @brief Factory function to get instance of derived classes of class
 FaceDetectRecog
 *
 * @param model_name model name of FaceDetectRecog class, it is a fake model with content of 3 real model names;
 * @param need_preprocess Normalize with mean/scale or not, default
 value is true.
 * @return An instance of FaceDetectRecog class.
 */
  static std::unique_ptr<FaceDetectRecog> create(const std::string &model_name,
                                                 bool need_preprocess = true);

  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit FaceDetectRecog();
  FaceDetectRecog(const FaceDetectRecog &) = delete;
  FaceDetectRecog &operator=(const FaceDetectRecog &) = delete;

 public:
  virtual ~FaceDetectRecog();
  /**
   * @endcond
   */
  /**
   * @brief Function to get InputWidth of the facedetectrecog network (input image
   * cols).
   *
   * @return InputWidth of the facedetectrecog network
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the facedetectrecog network (input image
   *rows).
   *
   *@return InputHeight of the facedetectrecog network.
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
   * @brief Function to get detect threshold.
   * @return The detect threshold , the value range from 0 to 1.
   */
  virtual float getThreshold() const = 0;

  /**
   * @brief Function of update detect threshold.
   * @note The detection results will filter by detect threshold (score >=
   * threshold).
   * @param threshold The detect threshold,the value range from 0 to 1.
   */
  virtual void setThreshold(float threshold) = 0;

  /**
   * @brief Function to get running result of the facedetectrecog network.
   *
   * @param image Input Data ,input image (cv::Mat) 
   *
   * @return The float result of face detect and recog 
   *
   */
  virtual FaceDetectRecogFloatResult run(const cv::Mat &image) = 0;
  virtual FaceDetectRecogFixedResult run_fixed(const cv::Mat &image) = 0;

  /**
   * @brief Function to get running results of the facedetectrecog neural network in
   * batch mode.
   *
   * @param images Input data of input images (std:vector<cv::Mat>). The size of
   * input images equals batch size obtained by get_input_batch. 
   *
   * @return The vector of FaceDetectRecogFloatResult.
   *
   */
  virtual std::vector<FaceDetectRecogFloatResult> run(
      const std::vector<cv::Mat> &images) = 0;
  virtual std::vector<FaceDetectRecogFixedResult> run_fixed(
      const std::vector<cv::Mat> &images) = 0;
};

}  // namespace ai
}  // namespace vitis

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
