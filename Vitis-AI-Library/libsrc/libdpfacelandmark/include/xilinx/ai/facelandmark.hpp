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
 * Filename: facelandmark.hpp
 *
 * Description:
 * This network is used to detecting five key points, gender and age of a face
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <utility>
#include <xilinx/ai/nnpp/facelandmark.hpp>

namespace xilinx {
namespace ai {


/**
 * @brief Base class for detecting five key points,and score from a
 face image (cv::Mat).
 *
 * Input a face image (cv::Mat).
 *
 * Output score, five key points of the face.
 *
 * @note Usually the input image contains only one face, when contains multiple
 faces will return the highest score.
 *
 * Sample code:
 * @code
   cv:Mat image = cv::imread("sample_facelandmark.jpg");
   auto landmark  = xilinx::ai::FaceLandmark::create("face_landmark");
   auto result = landmark->run(image);
   float score = result.score;
   auto points = result.points;
   for(int i = 0; i< 5 ; ++i){
       auto x = points[i].frist  * image.cols;
       auto y = points[i].second * image.rows;
   }
   @endcode
 *
 * Display of the model results:
 * @image latex images/sample_facelandmark_result.jpg "result image"
 image" width=100px
 */
class FaceLandmark {
public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   *FaceLandmark.
   * @param model_name Model name
   * @param need_preprocess  Normalize with mean/scale or not, default
   *value is true.
   *
   * @return An instance of FaceLandmark class.
   */
  static std::unique_ptr<FaceLandmark>
  create(const std::string &model_name,
         bool need_preprocess = true);

protected:
  explicit FaceLandmark();
  FaceLandmark(const FaceLandmark &other) = delete;

public:
  virtual ~FaceLandmark();

public:
  /**
   * @brief Function to get InputWidth of the landmark network (input image
   * cols).
   *
   * @return InputWidth of the face landmark network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the landmark network (input image
   *rows).
   *
   *@return InputHeight of the face landmark network.
   */
  virtual int getInputHeight() const = 0;

  /**
   * @brief Function of get running result of the face landmark network.
   *
   * Set data of a face(e.g data of cv::Mat) and get the five key points.
   *
   * @param input_image Input data of input image (cv::Mat) of detected by the
   *facedetect network and resized as inputwidth and inputheight.
   *
   * @return The struct of FaceLandmarkResult
   */
  virtual FaceLandmarkResult run(const cv::Mat &input_image) = 0;
};
/*!@} */
} // namespace ai
} // namespace xilinx

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
