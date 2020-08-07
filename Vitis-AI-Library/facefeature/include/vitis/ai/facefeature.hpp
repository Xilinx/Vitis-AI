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
  * Filename: facefeature.hpp
  *
  * Description:
  * This network is used to getting the features of a face
  * Please refer to document "Xilinx_AI_SDK_User_guide.pdf" for more details of
  * these APIs.
  */
#pragma once
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <vitis/ai/nnpp/facefeature.hpp>

namespace vitis {
namespace ai {

/**
 * @brief Base class for getting the features of a face image (cv::Mat).
 *
 * Input is a face image (cv::Mat).
 *
 * Output is the features of a face in the input image.
 *
 * @note Two interfaces are provided to get the float features or fixed
 features.  features is a vector has 512 elements.
 *
 * Float sample code :
 * @code
   cv:Mat image = cv::imread("test_face.jpg");
   auto network  = vitis::ai::FaceFeature::create("facerec_resnet20", true);
   auto result = network->run(image);
   @endcode
 *
 * Fixed sample code :
 * @code
   cv:Mat image = cv::imread("test_face.jpg");
   auto network  = vitis::ai::FaceFeature::create("facerec_resnet20", true);
   auto result = network->run_fixed(image);
   @endcode
 *
 * Similarity calculation formula :
 *
 * \f$\rho = \frac{\vec{a} \cdot \vec{b}}{\sqrt{\sum_{i=1}^{n} {a_i}^2}
  \cdot \sqrt{\sum_{i=1}^{n} {b_i}^2}}\f$
 *
 *
 * Fixed compare code :
 * @code
   float feature_norm(const int8_t *feature) {
      int sum = 0;
      for (int i = 0; i < 512; ++i) {
          sum += feature[i] * feature[i];
      }
      return 1.f / sqrt(sum);
   }

  /// This function is used for computing dot product of two vector
  static float feature_dot(const int8_t *f1, const int8_t *f2) {
     int dot = 0;
     for (int i = 0; i < 512; ++i) {
        dot += f1[i] * f2[i];
     }
     return (float)dot;
  }

  float feature_compare(const int8_t *feature, const int8_t *feature_lib){
     float norm = feature_norm(feature);
     float feature_norm_lib = feature_norm(feature_lib);
     return feature_dot(feature, feature_lib) * norm * feature_norm_lib;
  }

  /// This function is used for model "facerec_resnet20" 
  float score_map_l20(float score) { return 1.0 / (1 + exp(-12.4 * score + 3.763)); }

  /// This function is used for type "facerec_resnet64"
  float score_map_l64(float score) { return 1.0 / (1 + exp(-17.0836 * score + 5.5707)); }

  @endcode
 *
 * Display of the compare result with a set of images:
 * @image latex images/sample_facecompare_result "facecompare result image" width=300px
 *
 *
 */
class FaceFeature {
public:
  /**
   * @brief Factory function to get an instance of derived classes of class
   *FaceFeature.
   * @param model_name  Model name
   *
   * @param need_preprocess  Normalize with mean/scale or not, default value is true.
   *
   * @return An instance of FaceFeature class.
   */
  static std::unique_ptr<FaceFeature> create(const std::string &model_name, bool need_preprocess = true);

protected:
  explicit FaceFeature();
  FaceFeature(const FaceFeature &) = delete;
  FaceFeature &operator=(const FaceFeature &) = delete;

public:
  virtual ~FaceFeature();

  /**
   * @brief Function to get InputWidth of the feature network (input image cols).
   *
   * @return InputWidth of the feature network.
   */
  virtual int getInputWidth() const = 0;

  /**
   *@brief Function to get InputHeigth of the feature network (input image rows).
   *
   *@return InputHeight of the feature network.
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
   * @brief Function of get running result of the feature network.
   *
   * @param img Input data for image (cv::Mat) detected by the facedetect network and then retated and aligned.
   *
   *
   * @return FaceFeatureFloatResult
   */
  virtual FaceFeatureFloatResult run(const cv::Mat &img) = 0;

  /**
   * @brief Function of get running result of the feature network.
   *
   * @param img Input Data of input image (cv::Mat) of detected counterpart
   * and resized as InputWidth and InputHeight.
   *
   * @return FaceFeatureFixedResult 
   */
  virtual FaceFeatureFixedResult run_fixed(const cv::Mat &img) = 0;

  /**
   * @brief Function of get running result of the feature network
   * in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>) detected by the facedetect network and then retated and aligned.
   * The size of input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of FaceFeatureFloatResult.
   */

  virtual std::vector<FaceFeatureFloatResult> run(const std::vector<cv::Mat> &img) = 0;

  /**
   * @brief Function of get running result of the feature network
   * in batch mode.
   *
   * @param images Input data of batch input images (vector<cv::Mat>) detected by the facedetect network and then retated and aligned.
   * The size of input images equals batch size obtained by get_input_batch.
   *
   * @return The vector of FaceFeatureFixedResult.
   */
  virtual std::vector<FaceFeatureFixedResult> run_fixed(const std::vector<cv::Mat> &img) = 0;
};
/*!@} */
}
}


// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
