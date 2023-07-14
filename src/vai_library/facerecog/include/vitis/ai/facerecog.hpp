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
 * Filename: facerecog.hpp
 *
 * Description:
 * This library is used to getting features of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */

#pragma once
#include <memory>
#include <utility>
#include <opencv2/core.hpp>
#include <vector>
#include <array>
#include <vitis/ai/facefeature.hpp>

namespace xir {
  class Attrs;
};

namespace vitis {
namespace ai {

/**
 * @struct FaceRecogFloatResult
 * @brief Struct of the result returned by the facerecog network, the features type is float.
 */
struct FaceRecogFloatResult {
  ///width of a input image.
  int width;
  /// height of a input image.
  int height;
  ///Face confidence,the value range from 0 to 1.
  using vector_t = std::array<float, 512>;
  ///Face features, the float array has 512 elements.
  /// the 512 dimention array
  std::unique_ptr<vector_t> feature;
};
/**
 * @struct FaceRecogFixedResult
 * @brief Struct of the result returned by the facerecog network , the features type is fixed.
 */
struct FaceRecogFixedResult {
  ///width of a input image.
  int width;
  /// height of a input image.
  int height;
  ///the fix point
  float scale;
  using vector_t = std::array<int8_t, 512>;
  ///Face features, the float array has 512 elements.
  /// the 512 dimention array
  std::unique_ptr<vector_t> feature;
};

/**
 * @brief Base class for getting features in a face image
 (cv:Mat).
 *
 * Input is a face image (cv::Mat).
 *
 * Output is features of a face in the input image.
 *
 * @note Facedetect network is depended on Landmark network and Feature network.
 *
 * smaple code:
 * @code
   cv::Mat image = cv::imread("abc.jpg");
   auto densebox_detect = vitis::ai::FaceDetect::create(
      "densebox_640_360);
   auto result = densebox_detect->run(image);
   auto recog = vitis::ai::FaceRecog::create("facerec_resnet20");
   int i = 0;
   while(i < result.size()){
    auto e = vitis::ai::FaceRecog::expand_and_align(image.cols, image.rows,
       r[i].x * images.cols,
       r[i].y * images.rows,
       r[i].width * images.cols,
       r[i].height * images.rows,
       0.2,0.2,
       16, 8);
    auto result_tuple =
    recog->run(image(e.first), e1.second.x, e1.second.y,
                       e1.second.width, e1.second.height);
    auto confidence = result_tuple.score;
    auto features_normal = result_tuple.features;
    auto gender = result_tuple.gender;
    auto age = result_tuple.age;
    i++;
   }
  @endcode
 *
 */
class FaceRecog {
 public :
   /**
    * @brief Function to enlarge the detect.
    * @param width Origin width of an image.
    * @param height Origin height of an image.
    * @param x, y, w, h output of densebox.
    * @param ratio_x enlarge by ratio_x, 20% by default.
    * @param ratio_y enlarge by ratio_y, 20% by default.
    * @param align_x aligned pixels.
    * @param align_y aligned pixels.
    * @return the first is enlarged bounding box, the second is the relative bounding box relative to the first.
    */
   static std::pair<cv::Rect, cv::Rect>
   expand_and_align(int width, int height, int x, int y, int w, int h,
                    float ratio_x,float ratio_y, int align_x, int align_y);
   /**
    * @brief Factory function to get an instance of derived classes of class
    *Recog.
    *
    * @param feature_model_name Face feature model name 
    * @param need_preprocess  normalize with mean/scale or not, true
   *by default.
    *
    * @return An instance of FaceRecog class.
    */
   static std::unique_ptr<FaceRecog> create(const std::string &feature_network_name, bool need_preprocess = true);

   static std::unique_ptr<FaceRecog> create(const std::string &feature_network_name, 
                                            xir::Attrs *attrs,
                                            bool need_preprocess = true);
   /**
    * @brief Factory function to get an instance of derived classes of class
    *Recog.
    *
    * @param landmark_model_name Face landmark model name 
    * @param feature_model_name Face feature model name 
    * @param need_preprocess  normalize with mean/scale or not, true
   *by default.
    *
    * @return An instance of FaceRecog class.
    */
   static std::unique_ptr<FaceRecog> create(const std::string &landmark_network_name,
                                            const std::string &feature_network_name,
                                            bool need_preprocess = true);

   static std::unique_ptr<FaceRecog> create(const std::string &landmark_network_name,
                                            const std::string &feature_network_name,
                                            xir::Attrs *attrs,
                                            bool need_preprocess = true);
 protected:
   explicit FaceRecog();
   FaceRecog(const FaceRecog &other) = delete;

 public:
   virtual ~FaceRecog();

   /**
    * @brief Function to get InputWidth of the facerecog network (input image
    *cols).
    *
    * @return InputWidth of the facerecog network
    */
   virtual int getInputWidth() const = 0;

   /**
    *@brief Function to get InputHeigth of the facerecog network (input image
    *rows).
    *
    *@return InputHeight of the facerecog network.
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
    * @brief Function of get running result (float feature) of the facerecog network.
    *
    * @param img  A face image of after expand and align.
    * @param inner_x x-coordinate relative to the input img.
    * @param inner_y y-coordinate relative to the input img.
    * @param inner_w face width.
    * @param inner_h face heigth.
    *
    * @return the float features, gender and age of a face , features is a
    *vector has 512 elements.
    */
   virtual FaceRecogFloatResult run(const cv::Mat &img, int inner_x, int inner_y,
                           int inner_w, int inner_h) = 0;

   virtual std::vector<FaceRecogFloatResult> run(
                    const std::vector<cv::Mat> &imgs, 
                    const std::vector<cv::Rect> &inner_bboxes) = 0;

   /**
    * @brief Function of get running result (float feature) of the facerecog network.
    *
    * @param img A face image of after expand and align.
    * @param inner_x x-coordinate relative to the input img.
    * @param inner_y y-coordinate relative to the input img.
    * @param inner_w face width.
    * @param inner_h face heigth.
    *
    * @return the fixed features, gender and age of a face , features is a
    *vector has 512 elements.
    */
   virtual FaceRecogFixedResult run_fixed(const cv::Mat &img, int inner_x,
                                      int inner_y, int inner_w,
                                      int inner_h) = 0;

   virtual std::vector<FaceRecogFixedResult> run_fixed(
                    const std::vector<cv::Mat> &imgs, 
                    const std::vector<cv::Rect> &inner_bboxes) = 0;
 };
 /*!@} */
}
}

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: utf-8-unix
// End:
