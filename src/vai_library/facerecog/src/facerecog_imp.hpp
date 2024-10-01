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

#pragma once
#include <cstdint>
#include <vitis/ai/facelandmark.hpp>
#include <vitis/ai/facefeature.hpp>
#include "vitis/ai/facerecog.hpp"
//DEF_ENV_PARAM()

namespace vitis {
namespace ai {

class FaceRecogImp : public FaceRecog {
 public:
  explicit FaceRecogImp(const std::string &feature_model_name, bool need_preprocess);
  explicit FaceRecogImp(const std::string &feature_model_name,
                        xir::Attrs *attrs,
                        bool need_preprocess);
  explicit FaceRecogImp(const std::string &landmark_model_name, 
                        const std::string &feature_model_name, 
                        bool need_preprocess);
  explicit FaceRecogImp(const std::string &landmark_model_name, 
                        const std::string &feature_model_name, 
                        xir::Attrs *attrs,
                        bool need_preprocess);
  /// Destructor
  virtual ~FaceRecogImp();

  /// Input width(image cols)
  virtual int getInputWidth() const override;
  /// Input height(image rows)
  virtual int getInputHeight() const override;

  virtual size_t get_input_batch() const override;

  /// set an image of after and relative rect
  virtual FaceRecogFloatResult run(const cv::Mat &img, int inner_x, int inner_y,
                          int inner_w, int inner_h) override;

  virtual std::vector<FaceRecogFloatResult> run(const std::vector<cv::Mat> &imgs, 
                                                const std::vector<cv::Rect> &inner_bboxes) override;

  /// set an image of after and relative rect
  virtual FaceRecogFixedResult run_fixed(const cv::Mat &img, int inner_x,
                                     int inner_y, int inner_w, int inner_h) override;

  virtual std::vector<FaceRecogFixedResult> run_fixed(const std::vector<cv::Mat> &imgs, 
                                                      const std::vector<cv::Rect> &inner_bboxes) override;


 private:
   FaceRecogFixedResult _runNormalFixed(const cv::Mat &img_expanded, int inner_x,
                                    int inner_y, int inner_w, int inner_h);
   std::vector<FaceRecogFixedResult> _runNormalFixed(const std::vector<cv::Mat> &img_expanded, 
                                                     const std::vector<cv::Rect> &inner_bboxes);

   std::unique_ptr<vitis::ai::FaceLandmark> landmark_;
   std::unique_ptr<vitis::ai::FaceFeature> feature_;
};
/*!@} */
}
}
