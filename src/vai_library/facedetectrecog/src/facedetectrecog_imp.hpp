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
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/facedetectrecog.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/facelandmark.hpp>
#include <vitis/ai/facefeature.hpp>

using std::tuple;
using std::vector;

namespace vitis {
namespace ai {

class FaceDetectRecogImp : public FaceDetectRecog {
  public:
    FaceDetectRecogImp(const std::string &detect_model_name, 
                  const std::string &landmark_model_name, 
                  const std::string &feature_model_name, 
                  bool need_preprocess);

    FaceDetectRecogImp(const std::string &model_name, 
                  bool need_preprocess);

    /// Destructor
    virtual ~FaceDetectRecogImp();
    
    /// Input width(image cols)
    virtual int getInputWidth() const override;
    /// Input height(image rows)
    virtual int getInputHeight() const override;

    virtual size_t get_input_batch() const override;

    /// Set an image and get positions, scores and features of faces in the image
    virtual FaceDetectRecogFixedResult run_fixed(const cv::Mat &input_image) override;

    virtual FaceDetectRecogFloatResult run(const cv::Mat &input_image) override;

    /// Set an image list and get positions, scores and features of faces in the image
    virtual std::vector<FaceDetectRecogFixedResult> run_fixed(
        const std::vector<cv::Mat> &input_images) override;
    virtual std::vector<FaceDetectRecogFloatResult> run(
        const std::vector<cv::Mat> &input_images) override;
    /// Get detect threshold
    virtual float getThreshold() const override;
    /// Set detect threshold
    virtual void setThreshold(float threshold) override;

  private:
    std::vector<FaceFeatureFixedResult> run_recog_fixed_batch_internal(
        const std::vector<cv::Mat> &imgs_expanded,
        const std::vector<cv::Rect> &inner_bboxes);
    std::vector<FaceDetectRecogFixedResult> run_fixed_internal(
        const std::vector<cv::Mat> &input_images);
    std::unique_ptr<vitis::ai::FaceDetect> detect_;
    std::unique_ptr<vitis::ai::FaceLandmark> landmark_;
    std::unique_ptr<vitis::ai::FaceFeature> feature_;
};

}  // namespace ai
}  // namespace vitis
