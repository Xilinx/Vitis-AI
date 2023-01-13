/*
 * Copyright 2019 xilinx Inc.
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
#include "./carplaterecog_imp.hpp"

#include <glog/logging.h>

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xir/attrs/attrs.hpp>
namespace vitis {
namespace ai {

CarPlateRecogImp::CarPlateRecogImp(
                             const std::string &cardetect_model,
                             const std::string &platedetect_model,
                             const std::string &platerecog_model,
                             bool need_preprocess)
    : ssd_{vitis::ai::SSD::create(cardetect_model, need_preprocess)},
      plate_recog_{
          vitis::ai::PlateRecog::create(platedetect_model, platerecog_model, need_preprocess)} {
}


CarPlateRecogImp::CarPlateRecogImp(
                             const std::string &cardetect_model,
                             const std::string &platedetect_model,
                             const std::string &platerecog_model,
                             xir::Attrs *attrs,
                             bool need_preprocess)
    : ssd_{vitis::ai::SSD::create(cardetect_model, attrs, need_preprocess)},
      plate_recog_{
          vitis::ai::PlateRecog::create(platedetect_model, platerecog_model, attrs, need_preprocess)} {
}

CarPlateRecogImp::~CarPlateRecogImp() {}

int CarPlateRecogImp::getInputWidth() const {
  return ssd_->getInputWidth();
}

int CarPlateRecogImp::getInputHeight() const {
  return ssd_->getInputHeight();
}

size_t CarPlateRecogImp::get_input_batch() const {
  return ssd_->get_input_batch();
}

CarPlateRecogResult CarPlateRecogImp::run(const cv::Mat &input_image) {
  auto cardet_result = ssd_->run(input_image);
  std::vector<std::pair<SSDResult::BoundingBox,PlateRecogResult>> plate_results;
  for (auto& box: cardet_result.bboxes) {
    int label = box.label;

    float fxmin = box.x * input_image.cols;
    float fymin = box.y * input_image.rows;
    float fxmax = fxmin + box.width * input_image.cols;
    float fymax = fymin + box.height * input_image.rows;
    float confidence = box.score;

    int xmin = round(fxmin * 100.0) / 100.0;
    int ymin = round(fymin * 100.0) / 100.0;
    int xmax = round(fxmax * 100.0) / 100.0;
    int ymax = round(fymax * 100.0) / 100.0;
    
    xmin = std::min(std::max(xmin, 0), input_image.cols);
    xmax = std::min(std::max(xmax, 0), input_image.cols);
    ymin = std::min(std::max(ymin, 0), input_image.rows);
    ymax = std::min(std::max(ymax, 0), input_image.rows);
    auto bheight =  ymax - ymin;
    auto bwidth = xmax - xmin;
    std::cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\t" << bheight << "\t" << bwidth << "\n";
    if (label == 1) {
      auto car_img = input_image(cv::Rect2i{xmin, ymin, bwidth, bheight});
      auto rs = plate_recog_->run(car_img); 
      plate_results.push_back(std::make_pair(box, rs));
    }
  }
  return CarPlateRecogResult{getInputWidth(),                           //
                          getInputHeight(),                          //
                          plate_results};
}


std::vector<CarPlateRecogResult> CarPlateRecogImp::run(const std::vector<cv::Mat> &input_images) {
  std::vector<cv::Mat> images;
  std::vector<CarPlateRecogResult> nrs;
  auto cardet_result = ssd_->run(input_images);
  for(size_t i = 0; i < input_images.size(); i++) {
    std::vector<std::pair<SSDResult::BoundingBox,PlateRecogResult>> plate_results;
    std::pair<std::vector<SSDResult::BoundingBox>, std::vector<cv::Mat>> car_mat;
    for (auto& box: cardet_result[i].bboxes) {
      int label = box.label;
      float fxmin = box.x * input_images[i].cols;
      float fymin = box.y * input_images[i].rows;
      float fxmax = fxmin + box.width * input_images[i].cols;
      float fymax = fymin + box.height * input_images[i].rows;
      float confidence = box.score;

      int xmin = round(fxmin * 100.0) / 100.0;
      int ymin = round(fymin * 100.0) / 100.0;
      int xmax = round(fxmax * 100.0) / 100.0;
      int ymax = round(fymax * 100.0) / 100.0;
    
      xmin = std::min(std::max(xmin, 0), input_images[i].cols);
      xmax = std::min(std::max(xmax, 0), input_images[i].cols);
      ymin = std::min(std::max(ymin, 0), input_images[i].rows);
      ymax = std::min(std::max(ymax, 0), input_images[i].rows);
      auto bheight =  ymax - ymin;
      auto bwidth = xmax - xmin;
      if (0) {
      std::cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\t" << bheight << "\t" << bwidth << "\n";
      }
      if (label == 1) {
        auto car_img = input_images[i](cv::Rect2i{xmin, ymin, bwidth, bheight});
        car_mat.second.push_back(car_img);
        car_mat.first.push_back(box);
        // auto rs = plate_recog_->run(car_img); 
        // plate_results.push_back(std::make_pair(box, rs));
      }
    }
    auto rs = plate_recog_->run(car_mat.second);
    for(size_t i = 0; i < rs.size(); i++) {
      plate_results.push_back(std::make_pair(car_mat.first[i], rs[i]));        
    }
    nrs.push_back( CarPlateRecogResult{getInputWidth(),
                          getInputHeight(),                          //
                          plate_results});
  }
  return nrs;
}

}  // namespace ai
}  // namespace vitis
