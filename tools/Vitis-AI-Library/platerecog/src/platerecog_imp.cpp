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
#include "./platerecog_imp.hpp"

#include <glog/logging.h>

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
namespace vitis {
namespace ai {

PlateRecogImp::PlateRecogImp(const std::string &platedetect_model,
                             const std::string &platerecog_model,
                             bool need_preprocess)
    : plate_num_{vitis::ai::PlateNum::create(platerecog_model,
                                             need_preprocess)},
      plate_detect_{
          vitis::ai::PlateDetect::create(platedetect_model, need_preprocess)} {}

PlateRecogImp::PlateRecogImp(const std::string &platedetect_model,
                             const std::string &platerecog_model,
                             xir::Attrs *attrs,
                             bool need_preprocess)
    : plate_num_{vitis::ai::PlateNum::create(platerecog_model, attrs,
                                             need_preprocess)},
      plate_detect_{
          vitis::ai::PlateDetect::create(platedetect_model, need_preprocess)} {}

PlateRecogImp::~PlateRecogImp() {}

int PlateRecogImp::getInputWidth() const {
  return plate_detect_->getInputWidth();
}

int PlateRecogImp::getInputHeight() const {
  return plate_detect_->getInputHeight();
}


size_t PlateRecogImp::get_input_batch() const {
  return plate_detect_->get_input_batch();
}

std::vector<PlateRecogResult> PlateRecogImp::run(const std::vector<cv::Mat> &input_image) {
  auto batch_size = get_input_batch();
  size_t ind = 0;
  std::vector<cv::Mat> input_tmp;
  std::vector<PlateDetectResult> det_results;
  while(ind < input_image.size()) {
    size_t next = std::min(ind + batch_size, input_image.size());
    for(size_t i = ind; i < next; i++) {
      input_tmp.push_back(input_image[i]);
    }
    auto det_result = plate_detect_->run(input_tmp);
    det_results.insert(det_results.end(), det_result.begin(), det_result.end());
    ind = ind + batch_size;
    input_tmp.clear();
  }

  std::vector<PlateRecogResult> nrs;
  for (size_t i = 0u; i < input_image.size(); i++) {
    float confidence = det_results[i].box.score;
    int x = (int)(det_results[i].box.x * input_image[i].cols);
    int y = (int)(det_results[i].box.y * input_image[i].rows);
    int width = (int)(det_results[i].box.width * input_image[i].cols);
    int height = (int)(det_results[i].box.height * input_image[i].cols);

    x = x < 0 ? 0 : x;
    y = y < 0 ? 0 : y;

  // check
    width = input_image[i].cols > (x + width) ? width : (input_image[i].cols - x - 1);
    height =
        input_image[i].rows > (y + height) ? height : (input_image[i].rows - y - 1);

    /*DLOG(INFO) << "PlateRecog::imagesize " << input_image[i].cols << "x"
             << input_image[i].rows << " "
             << "x " << x << " "               //
             << "y " << y << " "               //
             << "width " << width << " "       //
             << "height " << height << " "     //
             << "score " << confidence << " "  //
             << std::endl;
    */
    if (width == 0 || height == 0 || confidence < 0.1) {
      //DLOG(INFO) << "Skip Plate Recog, plate detction confidence = "
      //        << det_results[i].box.score;
      nrs.push_back(PlateRecogResult{getInputWidth(),                           //
                            getInputHeight(),                          //
                            PlateRecogResult::BoundingBox{confidence,  //
                                                          x,           //
                                                          y,           //
                                                          width,       //
                                                          height},     //
                            "NONE", "NONE"});
    }
    cv::Mat plate_image =
      input_image[i](cv::Rect_<float>(cv::Point(x, y), cv::Size{width, height}));
    auto num = plate_num_->run(plate_image);

    std::string plate_number = num.plate_number;
    std::string plate_color = num.plate_color;

    // DLOG(INFO) << "PlateRecog:: PlateNum : "
    //          << "plate_number " << plate_number << " "  //
    //          << "plate_color " << plate_color << " "    //
    //          << std::endl;
    nrs.push_back(PlateRecogResult{getInputWidth(),                           //
                          getInputHeight(),                          //
                          PlateRecogResult::BoundingBox{confidence,  //
                                                        x,           //
                                                        y,           //
                                                        width,       //
                                                        height},     //
                          plate_number,                              //
                          plate_color});
  }
  return nrs;
}
PlateRecogResult PlateRecogImp::run(const cv::Mat &input_image) {
  auto det_result = plate_detect_->run(input_image);

  float confidence = det_result.box.score;
  int x = (int)(det_result.box.x * input_image.cols);
  int y = (int)(det_result.box.y * input_image.rows);
  int width = (int)(det_result.box.width * input_image.cols);
  int height = (int)(det_result.box.height * input_image.cols);

  x = x < 0 ? 0 : x;
  y = y < 0 ? 0 : y;

  // check
  width = input_image.cols > (x + width) ? width : (input_image.cols - x - 1);
  height =
      input_image.rows > (y + height) ? height : (input_image.rows - y - 1);

  DLOG(INFO) << "PlateRecog::imagesize " << input_image.cols << "x"
             << input_image.rows << " "
             << "x " << x << " "               //
             << "y " << y << " "               //
             << "width " << width << " "       //
             << "height " << height << " "     //
             << "score " << confidence << " "  //
             << std::endl;

  if (width == 0 || height == 0 || confidence < 0.1) {
    LOG(INFO) << "Skip Plate Recog, plate detction confidence = "
              << det_result.box.score;
    return PlateRecogResult{getInputWidth(),                           //
                            getInputHeight(),                          //
                            PlateRecogResult::BoundingBox{confidence,  //
                                                          x,           //
                                                          y,           //
                                                          width,       //
                                                          height},     //
                            "NONE", "NONE"};
  }
  cv::Mat plate_image =
      input_image(cv::Rect_<float>(cv::Point(x, y), cv::Size{width, height}));
  auto num = plate_num_->run(plate_image);

  std::string plate_number = num.plate_number;
  std::string plate_color = num.plate_color;

  DLOG(INFO) << "PlateRecog:: PlateNum : "
             << "plate_number " << plate_number << " "  //
             << "plate_color " << plate_color << " "    //
             << std::endl;

  return PlateRecogResult{getInputWidth(),                           //
                          getInputHeight(),                          //
                          PlateRecogResult::BoundingBox{confidence,  //
                                                        x,           //
                                                        y,           //
                                                        width,       //
                                                        height},     //
                          plate_number,                              //
                          plate_color};
}
}  // namespace ai
}  // namespace vitis
