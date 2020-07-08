#include "./platerecog_imp.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
namespace vitis {
namespace ai {

PlateRecogImp::PlateRecogImp(const std::string &platedetect_model,
                          const std::string &platerecog_model,
                          bool need_preprocess)
    : plate_num_{ vitis::ai::PlateNum::create(platerecog_model, need_preprocess) },
      plate_detect_{ vitis::ai::PlateDetect::create(platedetect_model, need_preprocess) } {}

PlateRecogImp::~PlateRecogImp() {}


int PlateRecogImp::getInputWidth() const { return plate_detect_->getInputWidth(); }

int PlateRecogImp::getInputHeight() const { return plate_detect_->getInputHeight(); }



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
             << "x " << x << " "                   //
             << "y " << y << " "                   //
             << "width " << width << " "           //
             << "height " << height << " "         //
             << "score " << confidence << " " //
             << std::endl;

  if (width == 0 || height == 0 || confidence < 0.1) {
    LOG(INFO) << "Skip Plate Recog, plate detction confidence = "
              << det_result.box.score;
    return PlateRecogResult{ getInputWidth(),                           //
                             getInputHeight(),                          //
                             PlateRecogResult::BoundingBox{ confidence, //
                                                            x,          //
                                                            y,          //
                                                            width,      //
                                                            height },   //
                             "",
                             "" };
  }
  cv::Mat plate_image = input_image(cv::Rect_<float>(cv::Point(x, y), cv::Size{width, height}));
  auto num = plate_num_->run(plate_image);

  std::string plate_number = num.plate_number;
  std::string plate_color = num.plate_color;

  DLOG(INFO) << "PlateRecog:: PlateNum : "
             << "plate_number " << plate_number << " " //
             << "plate_color " << plate_color << " "   //
             << std::endl;

  return PlateRecogResult{ getInputWidth(),                           //
                           getInputHeight(),                          //
                           PlateRecogResult::BoundingBox{ confidence, //
                                                          x,          //
                                                          y,          //
                                                          width,      //
                                                          height },   //
                           plate_number,                              //
                           plate_color };
}
}
}
