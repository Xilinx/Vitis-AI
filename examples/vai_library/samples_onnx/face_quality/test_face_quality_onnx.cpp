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

#include <glog/logging.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#if _WIN32
#include <codecvt>
#include <locale>
using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;
#endif
#include <vitis/ai/profiling.hpp>
#include "./face_quality_onnx.hpp"

cv::Mat process_result(cv::Mat image, const FaceQualityOnnxResult& result,
                       bool is_jpeg) {
  auto points = result.points;
  auto score = result.score;

  LOG_IF(INFO, is_jpeg) << "score : " << score << " points ";  //
  for (int i = 0; i < 5; ++i) {
    LOG_IF(INFO, is_jpeg) << points[i].first << " " << points[i].second << " ";
    auto point = cv::Point{static_cast<int>(points[i].first * image.cols),
                           static_cast<int>(points[i].second * image.rows)};
    cv::circle(image, point, 3, cv::Scalar(255, 8, 18), -1);
  }
  return image;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> " << std::endl;
    abort();
  }
#if _WIN32
  auto model_name = strconverter.from_bytes(std::string(argv[1]));
#else
  auto model_name = std::string(argv[1]);
#endif

  cv::Mat image = cv::imread(argv[2]);
  CHECK(!image.empty()) << "cannot read image from " << argv[2];

  auto face_quality_onnx = FaceQualityOnnx::create(model_name);
  auto batch = face_quality_onnx->get_input_batch();
  std::vector<cv::Mat> images(batch);
  for (auto i = 0u; i < batch; ++i) {
    image.copyTo(images[i]);
  }

  __TIC__(ONNX_RUN)
  auto results = face_quality_onnx->run(images);
  __TOC__(ONNX_RUN)
  for (auto i = 0u; i < results.size(); i++) {
    LOG(INFO) << "batch " << i;
    auto image = process_result(images[i], results[i], true);
    auto out_file = std::to_string(i) + "_result.jpg";
    cv::imwrite(out_file, image);
  }

  return 0;
}

