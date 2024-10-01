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
#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vitis/ai/facelandmark.hpp>
using namespace vitis::ai;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << " usage: " << argv[0] << " <img_url> [<img_url> ...]"
              << std::endl;  //
    abort();
  }

  auto landmark = FaceLandmark::create(argv[1], true);
  if (!landmark) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  int width = landmark->getInputWidth();
  int height = landmark->getInputHeight();

  std::cout << "InputWidth " << width << " "
            << "InputHeight " << height << " " << std::endl;

  for (int i = 2; i < argc; i++) {
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
              << "image " << argv[i] << " " << std::endl;
    cv::Mat image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      continue;
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    // cv::imwrite("resize_after.jpg", img_resize);
    // std::cout << "resize success,will be run!"<< std::endl; //

    auto result = landmark->run(img_resize);
    auto points = result.points;

    std::cout << "points ";  //
    for (int i = 0; i < 5; ++i) {
      std::cout << points[i].first << " " << points[i].second << " "
                << std::endl;
      auto pt = cv::Point{(int)(points[i].first * img_resize.cols),
                          (int)(points[i].second * img_resize.rows)};
      std::cout << pt << std::endl;
      cv::circle(img_resize, pt, 3, cv::Scalar(0, 255, 255));
    }
    std::cout << std::endl;
    cv::imwrite("out.jpg", img_resize);
  }
  return 0;
}
