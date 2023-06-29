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
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "vitis/ai/facequality5pt.hpp"

int main(int argc,char *argv[]){
  if (argc < 3) {
    std::cout << "usage : " << argv[0] << " <model_name> <img_url> [<img_url> ...]"
              << std::endl;
    return -1;
  }

  auto qual_tester = vitis::ai::FaceQuality5pt::create(argv[1], true);
  if (!qual_tester) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  //auto qual_tester = vitis::ai::FaceQuality5pt::create(vitis::ai::FACE_QUALITY5PT_NORMAL,true);
  qual_tester->setMode(vitis::ai::FaceQuality5pt::Mode::DAY);

  int width = qual_tester->getInputWidth();
  int height = qual_tester->getInputHeight();

  for (int i = 2; i < argc; i++) {
    cv::Mat image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      continue;
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);

    auto result = qual_tester->run(img_resize);

    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
         << "image " << argv[i] << " "      //
         << " day mode "
         << "quality " << result.score << std::endl; //
    // Use night mode
    qual_tester->setMode(vitis::ai::FaceQuality5pt::Mode::NIGHT);
    result = qual_tester->run(img_resize);
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ":"
         << "image " << argv[i] << " "      //
         << " night mode "
         << "quality " << result.score << std::endl //
         << "points ";

    for (int i = 0; i < 5; ++i) {
        std::cout << result.points[i].first * img_resize.cols << " " 
                  << result.points[i].second * img_resize.rows << " ";
        auto pt = cv::Point{(int)(result.points[i].first * img_resize.cols),
                             (int)(result.points[i].second * img_resize.rows)};
         cv::circle(img_resize, pt, 3, cv::Scalar(0,255,255));
    }
    std::cout << std::endl;
    cv::imwrite("out.jpg", img_resize);
  }
  return 0;
}
