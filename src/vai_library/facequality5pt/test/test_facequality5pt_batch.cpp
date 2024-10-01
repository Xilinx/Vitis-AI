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

  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  auto batch = qual_tester->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

 
  std::vector<cv::Mat> new_images;
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    cv::Mat img_resize;
    cv::resize(batch_images[batch_idx], img_resize, cv::Size(width, height), 0,
               0, cv::INTER_LINEAR);
    cv::imwrite("resize_after_" + batch_images_names[batch_idx], img_resize);
    new_images.push_back(img_resize);
    // std::cout << "resize success,will be run!"<< std::endl; //
  }

  auto results = qual_tester->run(new_images);
  std::cout << std::endl;

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
         << "image_name: " << batch_images_names[batch_idx] << " "  //
         << " day mode "
         << "quality " << results[batch_idx].score << " " //
         << "points ";

    for (int i = 0; i < 5; ++i) {
      std::cout << results[batch_idx].points[i].first * new_images[batch_idx].cols << " " 
                << results[batch_idx].points[i].second * new_images[batch_idx].rows << " ";
      auto pt = cv::Point{(int)(results[batch_idx].points[i].first * new_images[batch_idx].cols),
                          (int)(results[batch_idx].points[i].second * new_images[batch_idx].rows)};
       cv::circle(new_images[batch_idx], pt, 3, cv::Scalar(0,255,255));
    }
    std::cout << std::endl;
    cv::imwrite("out_" + batch_images_names[batch_idx], new_images[batch_idx]);
  }
  return 0;
}
