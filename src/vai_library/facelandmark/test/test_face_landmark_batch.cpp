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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  auto det = vitis::ai::FaceLandmark::create(argv[1], true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

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
  auto batch = det->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  int width = det->getInputWidth();
  int height = det->getInputHeight();

  std::cout << "InputWidth " << width << " "
            << "InputHeight " << height << " " << std::endl;

  std::vector<cv::Mat> new_images;
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    cv::Mat img_resize;
    cv::resize(batch_images[batch_idx], img_resize, cv::Size(width, height), 0,
               0, cv::INTER_LINEAR);
    cv::imwrite("resize_after_" + batch_images_names[batch_idx], img_resize);
    new_images.push_back(img_resize);
    // std::cout << "resize success,will be run!"<< std::endl; //
  }

  auto results = det->run(new_images);
  std::cout << std::endl;
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
              << "image_name: " << batch_images_names[batch_idx] << " "  //
              << std::endl;
    std::cout << "points: " << std::endl;
    auto canvas = new_images[batch_idx];
    for (const auto &p : results[batch_idx].points) {
      std::cout << p.first << " " << p.second << " " << std::endl;
      auto pt = cv::Point{(int)(p.first * canvas.cols),
                          (int)(p.second * canvas.rows)};
      std::cout << pt << std::endl;
      cv::circle(canvas, pt, 3, cv::Scalar(0, 255, 255));
    }
    std::cout << std::endl;
    cv::imwrite("out_" + batch_images_names[batch_idx], canvas);
  }

  return 0;
}
