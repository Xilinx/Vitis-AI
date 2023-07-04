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
#include <vitis/ai/facefeature.hpp>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "usage : " << argv[0] << "<model_name>"
              << " <img_url> [<img_url> ...]" << std::endl;
    return -1;
  }

  auto facefeature = vitis::ai::FaceFeature::create(argv[1], true);
  if (!facefeature) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  std::vector<cv::Mat> arg_input_images;
  std::vector<std::string> arg_input_images_names;
  for (int i = 2; i < argc; i++) {
    cv::Mat image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(image);
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_images_names;
  auto batch = facefeature->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  int width = facefeature->getInputWidth();
  int height = facefeature->getInputHeight();

  std::vector<cv::Mat> new_images;
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    cv::Mat img_resize;

    cv::resize(batch_images[batch_idx], img_resize, cv::Size(width, height), 0,
               0, cv::INTER_LINEAR);
    cv::imwrite("resize_after_" + std::to_string(batch_idx) +
                    batch_images_names[batch_idx],
                img_resize);
    new_images.push_back(img_resize);
  }

  auto results = facefeature->run(new_images);
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
              << "image_name: " << batch_images_names[batch_idx] << " "  //
              << " float_features:";                                     //
    for (auto feature : *(results[batch_idx].feature)) {
      std::cout << feature << " ";  //
    }
    std::cout << std::endl;
  }

  auto results_fixed = facefeature->run_fixed(new_images);
  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index: " << batch_idx << "   "                   //
              << "image_name: " << batch_images_names[batch_idx] << " "  //
              << " fixed_features:";                                     //
    for (float feature : *(results_fixed[batch_idx].feature)) {
      std::cout << feature << " ";  //
    }
    std::cout << std::endl;
  }
  return 0;
}
