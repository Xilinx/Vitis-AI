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
#include <google/protobuf/text_format.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "vitis/ai/general.hpp"
using namespace std;

static void usage() {
  std::cout << "usage: test_general <model_name> <img_file> [<image_url> ...]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    usage();
    return 1;
  }

  auto model = vitis::ai::General::create(argv[1], true);

  if (!model) {
    std::cerr
        << "no such model, ls -l /usr/share/vitis-ai-library to see available "
           "models"
        << std::endl;
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
    std::abort();
  }

  auto total_images = 5u;
  for (auto count = 0u; count < 2u; ++count) {
    std::cout << "test number: " << count << std::endl;
    auto img_idx = 0u;
    do {
      std::vector<cv::Mat> batch_images;
      std::vector<std::string> batch_images_names;
      auto batch = model->get_input_batch();
      std::cout << "batch size = " << batch << std::endl;
      for (auto batch_idx = 0u; batch_idx < batch && img_idx < total_images;
           batch_idx++, img_idx++) {
        batch_images.push_back(
            arg_input_images[img_idx % arg_input_images.size()]);
        batch_images_names.push_back(
            arg_input_images_names[img_idx % arg_input_images.size()]);
      }
      auto results = model->run(batch_images);

      for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
        std::cout << "batch_index " << batch_idx << " "                     //
                  << "image_name " << batch_images_names[batch_idx] << " "  //
                  << std::endl;
        std::cout << "result = " << results[batch_idx].DebugString()
                  << std::endl;
      }
    } while (img_idx < total_images);
  }
  return 0;
}
