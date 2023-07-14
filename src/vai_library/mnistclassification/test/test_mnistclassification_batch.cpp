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

#include <sys/stat.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <algorithm>
#include <vitis/ai/mnistclassification.hpp>

using namespace cv;
using namespace std;

std::vector<std::string> obj = {"zero","one","two","three","four","five","six","seven","eight","nine"};

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <modelname> <img_url> ... <img_url>" << std::endl;  //
    abort();
  }
  auto net = vitis::ai::MnistClassification::create(argv[1]);
  if (!net) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  std::vector<cv::Mat> arg_input_images;
  std::vector<cv::Size> arg_input_images_size;
  std::vector<std::string> arg_input_images_names;
  for (auto i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    arg_input_images.push_back(img);
    arg_input_images_size.push_back(img.size());
    arg_input_images_names.push_back(argv[i]);
  }

  if (arg_input_images.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  auto batch = net->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < batch; batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
  }

  auto result = net->run(batch_images);
  for (auto batch_idx = 0u; batch_idx < result.size(); batch_idx++) {
    std::cout << "result for " << batch_idx << " :  " << obj[result[batch_idx].classIdx] << "\n";
  }

  return 0;
}

