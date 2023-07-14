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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/segmentation.hpp>

static std::string get_file_name(std::string full_name) {
  std::string name = full_name.substr(0, full_name.rfind("."));
  auto pos = name.find_last_of('/');
  return name.substr(pos + 1);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  auto det = vitis::ai::Segmentation::create(argv[1]);
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

  auto results = det->run_8UC1(batch_images);

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    for (auto y = 0; y < results[batch_idx].segmentation.rows; y++) {
      for (auto x = 0; x < results[batch_idx].segmentation.cols; x++) {
        results[batch_idx].segmentation.at<uchar>(y, x) *= 10;
      }
    }
    std::string file_name = get_file_name(batch_images_names[batch_idx]);
    cv::imwrite("result_" + file_name + std::to_string(batch_idx) + ".jpg",
                results[batch_idx].segmentation);
  }

  auto results_show = det->run_8UC3(batch_images);

  for (auto batch_idx = 0u; batch_idx < results_show.size(); batch_idx++) {
    resize(results_show[batch_idx].segmentation,
           results_show[batch_idx].segmentation,
           cv::Size{results_show[batch_idx].width,
                    results_show[batch_idx].height});
    std::string file_name = get_file_name(batch_images_names[batch_idx]);
    cv::imwrite("result_show_" + file_name + std::to_string(batch_idx) + ".jpg",
                results_show[batch_idx].segmentation);
  }

  return 0;
}
