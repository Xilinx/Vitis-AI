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
#include <opencv2/opencv.hpp>
#include <vitis/ai/ssd.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0]
              << " <model_name> <image_url> ... <image_url> " << std::endl;
    abort();
  }

  auto ssd = vitis::ai::SSD::create(argv[1], true);
  if (!ssd) { // supress coverity complain
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
  // auto batch = ssd->get_input_batch();
  for (auto batch_idx = 0u; batch_idx < arg_input_images.size(); batch_idx++) {
    batch_images.push_back(
        arg_input_images[batch_idx % arg_input_images.size()]);
    batch_images_names.push_back(
        arg_input_images_names[batch_idx % arg_input_images.size()]);
  }

  auto results = ssd->run(batch_images);

  for (auto batch_idx = 0u; batch_idx < results.size(); batch_idx++) {
    std::cout << "batch_index " << batch_idx << " "                     //
              << "image_name " << batch_images_names[batch_idx] << " "  //
              << std::endl;
    for (auto& box : results[batch_idx].bboxes) {
      int label = box.label;
      float fxmin = box.x * batch_images[batch_idx].cols;
      float fymin = box.y * batch_images[batch_idx].rows;
      float fxmax = fxmin + box.width * batch_images[batch_idx].cols;
      float fymax = fymin + box.height * batch_images[batch_idx].rows;
      float confidence = box.score;

      int xmin = round(fxmin * 100.0) / 100.0;
      int ymin = round(fymin * 100.0) / 100.0;
      int xmax = round(fxmax * 100.0) / 100.0;
      int ymax = round(fymax * 100.0) / 100.0;

      xmin = std::min(std::max(xmin, 0), batch_images[batch_idx].cols);
      xmax = std::min(std::max(xmax, 0), batch_images[batch_idx].cols);
      ymin = std::min(std::max(ymin, 0), batch_images[batch_idx].rows);
      ymax = std::min(std::max(ymax, 0), batch_images[batch_idx].rows);

      std::cout << "RESULT-" << batch_idx << ": " << label << "\t" << xmin
                << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t"
                << confidence << "\n";
    }
    std::cout << std::endl;
  }

  return 0;
}
