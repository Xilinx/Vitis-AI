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
#include <vitis/ai/tfssd.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << " usage: " << argv[0] << " <model_name> "
              << " [<img_url> ... ]" << std::endl;
    abort();
  }

  auto tfssd = vitis::ai::TFSSD::create(argv[1], true);
  if (!tfssd) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   
  auto batch = tfssd->get_input_batch();
  int width = tfssd->getInputWidth();
  int height = tfssd->getInputHeight();
  std::vector<cv::Mat> images;
  std::vector<std::string> names;

  for (auto i = 2; i < argc; i++) {
    auto img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    images.push_back(img);
    names.push_back(argv[i]);
  }

  if (images.empty()) {
    std::cerr << "No image load cussess !" << std::endl;
    abort();
  }

  std::vector<cv::Mat> batch_images;
  std::vector<std::string> batch_names;
  for (auto i = 0u; i < batch; i++) {
    batch_images.push_back(images[i % images.size()]);
    batch_names.push_back(names[i % images.size()]);
  }

  std::cout << "width " << width << " "                        //
            << "height " << height << " "                      //
            << "batch " << batch << " "                        //
            << "images.size() " << batch_images.size() << " "  //
            << std::endl;
  auto res = tfssd->run(batch_images);

  auto pos = 0;
  for (auto &results : res) {
    std::cout << "idx: " << pos << "  file: " << batch_names[pos] << std::endl;
    for (auto &box : results.bboxes) {
      int label = box.label;

      float fxmin = box.x * batch_images[pos].cols;
      float fymin = box.y * batch_images[pos].rows;
      float fxmax = fxmin + box.width * batch_images[pos].cols;
      float fymax = fymin + box.height * batch_images[pos].rows;
      float confidence = box.score;

      int xmin = round(fxmin * 100.0) / 100.0;
      int ymin = round(fymin * 100.0) / 100.0;
      int xmax = round(fxmax * 100.0) / 100.0;
      int ymax = round(fymax * 100.0) / 100.0;

      xmin = std::min(std::max(xmin, 0), batch_images[pos].cols);
      xmax = std::min(std::max(xmax, 0), batch_images[pos].cols);
      ymin = std::min(std::max(ymin, 0), batch_images[pos].rows);
      ymax = std::min(std::max(ymax, 0), batch_images[pos].rows);

      cout << "  Label:" << label << "\t" << xmin << "\t" << ymin << "\t"
           << xmax << "\t" << ymax << "\t" << confidence << "\n";
    }
    pos++;
  }
  return 0;
}
