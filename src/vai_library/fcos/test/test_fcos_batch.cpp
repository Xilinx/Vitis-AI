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

#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/fcos.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  vector<Mat> imgs;
  vector<string> imgs_names;
  for (int i = 1; i < argc; i++) {
    auto img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    imgs.push_back(img);
    imgs_names.push_back(argv[i]);
  }
  if (imgs.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }
  auto model = vitis::ai::FCOS::create(argv[1]);
  auto batch = static_cast<int>(model->get_input_batch());
  int imgs_size = static_cast<int>(imgs.size());
  vector<Mat> inputs;
  for (int i = 0, j = -1; i < imgs_size; i++) {
    inputs.push_back(imgs[i]);
    j++;
    if (j < batch - 1 && i < imgs_size - 1) {
      continue;
    }

    auto result_vec = model->run(inputs);
    for (int k = 0; k < static_cast<int>(inputs.size()); k++) {
      cout << "batch_index " << k << " "                     //
           << "image_name " << imgs_names[i - j + k] << " "  //
           << std::endl;

      for (auto& result : result_vec[k].bboxes) {
        int label = result.label;
        auto& box = result.box;

        cout << "RESULT: " << label << "\t" << std::fixed
             << std::setprecision(2) << box[0] << "\t" << box[1] << "\t"
             << box[2] << "\t" << box[3] << "\t" << std::setprecision(6)
             << result.score << "\n";
        rectangle(inputs[k], Point(box[0], box[1]), Point(box[2], box[3]),
                  Scalar(0, 255, 0), 1, 1, 0);
      }
      imwrite(imgs_names[i - j + k] + "_result.jpg", inputs[k]);
    }
    inputs.clear();
    j = -1;
  }
  return 0;
}
