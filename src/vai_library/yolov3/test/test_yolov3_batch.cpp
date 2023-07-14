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

#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vitis/ai/yolov3.hpp>

#include "../src/utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url> [<image_url> ...]" << std::endl;
    abort();
  }

  vector<Mat> imgs;
  vector<string> imgs_names;
  for (int i = 2; i < argc; i++) {
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

  auto model = vitis::ai::YOLOv3::create(argv[1]);
  if (!model) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
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
      for (const auto &bbox : result_vec[k].bboxes) {
        int label = bbox.label;
        float xmin = bbox.x * inputs[k].cols + 1;
        float ymin = bbox.y * inputs[k].rows + 1;
        float xmax = xmin + bbox.width * inputs[k].cols;
        float ymax = ymin + bbox.height * inputs[k].rows;
        if (xmin < 0.) xmin = 1.;
        if (ymin < 0.) ymin = 1.;
        if (xmax > inputs[k].cols) xmax = inputs[k].cols;
        if (ymax > inputs[k].rows) ymax = inputs[k].rows;
        float confidence = bbox.score;

        cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
             << xmax << "\t" << ymax << "\t" << confidence << "\n";
        rectangle(inputs[k], Point(xmin, ymin), Point(xmax, ymax),
                  Scalar(0, 255, 0), 1, 1, 0);
      }
      imwrite(imgs_names[i - j + k] + "_result.jpg", inputs[k]);
    }
    inputs.clear();
    j = -1;
  }
  return 0;
}
