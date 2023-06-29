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
    cerr << "usage: " << argv[0] << " <model_name> <image_file_url> " << endl;
    abort();
  }
  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[1] << endl;
    abort();
  }
  auto model = vitis::ai::FCOS::create(argv[1], true);

  auto results = model->run(img);

  for (auto& result : results.bboxes) {
    int label = result.label;
    auto& box = result.box;

    cout << "RESULT: " << label << "\t" << std::fixed << std::setprecision(2)
         << box[0] << "\t" << box[1] << "\t" << box[2] << "\t" << box[3] << "\t"
         << std::setprecision(6) << result.score << "\n";
    rectangle(img, Point(box[0], box[1]), Point(box[2], box[3]),
              Scalar(0, 255, 0), 1, 1, 0);
  }
  imwrite("result.jpg", img);

  return 0;
}
