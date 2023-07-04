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
#include <vitis/ai/efficientdet_d2.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>"
              << " <image_url>" << std::endl;
    abort();
  }

  string kernel = argv[1];
  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto detector = vitis::ai::EfficientDetD2::create(kernel, true);
  if (!detector) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  

  int width = detector->getInputWidth();
  int height = detector->getInputHeight();

  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;

  auto results = detector->run(img);
  for (auto& box : results.bboxes) {
    int label = box.label;

    float fxmin = box.x * img.cols;
    float fymin = box.y * img.rows;
    float fxmax = fxmin + box.width * img.cols;
    float fymax = fymin + box.height * img.rows;
    float confidence = box.score;

    int xmin = round(fxmin * 100.0) / 100.0;
    int ymin = round(fymin * 100.0) / 100.0;
    int xmax = round(fxmax * 100.0) / 100.0;
    int ymax = round(fymax * 100.0) / 100.0;

    xmin = std::min(std::max(xmin, 0), img.cols);
    xmax = std::min(std::max(xmax, 0), img.cols);
    ymin = std::min(std::max(ymin, 0), img.rows);
    ymax = std::min(std::max(ymax, 0), img.rows);

    cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
              1, 0);
  }
  cv::imwrite("result.jpg", img);

  return 0;
}
