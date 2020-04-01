/*
 * Copyright 2019 Xilinx Inc.
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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <model_name>" <<" <image_url>" << std::endl;
    abort();
  }

  string kernel = argv[1];
  Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto ssd =
      vitis::ai::SSD::create(kernel, true);

  int width = ssd->getInputWidth();
  int height = ssd->getInputHeight();

  std::cout << "width " << width << " "   //
            << "height " << height << " " //
            << std::endl;

  cv::Mat img_resize;

  cv::resize(img, img_resize, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  auto results = ssd->run(img_resize);
  for (auto &box : results.bboxes) {
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
    if (label == 1) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1,
                1, 0);
    } else if (label == 2) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0), 1,
                1, 0);
    } else if (label == 3) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1,
                1, 0);
    } else if (label == 4) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 255),
                1, 1, 0);
    }
  }

  return 0;
}
