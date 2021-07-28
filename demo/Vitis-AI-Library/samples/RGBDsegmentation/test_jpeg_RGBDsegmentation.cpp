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
#include <vitis/ai/RGBDsegmentation.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0]
         << " <model_name> <image_bgr_file_url> <image_hha_file_url> " << endl;
    abort();
  }
  Mat img_bgr = cv::imread(argv[2]);
  Mat img_hha = cv::imread(argv[3]);
  if (img_bgr.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }
  if (img_hha.empty()) {
    cerr << "cannot load " << argv[3] << endl;
    abort();
  }
  auto segmentation = vitis::ai::RGBDsegmentation::create(argv[1], true);

  auto result = segmentation->run(img_bgr, img_hha);

  imwrite(string(argv[1]) + "_result.png", result.segmentation);
  cout << "The result is written in " << argv[1] << "_result.png" << endl;

  auto get_color = [](map<uint8_t, Vec3b>& colors, uint8_t i) {
    if (!colors.count(i)) {
      auto id = i;
      uint8_t r = 0, g = 0, b = 0;
      for (uint8_t j = 0; j < 7; j++) {
        r = r ^ (((id >> 0) & 1) << (7 - j));
        g = g ^ (((id >> 1) & 1) << (7 - j));
        b = b ^ (((id >> 2) & 1) << (7 - j));
        id = id >> 3;
      }
      colors[i] = Vec3b(b, g, r);
    }
    return colors[i];
  };

  Mat result_color(result.segmentation.size(), CV_8UC3);
  map<uint8_t, Vec3b> colors;
  for (int i = 0; i < result_color.rows; i++) {
    for (int j = 0; j < result_color.cols; j++) {
      result_color.at<Vec3b>(i, j) =
          get_color(colors, result.segmentation.at<uchar>(i, j));
    }
  }

  imwrite(string(argv[1]) + "_color_result.png", result_color);
  cout << "The result is written in " << argv[1] << "_color_result.png" << endl;
  return 0;
}
