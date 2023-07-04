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
#include <vitis/ai/RGBDsegmentation.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "usage: " << argv[0]
         << "<model_name> <image_bgr_file_url> <image_hha_file_url> " << endl;
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
  if (!segmentation) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  auto result = segmentation->run(img_bgr, img_hha);

  imwrite(string(argv[1]) + "_result.png", result.segmentation);
  cout << "The result is written in " << argv[1] << "_result.png" << endl;

  return 0;
}
