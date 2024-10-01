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

#include <sys/stat.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/medicalsegcell.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << " usage: " << argv[0] << " <model_name> <result_file> <img_url>" << std::endl;  //
    abort();
  }

  Mat img = cv::imread(argv[3]);
  if (img.empty()) {
    cerr << "cannot load " << argv[3] << endl;
    abort();
  }

  auto seg = vitis::ai::MedicalSegcell::create(argv[1] );
  auto result = seg->run(img);
  cv::imwrite(argv[2], result.segmentation);

  return 0;
}

