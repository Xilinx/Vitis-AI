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
#include <stdio.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/carplaterecog.hpp>

using namespace std;
using namespace vitis::ai;

int main(int argc, char *argv[]) {
  auto det = vitis::ai::CarPlateRecog::create(argv[1], argv[2], argv[3], true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }   

  for (int i = 4; i < argc; i++) {
    auto image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      abort();
    }
    auto result = det->run(image);
    for (auto &r: result.platerecogs) {
      std::cout << "car pos: " << r.first.x <<  "\t" <<r.first.y << "\t" << r.first.width << "\t" << r.first.height << "\t license plate number: "<< r.second.plate_number << std::endl;
    }
  }

  return 0;
}
