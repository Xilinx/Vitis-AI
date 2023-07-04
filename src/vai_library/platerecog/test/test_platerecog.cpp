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
#include <vitis/ai/platerecog.hpp>

using namespace std;
using namespace vitis::ai;

int main(int argc, char *argv[]) {
  auto det = vitis::ai::PlateRecog::create(argv[1], argv[2], true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }

  for (int i = 3; i < argc; i++) {
    auto image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      abort();
    }
    auto result = det->run(image);
    std::cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "argv[i] " << argv[i] << " "                                 //
              << "plate.score " << result.box.score << " "                    //
              << "plate.x " << result.box.x << " "                            //
              << "plate.y " << result.box.y << " "                            //
              << "plate.width " << result.box.width << " "                    //
              << "plate.height " << result.box.height << " "                  //
              << "result.plate_color " << result.plate_color << " "           //
              << "result.plate_number " << result.plate_number << " "         //
              << std::endl;

    auto rect =
        cv::Rect{cv::Point{(int)(result.box.x), (int)(result.box.y)},
                 cv::Size{(int)(result.box.width), (int)(result.box.height)}};
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255));
    cv::imwrite("plate_num_det.jpg", image);
  }

  return 0;
}
