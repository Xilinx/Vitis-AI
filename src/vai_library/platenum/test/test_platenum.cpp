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
#include <memory>
#include <opencv2/opencv.hpp>
#include <vitis/ai/platenum.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  cout << "init " << endl;
  auto det = vitis::ai::PlateNum::create(argv[1], true);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  cout << "init end" << endl;
  int width = det->getInputWidth();
  int height = det->getInputHeight();
  std::cout << "width " << width << " "    //
            << "height " << height << " "  //
            << std::endl;
  cout << "get hw" << endl;
  for (int i = 2; i < argc; i++) {
    auto image = cv::imread(argv[i]);
    if (image.empty()) {
      std::cout << "cannot load " << argv[i] << std::endl;
      abort();
    }

    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0,
               cv::INTER_LINEAR);
    cout << "start process " << endl;
    auto result = det->run(img_resize);
    std::cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "argv[i] " << argv[i] << " "                                 //
              << "result.width " << result.width << " "                       //
              << "result.height " << result.height << " "                     //
              << "result.plate_color " << result.plate_color << " "           //
              << "result.plate_number " << result.plate_number << " "         //
              << std::endl;
  }
  return 0;
}
