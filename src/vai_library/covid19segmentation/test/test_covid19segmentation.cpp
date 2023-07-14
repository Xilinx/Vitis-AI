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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/covid19segmentation.hpp>

using namespace std;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage :" << argv[0] << " <modelname> <image_url> " << std::endl;
    abort();
  }

  auto det = vitis::ai::Covid19Segmentation::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto image = cv::imread(argv[2]);                     // Load an input image;
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }
  cout << "Run and get a visualization result" << endl;
  auto resultshow = det->run_8UC3(image);
  resize(
      resultshow.positive_classification, resultshow.positive_classification,
      cv::Size{
          resultshow.width,
          resultshow.height});  // Resize the result Mat as same as input size;
  cv::imwrite("classification.jpg",
              resultshow.positive_classification);  // Save the result as an image;
  resize(
      resultshow.infected_area_classification, resultshow.infected_area_classification,
      cv::Size{
          resultshow.width,
          resultshow.height});  // Resize the result Mat as same as input size;
  cv::imwrite("infected_area.jpg",
              resultshow.infected_area_classification);  // Save the result as an image;
  return 0;
}
