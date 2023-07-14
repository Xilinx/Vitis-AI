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
#include <vitis/ai/segmentation.hpp>

using namespace std;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage :" << argv[0] << "<image_url>" << std::endl;
    abort();
  }

  auto det = vitis::ai::Segmentation::create(argv[1]);  // Init
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }
  auto image = cv::imread(argv[2]);                     // Load an input image;
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }
  cout << "Run and get the result" << endl;
  auto result = det->run_8UC1(image);

  for (auto y = 0; y < result.segmentation.rows; y++) {
    for (auto x = 0; x < result.segmentation.cols; x++) {
      result.segmentation.at<uchar>(y, x) *= 10;
    }
  }
  // for (size_t i = 0; i < result.segmentation.data().size(); i++) {
  // result.segmentation[i] = result.segmentation[i] * 10;

  // It's hard to see the result clearly Because the original result is range
  // from 0 to number of classes - 1. So I multiply the result by a scale to
  // look brightly.
  //}
  // cv::Mat img(result.rows, result.cols, CV_8UC1, result.segmentation.data());
  // //Init a cv::Mat to save the segmentation result
  cv::imwrite("segres.jpg",
              result.segmentation);  // Save the result as an image;
  cout << "Run and get a visualization result" << endl;
  auto resultshow = det->run_8UC3(image);
  resize(
      resultshow.segmentation, resultshow.segmentation,
      cv::Size{
          resultshow.width,
          resultshow.height});  // Resize the result Mat as same as input size;
  cv::imwrite("segres2.jpg",
              resultshow.segmentation);  // Save the result as an image;
  return 0;
}
