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
#include <vitis/ai/multitask.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  auto det = vitis::ai::MultiTask::create(argv[1]);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  auto image = cv::imread(argv[2]);
  cout << "read img" << endl;
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  auto result = det->run_8UC1(image);
  for (auto y = 0; y < result.segmentation.rows; y++) {
    for (auto x = 0; x < result.segmentation.cols; x++) {
      result.segmentation.at<uchar>(y, x) *= 10;
    }
  }
  cv::imwrite("segres.jpg", result.segmentation);

  auto resultshow = det->run_8UC3(image);
  resize(resultshow.segmentation, resultshow.segmentation,
         cv::Size(resultshow.width, resultshow.height));
  cout << resultshow.vehicle.size() << endl;
  for (size_t i = 0; i < resultshow.vehicle.size(); i++) {
    int xmin = resultshow.vehicle[i].x * resultshow.segmentation.cols;
    int ymin = resultshow.vehicle[i].y * resultshow.segmentation.rows;
    // int xmax = xmin + resultshow.vehicle[i].width *
    // resultshow.segmentation.cols;
    // int ymax = ymin + resultshow.vehicle[i].height *
    // resultshow.segmentation.rows;
    // rectangle(resultshow.segmentation, Point(xmin, ymin), Point(xmax,
    // ymax), Scalar(185, 181, 178), 2,
    //           1, 0);
    int width = resultshow.vehicle[i].width * resultshow.segmentation.cols;
    int height = resultshow.vehicle[i].height * resultshow.segmentation.rows;
    rectangle(resultshow.segmentation, Rect_<int>(xmin, ymin, width, height),
              Scalar(185, 181, 178), 2, 1, 0);
    cout << "car: " << i                                      //
         << " score: " << resultshow.vehicle[i].score << " "  //
         << " orientation " << resultshow.vehicle[i].angle << endl;
  }
  cv::imwrite("segres2.jpg", resultshow.segmentation);
  return 0;
}
