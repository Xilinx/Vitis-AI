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
#include <vitis/ai/multitaskv3.hpp>
extern int GLOBAL_ENABLE_C_SOFTMAX;
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  GLOBAL_ENABLE_C_SOFTMAX = 1;
  auto det = vitis::ai::MultiTaskv3::create(argv[1]);
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

  auto result = det->run_8UC3(image);
  //for (auto y = 0; y < result.segmentation.rows; y++) {
  //  for (auto x = 0; x < result.segmentation.cols; x++) {
  //    result.segmentation.at<uchar>(y, x) *= 13;
  //  }
  //}
  //for (auto y = 0; y < result.drivable.rows; y++) {
  //  for (auto x = 0; x < result.drivable.cols; x++) {
  //    result.drivable.at<uchar>(y, x) *= 40;
  //  }
  //}
  //for (auto y = 0; y < result.lane.rows; y++) {
  //  for (auto x = 0; x < result.lane.cols; x++) {
  //    result.lane.at<uchar>(y, x) *= 40;
  //  }
  //}
  cv::imwrite("drivale.jpg", result.drivable);
  cv::imwrite("lane.jpg", result.lane);
  cv::imwrite("depth.png", result.depth);

  cv::imwrite("segmentation.jpg", result.segmentation);
  for (size_t i = 0; i < result.vehicle.size(); i++) {
    int xmin = result.vehicle[i].x * image.cols;
    int ymin = result.vehicle[i].y * image.rows;
    int width = result.vehicle[i].width * image.cols;
    int height = result.vehicle[i].height * image.rows;
    rectangle(image, Rect_<int>(xmin, ymin, width, height),
              Scalar(185, 181, 178), 2, 1, 0);
    cout << "label " << result.vehicle[i].label<< " "                                      //
         << " score: " << result.vehicle[i].score << " xmin: "  //
         << xmin <<" ymin: " << ymin << " width: " << width << " height: " << height << endl;
  }
  cv::imwrite("detect.jpg", image);
  return 0;
}
