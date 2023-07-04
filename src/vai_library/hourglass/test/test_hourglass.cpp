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
#include <vitis/ai/hourglass.hpp>

using namespace std;
using namespace cv;
using Result = vitis::ai::HourglassResult::PosePoint;

int main(int argc, char *argv[]) {
  auto image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "cannot load " << argv[2] << std::endl;
    abort();
  }
  auto det = vitis::ai::Hourglass::create(argv[1]);
  if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 6},  {3, 6},  {3, 4}, {4, 5},
                                 {6, 7},   {7, 8},  {8, 9}, {7, 12},
                                 {12, 11}, {11, 10}, {7, 13}, {13, 14}, {14, 15}};
  
  auto results = det->run(image.clone());
  for (size_t i = 0; i < results.poses.size(); ++i) {
    cout<< results.poses[i].point<<endl;
    if (results.poses[i].type == 1) {
      cv::putText(image, to_string(i),results.poses[i].point, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 1, 0);
      cv::circle(image, results.poses[i].point, 5, cv::Scalar(0, 255, 0),
                 -1);
    }
  }
  for (size_t i = 0; i < limbSeq.size(); ++i) {
    Result a = results.poses[limbSeq[i][0]];
    Result b = results.poses[limbSeq[i][1]];
    if (a.type == 1 && b.type == 1) {
      cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
    }
  }
  cv::imwrite("sample_hourglass_result.jpg", image);
  return 0;
}
