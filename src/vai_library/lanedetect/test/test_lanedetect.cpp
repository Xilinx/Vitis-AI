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
#include <vitis/ai/lanedetect.hpp>

using namespace std;
using namespace cv;
using namespace vitis::ai;

int main(int argc, char *argv[]) {
  auto det = vitis::ai::RoadLine::create(argv[1]);
  if (!det) { // supress coverity complain
     std::cerr <<"create error\n";
     abort();
  }

  auto image = cv::imread(argv[2]);
  //    Mat image;
  //    resize(img, image, Size(640, 480));
  if (image.empty()) {
    cerr << "cannot load " << argv[2] << endl;
    abort();
  }

  vector<int> color1 = {0, 255, 0, 0, 100, 255};
  vector<int> color2 = {0, 0, 255, 0, 100, 255};
  vector<int> color3 = {0, 0, 0, 255, 100, 255};

  RoadLineResult results = det->run(image);
  for (auto &line : results.lines) {
    vector<Point> points_poly = line.points_cluster;
    // for (auto &p : points_poly) {
    //  std::cout << p.x << " " << (int)p.y << std::endl;
    //}
    int type = line.type < 5 ? line.type : 5;
    if (type == 2 && points_poly[0].x < image.rows * 0.5) continue;
    cv::polylines(image, points_poly, false,
                  Scalar(color1[type], color2[type], color3[type]), 3, LINE_AA,
                  0);
  }
  cv::imwrite("roadline_results.jpg", image);
  // waitKey(0);
  return 0;
}
