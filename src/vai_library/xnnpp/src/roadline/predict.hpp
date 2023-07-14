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

//
// Created by sheng on 2018/5/18.
//

#ifndef SRC_PREDICT_H
#define SRC_PREDICT_H

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ipm_info.hpp"

#define CNUM 20

using namespace std;
using namespace cv;

namespace vitis {
namespace nnpp {
namespace roadline {

class Predict {
 public:
  Predict(float ipm_width, float ipm_height);
  ~Predict() {}
  bool curve_fit(std::vector<cv::Point> &key_point, int n, cv::Mat &A);
  void findLocalmaximum(vector<int> &datase, vector<cv::Point_<int>> &seed);
  void cluster(const vector<int> &outImage, vector<int> &clusters);
  void voteClassOfClusters(const vector<int> &datase,
                           const vector<int> &clusters, vector<int> &types);
  int majorityElement(const vector<int> &nums);
  int getMaxX(const vector<cv::Point> &points);
  int getMinX(const vector<cv::Point> &points);
  cv::Scalar getScalarOfType(int type);

 private:
  float ipm_width_;
  float ipm_height_;
};

}  // namespace roadline
}  // namespace nnpp
}  // namespace vitis

#endif  // SRC_PREDICT_H
