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
#pragma once
using namespace cv;
using namespace std;
using Result = vitis::ai::OpenPoseResult::PosePoint;

static cv::Mat process_result(cv::Mat& image, vitis::ai::OpenPoseResult results,
                              bool is_jpeg) {
  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 3},  {3, 4}, {1, 5},
                                 {5, 6},  {6, 7},   {1, 8},  {8, 9}, {9, 10},
                                 {1, 11}, {11, 12}, {12, 13}};
  for (size_t k = 1; k < results.poses.size(); ++k) {
    for (size_t i = 0; i < results.poses[k].size(); ++i) {
      if (results.poses[k][i].type == 1) {
        cv::circle(image, results.poses[k][i].point, 5, cv::Scalar(0, 255, 0),
                   -1);
      }
    }
    for (size_t i = 0; i < limbSeq.size(); ++i) {
      Result a = results.poses[k][limbSeq[i][0]];
      Result b = results.poses[k][limbSeq[i][1]];
      if (a.type == 1 && b.type == 1) {
        cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
      }
    }
  }
  return image;
}
