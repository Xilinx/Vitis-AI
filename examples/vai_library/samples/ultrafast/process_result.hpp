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
#include <sys/stat.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "vitis/ai/ultrafast.hpp"

using namespace cv;
using namespace std;

Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(255, 255, 0), Scalar(0, 0, 255) };

static cv::Mat process_result(
  cv::Mat &img, const vitis::ai::UltraFastResult &result, bool is_jpeg) {

  int iloop = 0;
  for(auto &lane: result.lanes) {
     std::cout <<"lane: " << iloop << "\n";
     for(auto &v: lane) {
        if(v.first >0) {
          cv::circle(img, cv::Point(v.first, v.second), 5, colors[iloop], -1);
        }
        std::cout << "    ( " << v.first << ", " << v.second << " )\n";
     }
     iloop++;
  }

  return img;
}

