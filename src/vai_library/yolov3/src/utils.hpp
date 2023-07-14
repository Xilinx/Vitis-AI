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
/*
  The following source code derives from Darknet
*/

#include <math.h>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace vitis {
namespace ai {
namespace yolov3 {

typedef struct {
  int w;
  int h;
  int c;
  float* data;
} image;

void convertInputImage(const cv::Mat& frame, int width, int height, int channel,
                       float scale, int8_t* data);
//# Overload method with float input data for DPUV1
void convertInputImage(const cv::Mat& frame, int width, int height, int channel,
                       float scale, float* data);

cv::Mat letterbox(const cv::Mat& im, int w, int h);
cv::Mat letterbox_tf(const cv::Mat& im, int w, int h);

}  // namespace yolov3
}  // namespace ai
}  // namespace vitis
