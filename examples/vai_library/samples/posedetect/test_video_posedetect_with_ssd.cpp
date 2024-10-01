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
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/posedetect.hpp>

#include "./process_result.hpp"
#include "./ssd_posedetect.hpp"
using namespace std;
cv::Mat process_result_ssd(
    cv::Mat &image, const std::vector<vitis::ai::PoseDetectResult> &results,
    bool is_jpeg) {
  for (auto &result : results) {
    process_result(image, result, is_jpeg);
  }
  return image;
}

using namespace std;
int main(int argc, char *argv[]) {
  return vitis::ai::main_for_video_demo(
      argc, argv, [] { return vitis::ai::SSDPoseDetect::create(); },
      process_result_ssd);
}
