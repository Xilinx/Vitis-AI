/*
 * Copyright 2019 Xilinx Inc.
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

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(IMSHOW, "1");
DEF_ENV_PARAM(MAX_FPS, "0");
using namespace std;
int main(int argc, char* argv[]) {
  auto video_file_ = string(argv[1]);
  auto is_camera =
      video_file_.size() == 1 && video_file_[0] >= '0' && video_file_[0] <= '9';
  auto video_stream = std::unique_ptr<cv::VideoCapture>(
      is_camera ? new cv::VideoCapture(std::stoi(video_file_))
                : new cv::VideoCapture(video_file_));
  auto& cap = *video_stream.get();
  if (!cap.isOpened()) {
    LOG(ERROR) << "cannot open file " << video_file_;
    return -1;
  }
  int f = 0;
  auto now = std::chrono::steady_clock::now();
  while (true) {
    cv::Mat image;
    cap >> image;
    if (ENV_PARAM(MAX_FPS) != 0) {
      cap.set(cv::CAP_PROP_FPS, ENV_PARAM(MAX_FPS));
    }
    auto video_ended = image.empty();
    if (video_ended) {
      return 0;
    }
    if (ENV_PARAM(IMSHOW)) {
      cv::imshow("Video", image);
      auto key = cv::waitKey(1);
      if (key == 27) {
        break;
      }
    }
    auto prop_fps = cap.get(cv::CAP_PROP_FPS);
    auto e = std::chrono::steady_clock::now();
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(e - now).count();
    LOG(INFO) << "frame " << f << " FPS: "
              << static_cast<float>(f) / static_cast<float>(us) * 1e6
              << " prop_fps = " << prop_fps;
    f = f + 1;
  }
  return 0;
}
