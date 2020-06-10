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
#include <iostream>
#include <memory>
#include <string>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
int main(int argc, char *argv[]) {
  auto video_file_ = string(argv[1]);
  auto is_camera =
      video_file_.size() == 1 && video_file_[0] >= '0' && video_file_[0] <= '9';
  auto video_stream = std::unique_ptr<cv::VideoCapture>(
      is_camera ? new cv::VideoCapture(std::stoi(video_file_))
                : new cv::VideoCapture(video_file_));
  auto &cap = *video_stream.get();
  if (!cap.isOpened()) {
    LOG(ERROR) << "cannot open file " << video_file_;
    return -1;
  }
  while (true) {
    cv::Mat image;
    cap >> image;
    auto video_ended = image.empty();
    if (video_ended) {
      return 0;
    }
    cv::imshow("Video", image);
    auto key = cv::waitKey(1);
    if (key == 27) {
      break;
    }
  }
  return 0;
}
