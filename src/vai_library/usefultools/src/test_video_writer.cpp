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

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(IMSHOW, "1");
DEF_ENV_PARAM(MAX_FPS, "0");
using namespace std;

std::unique_ptr<cv::VideoCapture> open_input(const std::string& video_file_) {
  auto is_camera =
      video_file_.size() == 1 && video_file_[0] >= '0' && video_file_[0] <= '9';
  auto video_stream = std::unique_ptr<cv::VideoCapture>(
      is_camera ? new cv::VideoCapture(std::stoi(video_file_))
                : new cv::VideoCapture(video_file_));
  auto& cap = *video_stream.get();
  if (!cap.isOpened()) {
    LOG(FATAL) << "cannot open file " << video_file_;
    return nullptr;
  }
  return video_stream;
}

std::unique_ptr<cv::VideoWriter> open_output(const std::string& video_file_,
                                             int width, int height) {
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      video_file_, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& cap = *video_stream.get();
  if (!cap.isOpened()) {
    LOG(FATAL) << "cannot open file " << video_file_;
    return nullptr;
  }
  return video_stream;
}

int main(int argc, char* argv[]) {
  auto video_stream = open_input(std::string(argv[1]));
  // open output
  auto width = stoi(argv[3]);
  auto height = stoi(argv[4]);
  auto video_sink = open_output(string(argv[2]), width, height);
  auto& cap = *video_stream.get();
  auto& sink = *video_sink.get();
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
    auto prop_fps = cap.get(cv::CAP_PROP_FPS);
    auto e = std::chrono::steady_clock::now();
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(e - now).count();
    LOG(INFO) << "frame " << f << " FPS: "
              << static_cast<float>(f) / static_cast<float>(us) * 1e6
              << " prop_fps = " << prop_fps << " size " << image.rows << "x"
              << image.cols;
    cv::Mat img_resize;
    cv::resize(image, img_resize, cv::Size(width, height), 0, 0);
    cv::rectangle(
        img_resize,
        cv::Rect{cv::Point(10, 10), cv::Size(30 + f % 300, 30 + f % 300)},
        cv::Scalar(0, 0, 255), 5);
    cv::rectangle(
        img_resize,
        cv::Rect{cv::Point(20, 20), cv::Size(40 + f % 300, 40 + f % 300)},
        cv::Scalar(0, 255, 0), 5);
    cv::rectangle(
        img_resize,
        cv::Rect{cv::Point(30, 30), cv::Size(50 + f % 300, 50 + f % 300)},
        cv::Scalar(255, 0, 0), 5);
    if (f == 60) {
      cv::imwrite("test.jpg", image);
      cv::imwrite("test_resized.jpg", img_resize);
    }
    sink << img_resize;
    f = f + 1;
  }
  return 0;
}
