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
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(MAX_FPS, "0");
DEF_ENV_PARAM(IMSHOW, "1");
using namespace std;

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int width, int height, std::string pipeline) {
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "cannot open gst: " << pipeline;
    return nullptr;
  } else {
    LOG(INFO) << "video writer is created: " << width << "x" << height << " "
              << pipeline;
  }
  return video_stream;
}
static void usage(const char* prog) {
  cerr << "usage: \n"  //
       << " # e.g. open USB camera, i.e. /dev/video0 and play back on a GUI "
          "window. X server is required.\n"                              //
       << "\t" << prog << " 0\n"                                         //
       << " # e.g. open sample.avi and and play back on a GUI window\n"  //
       << "\t" << prog << " sample.avi\n"                                //
       << " # e.g. usb camera 0 and play back with GStreamer\n"          //
       << "\t" << prog
       << " 0 1920 1080 'appsrc ! videoconvert ! video/x-raw, width=1920, "
          "height=1080 ! "
          "kmssink driver-name=xlnx plane-id=36 sync=false'\n"  //
       << "\t" << prog
       << " /workspace/aisw/apps/seg_and_pose_detect/seg_960_540.avi 960 540 "
          "'appsrc ! video/x-raw ! videoconvert ! autovideosink'"  //
       << endl;
}
using namespace std;
int main(int argc, char* argv[]) {
  if (argc <= 1) {
    usage(argv[0]);
    return 0;
  }
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
  std::unique_ptr<cv::VideoWriter> writer = nullptr;
  if (argc >= 5) {
    auto width = std::atoi(argv[2]);
    auto height = std::atoi(argv[3]);
    auto pipe = std::string(argv[4]);
    writer = maybe_create_gst_video_writer(width, height, pipe);
  }
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
    if (writer) {
      (*writer) << image;
    } else if (ENV_PARAM(IMSHOW)) {
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
    auto width = image.cols;
    auto height = image.rows;

    LOG(INFO) << "frame " << f << " FPS: "
              << static_cast<float>(f) / static_cast<float>(us) * 1e6 << " "
              << width << "x" << height << " prop_fps = " << prop_fps;
    f = f + 1;
  }
  return 0;
}
