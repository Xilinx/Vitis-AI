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
#include <glog/logging.h>
#include <opencv2/imgproc/types_c.h>
#include <signal.h>
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <stack>
#include <thread>
#include <type_traits>
#include <vitis/ai/bounded_queue.hpp>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM_2(
    DEMO_VIDEO_WRITER,
    "appsrc ! videoconvert ! queue ! kmssink "
    "driver-name=xlnx plane-id=39 fullscreen-overlay=false sync=false",
    std::string)
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")
DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");

#ifndef USE_DRM
#define USE_DRM 0
#endif
#if USE_DRM
#include "./dpdrm.hpp"
#endif

// set the layout
inline std::vector<cv::Rect>& gui_layout() {
  static std::vector<cv::Rect> rects;
  return rects;
}
// set the wallpaper
inline cv::Mat& gui_background() {
  static cv::Mat img;
  return img;
}

inline std::vector<cv::Size>& each_channel_mosaik_size() {
  static std::vector<cv::Size> msize;
  return msize;
}

namespace vitis {
namespace ai {
// Read a video without doing anything
struct VideoByPass {
 public:
  int run(const cv::Mat& input_image) { return 0; }
};

// Do nothing after after excuting
inline cv::Mat process_none(cv::Mat image, int fake_result, bool is_jpeg) {
  return image;
}

// A struct that can storage data and info for each frame
struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  float max_fps;
  float fps;
  int belonging;
  int mosaik_width;
  int mosaik_height;
  int horizontal_num;
  int vertical_num;
  cv::Rect_<int> local_rect;
  cv::Rect_<int> page_layout;
  std::string channel_name;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;
struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }
  static void stop_all() {
    for (auto& th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto& th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Thread num " << all_threads().size();
    for (auto& th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread* me) { return me->main(); }
  void main() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is started";
    while (!stop_) {
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;

  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT A Thread";
    all_threads().push_back(this);
  }

  virtual ~MyThread() {  //
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is starting";
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "waiting for [" << name() << "] ended";
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};

// std::vector<MyThread *> MyThread::all_threads_;
struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string& video_file, queue_t* queue)
      : MyThread{},
        channel_id_{channel_id},
        video_file_{video_file},
        frame_id_{0},
        video_stream_{},
        queue_{queue} {
    open_stream();
    auto& cap = *video_stream_.get();
    if (is_camera_) {
      cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    }
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    auto& cap = *video_stream_.get();
    cv::Mat image;
    cap >> image;
    auto video_ended = image.empty();
    if (video_ended) {
      // loop the video
      open_stream();
      return 0;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    if (queue_->size() > 0 && is_camera_ == true) {
      return 0;
    }
    while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image},
                         std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
  }

  void open_stream() {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
                   : new cv::VideoCapture(video_file_));
    if (!video_stream_->isOpened()) {
      LOG(FATAL)
          << "[UNILOG][FATAL][VAILIB_DEMO_VIDEO_OPEN_ERROR][Can not open "
             "video stream!]  video name: "
          << video_file_;
      stop();
    }
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  std::unique_ptr<cv::VideoCapture> video_stream_;
  queue_t* queue_;
  bool is_camera_;
};

std::vector<std::vector<cv::Mat>> mosaik_images;

struct ClassificaitonDecodeThread : public MyThread {
  ClassificaitonDecodeThread(int channel_id, std::string channel_name,
                             const std::vector<cv::String>& namelist,
                             queue_t* queue, int page_num,
                             cv::Rect_<int> page_layout, cv::Size mosaik_size)
      : MyThread{},
        channel_id_{channel_id},
        channel_name_{channel_name},
        frame_id_{0},
        queue_{queue},
        page_num_{page_num},
        page_layout_{page_layout},
        mosaik_width_{mosaik_size.width},
        mosaik_height_{mosaik_size.height} {
    load_image(namelist);
    get_mosaik(mosaik_images[channel_id_]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "init end";
  }

  virtual ~ClassificaitonDecodeThread() {}

  // struct FrameInfo {
  //   int channel_id;
  //   unsigned long frame_id;
  //   cv::Mat mat;
  //   float max_fps;
  //   float fps;
  //   int belonging;
  //   int mosaik_width;
  //   int mosaik_height;
  //   int horizontal_num;
  //   int vertical_num;
  // };

  virtual int run() override {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    while (!queue_->push(
        FrameInfo{channel_id_, frame_id_++,
                  std::get<0>(mosaik_part_[frame_id_ % mosaik_part_.size()]), 0,
                  0, std::get<1>(mosaik_part_[frame_id_ % mosaik_part_.size()]),
                  mosaik_width_, mosaik_height_, horizontal_num_, vertical_num_,
                  std::get<2>(mosaik_part_[frame_id_ % mosaik_part_.size()]),
                  page_layout_, channel_name_},
        std::chrono::milliseconds(500))) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "send a image ";
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"ClassificationDedodeThread-"} +
           std::to_string(channel_id_);
  }

  void load_image(const std::vector<cv::String>& namelist) {
    for (auto& name : namelist) {
      auto image = cv::imread(name);
      cv::resize(image, image, cv::Size(224, 224));
      image_lib_.push_back(image);
    }
  }

  void get_mosaik(std::vector<cv::Mat>& mosaik_image) {
    int tmp_horizontal_num = page_layout_.width / mosaik_width_;
    int tmp_vertical_num = page_layout_.height / mosaik_height_;
    horizontal_num_ = tmp_horizontal_num;
    vertical_num_ = tmp_vertical_num;
    /*
    if (horizontal_num_ == 0 || vertical_num_ == 0) {
    }
    */
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "get mosaik " << page_num_;
    mosaik_image.resize(page_num_);
    for (int i = 0; i < page_num_; i++) {
      mosaik_image[i] = cv::Mat(1, 1, CV_8UC3);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "resize mosaik " << page_layout_.width << " " << tmp_horizontal_num
          << " " << tmp_vertical_num;
      cv::resize(mosaik_image[i], mosaik_image[i],
                 cv::Size(page_layout_.width, page_layout_.height));
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "resize mosaik 2";
      for (int j = 0; j < tmp_vertical_num; j++) {
        for (int k = 0; k < tmp_horizontal_num; k++) {
          cv::Mat img = image_lib_[(i * tmp_vertical_num * tmp_horizontal_num +
                                    j * tmp_horizontal_num + k) %
                                   image_lib_.size()];
          // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << (i * tmp_vertical_num *
          // tmp_horizontal_num + j * tmp_horizontal_num + k) %
          // image_lib_.size() << " " << (i * tmp_vertical_num *
          // tmp_horizontal_num + j * tmp_horizontal_num + k);

          cv::Mat mosaik;
          cv::resize(img, mosaik, cv::Size(mosaik_width_, mosaik_height_));
          auto rect_tmp = cv::Rect_<int>(k * mosaik_width_, j * mosaik_height_,
                                         mosaik_width_, mosaik_height_);
          auto rect = cv::Rect_<int>(page_layout_.x + rect_tmp.x,
                                     page_layout_.y + rect_tmp.y,
                                     rect_tmp.width, rect_tmp.height);
          mosaik_part_.push_back(std::make_tuple(img, i, rect));
          // LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << rect;
          mosaik.copyTo(mosaik_image[i](rect_tmp));
        }
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << mosaik_image[i].size();
      cv::imwrite("./mosaik/" + std::to_string(i) + "_mosaik.jpg",
                  mosaik_image[i]);
    }
  }

  std::vector<cv::Mat> image_lib_;
  std::vector<std::tuple<cv::Mat, int, cv::Rect_<int>>> mosaik_part_;

  int channel_id_=0;
  std::string channel_name_;
  unsigned long frame_id_=0;
  queue_t* queue_=NULL;
  int per_page_=0;
  int page_num_=0;
  cv::Rect_<int> page_layout_;
  int mosaik_width_=0;
  int mosaik_height_=0;
  int horizontal_num_=0;
  int vertical_num_=0;
};

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int width, int height) {
  if (!ENV_PARAM(DEMO_USE_VIDEO_WRITER)) {
    return nullptr;
  }
  auto pipeline = ENV_PARAM(DEMO_VIDEO_WRITER);
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_GST_ERROR][Failed to open "
                  "gstreamer!] cannot open "
               << pipeline;
    return nullptr;
  } else {
    LOG(INFO) << "video writer is created: " << width << "x" << height << " "
              << pipeline;
  }
  return video_stream;
}

struct GuiThread : public MyThread {
  static std::shared_ptr<GuiThread> instance() {
    static std::weak_ptr<GuiThread> the_instance;
    std::shared_ptr<GuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<GuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
#if USE_DRM
    vitis::ai::imshow_open();
    cv::Mat img = gui_background();
    imshow_set_background(img);
#endif
    return ret;
  }

  GuiThread()
      : MyThread{},
        queue_{
            new queue_t{
                10}  // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0},
        video_writer_{maybe_create_gst_video_writer(
            ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH),
            ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT))} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
  }
  virtual ~GuiThread() {  //
#if USE_DRM
    vitis::ai::imshow_close();
#endif
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    while (!queue_->empty()) {
      queue_->pop(frame_info);
      frames_[frame_info.channel_id].frame_info = frame_info;
      frames_[frame_info.channel_id].dirty = true;
    }
  }
  virtual int run() override {
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 10) {
        // inactive for 5 second, stop
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
        return 1;
      } else {
        return 0;
      }
    }
    inactive_counter_ = 0;
    frames_[frame_info.channel_id].frame_info = frame_info;
    frames_[frame_info.channel_id].dirty = true;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
    clean_up_queue();
#if USE_DRM
    bool all_dirty = true;
    for (auto& f : frames_) {
      all_dirty = all_dirty && f.second.dirty;
    }
    if (!all_dirty) {
      // only show frames until all channels are dirty
      return 0;
    }
    auto width = modeset_get_fb_width();
    auto height = modeset_get_fb_height();
    auto screen_size = cv::Size{width, height};
    auto sizes = std::vector<cv::Size>(frames_.size());
    std::transform(frames_.begin(), frames_.end(), sizes.begin(),
                   [](const decltype(frames_)::value_type& a) {
                     return a.second.frame_info.mat.size();
                   });
    std::vector<cv::Rect> rects;
    rects = gui_layout();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "rects size is  " << rects.size();

    for (const auto& rect : rects) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "screen " << screen_size << "; r = " << rect;
      if ((rect.x + rect.width > width) || (rect.y + rect.height > height) ||
          (rect.x + rect.width < 1) || (rect.y + rect.height < 1)) {
        LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_OUT_OF_BOUNDARY][Gui rects "
                      "out of boundary!]";
      }
    }
    int c = 0;
    for (auto& f : frames_) {
      vitis::ai::imshow(rects[c], f.second.frame_info.mat);
      f.second.dirty = false;
      c++;
    }
    vitis::ai::imshow_update();
#else
    bool any_dirty = false;
    for (auto& f : frames_) {
      if (f.second.dirty) {
        if (video_writer_ == nullptr) {
          cv::imshow(std::string{"CH-"} +
                         std::to_string(f.second.frame_info.channel_id),
                     f.second.frame_info.mat);
        } else {
          *video_writer_ << f.second.frame_info.mat;
        }
        f.second.dirty = false;
        any_dirty = true;
      }
    }
    if (video_writer_ == nullptr) {
      if (any_dirty) {
        auto key = cv::waitKey(1);
        if (key == 27) {
          return 1;
        }
      }
    }
#endif
    clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_=0;
  struct FrameCache {
    bool dirty=false;
    FrameInfo frame_info;
  };
  std::map<int, FrameCache> frames_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
};  // namespace ai

struct GridGuiThread : public MyThread {
  static std::shared_ptr<GridGuiThread> instance() {
    static std::weak_ptr<GridGuiThread> the_instance;
    std::shared_ptr<GridGuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<GridGuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
#if USE_DRM
    vitis::ai::imshow_open();
    cv::Mat img = gui_background();
    imshow_set_background(img);
#endif
    return ret;
  }

  GridGuiThread()
      : MyThread{},
        queue_{
            new queue_t{
                60}  // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
  }
  virtual ~GridGuiThread() {  //
#if USE_DRM
    vitis::ai::imshow_close();
#endif
  }
  // void clean_up_queue() {
  //   FrameInfo frame_info;
  //   while (!queue_->empty()) {
  //     queue_->pop(frame_info);
  //     frames_[frame_info.channel_id].frame_info = frame_info;
  //     frames_[frame_info.channel_id].dirty = true;
  //   }
  // }
  virtual int run() override {  // GRID
    while (!queue_->empty()) {
      FrameInfo frame_info;
      if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
        inactive_counter_++;
        if (inactive_counter_ > 50) {
          // inactive for 5 second, stop
          LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
          return 1;
        } else {
          return 0;
        }
      }
      inactive_counter_ = 0;
      frames_[frame_info.channel_id].all_frame_info.push_back(frame_info);
      // frames_[frame_info.channel_id].dirty = true;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << " gridgui queue size " << queue_->size()
          << ", state = " << (is_stopped() ? "stopped" : "running");
    }
    // clean_up_queue();
#if USE_DRM
    // bool all_dirty = true;   // GRID
    // for (auto &f : frames_) {
    // all_dirty = all_dirty && f.second.dirty;
    // }
    // if (!all_dirty) {
    // only show frames until all channels are dirty
    // return 0;
    // }
    auto width = modeset_get_fb_width();
    auto height = modeset_get_fb_height();
    auto screen_size = cv::Size{width, height};
    // auto sizes = std::vector<cv::Size>(frames_.size());
    // std::transform(frames_.begin(), frames_.end(), sizes.begin(),
    //                [](const decltype(frames_)::value_type &a) {
    //                  return a.second.frame_info.mat.size();
    //                });
    std::vector<cv::Rect> rects;
    rects = gui_layout();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "rects size is  " << rects.size();

    for (const auto& rect : rects) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "screen " << screen_size << "; r = " << rect;
      if ((rect.x + rect.width > width) || (rect.y + rect.height > height) ||
          (rect.x + rect.width < 1) || (rect.y + rect.height < 1)) {
        LOG(FATAL)
            << "[UNILOG][FATAL][VAILIB_DEMO_OUT_OF_BOUNDARY][out of boundary!]";
      }
    }
    // int c = 0;
    // for (auto &f : frames_) {  // GRID
    //   vitis::ai::imshow(rects[c], f.second.frame_info.mat);
    //   f.second.dirty = false;
    //   c++;
    // }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "current frame size is " << frames_.size();
    for (auto& f : frames_) {
      for (auto& frame_info : f.second.all_frame_info) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
            << "local rect: " << frame_info.local_rect;
        vitis::ai::imshow(frame_info.local_rect, frame_info.mat);
      }
      auto tmp_frame_info = f.second.all_frame_info[0];
      auto tmp_width = 96;
      cv::Mat info_mat(cv::Size(tmp_width * 3, 16), CV_8UC3,
                       cv::Scalar(255, 255, 255));
      cv::Mat name_mat = info_mat(cv::Rect_<int>(0, 0, tmp_width * 1.5, 16));
      cv::Mat fps_mat =
          info_mat(cv::Rect_<int>(tmp_width * 1.7, 0, tmp_width, 16));
      cv::putText(name_mat, tmp_frame_info.channel_name, cv::Point(3, 11),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, 1);
      cv::putText(fps_mat,
                  std::string("fps: ") + std::to_string(tmp_frame_info.fps),
                  cv::Point(3, 11), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 0), 1, 1);
      vitis::ai::imshow(
          cv::Rect_<int>(tmp_frame_info.page_layout.x,
                         tmp_frame_info.page_layout.y, tmp_width * 3, 16),
          info_mat);
    }

    // if (frames_[0].all_frame_info.size() == 0)
    //   return 0;
    // int mosaik_width = frames_[0].all_frame_info[0].mosaik_width;
    // cv::Mat fps_mat(cv::Size(mosaik_width, 16), CV_8UC3, cv::Scalar(255, 255,
    // 255)); cv::putText(fps_mat, std::string("fps: ") +
    // std::to_string(frames_[0].all_frame_info[0].fps),
    //               cv::Point(1, 11), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    //               cv::Scalar(0, 0, 0), 1, 1);
    // vitis::ai::imshow(cv::Rect_<int>(0, 0, mosaik_width, 16), fps_mat);
    vitis::ai::imshow_update();
    for (auto& f : frames_) {
      for (auto& frame_info : f.second.all_frame_info) {
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
            << "local rect: " << frame_info.local_rect;
        vitis::ai::imshow(frame_info.local_rect, frame_info.mat);
      }
      auto tmp_frame_info = f.second.all_frame_info[0];
      auto tmp_width = 96;
      cv::Mat info_mat(cv::Size(tmp_width * 3, 16), CV_8UC3,
                       cv::Scalar(255, 255, 255));
      cv::Mat name_mat = info_mat(cv::Rect_<int>(0, 0, tmp_width * 1.5, 16));
      cv::Mat fps_mat =
          info_mat(cv::Rect_<int>(tmp_width * 1.7, 0, tmp_width, 16));
      cv::putText(name_mat, tmp_frame_info.channel_name, cv::Point(3, 11),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, 1);
      cv::putText(fps_mat,
                  std::string("fps: ") + std::to_string(tmp_frame_info.fps),
                  cv::Point(3, 11), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 0), 1, 1);
      vitis::ai::imshow(
          cv::Rect_<int>(tmp_frame_info.page_layout.x,
                         tmp_frame_info.page_layout.y, tmp_width * 3, 16),
          info_mat);
    }

    // vitis::ai::imshow(cv::Rect_<int>(0, 0, mosaik_width, 16), fps_mat);
    for (auto& f : frames_) {
      f.second.all_frame_info.clear();
    }
#else
    // bool any_dirty = false;
    // for (auto &f : frames_) {
    //   if (f.second.dirty) {
    //     cv::imshow(std::string{"CH-"} +
    //                    std::to_string(f.second.frame_info.channel_id),
    //                f.second.frame_info.mat);
    //     f.second.dirty = false;
    //     any_dirty = true;
    //   }
    // }
    // if (any_dirty) {
    //   auto key = cv::waitKey(1);
    //   if (key == 27) {
    //     return 1;
    //   }
    // }
#endif
    // clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_=0;
  struct FrameCache {
    bool dirty=false;
    std::vector<FrameInfo> all_frame_info;
  };
  std::map<int, FrameCache> frames_;
};

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual cv::Mat run(cv::Mat& input) = 0;
};

// Execute each lib run function and processor your implement
template <typename dpu_model_type_t, typename ProcessResult>
struct DpuFilter : public Filter {
  DpuFilter(std::unique_ptr<dpu_model_type_t>&& dpu_model,
            const ProcessResult& processor)
      : Filter{}, dpu_model_{std::move(dpu_model)}, processor_{processor} {
    LOG(INFO) << "DPU model size=" << dpu_model_->getInputWidth() << "x"
              << dpu_model_->getInputHeight();
  }
  virtual ~DpuFilter() {}
  cv::Mat run(cv::Mat& image) override {
    auto result = dpu_model_->run(image);
    return processor_(image, result, false);
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
  const ProcessResult& processor_;
};
template <typename FactoryMethod, typename ProcessResult>
std::unique_ptr<Filter> create_dpu_filter(const FactoryMethod& factory_method,
                                          const ProcessResult& process_result) {
  using dpu_model_type_t = typename decltype(factory_method())::element_type;
  return std::unique_ptr<Filter>(new DpuFilter<dpu_model_type_t, ProcessResult>(
      factory_method(), process_result));
}

// Execute dpu filter
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<Filter>&& filter, queue_t* queue_in,
            queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        filter_{std::move(filter)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }
  virtual ~DpuThread() {}

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }
    if (filter_) {
      frame.mat = filter_->run(frame.mat);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t* queue_in_=NULL;
  queue_t* queue_out_=NULL;
  std::string suffix_;
};

// Implement sorting thread
struct SortingThread : public MyThread {
  SortingThread(queue_t* queue_in, queue_t* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT SORTING";
  }
  virtual ~SortingThread() {}
  virtual int run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond =
        std::function<bool(const FrameInfo&)>{[frame_id](const FrameInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    if (!queue_in_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto now = std::chrono::steady_clock::now();
    float fps = -1.0f;
    long duration = 0;
    if (!points_.empty()) {
      auto end = points_.back();
      duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;
      auto x = 10;
      auto y = 20;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_=NULL;
  queue_t* queue_out_=NULL;
  unsigned long frame_id_=0;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_=0.0;
  float max_fps_=0.0;
};
inline void usage_video(const char* progname) {
  std::cout << "usage: " << progname << "      -t <num_of_threads>\n"
            << "      <video file name>\n"
            << std::endl;
  return;
}
/*
  global command line options
 */
static std::vector<int> g_num_of_threads;
static std::vector<std::string> g_avi_file;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  while ((opt = getopt(argc, argv, "c:t:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads.emplace_back(std::stoi(optarg));
        break;
      case 'c':  // how many channels
        break;   // do nothing. parse it in outside logic.
      default:
        usage_video(argv[0]);
        exit(1);
    }
  }
  for (int i = optind; i < argc; ++i) {
    g_avi_file.push_back(std::string(argv[i]));
  }
  if (g_avi_file.empty()) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_num_of_threads.empty()) {
    // by default, all channels has at least one thread
    g_num_of_threads.emplace_back(1);
  }
  return;
}

// Entrance of single channel video demo
template <typename FactoryMethod, typename ProcessResult>
int main_for_video_demo(int argc, char* argv[],
                        const FactoryMethod& factory_method,
                        const ProcessResult& process_result,
                        int start_pos = 1) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv, start_pos);
  {
#if USE_DRM
    cv::VideoCapture video_cap(g_avi_file[0]);
    std::string file_name(g_avi_file[0]);
    bool is_camera =
        file_name.size() == 1 && file_name[0] >= '0' && file_name[0] <= '9';
    if (is_camera) {
      gui_layout() = {{0, 0, 640, 360}};
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "Using camera";
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "Using file";
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "width " << video_cap.get(3);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "height " << video_cap.get(4);
      gui_layout() = {{0, 0, (int)video_cap.get(3), (int)video_cap.get(4)}};
    }
    video_cap.release();
#else
    if (ENV_PARAM(DEMO_USE_VIDEO_WRITER) == 0) {
      cv::moveWindow(std::string{"CH-"} + std::to_string(0), 500, 500);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "Window name " << std::string{"CH-"} + std::to_string(0);
    }
#endif
    auto channel_id = 0;
    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    auto decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{channel_id, g_avi_file[0], decode_queue.get()});
    auto dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    auto sorting_queue =
        std::unique_ptr<queue_t>(new queue_t(5 * g_num_of_threads[0]));
    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    for (int i = 0; i < g_num_of_threads[0]; ++i) {
      dpu_thread.emplace_back(new DpuThread(
          create_dpu_filter(factory_method, process_result), decode_queue.get(),
          sorting_queue.get(), std::to_string(i)));
    }
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));
    // start everything
    MyThread::start_all();
    gui_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

// A class can create a video channel
struct Channel {
  Channel(size_t ch, const std::string& avi_file,
          const std::function<std::unique_ptr<Filter>()>& filter,
          int n_of_threads) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "create channel " << ch << " for " << avi_file;
    auto channel_id = ch;
    decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)channel_id, avi_file, decode_queue.get()});
    dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(5 * n_of_threads));
    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix =
          avi_file + "-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
};

// Entrance of multi-channel video demo
inline int main_for_video_demo_multiple_channel(
    int argc, char* argv[],
    const std::vector<std::function<std::unique_ptr<Filter>()>>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  auto gui_thread = GuiThread::instance();
  std::vector<Channel> channels;
  channels.reserve(filters.size());
  for (auto ch = 0u; ch < filters.size(); ++ch) {
    channels.emplace_back(ch, g_avi_file[ch % g_avi_file.size()], filters[ch],
                          g_num_of_threads[ch % g_num_of_threads.size()]);
  }
  // start everything
  MyThread::start_all();
  gui_thread->wait();
  MyThread::stop_all();
  MyThread::wait_all();
#if USE_DRM
  imshow_save_screen("prtscmultivideo.jpg");
#endif
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

// ClassificaitonDecodeThread(int channel_id, const std::vector<std::string>
// &namelist, queue_t *queue, int page_num, int page_height, int page_width)

struct Classification_Channel {
  Classification_Channel(size_t ch, const std::vector<cv::String>& image_list,
                         const std::function<std::unique_ptr<Filter>()>& filter,
                         std::string channel_name, int n_of_threads,
                         int page_num, cv::Rect_<int> layout,
                         cv::Size mosaik_size) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "create channel " << ch << " for ";
    auto channel_id = ch;
    decode_queue = std::unique_ptr<queue_t>(new queue_t(10 * n_of_threads));
    decode_thread = std::unique_ptr<ClassificaitonDecodeThread>(
        new ClassificaitonDecodeThread{(int)channel_id, channel_name,
                                       image_list, decode_queue.get(), page_num,
                                       layout, mosaik_size});
    dpu_thread = std::vector<std::unique_ptr<DpuThread>>{};
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(8 * n_of_threads));
    auto gui_thread = GridGuiThread::instance();
    auto gui_queue = gui_thread->getQueue();

    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix = "file-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, "file-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<ClassificaitonDecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
};


static void get_valid_file_path(std::string filename, std::string filedir, std::string& result){
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    std::string current_dir(cwd);
    LOG_IF(INFO, false) << "Current working directory: " << current_dir << std::endl;
    //get absolute path
    std::string file_tmp;
    if(filedir[0] == '~'){
      file_tmp =  "/home/root" + filedir.substr(1);
    }
    else if (filedir[0] != '/'){
      file_tmp = current_dir + "/" + filedir;
    }
    else{
      file_tmp = filedir;
    }
    //remove ./ ../ in the absolute path
    std::stack<std::string> stk;
    std::string segment;
    std::stringstream ss(file_tmp);

    while (std::getline(ss, segment, '/')) {
        if (segment == ".." && !stk.empty()) {
            stk.pop();
        } else if (segment != "." && segment != ".." && !segment.empty()) {
            stk.push(segment);
        }
    }

    std::vector<std::string> dirs;
    while (!stk.empty()) {
        dirs.push_back(stk.top());
        stk.pop();
    }

    for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) {
        result += "/";
        result += *it;
    }

    result = result + "/" + filename;
  } else {
      std::cerr << "Failed to get current working directory." << std::endl;
  }
  

}

int main_for_classification_demo(
    int argc, char* argv[],
    const std::vector<std::pair<std::function<std::unique_ptr<Filter>()>,
                                std::string>>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  auto gui_thread = GridGuiThread::instance();
  parse_opt(argc, argv);
  std::vector<cv::Size> mosaik_size;
  std::vector<cv::Rect_<int>> layout;
  if (gui_layout().size() == 0) {
    if (filters.size() == 1) {
      layout.push_back(cv::Rect_<int>(0, 0, 1920, 1080));
      mosaik_size = {cv::Size(96, 108)};
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "1 channel mode";
    } else if (filters.size() == 2) {
      layout.push_back(cv::Rect_<int>(0, 0, 960, 1080));
      layout.push_back(cv::Rect_<int>(960, 0, 960, 1080));
      mosaik_size = {cv::Size(96, 108), cv::Size(96, 108)};
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "2 channels mode";
    } else if (filters.size() == 4) {
      layout.push_back(cv::Rect_<int>(0, 0, 960, 540));
      layout.push_back(cv::Rect_<int>(960, 0, 960, 540));
      layout.push_back(cv::Rect_<int>(0, 540, 960, 540));
      layout.push_back(cv::Rect_<int>(960, 540, 960, 540));
      mosaik_size = {cv::Size(96, 108), cv::Size(96, 108), cv::Size(96, 108),
                     cv::Size(96, 108)};
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "4 channels mode";
    } else {
      LOG(INFO)
          << "unsupported Filter num in auto mode, only support 1, 2 or 4";
      return 0;
    }
  } else {
    if (gui_layout().size() >= filters.size()) {
      layout = gui_layout();
      mosaik_size.resize(filters.size());
      for (auto i = 0u; i < filters.size(); i++) {
        if (each_channel_mosaik_size().size() > 0) {
          mosaik_size[i] =
              each_channel_mosaik_size()[i % each_channel_mosaik_size().size()];
        } else {
          mosaik_size[i] = cv::Size(96, 108);
        }
      }
    } else {
      LOG(FATAL)
          << "UNILOG][FATAL][VAILIB_DEMO_CANVAS_ERROR][Canvas image size is "
             "too small!]";
    }
  }
  std::vector<Classification_Channel> channels;
  mosaik_images.resize(filters.size());
  std::vector<std::vector<cv::String>> image_lists(filters.size());

  channels.reserve(filters.size());
  for (auto ch = 0u; ch < filters.size(); ++ch) {
    cv::glob(g_avi_file[ch % g_avi_file.size()], image_lists[ch]);
    channels.emplace_back(ch, image_lists[ch], filters[ch].first,
                          filters[ch].second,
                          g_num_of_threads[ch % g_num_of_threads.size()], 20,
                          layout[ch], mosaik_size[ch]);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "start all";
  MyThread::start_all();
  gui_thread->wait();
  MyThread::stop_all();
  MyThread::wait_all();
#if USE_DRM
  imshow_save_screen("prtscmulticlassification.jpg");
#endif
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

static inline void usage_jpeg(const char* progname) {
  std::cout << "usage : " << progname << " <img_url> [<img_url> ...]"
            << std::endl;
}

// Entrance of jpeg demo
template <typename FactoryMethod, typename ProcessResult>
int main_for_jpeg_demo(int argc, char* argv[],
                       const FactoryMethod& factory_method,
                       const ProcessResult& process_result, int start_pos = 1) {
  if (argc <= 1) {
    usage_jpeg(argv[0]);
    exit(1);
  }

  auto model = factory_method();
  if (ENV_PARAM(SAMPLES_ENABLE_BATCH)) {
    std::vector<std::string> image_files;
    for (int i = start_pos; i < argc; ++i) {
      image_files.push_back(std::string(argv[i]));
    }
    if (image_files.empty()) {
      std::cerr << "no input file" << std::endl;
      exit(1);
    }

    auto batch = model->get_input_batch();
    if (ENV_PARAM(SAMPLES_BATCH_NUM)) {
      unsigned int batch_set = ENV_PARAM(SAMPLES_BATCH_NUM);
      assert(batch_set <= batch);
      batch = batch_set;
    }
    std::vector<std::string> batch_files(batch);
    std::vector<std::string> batch_filenames(batch);
    std::vector<cv::Mat> images(batch);
    for (auto index = 0u; index < batch; ++index) {
      auto& file_init = image_files[index % image_files.size()];
      std::size_t pos = file_init.find_last_of('/');
      std::size_t dot_pos = file_init.find_last_of('.');

      // get basename of image
      //get file name of image
      if (pos == std::string::npos){
        const std::string basename = file_init.substr(0, dot_pos);
        const std::string filename = file_init;
        batch_files[index] = basename;
        batch_filenames[index] = filename;
      }else{
        const std::string basename = file_init.substr(pos + 1, dot_pos - pos - 1);
        const std::string filename = file_init.substr(pos+1);
        batch_files[index] = basename;
        batch_filenames[index] = filename;
      }
      

      //get valid path of image;
      std::string valid_path = "";
      //get working dir
      std::string filedir = "";
      if(pos != std::string::npos)
        filedir = file_init.substr(0,pos);
      LOG_IF(INFO, false) << "filedir: " << filedir <<std::endl
            << "filename: " << batch_filenames[index] << std::endl
            << "basename: " << batch_files[index] << std::endl;
      get_valid_file_path(batch_filenames[index], filedir , valid_path);
      const auto& file = valid_path;
      LOG_IF(INFO, false) << "valid image file path: " << file << std::endl;


      images[index] = cv::imread(file);
      CHECK(!images[index].empty()) << "cannot read image from " << file;
    }

    auto results = model->run(images);

    assert(results.size() == batch);
    for (auto i = 0u; i < results.size(); i++) {
      LOG(INFO) << "batch: " << i << "     image: " << batch_filenames[i] ;
      auto image = process_result(images[i], results[i], true);
      auto out_file = std::to_string(i) + "_" +
                      batch_files[i] +
                      "_result.jpg";
      cv::imwrite(out_file, image);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "result image write to " << out_file;
      std::cout << std::endl;
    }
  } else {
    for (int i = start_pos; i < argc; ++i) {
      auto image_file_name = std::string{argv[i]};
      auto image = cv::imread(image_file_name);
      if (image.empty()) {
        LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_IMAGE_LOAD_ERROR][Failed to "
                      "load image!]cannot load "
                   << image_file_name << std::endl;
        abort();
      }
      auto result = model->run(image);
      image = process_result(image, result, true);
      auto out_file =
          image_file_name.substr(0, image_file_name.size() - 4) + "_result.jpg";
      cv::imwrite(out_file, image);
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "result image write to " << out_file;
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

}  // namespace ai
}  // namespace vitis
