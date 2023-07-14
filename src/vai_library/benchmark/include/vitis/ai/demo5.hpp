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

#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <type_traits>
#include <vitis/ai/bounded_queue.hpp>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM_2(
    DEMO_VIDEO_WRITER,
    "appsrc ! videoconvert ! queue ! fpsdisplaysink video-sink=\"kmssink driver-name=xlnx plane-id=39\" sync=false -v ",
    std::string)
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")
DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");

#ifndef USE_DRM
#define USE_DRM 0
#endif
#if USE_DRM
#include "./dpdrm.hpp"
#endif


std::time_t getTimeStamp() {
    std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp=std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
    //std::time_t timestamp = std::chrono::system_clock::to_time_t(tp);
    return timestamp;
}

using std::to_string;
std::string gettm(std::time_t timestamp) {
    std::time_t milli = timestamp/*+ (std::time_t)8*60*60*1000*/;
    auto mTime = std::chrono::milliseconds(milli);
    auto tp=std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds>(mTime);
    auto tt = std::chrono::system_clock::to_time_t(tp);
    std::tm* now = std::gmtime(&tt);
    std::string time_now(to_string(now->tm_year+1900) + "/" + to_string(now->tm_mon+1) + "/" + to_string(now->tm_mday) + "  " + to_string(now->tm_hour) + ":" + to_string(now->tm_min) + ":" + to_string(now->tm_sec) + "." + to_string(milli%1000));
   return time_now;
}

std::vector<std::unique_ptr<cv::VideoCapture>> caps_;

// set the layout
size_t batch_size = 1;
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
  std::vector<cv::Mat> mats;
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
    //open_stream();
    //auto& cap = *video_stream_.get();
    //if (is_camera_) {
    //  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    //  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    //}
  }

  virtual ~DecodeThread() {}

  virtual int run() override {
    auto& cap = *caps_[channel_id_].get();
    FrameInfo frameinfo{channel_id_, ++frame_id_};
    for (size_t i = 0; i < batch_size; i++) {
      cv::Mat image;
      cap >> image;
      //LOG(INFO) << "channel_id " << channel_id_ << " size " << gui_layout()[channel_id_];
      //cv::resize(image, image, cv::Size(gui_layout()[channel_id_].tl().x, gui_layout()[channel_id_].tl().y));
      cv::resize(image, image, cv::Size(640, 360));
      //LOG(INFO) << "debug1 " << channel_id_;
      auto video_ended = image.empty();
      if (video_ended) {
        LOG(INFO) << video_ended;
              // loop the video

      LOG(INFO) << "debug2 " << channel_id_;
        return 0;
      }
      /*
      auto time_now = gettm(getTimeStamp());
      cv::putText(image, std::string("Timestamp0: ") + time_now,
                    cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
                    */
      frameinfo.mats.push_back(image);
    }
    usleep(10000);
      //LOG(INFO) << "debug3 " << channel_id_;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    if (queue_->size() > 5 && is_camera_ == true) {
      return 0;
    }
    while (!queue_->push(frameinfo,
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
      LOG(ERROR) << "cannot open file " << video_file_;
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
    for (auto name : namelist) {
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
      return 0;
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

  int channel_id_;
  std::string channel_name_;
  unsigned long frame_id_;
  queue_t* queue_;
  int per_page_;
  int page_num_;
  cv::Rect_<int> page_layout_;
  int mosaik_width_;
  int mosaik_height_;
  int horizontal_num_;
  int vertical_num_;
};

static std::string create_pipeline(int x, int y, int width, int height,
                                   int id) {
  std::ostringstream str;
  /*
  str << "appsrc"            //
      << " ! videoconvert "  //
      << " ! video/x-raw, width=" << width << ","
      << " height=" << height  //
      << "! kmssink driver-name=xlnx plane-id=" << 36 + id
      << " render-rectangle=\"<" << x << "," << y << "," << width << ","
      << height << ">\" sync=false";
      */
  if(id == 0) 
  str << "appsrc"          
      << " ! videoconvert " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:v_multi_scaler_" << 5
      << " ! video/x-raw, width=" << 1920 << ","
      << " height=" << 1080 //
      << " ! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 0 << "," << 0 << "," << 1920 << ","
      << 1080 << ">\" sync=false";
  else if(id == 1) 
  str << "appsrc"          
      << " ! videoconvert " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:v_multi_scaler_" << 1
      << " ! video/x-raw, width=" << 1920 << ","
      << " height=" << 1080 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 1920 << "," << 0 << "," << 1920 << ","
      << 1080 << ">\" sync=false";
  else if(id == 2) 
  str << "appsrc"          
      << " ! videoconvert " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:v_multi_scaler_" << 2
      << " ! video/x-raw, width=" << 1920 << ","
      << " height=" << 1080 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 0 << "," << 1080 << "," << 1920 << ","
      << 1080 << ">\" sync=false";
  else if(id == 3) 
  str << "appsrc"          
      << " ! videoconvert  " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:v_multi_scaler_" << 4
      << " ! video/x-raw, width=" << 1920 << ","
      << " height=" << 1080 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 1920 << "," << 1080 << "," << 1920 << ","
      << 1080 << ">\" sync=false";
  else
  str << "appsrc"          
      << " ! videoconvert ! videoscale " 
      << " ! video/x-raw, width=" << width * 2 << ","
      << " height=" << height * 2 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << x*2 << "," << y*2 << "," << width * 2 << ","
      << height * 2 << ">\" sync=false";

  return str.str();
}

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int x, int y, int width, int height, int id) {
  auto pipeline = create_pipeline(x, y, width, height, id);
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_GST_ERROR][failed to open gstreamer!] cannot open " << pipeline;
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
        inactive_counter_{0} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
  }
  virtual ~GuiThread() {  //
#if USE_DRM
    vitis::ai::imshow_close();
#endif
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    //while (!queue_->empty()) {
    while (!queue_->size() > 2) {
      queue_->pop(frame_info);
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
    if (!video_writer_) {
      auto width = frame_info.mat.cols;
      auto height = frame_info.mat.rows;
      auto& layout = gui_layout()[frame_info.channel_id];
      auto x = layout.tl().x;
      auto y = layout.tl().y;
      video_writer_ = maybe_create_gst_video_writer(x, y, width, height,
                                                    frame_info.channel_id);
      std::this_thread::sleep_for(std::chrono::microseconds((uint32_t)(1000 / (frame_info.fps + queue_->size()))));
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
    /*
    auto time_now = gettm(getTimeStamp());
    cv::putText(frame_info.mat, std::string("Timestamp2: ") + time_now,
                    cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
                    */
    *video_writer_ << frame_info.mat;
    clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
};  // namespace ai

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual std::vector<cv::Mat> run(std::vector<cv::Mat>& input) = 0;
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
  std::vector<cv::Mat> run(std::vector<cv::Mat>& images) override {
    auto results = dpu_model_->run(images);
    return processor_(images, results, false);
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
    //LOG(INFO) << "pop " << frame.channel_id ; 
    if (filter_) {
      std::vector<cv::Mat> mats = frame.mats; 
      frame.mats = filter_->run(mats);
      /*
      auto time_now = gettm(getTimeStamp());
      cv::putText(frame.mats[0], std::string("Timestamp1: ") + time_now,
                      cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(20, 20, 180), 2, 1);
                      */
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
  queue_t* queue_in_;
  queue_t* queue_out_;
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
      fps = fps * batch_size;
      auto x = 10;
      auto y = 20;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      // if (frame.mat.cols > 200)
      //   cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
      //               cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      //               cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    for (auto i = 0; i < batch_size; i++) {
      auto s_frame = frame;
      s_frame.mat = frame.mats[i];
      if (s_frame.mat.cols > 200)
        cv::putText(s_frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
      while (!queue_out_->push(s_frame, std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
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
    sorting_queue = std::unique_ptr<queue_t>(new queue_t(3 * n_of_threads));
    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix =
          avi_file + "-" + std::to_string(i) + "/" + std::to_string(ch);
      dpu_thread.emplace_back(new DpuThread{filter(), decode_queue.get(),
                                            sorting_queue.get(), suffix});
    }
    gui_thread = std::make_unique<GuiThread>();
    auto gui_queue = gui_thread->getQueue();
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));
  }

  std::unique_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::unique_ptr<DpuThread>> dpu_thread;
  std::unique_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
  std::unique_ptr<GuiThread> gui_thread;
};

// Entrance of multi-channel video demo
//

void enable_st(size_t ch, std::vector<std::string> g_avi_file) {
  caps_[ch] = std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(g_avi_file[ch % g_avi_file.size()]));
}

inline int main_for_video_demo_multiple_channel(
    int argc, char* argv[],
    const std::vector<std::function<std::unique_ptr<Filter>()>>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  auto gui_thread = GuiThread::instance();
  std::vector<Channel> channels;
  channels.reserve(filters.size());
  caps_.resize(filters.size());
  std::thread t1(enable_st, 0, std::ref(g_avi_file));
  std::thread t2(enable_st, 1, std::ref(g_avi_file));
  std::thread t3(enable_st, 2, std::ref(g_avi_file));
  std::thread t4(enable_st, 3, std::ref(g_avi_file));
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  sleep(10);
  for (auto ch = 0u; ch < filters.size(); ++ch) {
    channels.emplace_back(ch, g_avi_file[ch % g_avi_file.size()], filters[ch],
                          g_num_of_threads[ch % g_num_of_threads.size()]);
  }
  // start everything
  MyThread::start_all();
  std::string cmd = "";
  while (cmd != "exit") {
    //std::cout << ">>> " << std::flush;
    std::cin >> cmd;
  }
  MyThread::stop_all();
  MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

}  // namespace ai
}  // namespace vitis
