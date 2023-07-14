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
#include <fstream>
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

DEF_ENV_PARAM(ENABLE_MULTI_BATCH, "0")
DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEMO_USE_X, "0")
DEF_ENV_PARAM(DEBUG_VEK280_HDMI, "0")
DEF_ENV_PARAM(DEBUG_SHOW_LATENCY, "0")


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


// set the layout
inline std::vector<cv::Rect>& gui_layout() {
  static std::vector<cv::Rect> rects;
  return rects;
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
  std::string channel_name;
  std::vector<cv::Mat> mats;
  std::time_t timestamp;
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
	  << "decode output queue capacity:" << queue_->capacity(); 
    
  }

  virtual ~DecodeThread() {}

  virtual int run()  override {
    auto timestamp = getTimeStamp();
    auto& cap = *video_stream_.get();
    cv::Mat image;
    cap >> image;
    auto video_ended = image.empty();
    if (video_ended) {
      // loop the video
      LOG(INFO) << "reopen stream : " << video_file_;
      open_stream();
      return 0;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode output queue size " << queue_->size();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "ch:" << channel_id_ << " frame size:" << image.size();
    //if (queue_->size() > 0 && is_camera_ == true) {
    //  return 0;
    //}
    FrameInfo frameinfo{channel_id_, ++frame_id_, image};
    frameinfo.timestamp = timestamp;
    while (!queue_->push(frameinfo, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "push frame id: " << frame_id_ << " to decode output queue";
    usleep(1000);
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

static std::string create_pipeline(int x, int y, int width, int height,
                                   int id) {
  std::ostringstream str;
  if (ENV_PARAM(DEMO_USE_X)) {
    str << "appsrc"            //
        << " ! videoconvert "  //
        << " ! ximagesink ";
    return str.str(); 
  }

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
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:{v_multi_scaler_1}"
      << " ! video/x-raw, width=" << 960 << ","
      << " height=" << 540 //
      << " ! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 0 << "," << 0 << "," << 960 << ","
      << 540 << ">\" sync=false";
  else if(id == 1) 
  str << "appsrc"          
      << " ! videoconvert " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:{v_multi_scaler_2}" 
      << " ! video/x-raw, width=" << 960 << ","
      << " height=" << 540 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 960 << "," << 0 << "," << 960 << ","
      << 540 << ">\" sync=false";
  else if(id == 2) 
  str << "appsrc"          
      << " ! videoconvert " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:{v_multi_scaler_3}"
      << " ! video/x-raw, width=" << 960 << ","
      << " height=" << 540 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 0 << "," << 540 << "," << 960 << ","
      << 540 << ">\" sync=false";
  else if(id == 3) 
  str << "appsrc"          
      << " ! videoconvert  " 
      << " ! vvas_xabrscaler xclbin-location=\"/run/media/mmcblk0p1/dpu.xclbin\" kernel-name=v_multi_scaler:{v_multi_scaler_4}"
      << " ! video/x-raw, width=" << 960 << ","
      << " height=" << 540 //
      << "! kmssink driver-name=xlnx plane-id=" << 34 + id
      << " render-rectangle=\"<" << 960 << "," << 540 << "," << 960 << ","
      << 540 << ">\" sync=false";
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
  std::string pipeline;
  std::string file_prefix = "gst_writer_";
  auto file_suffix = ".cmd";
  if (ENV_PARAM(DEBUG_VEK280_HDMI)) {
    //std::ostringstream str;
    //str << "appsrc"          
    //  << " ! videoconvert ! videoscale " 
    //  << " ! video/x-raw, width=" << width << ","
    //  << " height=" << height  //
    //  << "! kmssink driver-name=xlnx plane-id=" << 34 + id
    //  << " render-rectangle=\"<" << x << "," << y << "," << width  << ","
    //  << height  << ">\" sync=false";
    //pipeline = str.str();
    std::string file_name = file_prefix + std::to_string(id) + file_suffix;
    std::ifstream in(file_name.c_str());
    if (in.is_open()) {
      std::stringstream str;
      str << in.rdbuf();
      pipeline = str.str();
    } else {
      LOG(ERROR) << "read gst command file:" << file_name << " fail!";
      exit(0);
    }

  } else {
    pipeline = create_pipeline(x, y, width, height, id);
  }
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "[UNILOG][FATAL][VAILIB_DEMO_GST_ERROR][failed to open gstreamer!] cannot open " << pipeline;
    return nullptr;
  } else {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "video writer is created: " << width << "x" << height << " "
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui running"; 
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
      //std::this_thread::sleep_for(std::chrono::microseconds((uint32_t)(1000 / (frame_info.fps + queue_->size()))));
      //std::this_thread::sleep_for(std::chrono::microseconds(50));
      std::this_thread::sleep_for(std::chrono::microseconds(1));
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
    auto time_begin = getTimeStamp();
    *video_writer_ << frame_info.mat;
    auto time_end = getTimeStamp();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
	  << name() << " show time:" << time_end - time_begin;
    if (ENV_PARAM(DEBUG_SHOW_LATENCY)) {
      auto now = getTimeStamp();
      auto latency = now - frame_info.timestamp;
      LOG(INFO)
          << "gui show ch:" << frame_info.channel_id
          << " , frame:" << frame_info.frame_id << " latency:" << latency;
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "gui show ch:" << frame_info.channel_id
          << " , frame:" << frame_info.frame_id;
    }
    clean_up_queue();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << name() << " clean up queue";

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
      //std::vector<cv::Mat> mats = frame.mats; 
      //frame.mats = filter_->run(mats);
      std::vector<cv::Mat> mats(1);
      mats[0] = frame.mat;
      auto result_mats = filter_->run(mats);
      frame.mat = result_mats[0];
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

struct DpuBatchThread : public MyThread {
  DpuBatchThread(std::unique_ptr<Filter>&& filter, queue_t* queue_in,
      std::vector<std::unique_ptr<queue_t>>& sorting_queues,
      const std::string& suffix)
      : MyThread{},
        filter_{std::move(filter)},
        queue_in_{queue_in},
        sorting_queues_{sorting_queues},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }
  virtual ~DpuBatchThread() {}

  virtual int run() override {
    FrameInfo frame;
    int batch =  sorting_queues_.size();
    std::vector<FrameInfo> batch_frames(batch);
    int valid_num = 0;
    int pop_count = 0;
    while (valid_num < batch && pop_count < 6) {
      pop_count++;
      if (!queue_in_->pop(frame, std::chrono::milliseconds(50))) {
	continue;
      }
      batch_frames[valid_num] = frame;
      valid_num++; 
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "valid num:" << valid_num;

    if (filter_) {
      std::vector<cv::Mat> mats(valid_num);
      for (auto i = 0; i < valid_num; ++i) {
	mats[i] = batch_frames[i].mat;
      }
      auto result_mats = filter_->run(mats);

      for (auto i = 0; i < valid_num; ++i) {
	batch_frames[i].mat = result_mats[i];
      }
      /*
      auto time_now = gettm(getTimeStamp());
      cv::putText(frame.mats[0], std::string("Timestamp1: ") + time_now,
                      cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(20, 20, 180), 2, 1);
                      */
    }

    int push_cnt = 0;
    std::vector<bool> push_ok(valid_num, false);
    while (push_cnt != valid_num) {
      for (auto i = 0; i < valid_num; ++i) {
	if (push_ok[i]) {
          continue;
	}
        auto ch = batch_frames[i].channel_id;
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
            << "dpu output queue of ch:" << ch << " size " << sorting_queues_[ch]->size();

        if (!sorting_queues_[ch]->push(batch_frames[i], std::chrono::milliseconds(50))) {
          if (is_stopped()) {
            return -1;
          }
        } else {
          push_cnt++;
	  push_ok[i] = true;
          LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
               << "push sorting queue[" << ch << "] done, push count:" << push_cnt;
	}
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t* queue_in_;
  std::vector<std::unique_ptr<queue_t>>& sorting_queues_;
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << name() << "cond id:" << frame_id;
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
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id //<< " now: " << now << " end: " << end 
	<< " total:" << total << " duration:" << duration << "  FPS: " << fps;
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
        << " frame id " << frame.frame_id << " sorting output queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }

    if (frame.mat.cols > 200) {
      cv::putText(frame.mat, std::string("FPS: ") + std::to_string(frame.fps),
                  cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(20, 20, 180), 2, 1);
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(50))) {
      if (is_stopped()) {
        return -1;
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

inline int main_for_video_demo_multiple_channel(
    int argc, char* argv[],
    const std::vector<std::function<std::unique_ptr<Filter>()>>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  //auto gui_thread = GuiThread::instance();

  std::vector<Channel> channels;
  channels.reserve(filters.size());
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

inline int main_for_video_demo_batch(
    int argc, char* argv[],
    const std::vector<std::function<std::unique_ptr<Filter>()>>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  //auto gui_thread = GuiThread::instance();

  //auto batch = 1; 
  auto batch = g_avi_file.size(); 

  auto dpu_thread_num = g_num_of_threads[0];
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "filters.size() :" << filters.size();
  std::vector<std::unique_ptr<DecodeThread>> decode_thread(batch);
  std::vector<std::unique_ptr<queue_t>> decode_queue(batch);
  std::vector<std::unique_ptr<SortingThread>> sorting_thread(batch);
  std::vector<std::unique_ptr<GuiThread>> gui_thread(batch);
  std::vector<std::unique_ptr<queue_t>> sorting_queue(batch);

  decode_queue[0] = std::unique_ptr<queue_t>{new queue_t{5 * batch}};
  for (auto i = 0; i < batch; ++i) {
    decode_thread[i] = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)i, g_avi_file[i], decode_queue[0].get()});
    gui_thread[i] = std::make_unique<GuiThread>();
    auto gui_queue = gui_thread[i]->getQueue();
    sorting_queue[i] = std::unique_ptr<queue_t>(new queue_t(3 * dpu_thread_num));
    sorting_thread[i] = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue[i].get(), gui_queue, g_avi_file[i] + "-" + std::to_string(i)));
    
  }

  std::vector<std::unique_ptr<DpuBatchThread>> dpu_thread;
  for (auto i = 0; i < dpu_thread_num; ++i) {
      auto suffix =
          std::to_string(i) + "/" + std::to_string(dpu_thread_num);
      dpu_thread.emplace_back(new DpuBatchThread{filters[0](), decode_queue[0].get(), 
                                            std::ref(sorting_queue), suffix});
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
