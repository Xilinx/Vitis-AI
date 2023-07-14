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
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEBUG_DEMO_THREAD, "0")
DEF_ENV_PARAM(DEMO_USE_X, "0")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM_2(
    DEMO_VIDEO_WRITER,
    "appsrc ! videoconvert ! queue ! kmssink "
    "driver-name=xlnx plane-id=39 fullscreen-overlay=false sync=false",
    std::string)
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")
DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
static bool exiting = false;
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
  static void signal_handler(int) {
    exiting = true;
    stop_all();
  }
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO_THREAD))
        << "thread [" << name() << "] is ended";
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO_THREAD))
        << "thread [" << name() << "] is starting";
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO_THREAD))
        << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO_THREAD))
          << "waiting for [" << name() << "] ended";
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};
std::string to_string(const std::vector<int>& ids) {
  std::ostringstream str;
  str << "[";
  for (auto i = 0u; i < ids.size(); ++i) {
    if (i != 0u) {
      str << ",";
    }
    str << ids[i];
  }
  str << "]";
  return str.str();
}
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
    std::string key = std::to_string((intptr_t)(void*)queue);
    fair_ = vitis::ai::WeakStore<std::string, Fair>::create(key);
    fair_->channel_ids_.push_back(channel_id);
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
        << "decode out queue size " << queue_->size() << " ch=" << channel_id_;
    /*if (queue_->size() > 0 && is_camera_ == true) {
      return 0;
      }*/
    do {
      auto fair = fair_.get();
      auto ch_id = channel_id_;
      std::unique_lock<std::mutex> lock(fair->mtx_);
      if (exiting == true) {
        break;
      }
      fair->cv_.wait(lock, [fair, ch_id]() {
        return fair->channel_ids_[fair->cur_] == ch_id || exiting == true;
      });
      if (exiting == true) {
        break;
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << " " << to_string(fair->channel_ids_) << " ch=" << ch_id
          << " cur=" << fair->cur_;
      while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image},
                           std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
      fair->cur_++;
      if (fair->cur_ >= fair->channel_ids_.size()) {
        fair->cur_ = 0u;
      }
    } while (0);
    fair_->cv_.notify_all();
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
  struct Fair {
    std::mutex mtx_;
    std::vector<int> channel_ids_;
    std::condition_variable cv_;
    size_t cur_;
  };
  std::shared_ptr<Fair> fair_;
};

static std::string create_pipeline(int x, int y, int width, int height,
                                   int id) {
  std::ostringstream str;
  if (ENV_PARAM(DEMO_USE_X)) {
    str << "appsrc"            //
        << " ! videoconvert "  //
        << " ! ximagesink ";
  } else {
    str << "appsrc"            //
        << " ! videoconvert "  //
        << " ! video/x-raw, width=" << width << ","
        << " height=" << height  //
        << "! kmssink driver-name=xlnx plane-id=" << 36 + id
        << " render-rectangle=\"<" << x << "," << y << "," << width << ","
        << height << ">\" sync=false";
  }
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
    while (!queue_->empty()) {
      queue_->pop(frame_info);
    }
  }
  virtual int run() override {
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 20) {
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
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
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
  virtual size_t get_input_batch() = 0;
  virtual std::vector<cv::Mat> run(const std::vector<cv::Mat>& input) = 0;
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
  std::vector<cv::Mat> run(const std::vector<cv::Mat>& images) override {
    auto results = dpu_model_->run(images);
    std::vector<cv::Mat> ret;
    auto i = 0;
    for (auto image : images) {
      ret.push_back(processor_(image, results[i], false));
      i = i + 1;
    }
    return ret;
  }
  virtual size_t get_input_batch() override {
    return dpu_model_->get_input_batch();
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
  DpuThread(const std::function<std::unique_ptr<Filter>()>& filter_factory,
            queue_t* queue_in, const std::string& suffix)
      : MyThread{},
        filter_{filter_factory()},
        queue_in_{queue_in},
        suffix_{suffix},
        batch_{filter_->get_input_batch()} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
    for (auto i = 0u; i < 32u; ++i) {
      queue_out_.push_back(
          vitis::ai::WeakStore<size_t, queue_t>::create(i, 5 * 5));
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "INIT DPU: QUEUE IN: " << (void*)queue_in_;
    for (auto i = 0u; i < 5u; ++i) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "INIT DPU: QUEUE OUT: " << (void*)queue_out_[i].get();
    }
    queue_in_lock_ =
        vitis::ai::WeakStore<queue_t*, std::mutex>::create(queue_in_);
  }
  virtual ~DpuThread() {}

  virtual int run() override {
    std::vector<FrameInfo> frames(batch_);
    {
      // std::lock_guard<std::mutex> lock(*queue_in_lock_);
      for (auto b = 0u; b < batch_; ++b) {
        if (!queue_in_->pop(frames[b], std::chrono::milliseconds(500))) {
          return 0;
        }
      }
    }
    auto images = std::vector<cv::Mat>();
    auto ret_images = std::vector<cv::Mat>();
    for (auto b = 0u; b < batch_; ++b) {
      images.push_back(frames[b].mat);
    }

    ret_images = filter_->run(images);

    for (auto b = 0u; b < batch_; ++b) {
      frames[b].mat = ret_images[b];
    }

    for (auto b = 0u; b < batch_; ++b) {
      auto ch = frames[b].channel_id;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "dpu out queue size " << queue_out_[ch]->size() << " ch = " << ch
          << " suffix=" << suffix_;
      while (!queue_out_[ch]->push(frames[b], std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> filter_;
  queue_t* queue_in_;
  std::shared_ptr<std::mutex> queue_in_lock_;
  std::vector<std::shared_ptr<queue_t>> queue_out_;
  std::string suffix_;
  size_t batch_;
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
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "INIT SORTING: this= " << (void*)this;
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
      if (duration != 0) {
        max_fps_ = std::max(max_fps_, fps_);
      }
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat,
                    std::string("FPS: ") + std::to_string(fps)  // + "/" +
                    // std::to_string(max_fps_),
                    ,
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    while (duration > 5000) {  // sliding window for 2 seconds.
      points_.pop_back();
      auto end = points_.back();
      duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
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
static std::vector<std::string> g_avi_file;
static std::vector<std::string> g_models;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  while ((opt = getopt(argc, argv, "c:t:m:v:")) != -1) {
    switch (opt) {
      case 'm':
        g_models.emplace_back(std::string(optarg));
        break;
      case 'v':
        g_avi_file.emplace_back(std::string(optarg));
        break;
      case 'c':  // how many channels
        break;   // do nothing. parse it in outside logic.
      default:
        usage_video(argv[0]);
        exit(1);
    }
  }
  CHECK_EQ(g_avi_file.size(), g_models.size())
      << "num of  -m and -v must be same";
  return;
}

struct DpuFilterDesciption {
  std::string name;
  std::function<std::unique_ptr<Filter>()> filter;
  size_t num_of_threads;
};

const DpuFilterDesciption& find_fileter_description(
    const std::vector<DpuFilterDesciption>& filters, const std::string& name) {
  auto it = std::find_if(
      filters.begin(), filters.end(),
      [name](const DpuFilterDesciption& f) { return f.name == name; });
  CHECK(it != filters.end())
      << "cannot find fitler description: name = " << name;
  return *it;
}
// A class can create a video channel
struct Channel {
  Channel(size_t ch, const std::string& avi_file, const std::string& model,
          const std::vector<DpuFilterDesciption>& filters) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "create channel " << ch << " for "
                                        << avi_file << " model = " << model;
    auto& filter = find_fileter_description(filters, model);
    auto n_of_threads = filter.num_of_threads;
    auto channel_id = ch;
    decode_queue =
        vitis::ai::WeakStore<std::string, queue_t>::create(filter.name, 100);
    decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{(int)channel_id, avi_file, decode_queue.get()});
    dpu_thread = std::vector<std::shared_ptr<DpuThread>>{};
    sorting_queue =
        vitis::ai::WeakStore<size_t, queue_t>::create(ch, 32 * n_of_threads);
    for (int i = 0; i < n_of_threads; ++i) {
      auto suffix = filter.name + "-" + std::to_string(i) + "/" +
                    std::to_string(n_of_threads);
      dpu_thread.emplace_back(
          vitis::ai::WeakStore<std::string, DpuThread>::create(
              suffix, filter.filter, decode_queue.get(), suffix));
    }
    gui_thread = std::make_unique<GuiThread>();
    auto gui_queue = gui_thread->getQueue();
    sorting_thread = std::unique_ptr<SortingThread>(new SortingThread(
        sorting_queue.get(), gui_queue, avi_file + "-" + std::to_string(ch)));

    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Channel is create:"
        << "ch " << ch << " "                                     //
        << "sorting_queue " << (void*)sorting_queue.get() << " "  //
        << "decoding_queue " << (void*)decode_queue.get() << " "  //
        ;
  }

  std::shared_ptr<queue_t> decode_queue;
  std::unique_ptr<DecodeThread> decode_thread;
  std::vector<std::shared_ptr<DpuThread>> dpu_thread;
  std::shared_ptr<queue_t> sorting_queue;
  std::unique_ptr<SortingThread> sorting_thread;
  std::unique_ptr<GuiThread> gui_thread;
};  // namespace ai

// Entrance of multi-channel video demo
inline int main_for_video_demo_multiple_channel(
    int argc, char* argv[], const std::vector<DpuFilterDesciption>& filters) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv);
  auto gui_thread = GuiThread::instance();
  std::vector<Channel> channels;
  channels.reserve(filters.size());
  for (auto ch = 0u; ch < g_avi_file.size(); ++ch) {
    channels.emplace_back(ch, g_avi_file[ch], g_models[ch], filters);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // start everything
  MyThread::start_all();
  LOG(INFO) << "press Ctrl-C to exit....";
  while (!exiting) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG(INFO) << "waiting all thread to shutdown....";
  MyThread::stop_all();
  MyThread::wait_all();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}

}  // namespace ai
}  // namespace vitis
