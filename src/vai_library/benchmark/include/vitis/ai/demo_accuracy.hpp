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
#include <signal.h>
#include <unistd.h>

#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <type_traits>
#include <vitis/ai/bounded_queue.hpp>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_DEMO, "0")

namespace vitis {
namespace ai {

// A struct that can storage data and info for each frame
struct FrameInfo {
  unsigned long frame_id;
  cv::Mat mat;
  std::string single_name;
  int w;
  int h;
};

// A struct that can storage dpu output data and info for each frame
struct DpuResultInfo {
  unsigned long frame_id;
  std::shared_ptr<void> result_ptr;
  std::string single_name;
  int w;
  int h;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;
using queue_dpu = vitis::ai::BoundedQueue<DpuResultInfo>;
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

int g_last_frame_id = 0;
struct ReadImagesThread : public MyThread {
  ReadImagesThread(const std::string& images_list_file, queue_t* queue)
      : MyThread{},
        images_list_file_{images_list_file},
        frame_id_{0},
        queue_{queue} {}

  virtual ~ReadImagesThread() {}

  virtual void set_filename_and_queue(const std::string& images_list_file,
                                      queue_t* queue) {
    images_list_file_ = images_list_file;
    queue_ = queue;
  }

  virtual int run() override {
    std::ifstream fs(images_list_file_);
    std::string line;
    std::string single_name;
    while (getline(fs, line)) {
      auto image = cv::imread(line);
      if (image.empty()) {
        std::cerr << "cannot read image: " << line;
        continue;
      }
      single_name = get_single_name(line);
      int w = image.cols;
      int h = image.rows;
      auto frame = FrameInfo{++frame_id_, std::move(image), single_name, w, h};
      while (!queue_->push(frame, std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "push image frame_id " << frame_id_ << ",read images queue size "
          << queue_->size();
    }
    g_last_frame_id = frame_id_;
    return -1;
  }

  virtual std::string name() override { return std::string{"ReadImageThread"}; }

  std::string get_single_name(const std::string& line) {
    std::size_t found = line.rfind('/');
    if (found != std::string::npos) {
      return line.substr(found + 1);
    }
    return line;
  }

  std::string images_list_file_;
  unsigned long frame_id_;
  std::string single_name_;
  queue_t* queue_;
};

struct AccThread : public MyThread {
  AccThread()
      : MyThread{},
        queue_{
            new queue_dpu{
                30}  // assuming show acc is not bottleneck, 30 is high enough
        } {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT " << name();
  }
  virtual ~AccThread() {  //
  }
  virtual int run() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "acc queue size : " << queue_->size();
    return 0;
  }

  virtual std::string name() override { return std::string{"AccShowThread"}; }

  queue_dpu* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_dpu> queue_;
};

struct Filter {
  explicit Filter() {}
  virtual ~Filter() {}
  virtual std::shared_ptr<void> run(cv::Mat& input) = 0;
  virtual std::vector<std::shared_ptr<void>> run(
      std::vector<cv::Mat>& inputs) = 0;
  virtual size_t get_batch_size() = 0;
};

template <typename dpu_model_type_t>
struct DpuRunFilter : public Filter {
  DpuRunFilter(std::unique_ptr<dpu_model_type_t>&& dpu_model)
      : dpu_model_{std::move(dpu_model)} {}
  virtual ~DpuRunFilter() {}
  virtual std::shared_ptr<void> run(cv::Mat& image) override {
    auto* model_result = new auto(dpu_model_->run(image));
    return std::shared_ptr<void>(model_result);
  }
  virtual std::vector<std::shared_ptr<void>> run(
      std::vector<cv::Mat>& images) override {
    std::vector<std::shared_ptr<void>> rets;
    auto results = dpu_model_->run(images);
    for (auto i = 0u; i < results.size(); ++i) {
      rets.emplace_back(std::shared_ptr<void>(new auto(results[i])));
    }
    return rets;
  }
  virtual size_t get_batch_size() override {
    return dpu_model_->get_input_batch();
  }
  std::unique_ptr<dpu_model_type_t> dpu_model_;
};
template <typename FactoryMethod>
std::unique_ptr<Filter> create_rundpu_filter(
    const FactoryMethod& factory_method) {
  using dpu_model_type_t = typename decltype(factory_method())::element_type;
  return std::unique_ptr<Filter>(
      new DpuRunFilter<dpu_model_type_t>(factory_method()));
}

bool g_is_completed = false;
// Execute dpu run filter
struct DpuRunThread : public MyThread {
  DpuRunThread(std::unique_ptr<Filter>&& dpu_filter, queue_t* queue_in,
               queue_dpu* queue_out, const std::string& suffix)
      : MyThread{},
        dpu_filter_{std::move(dpu_filter)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT " << name();
  }
  virtual ~DpuRunThread() {}

  virtual int run() override {
    size_t batch = dpu_filter_->get_batch_size();
    std::vector<FrameInfo> frames;
    std::vector<cv::Mat> images;
    DpuResultInfo dpu_result;

    while (frames.size() < batch) {
      FrameInfo frame;
      if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
        if (frames.size() > 0 &&
            (int(frames.rbegin()->frame_id) == g_last_frame_id)) {
          g_is_completed = true;
        }
        if (g_is_completed) break;
        if (is_stopped()) return -1;
        continue;
      }
      if (int(frame.frame_id) == g_last_frame_id) g_is_completed = true;

      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "thread [" << name() << "] read image frame_id: " << frame.frame_id
          << " " << frame.mat.cols << " " << frame.mat.rows;
      frames.emplace_back(frame);
    }
    for (auto& f : frames) {
      images.emplace_back(f.mat);
    }
    auto result_ptrs = dpu_filter_->run(images);
    for (auto i = 0u; i < frames.size(); ++i) {
      dpu_result.frame_id = frames[i].frame_id;
      dpu_result.result_ptr = result_ptrs[i];
      dpu_result.single_name = frames[i].single_name;
      dpu_result.w = frames[i].w;
      dpu_result.h = frames[i].h;
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "thread [" << name()
          << "] dpu_result frame_id: " << dpu_result.frame_id
          << " ,dpu queue size :" << queue_out_->size();
      if (int(dpu_result.frame_id) == g_last_frame_id) g_is_completed = true;
      while (!queue_out_->push(dpu_result, std::chrono::milliseconds(500))) {
        if (is_stopped()) {
          return -1;
        }
      }
    }
    if (g_is_completed) return -1;
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<Filter> dpu_filter_;
  queue_t* queue_in_;
  queue_dpu* queue_out_;
  std::string suffix_;
};

// Implement sorting thread
struct SortDpuThread : public MyThread {
  SortDpuThread(queue_dpu* queue_in, queue_dpu* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT SORTING";
  }
  virtual ~SortDpuThread() {}
  virtual int run() override {
    DpuResultInfo dpu_result;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond = std::function<bool(const DpuResultInfo&)>{
        [frame_id](const DpuResultInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
    if (!queue_in_->pop(dpu_result, cond, std::chrono::milliseconds(500))) {
      frame_id_--;
      return 0;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << "in queue size " << queue_in_->size() << ", frame id "
        << dpu_result.frame_id << " sorting queue size " << queue_out_->size();

    while (!queue_out_->push(dpu_result, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_dpu* queue_in_;
  queue_dpu* queue_out_;
  unsigned long frame_id_ = 0;
  std::string suffix_;
};

inline void usage_accuracy(const char* progname) {
  std::cout << "usage: " << progname << " "
            << "      <model name>\n"
            << "      <input image list file>\n"
            << "      <output file name>\n"
            << "      -t <num_of_threads>\n"
            << std::endl;
  return;
}
/*
  global command line options
 */
static int g_num_of_threads;
static std::string g_input_file;

inline void parse_opt(int argc, char* argv[], int start_pos = 1) {
  int opt = 0;
  optind = start_pos;
  while ((opt = getopt(argc, argv, "c:t:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads = std::stoi(optarg);
        break;
      case 'c':  // how many channels
        break;   // do nothing. parse it in outside logic.
      default:
        usage_accuracy(argv[0]);
        exit(1);
    }
  }
  g_input_file = std::string(argv[optind]);
  if (g_input_file.empty()) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_num_of_threads == 0) {
    // by default, all channels has at least one thread
    g_num_of_threads = 1;
  }
  return;
}

// Entrance of accuracy demo
template <typename FactoryMethod>
int main_for_accuracy_demo(
    int argc, char* argv[], const FactoryMethod& factory_method,
    std::shared_ptr<AccThread> acc_thread, int start_pos = 1,
    std::shared_ptr<ReadImagesThread> read_thread = NULL) {
  signal(SIGINT, MyThread::signal_handler);
  parse_opt(argc, argv, start_pos);
  {
    auto images_queue = std::unique_ptr<queue_t>{new queue_t{50}};
    if (read_thread == NULL) {
      read_thread = std::shared_ptr<ReadImagesThread>(
          new ReadImagesThread{g_input_file, images_queue.get()});
    } else {
      read_thread->set_filename_and_queue(g_input_file, images_queue.get());
    }
    auto dpu_run_thread = std::vector<std::unique_ptr<DpuRunThread>>{};
    auto sorting_queue =
        std::unique_ptr<queue_dpu>(new queue_dpu(500 * g_num_of_threads));
    auto acc_queue = acc_thread->getQueue();
    for (int i = 0; i < g_num_of_threads; ++i) {
      dpu_run_thread.emplace_back(new DpuRunThread(
          create_rundpu_filter(factory_method), images_queue.get(),
          sorting_queue.get(), std::to_string(i)));
    }
    auto sorting_thread = std::unique_ptr<SortDpuThread>(
        new SortDpuThread(sorting_queue.get(), acc_queue, std::to_string(0)));
    // start everything
    MyThread::start_all();
    acc_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
  return 0;
}
}  // namespace ai
}  // namespace vitis
