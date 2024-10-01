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

#include <cassert>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>
#include <thread>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_list.hpp>
#include <vitis/ai/stat_samples.hpp>
#include <vitis/ai/time_measure.hpp>
DEF_ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME, "0");
namespace vitis {
namespace ai {

struct BenchMarkResult {
  long ret;
  StatSamples e2eSamples;
  StatSamples dpuSamples;
};

std::mutex g_mtx;
int g_num_of_threads = 1;
int g_num_of_seconds = 30;
std::string g_list_name = "image.list";
std::string g_report_file_name = "";
long g_total = 0;
double g_e2e_mean = 0.0;
double g_dpu_mean = 0.0;
bool g_stop = false;
std::atomic<int> _counter(0);
long act_time = 30000000;

template <typename T>
inline BenchMarkResult thread_main_for_performance(const ImageList* image_list,
                                                   std::unique_ptr<T>&& model) {
  std::unique_lock<std::mutex> lock_t(g_mtx);
  lock_t.unlock();
  long ret = 0;
  StatSamples e2e_stat_samples(10000);
  StatSamples dpu_stat_samples(10000);
  while (!g_stop) {
    vitis::ai::TimeMeasure::getThreadLocalForDpu().reset();
    auto start = std::chrono::steady_clock::now();
    auto batch = model->get_input_batch();
    std::vector<std::pair<cv::Mat, cv::Mat>> imgs;
    imgs.reserve(batch);
    for (auto n = 0u; n < batch; ++n) {
      auto inputs =
          std::make_pair((*image_list)[2 * ret], (*image_list)[2 * ret + 1]);
      ret++;
      imgs.push_back(inputs);
    }
    model->run(imgs);
    auto end = std::chrono::steady_clock::now();
    auto end2endtime =
        int(std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());
    auto dputime = vitis::ai::TimeMeasure::getThreadLocalForDpu().get();

    e2e_stat_samples.addSample(end2endtime);
    dpu_stat_samples.addSample(dputime);
    _counter += 1;
  }
  return BenchMarkResult{ret, std::move(e2e_stat_samples),
                         std::move(dpu_stat_samples)};
}

static void signal_handler(int signal) { g_stop = true; }
static void usage() {
  std::cout << "usage: env dpbenchmark \n"
               " -l <log_file_name> \n"
               " -t <num_of_threads> \n"
               " -s <num_of_seconds> \n"
               " <image list file> \n"
            << std::endl;
}
inline void parse_opt(int argc, char* argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "t:s:l:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads = std::stoi(optarg);
        break;
      case 's':
        g_num_of_seconds = std::stoi(optarg);
        break;
      case 'l':
        g_report_file_name = optarg;
        break;
      default:
        usage();
        exit(1);
    }
  }
  if (optind >= argc) {
    std::cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  g_list_name = argv[argc - 1];
  return;
}

static void report(std::ostream* p_out) {
  std::ostream& out = *p_out;
  float sec = (float)act_time / 1000000.0;
  float fps = ((float)g_total) / sec;
  out << "FPS=" << fps << "\n";
  out << "E2E_MEAN=" << g_e2e_mean << "\n";
  out << "DPU_MEAN=" << g_dpu_mean << "\n";
  out << std::flush;
  return;
}

static void report_for_mt(std::ostream* p_out) {
  std::ostream& out = *p_out;
  float sec = (float)act_time / 1000000.0;
  float fps = ((float)g_total) / sec;
  out << "FPS=" << fps << "\n";
  out << std::flush;
  return;
}
int total_step = 0;
int step = 10;
static void report_step(std::ostream* p_out) {
  std::ostream& out = *p_out;
  float fps = ((float)total_step) / ((float)step);
  out << "step " << step << "FPS=" << fps << "\n";
  out << std::flush;
  return;
}

template <typename T>
inline int main_for_performance(int argc, char* argv[], T factory_method) {
  parse_opt(argc, argv);
  ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME) = 1;
  auto lazy_load_image = false;
  auto image_list =
      std::unique_ptr<ImageList>(new ImageList(g_list_name, lazy_load_image));
  if (image_list->empty()) {
    LOG(FATAL) << "list of images are empty [" << image_list->to_string()
               << "]";
  }
  auto model = factory_method();
  using model_t = typename decltype(model)::element_type;
  auto width = model->get_input_width();
  auto height = model->get_input_height();
  image_list->resize_images(width, height);
  //
  std::vector<std::future<BenchMarkResult>> results;
  results.reserve(g_num_of_threads);

  std::ostream* report_fs = &std::cout;
  auto fs = std::unique_ptr<std::ostream>{};
  if (!g_report_file_name.empty()) {
    LOG(INFO) << "writing report to " << g_report_file_name;
    fs = std::unique_ptr<std::ostream>{
        new std::ofstream(g_report_file_name.c_str(), std::ofstream::out)};
    report_fs = fs.get();
  } else {
    LOG(INFO) << "writing report to <STDOUT>";
  }

  std::vector<decltype(model)> models;
  for (int i = 0; i < g_num_of_threads; ++i) {
    // every thread should have its own model object.
    if (i == 0) {
      // the first thread reuse the model which is already created.
      models.emplace_back(std::move(model));
    } else {
      models.emplace_back(factory_method());
    }
  }

  std::unique_lock<std::mutex> lock_main(g_mtx);
  for (int i = 0; i < g_num_of_threads; ++i) {
    results.emplace_back(std::async(std::launch::async,
                                    thread_main_for_performance<model_t>,  //
                                    image_list.get(),                      //
                                    std::move(models[i])));
  }
  signal(SIGALRM, signal_handler);
  alarm(g_num_of_seconds);

  auto exe_start = std::chrono::system_clock::now();
  lock_main.unlock();
  for (int i = 0; i < g_num_of_seconds; i = i + step) {
    LOG(INFO) << "waiting for " << i << "/" << g_num_of_seconds << " seconds, "
              << g_num_of_threads << " threads running";
    std::this_thread::sleep_for(std::chrono::milliseconds(step * 1000));
    total_step = _counter;
    _counter = 0;
    if (0) report_step(report_fs);
    // LOG(INFO) << "FPS : " << (float)(((float)total_5)/(float)step) ;
  }
  LOG(INFO) << "waiting for threads terminated";
  long total = 0;

  StatSamples e2eStatSamples(0);
  StatSamples dpuStatSamples(0);
  for (auto& r : results) {
    auto result = r.get();
    total = total + result.ret;
    e2eStatSamples.merge(result.e2eSamples);
    dpuStatSamples.merge(result.dpuSamples);
  }

  act_time = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::system_clock::now() - exe_start)
                 .count();
  g_e2e_mean = e2eStatSamples.getMean();
  g_dpu_mean = dpuStatSamples.getMean();

  g_total = total;
  /*std::ostream* report_fs = &std::cout;
  auto fs = std::unique_ptr<std::ostream>{};
  if(!g_report_file_name.empty()) {
      LOG(INFO) << "writing report to " << g_report_file_name;
      fs = std::unique_ptr<std::ostream>{
          new std::ofstream(g_report_file_name.c_str(), std::ofstream::out)};
      report_fs = fs.get();
  } else {
      LOG(INFO) << "writing report to <STDOUT>";
      }*/
  if (g_num_of_threads == 1) {
    report(report_fs);
  } else {
    report_for_mt(report_fs);
  }
  return 0;
}

}  // namespace ai
}  // namespace vitis
