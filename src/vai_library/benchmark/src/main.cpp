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
#include <signal.h>
#include <unistd.h>

#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vitis/base/dpu/time_measure.hpp>

#include "../include/benchmark.hpp"
#include "./image_list.hpp"
#include "./stat_samples.hpp"
using namespace std;

int g_num_of_threads = 1;
int g_num_of_seconds = 30;
string g_list_name = "image.list";
string g_output_dir = "";
string g_model_name = "";

string g_report_file_name = "";
string g_run_mode = "accuracy";
long g_total = 0;
double g_e2e_mean = 0.0;
double g_dpu_mean = 0.0;
bool g_stop = false;

static int main_accuracy();

static int main_performance();
static void signal_handler(int signal) { g_stop = true; }

static void usage() {
  cout << "usage: env DEEPHI_DPU_CONSUMING_TIME=1 dpbenchmark -r <run_mode> -t "
          "<num_of_threads> -s <num_of_seconds> "
          " -m <model name>"
          " <image list file> "
       << endl;
}

static void parse_opt(int argc, char *argv[]) {
  int opt = 0;

  while ((opt = getopt(argc, argv, "r:t:s:l:m:p:")) != -1) {
    switch (opt) {
      case 't':
        g_num_of_threads = std::stoi(optarg);
        break;
      case 's':
        g_num_of_seconds = std::stoi(optarg);
        break;
      case 'm':
        g_model_name = optarg;
        break;
      case 'l':
        g_report_file_name = optarg;
        break;
      case 'p':
        g_output_dir = optarg;
        break;
      case 'r':
        g_run_mode = optarg;
        if (g_run_mode == "performance" || g_run_mode == "accuracy") {
        } else {
          LOG(FATAL) << "run mode must be 'accuracy' or 'performance'";
        }
        break;

      default:
        usage();
        exit(1);
    }
  }
  if (optind >= argc) {
    cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  if (g_model_name.empty()) {
    cerr << "model name is unknown, please use -m to specify a model name"
         << endl;
    exit(EXIT_FAILURE);
  }
  g_list_name = argv[optind];
  return;
}

BenchMarkResult thread_main(const ImageList *image_list,
                            unique_ptr<vitis::benchmark::Benchmark> &&model) {
  long ret = 0;
  StatSamples e2e_stat_samples(10000);
  StatSamples dpu_stat_samples(10000);
  for (ret = 0; !g_stop; ++ret) {
    vitis::base::TimeMeasure::getThreadLocalForDpu().reset();
    auto start = std::chrono::steady_clock::now();
    model->run_performance((*image_list)[ret]);
    auto end = std::chrono::steady_clock::now();
    auto end2endtime =
        int(std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count());
    auto dputime = vitis::base::TimeMeasure::getThreadLocalForDpu().get();

    e2e_stat_samples.addSample(end2endtime);
    dpu_stat_samples.addSample(dputime);
    /*std::cout << "end2endtime "  << end2endtime << " " //
              << "dputime "  << dputime << " " //
              << std::endl;*/
  }
  /*std::cout << "ret " << ret << " " //
            << "e2e mean " << e2e_stat_samples.getMean() << " "
            << "dpu mean " << dpu_stat_samples.getMean() << " "
            << "e2e stdvar " << e2e_stat_samples.getStdVar() << " "
            << "dpu stdvar " << dpu_stat_samples.getStdVar() << " "
            << std::endl;
  */
  return BenchMarkResult{ret, std::move(e2e_stat_samples),
                         std::move(dpu_stat_samples)};
}

static void report(ostream *p_out) {
  ostream &out = *p_out;
  float fps = ((float)g_total) / ((float)g_num_of_seconds);
  out << "FPS=" << fps << "\n";
  out << "E2E_MEAN=" << g_e2e_mean << "\n";
  out << "DPU_MEAN=" << g_dpu_mean << "\n";
  out << flush;
  return;
}

static void report_for_mt(std::ostream *p_out) {
  std::ostream &out = *p_out;
  float sec = (float)act_time / 1000000.0;
  float fps = ((float)g_total) / sec;
  out << "FPS=" << fps << "\n";
  out << std::flush;
  return;
}
int main(int argc, char *argv[]) {
  parse_opt(argc, argv);
  if (g_run_mode == "performance") {
    return main_performance();
  } else if (g_run_mode == "accuracy") {
    return main_accuracy();
  }
  return 0;
}

static int main_performance() {
  auto lazy_load_image = false;
  auto image_list =
      std::unique_ptr<ImageList>(new ImageList(g_list_name, lazy_load_image));
  if (image_list->empty()) {
    LOG(FATAL) << "list of images are empty [" << image_list->to_string()
               << "]";
  }
  auto model = vitis::benchmark::create(g_model_name);
  auto width = model->getInputWidth();
  auto height = model->getInputHeight();
  image_list->resize_images(width, height);
  //
  vector<std::future<BenchMarkResult>> results;
  results.reserve(g_num_of_threads);

  for (int i = 0; i < g_num_of_threads; ++i) {
    // every thread should have its own model object.
    if (i == 0) {
      // the first thread reuse the model which is already created.
      results.emplace_back(
          std::async(thread_main, image_list.get(), std::move(model)));
    } else {
      results.emplace_back(std::async(thread_main, image_list.get(),
                                      vitis::benchmark::create(g_model_name)));
    }
  }
  signal(SIGALRM, signal_handler);
  alarm(g_num_of_seconds);

  for (int i = 0; i < g_num_of_seconds; i = i + 5) {
    LOG(INFO) << "waiting for " << i << "/" << g_num_of_seconds << " seconds, "
              << g_num_of_threads << " threads running";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }
  LOG(INFO) << "waiting for threads terminated";
  long total = 0;

  StatSamples e2eStatSamples(0);
  StatSamples dpuStatSamples(0);
  for (auto &r : results) {
    auto result = r.get();
    total = total + result.ret;
    /*
    std::cout << "r.get().e2eSamples " << e2eSamples.getMean() << " " //
              << "r.get().dpuSamples " << dpuSamples.getMean() << " " //
              << std::endl;
    */
    e2eStatSamples.merge(result.e2eSamples);
    dpuStatSamples.merge(result.dpuSamples);
  }

  /*std::cout << "E2E_MEAN=" << e2eStatSamples.getMean() << " "
           << std::endl;
 std::cout << "DPU_MEAN=" << dpuStatSamples.getMean() << " " << std::endl;
 */
  g_e2e_mean = e2eStatSamples.getMean();
  g_dpu_mean = dpuStatSamples.getMean();

  g_total = total;
  ostream *report_fs = &std::cout;
  auto fs = std::unique_ptr<ostream>{};
  if (!g_report_file_name.empty()) {
    LOG(INFO) << "writing report to " << g_report_file_name;
    fs = std::unique_ptr<ostream>{
        new std::ofstream(g_report_file_name.c_str(), std::ofstream::out)};
    report_fs = fs.get();
  } else {
    LOG(INFO) << "writing report to <STDOUT>";
  }
  if (g_num_of_threads==1){
    report(report_fs);
  }else{
    report_for_mt(report_fs);
  }
  return 0;
}

static int main_accuracy() {
  auto lazy_load_image = true;
  auto image_list =
      std::unique_ptr<ImageList>(new ImageList(g_list_name, lazy_load_image));
  if (image_list->empty()) {
    LOG(FATAL) << "list of images are empty [" << image_list->to_string()
               << "]";
  }
  ostream *report_fs = &std::cout;
  auto fs = std::unique_ptr<ostream>{};
  if (!g_report_file_name.empty()) {
    fs = std::unique_ptr<ostream>{
        new std::ofstream(g_report_file_name.c_str(), std::ofstream::out)};
    report_fs = fs.get();
    LOG(INFO) << "writing report to " << g_report_file_name;
  } else {
    LOG(INFO) << "writing report to <STDOUT>";
  }
  auto model = vitis::benchmark::create(g_model_name);
  for (auto i = 0u; i < image_list->size(); ++i) {
    model->run_accuracy(image_list->getName(i), (*image_list)[i], *report_fs);
  }
  return 0;
}
