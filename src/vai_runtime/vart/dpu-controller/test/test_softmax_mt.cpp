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
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vitis/ai/performance_test.hpp>

#include "vitis/ai/parse_value.hpp"
#include "vitis/ai/profiling.hpp"
#include "xir/sfm_controller.hpp"
using std::vector;
using std::cout;
using std::endl;
using std::stoi;
using xir::SfmController;
DEF_ENV_PARAM(DEBUG_TEST, "0");

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(input[i] * scale);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}

static vector<int8_t> random_vector_char(size_t sz) {
  static std::mt19937 rng(100);
  static std::uniform_int_distribution<int> dist;
  auto ret = vector<int8_t>(sz);
  for (auto i = 0u; i < ret.size(); ++i) {
    auto value = dist(rng);
    ret[i] = (int8_t)(value && 0xFF);
  }
  return ret;
}

// static vector<vector<int8_t>> random_vector_char(int num_of_threads, int cls,
//                                                  int group) {
//   auto ret = vector<vector<int8_t>>((size_t)num_of_threads);
//   for (auto& r : ret) {
//     r = random_vector_char((size_t)(cls * group));
//   }
//   return ret;
// }

static vector<float> generate_ref(const int8_t* input, float scale, int cls,
                                  int group) {
  auto ret = vector<float>((size_t)(cls * group));
  softmax_c(input, scale, cls, group, &ret[0]);
  return ret;
}

static void compare(int cls, int group, const signed char* input,
                    const float* output1, const float* output2) {
  int error = 0;
  for (auto g = 0; g < group; ++g) {
    for (auto i = 0; i < cls; ++i) {
      auto idx = g * cls + i;
      auto diff = output1[idx] - output2[idx];
      if ((diff != 0.0 && std::abs(diff) > 0.001)) {
        error = error + 1;
      }
    }
  }
  CHECK_EQ(error, 0);
}

class MyPerformanceTestRunner : public vitis::ai::PerformanceTestRunner {
 public:
  explicit MyPerformanceTestRunner(float scale, int cls, int group);
  virtual ~MyPerformanceTestRunner() = default;
  MyPerformanceTestRunner(const PerformanceTestRunner& other) = delete;
  MyPerformanceTestRunner& operator=(const PerformanceTestRunner& rhs) = delete;

 public:
  virtual void step(size_t idx, int thread_id) override;
  virtual size_t get_result() override;

 private:
  float scale_;
  int cls_;
  int group_;
  std::shared_ptr<xir::SfmController> sfm_;
  vector<int8_t> input_;
  vector<float> output_;
  size_t result_ = 0;
};

MyPerformanceTestRunner::MyPerformanceTestRunner(float scale, int cls,
                                                 int group)
    : scale_{scale},                                        //
      cls_{cls},                                            //
      group_{group},                                        //
      sfm_{xir::SfmController::get_instance()},             //
      input_{random_vector_char(cls_ * group_)},            //
      output_{generate_ref(&input_[0], scale, cls, group)}  //
{}

void MyPerformanceTestRunner::step(size_t idx, int thread_id) {
  vector<float> output1(cls_ * group_);
  sfm_->run(&input_[0], scale_, cls_, group_, &output1[0]);
  compare(cls_, group_, &input_[0], &output1[0], &output_[0]);
  result_ = result_ + 1u;
  return;
}

size_t MyPerformanceTestRunner::get_result() {
  LOG(INFO) << "result_ " << result_ << " "  //
      ;
  return result_;
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    cout << "usage: " << argv[0] << "<num_of_threads> <fixpos> <cls> <group>"
         << endl;
    return 0;
  }
  int num_of_threads = stoi(argv[1]);
  int fixpos = stoi(argv[2]);
  int cls = stoi(argv[3]);
  int group = stoi(argv[4]);
  float scale = std::exp2f(-1.0f * (float)fixpos);
  auto runners = vector<std::unique_ptr<vitis::ai::PerformanceTestRunner>>();
  for (auto i = 0; i < num_of_threads; ++i) {
    LOG(INFO) << "create runner ... " << i << "/" << num_of_threads;
    runners.emplace_back(
        std::make_unique<MyPerformanceTestRunner>(scale, cls, group));
  }
  return std::make_unique<vitis::ai::PerformanceTest>()->main(
      argc, argv, std::move(runners));
}
