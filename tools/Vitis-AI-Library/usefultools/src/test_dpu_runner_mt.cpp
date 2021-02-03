/*
 * Copyright 2019 Xilinx Inc.
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
#include <google/protobuf/message.h>
#include <openssl/md5.h>

#include <chrono>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vart/dpu/vitis_dpu_runner_factory.hpp>
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner_ext.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/performance_test.hpp>
#include <xir/tensor/tensor.hpp>

using Clock = chrono::steady_clock;

DEF_ENV_PARAM(DEBUG_TEST, "0");
DEF_ENV_PARAM(NUM_OF_REF, "4");
DEF_ENV_PARAM(THREAD_ADD_LOCK, "0");
DEF_ENV_PARAM(SAME_INPUT, "0");
DEF_ENV_PARAM(SAVE_INPUT_TO_FILE, "0");
DEF_ENV_PARAM(SAVE_ERROR_OUTPUT_TO_FILE, "0");
DEF_ENV_PARAM(COPY_INPUT, "1");
DEF_ENV_PARAM(COPY_OUTPUT, "1");
DEF_ENV_PARAM(ENABLE_MEMCMP, "1");

using namespace std;

class MyPerformanceTestRunner : public vitis::ai::PerformanceTestRunner {
 public:
  explicit MyPerformanceTestRunner(const string& filename,  //
                                   const string& kernel);
  virtual ~MyPerformanceTestRunner();
  MyPerformanceTestRunner(const PerformanceTestRunner& other) = delete;
  MyPerformanceTestRunner& operator=(const PerformanceTestRunner& rhs) = delete;

 public:
  virtual void step(size_t idx, int thread_id) override;
  virtual size_t get_result() override;
  vart::Runner* get_runner() { return runner_.get(); };

 private:
  unique_ptr<vart::Runner> runner_;
  const vector<vector<vector<char>>> inputs_;
  const vector<vector<vector<char>>> ref_outputs_;
  vector<vector<char>> output_buffers_;
  size_t result_ = 0;
  int error_counter_ = 0;
  int ok_counter_ = 0;
};

static vector<char> random_vector_char(size_t sz, int batch_size) {
  auto ret = vector<char>(sz);
  if (ENV_PARAM(SAME_INPUT)) {
    LOG(INFO) << "sz " << sz << " batch size " << batch_size;
    for (auto j = 0; j < batch_size; j++) {
      for (auto i = 0u; i < (sz / batch_size); ++i) {
        ret[j * (sz / batch_size) + i] = i % 100;
      }
    }
  } else {
    static mt19937 rng(100);
    static uniform_int_distribution<char> dist;
    for (auto i = 0u; i < sz; ++i) {
      ret[i] = dist(rng);
    }
  }
  return ret;
}

static vector<vector<char>> allocate_buffer(
    vector<const xir::Tensor*> tensors) {
  auto ret = vector<vector<char>>(tensors.size());
  for (auto i = 0u; i < ret.size(); ++i) {
    auto size_of_input = tensors[i]->get_element_num();
    ret[i] = random_vector_char(size_of_input, tensors[i]->get_shape()[0]);
  }
  return ret;
}

static vector<vector<vector<char>>> generate_inputs(vart::Runner* runner) {
  auto input_tensors = runner->get_input_tensors();
  auto sz = (size_t)ENV_PARAM(NUM_OF_REF);
  auto ret = vector<vector<vector<char>>>(sz);
  for (auto i = 0u; i < sz; ++i) {
    ret[i] = allocate_buffer(input_tensors);
  }
  return ret;
}

static void copy_input(const vector<char>& data, vart::TensorBuffer* tb) {
  size_t batch_size = tb->get_tensor()->get_shape()[0];
  for (auto i = 0u; i < batch_size; ++i) {
    uint64_t input_data = 0u;
    auto input_size = 0u;
    auto dims = vector<int>(tb->get_tensor()->get_shape().size(), 0);
    dims[0] = (int)i;
    tie(input_data, input_size) = tb->data(dims);
    auto size_per_batch = tb->get_tensor()->get_data_size() / batch_size;
    memcpy((char*)input_data, &data[i * size_per_batch], size_per_batch);
  }
  return;
}

static void copy_output(vector<char>& data, vart::TensorBuffer* tb) {
  size_t batch_size = tb->get_tensor()->get_shape()[0];
  for (auto i = 0u; i < batch_size; ++i) {
    uint64_t output_data = 0u;
    auto output_size = 0u;
    auto dims = vector<int>(tb->get_tensor()->get_shape().size(), 0);
    dims[0] = (int)i;
    tie(output_data, output_size) = tb->data(dims);
    auto size_per_batch = tb->get_tensor()->get_data_size() / batch_size;
    memcpy(&data[i * size_per_batch], (const void*)output_data, size_per_batch);
  }
  return;
}

static void copy_inputs(const vector<vector<char>>& data,
                        vector<vart::TensorBuffer*> tb) {
  CHECK_EQ(data.size(), tb.size());
  auto sz = data.size();
  for (auto i = 0u; i < sz; ++i) {
    copy_input(data[i], tb[i]);
  }
  return;
}

static void copy_outputs(vector<vector<char>>& data,
                         vector<vart::TensorBuffer*> tb) {
  CHECK_EQ(data.size(), tb.size());
  auto sz = data.size();
  for (auto i = 0u; i < sz; ++i) {
    copy_output(data[i], tb[i]);
  }
  return;
}

static void write_to_file(const char* buf, size_t size, const string& file) {
  auto mode = ios_base::out | ios_base::binary | ios_base::trunc;
  CHECK(ofstream(file, mode).write(buf, size).good())
      << " faild to write to " << file;
}
static void write_tensors(size_t batch_size, const vector<char>& tensor_buffers,
                          const string& file) {
  for (auto batch = 0u; batch < tensor_buffers.size() / batch_size; ++batch) {
    write_to_file(&tensor_buffers[batch * batch_size], batch_size,
                  file + "_batch_" + to_string(batch) + ".bin");
  }
}

static void write_tensors(size_t batch_size,
                          const vector<vector<vector<char>>>& tensor_buffers,
                          const string& file) {
  int c = 0;
  for (auto& t : tensor_buffers) {
    write_tensors(batch_size, t[0], file + "_c_" + to_string(c++));
  }
}
static vector<vector<vector<char>>> generate_outputs(
    vart::Runner* runner, const vector<vector<vector<char>>>& inputs) {
  auto output_tensors = runner->get_output_tensors();
  auto sz = inputs.size();
  auto num_of_batch = output_tensors[0]->get_shape()[0];
  auto ret = vector<vector<vector<char>>>(sz);
  for (auto i = 0u; i < sz; ++i) {
    ret[i] = allocate_buffer(output_tensors);
  }
  for (auto i = 0u; i < sz; ++i) {
    auto r = dynamic_cast<vart::RunnerExt*>(runner);
    auto dpu_inputs = r->get_inputs();
    auto dpu_outputs = r->get_outputs();
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "generating ref " << i << endl;
    copy_inputs(inputs[i], dpu_inputs);
    for (auto input : dpu_inputs) {
      input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                   input->get_tensor()->get_shape()[0]);
    }

    runner->execute_async(dpu_inputs, dpu_outputs);
    runner->wait(0, 0);
    for (auto output : dpu_outputs) {
      output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                   output->get_tensor()->get_shape()[0]);
    }

    copy_outputs(ret[i], dpu_outputs);
  }
  if (ENV_PARAM(SAVE_INPUT_TO_FILE)) {
    auto input_batch = inputs[0][0].size() / num_of_batch;
    auto output_batch = ret[0][0].size() / num_of_batch;
    write_tensors(input_batch, inputs, string("ref_input"));
    write_tensors(output_batch, ret, string("ref_ouput"));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "references are generated" << endl;
  return ret;
}

MyPerformanceTestRunner::MyPerformanceTestRunner(const string& filename,
                                                 const string& kernel)
    : runner_{vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel)},
      inputs_{generate_inputs(runner_.get())},
      ref_outputs_{generate_outputs(runner_.get(), inputs_)},
      output_buffers_{allocate_buffer(runner_->get_output_tensors())} {}

std::atomic<u_int64_t> errors_total = 0;
MyPerformanceTestRunner::~MyPerformanceTestRunner() {
  errors_total += error_counter_;
  LOG_IF(INFO, error_counter_) << "error_counter = " << error_counter_
                               << ",errors_total = " << errors_total;
}

void MyPerformanceTestRunner::step(size_t idx, int thread_id) {
  if (ENV_PARAM(THREAD_ADD_LOCK)) {
    static mutex mtx;
    lock_guard<mutex> lock(mtx);
  }
  CHECK_EQ(inputs_.size(), ref_outputs_.size());
  idx = idx % inputs_.size();
  auto output_tensors = runner_->get_output_tensors();
  auto r = dynamic_cast<vart::RunnerExt*>(runner_.get());
  auto dpu_inputs = r->get_inputs();
  auto dpu_outputs = r->get_outputs();
  if (ENV_PARAM(COPY_INPUT)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "copying input...";
    copy_inputs(inputs_[idx], dpu_inputs);
  }

  for (auto input : dpu_inputs) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }
  runner_->execute_async(dpu_inputs, dpu_outputs);
  runner_->wait(0, 0);
  for (auto output : dpu_outputs) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }
  if (ENV_PARAM(COPY_INPUT) && ENV_PARAM(COPY_OUTPUT)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "copying output...";
    copy_outputs(output_buffers_, dpu_outputs);
    if (ENV_PARAM(ENABLE_MEMCMP)) {
      for (auto i = 0u; i < output_buffers_.size(); ++i) {
        auto mem_size = output_buffers_[i].size();
        auto num_of_batch = output_tensors[i]->get_shape()[0];
        auto size_per_batch = mem_size / num_of_batch;
        CHECK_EQ(mem_size, ref_outputs_[idx][i].size());
        auto ok = true;
        LOG_IF(INFO, false)
            << " mem_size " << mem_size << " num_of_batch " << num_of_batch
            << " size_per_batch " << size_per_batch;
        for (auto batch_idx = 0; batch_idx < num_of_batch; ++batch_idx) {
          auto r = memcmp(&output_buffers_[i][batch_idx * size_per_batch],
                          &ref_outputs_[idx][i][batch_idx * size_per_batch],
                          size_per_batch);
          ok = ok && r == 0;
          LOG_IF(INFO, r != 0)
              << "thread " << thread_id << " batch " << batch_idx
              << " error_counter " << error_counter_ + 1 << " ok_counter "
              << (ok_counter_ + 1);
          if (r != 0 && ENV_PARAM(SAVE_ERROR_OUTPUT_TO_FILE)) {
            auto ref_file_name =
                string("ref_t") + to_string(thread_id) + string("_") +
                to_string(idx) + string("_batch_") + to_string(batch_idx) +
                string("_") + to_string(error_counter_ + 1) + string(".bin");
            auto out_file_name =
                string("out_t") + to_string(thread_id) + string("_") +
                to_string(idx) + string("_batch_") + to_string(batch_idx) +
                string("_") + to_string(error_counter_ + 1) + string(".bin");

            write_to_file(&ref_outputs_[idx][i][batch_idx * size_per_batch],
                          size_per_batch, ref_file_name);
            write_to_file(&output_buffers_[i][batch_idx * size_per_batch],
                          size_per_batch, out_file_name);
          }
        }
        if (ok) {
          ok_counter_++;
        } else {
          error_counter_++;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
            << "thread " << thread_id << "checking ok is " << ok
            << " idx =" << idx << ",i=" << i;
      }
    }
  }
  result_ = result_ + runner_->get_input_tensors()[0]->get_shape()[0];
  return;
}

size_t MyPerformanceTestRunner::get_result() { return result_; }

bool test_dpu_runner_mt(string filename, string kernel, uint32_t runner_num) {
  CHECK_GT(runner_num, 0);
  {
    auto runners = vector<unique_ptr<vitis::ai::PerformanceTestRunner>>();
    for (auto i = 0u; i < runner_num; ++i) {
      LOG(INFO) << "create runner ... " << i << "/" << runner_num;
      runners.emplace_back(
          make_unique<MyPerformanceTestRunner>(filename, kernel));
    }
    // not use
    int argc = 0;
    char* argv[] = {};
    make_unique<vitis::ai::PerformanceTest>()->main(argc, argv, move(runners));
  }
  return errors_total == 0;
}
