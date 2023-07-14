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
#include <google/protobuf/message.h>
#include <pybind11/pybind11.h>

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
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include "vitis/ai/graph_runner.hpp"

using Clock = std::chrono::steady_clock;

DEF_ENV_PARAM(DEBUG_TEST, "0");
DEF_ENV_PARAM(LOG_ERROR_COUNTER, "0");
DEF_ENV_PARAM(NUM_OF_REF, "4");
DEF_ENV_PARAM(THREAD_ADD_LOCK, "0");
DEF_ENV_PARAM(SAME_INPUT, "0");
DEF_ENV_PARAM(SAVE_INPUT_TO_FILE, "0");
DEF_ENV_PARAM(SAVE_ERROR_OUTPUT_TO_FILE, "0");
DEF_ENV_PARAM(COPY_INPUT, "1");
DEF_ENV_PARAM(COPY_OUTPUT, "1");
DEF_ENV_PARAM(ENABLE_MEMCMP, "0");
DEF_ENV_PARAM(ENABLE_SHUFFLE, "1");

using namespace std;

class MyPerformanceTestRunner : public vitis::ai::PerformanceTestRunner {
 public:
  explicit MyPerformanceTestRunner(const xir::Subgraph* subgraph);
  explicit MyPerformanceTestRunner(const xir::Subgraph* subgraph,
                                   const vector<string>& input_filenames,
                                   const vector<string>& output_filenames);
  virtual ~MyPerformanceTestRunner();
  MyPerformanceTestRunner(const PerformanceTestRunner& other) = delete;
  MyPerformanceTestRunner& operator=(const PerformanceTestRunner& rhs) = delete;

 public:
  virtual void step(size_t idx, int thread_id) override;
  void generate_output(uint32_t runner_num);
  void shuffle_batch();
  virtual size_t get_result() override;
  vart::Runner* get_runner() { return runner_.get(); };

 private:
  std::unique_ptr<xir::Attrs> attrs_;
  unique_ptr<vart::RunnerExt> runner_;
  vector<vector<vector<char>>> inputs_;
  vector<vector<vector<char>>> ref_outputs_;
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

// ret[input_idx][batch_idx * size_per_batch + idx]
static vector<vector<char>> allocate_buffer(
    vector<const xir::Tensor*> tensors) {
  auto ret = vector<vector<char>>(tensors.size());
  for (auto i = 0u; i < ret.size(); ++i) {
    auto size_of_input = tensors[i]->get_data_size();
    ret[i] = random_vector_char(size_of_input, tensors[i]->get_shape()[0]);
  }
  return ret;
}

// ret[N][input_idx][batch_idx * size_per_batch + idx]
static vector<vector<vector<char>>> generate_inputs(vart::Runner* runner) {
  auto input_tensors = runner->get_input_tensors();
  auto sz = (size_t)ENV_PARAM(NUM_OF_REF);
  auto ret = vector<vector<vector<char>>>(sz);
  for (auto i = 0u; i < sz; ++i) {
    ret[i] = allocate_buffer(input_tensors);
  }
  return ret;
}

static vector<vector<vector<char>>> fill_tensors(
    vector<const xir::Tensor*> tensors, const vector<string>& input_filenames) {
  auto tensors_size = tensors.size();
  auto sz = (size_t)ENV_PARAM(NUM_OF_REF);
  auto ret = vector<vector<vector<char>>>(sz);
  uint num_of_batch = tensors[0]->get_shape()[0];
  for (auto i = 0u; i < sz; ++i) {
    ret[i].resize(tensors_size);
    for (auto b = 0u; b < num_of_batch; ++b) {
      for (auto j = 0u; j < tensors_size; ++j) {
        auto element_num = tensors[j]->get_data_size();
        auto each_file_size = element_num / num_of_batch;
        ret[i][j].resize(element_num);
        auto& filename = input_filenames[(i * num_of_batch * tensors_size +
                                          b * tensors_size + j) %
                                         input_filenames.size()];
        auto flag = std::ifstream(filename)
                        .read((char*)ret[i][j].data() + each_file_size * b,
                              each_file_size)
                        .good();

        LOG_IF(INFO, !flag) << "fail to read! filename=" << filename;
      }
    }
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

inline void run(vart::Runner* runner,
                const vector<vector<vector<char>>>& inputs, size_t idx,
                vector<vector<char>>& output_buffers) {
  auto r = dynamic_cast<vart::RunnerExt*>(runner);
  if (r == nullptr) {
     return;
  }
  auto dpu_inputs = r->get_inputs();
  auto dpu_outputs = r->get_outputs();
  if (ENV_PARAM(COPY_INPUT)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "copying input...";
    copy_inputs(inputs[idx], dpu_inputs);
  }

  for (auto input : dpu_inputs) {
    input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
  }
  auto v = runner->execute_async(dpu_inputs, dpu_outputs);
  runner->wait((int)v.first, -1);
  for (auto output : dpu_outputs) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }
  if (ENV_PARAM(COPY_INPUT) && ENV_PARAM(COPY_OUTPUT)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "copying output...";
    copy_outputs(output_buffers, dpu_outputs);
  }
}
// inputs[N][input_index][batch_index*size_per_batch + data_index] all batches
// share the same vector<char>.
//
// return[N][output_index][batch_index*size_per_batch + data_index]
//
inline vector<vector<vector<char>>> generate_outputs(
    vart::Runner* runner, const vector<vector<vector<char>>>& inputs,
    uint32_t runner_num) {
  auto output_tensors = runner->get_output_tensors();
  auto sz = inputs.size();
  auto num_of_batch = output_tensors[0]->get_shape()[0];
  auto ret = vector<vector<vector<char>>>(sz);
  for (auto i = 0u; i < sz; ++i) {
    ret[i] = allocate_buffer(output_tensors);
  }

  for (auto i = 0u; i < sz; ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "generating ref " << i << endl;
    run(runner, inputs, i, ret[i]);
  }
  if (ENV_PARAM(SAVE_INPUT_TO_FILE)) {
    auto input_batch = inputs[0][0].size() / num_of_batch;
    auto output_batch = ret[0][0].size() / num_of_batch;
    write_tensors(input_batch, inputs,
                  string("ref_input_thread_") + to_string(runner_num));
    write_tensors(output_batch, ret,
                  string("ref_ouput_thread_") + to_string(runner_num));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "references are generated" << endl;
  return ret;
}

static std::vector<size_t> random_indices(size_t num_of_results) {
  auto indices = std::vector<size_t>(num_of_results);
  auto values = std::vector<size_t>(num_of_results);
  std::iota(indices.begin(), indices.end(), 0u);
  static mt19937 rng(100);
  static uniform_int_distribution<char> dist;
  for (auto i = 0u; i < num_of_results; ++i) {
    values[i] = dist(rng);
  }
  std::sort(indices.begin(), indices.end(),
            [values](size_t a, size_t b) { return values[a] < values[b]; });
  return indices;
}

static vector<vector<vector<char>>> clone_buffers(
    const vector<vector<vector<char>>>& in) {
  auto ret = vector<vector<vector<char>>>(in.size());
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = vector<vector<char>>(in[i].size());
    for (auto j = 0u; j < ret[i].size(); ++j) {
      ret[i][j] = in[i][j];
    }
  }
  return ret;
}

static void copy_ref_batch(const vector<vector<vector<char>>>& from,
                           vector<vector<vector<char>>>& to, size_t ref_idx,
                           size_t input_idx, size_t batch_idx,
                           size_t random_idx, size_t num_of_batch) {
  auto size_per_batch = from[ref_idx][input_idx].size() / num_of_batch;
  auto src = &from[ref_idx][input_idx][batch_idx * size_per_batch];
  auto new_ref_idx = random_idx / num_of_batch;
  auto new_batch_idx = random_idx % num_of_batch;
  auto dst = &to[new_ref_idx][input_idx][new_batch_idx * size_per_batch];
  memcpy(dst, src, size_per_batch);
  return;
}

void MyPerformanceTestRunner::shuffle_batch() {
  auto num_of_ref = inputs_.size();
  auto output_tensors = runner_->get_output_tensors();
  auto num_of_batchs = (size_t)output_tensors[0]->get_shape()[0];
  CHECK_EQ(num_of_ref, ref_outputs_.size());
  auto num_of_results = num_of_ref * num_of_batchs;
  auto indices = random_indices(num_of_results);
  auto origin_inputs = clone_buffers(inputs_);
  auto origin_outputs = clone_buffers(ref_outputs_);
  auto result_idx = 0u;
  for (auto ref_idx = 0u; ref_idx < num_of_ref; ++ref_idx) {
    for (auto batch_idx = 0u; batch_idx < num_of_batchs; ++batch_idx) {
      for (auto input_idx = 0u; input_idx < inputs_[ref_idx].size();
           ++input_idx) {
        // inputs[N][input_index][batch_index*size_per_batch + data_index] all
        // batches
        copy_ref_batch(origin_inputs, inputs_, ref_idx, input_idx, batch_idx,
                       indices[result_idx], num_of_batchs);
      }
      for (auto output_idx = 0u; output_idx < ref_outputs_[ref_idx].size();
           ++output_idx) {
        // outputs[N][output_index][batch_index*size_per_batch + data_index] all
        // batches
        copy_ref_batch(origin_outputs, ref_outputs_, ref_idx, output_idx,
                       batch_idx, indices[result_idx], num_of_batchs);
      }
      result_idx = result_idx + 1;
    }
  }
}

static std::unique_ptr<vart::RunnerExt> create_runner(
    const xir::Subgraph* subgraph, xir::Attrs* attrs) {
  auto ret = std::unique_ptr<vart::RunnerExt>();
  if (subgraph->is_root()) {
    ret = vitis::ai::GraphRunner::create_graph_runner(subgraph->get_graph(),
                                                      attrs);
  } else {
    ret = vart::RunnerExt::create_runner(subgraph, attrs);
  }
  return ret;
}

MyPerformanceTestRunner::MyPerformanceTestRunner(const xir::Subgraph* subgraph)
    : attrs_{xir::Attrs::create()},
      runner_{create_runner(subgraph, attrs_.get())},
      inputs_{generate_inputs(runner_.get())},
      // ref_outputs_{generate_outputs(runner_.get(), inputs_)},
      output_buffers_{allocate_buffer(runner_->get_output_tensors())} {}

void MyPerformanceTestRunner::generate_output(uint32_t runner_num) {
  ref_outputs_ = generate_outputs(runner_.get(), inputs_, runner_num);
}

MyPerformanceTestRunner::MyPerformanceTestRunner(
    const xir::Subgraph* subgraph, const vector<string>& input_filenames,
    const vector<string>& output_filenames)
    : attrs_{xir::Attrs::create()},
      runner_{create_runner(subgraph, attrs_.get())},
      inputs_{fill_tensors(runner_->get_input_tensors(), input_filenames)},
      ref_outputs_{
          fill_tensors(runner_->get_output_tensors(), output_filenames)},
      output_buffers_{allocate_buffer(runner_->get_output_tensors())} {}

std::atomic<u_int64_t> errors_total = 0;
MyPerformanceTestRunner::~MyPerformanceTestRunner() {
  errors_total += error_counter_;
  LOG_IF(INFO, ENV_PARAM(LOG_ERROR_COUNTER) && error_counter_)
      << "error_counter = " << error_counter_
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
  run(get_runner(), inputs_, idx, output_buffers_);
  if (ENV_PARAM(COPY_INPUT) && ENV_PARAM(COPY_OUTPUT) &&
      ENV_PARAM(ENABLE_MEMCMP)) {
    for (auto i = 0u; i < output_buffers_.size(); ++i) {
      auto mem_size = output_buffers_[i].size();
      auto num_of_batch = output_tensors[i]->get_shape()[0];
      auto size_per_batch = mem_size / num_of_batch;
      CHECK_EQ(mem_size, ref_outputs_[idx][i].size());
      auto ok = true;
      LOG_IF(INFO, false) << " mem_size " << mem_size << " num_of_batch "
                          << num_of_batch << " size_per_batch "
                          << size_per_batch;
      for (auto batch_idx = 0; batch_idx < num_of_batch; ++batch_idx) {
        auto r = memcmp(&output_buffers_[i][batch_idx * size_per_batch],
                        &ref_outputs_[idx][i][batch_idx * size_per_batch],
                        size_per_batch);
        ok = ok && r == 0;
        LOG_IF(INFO, ENV_PARAM(LOG_ERROR_COUNTER) && r != 0)
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
  result_ = result_ + runner_->get_input_tensors()[0]->get_shape()[0];
  return;
}

size_t MyPerformanceTestRunner::get_result() { return result_; }

bool test_dpu_runner_mt(const xir::Subgraph* subgraph, uint32_t runner_num,
                        const vector<string>& input_filenames,
                        const vector<string>& output_filenames) {
  pybind11::gil_scoped_release release;
  CHECK_GT(runner_num, 0);
  {
    auto runners = vector<unique_ptr<vitis::ai::PerformanceTestRunner>>();
    for (auto i = 0u; i < runner_num; ++i) {
      if (input_filenames.empty() || output_filenames.empty()) {
        runners.emplace_back(
            move(make_unique<MyPerformanceTestRunner>(subgraph)));
      } else {
        runners.emplace_back(make_unique<MyPerformanceTestRunner>(
            subgraph, input_filenames, output_filenames));
      }
    }
    if (input_filenames.empty() || output_filenames.empty()) {
      for (auto i = 0u; i < runner_num; ++i) {
        dynamic_cast<MyPerformanceTestRunner*>(runners[i].get())
            ->generate_output(i);
      }
    }
    if (ENV_PARAM(ENABLE_SHUFFLE)) {
      LOG(INFO) << "shuffle results for batch...";
      for (auto i = 0u; i < runner_num; ++i) {
        dynamic_cast<MyPerformanceTestRunner*>(runners[i].get())
            ->shuffle_batch();
      }
    } else {
      LOG(WARNING) << "shuffling result is disabled, some error might not be "
                      "triggerred.";
    }
    // not use
    int argc = 0;
    char* argv[] = {};
    make_unique<vitis::ai::PerformanceTest>()->main(argc, argv, move(runners));
  }

  pybind11::gil_scoped_acquire acquire;
  return errors_total == 0;
}
