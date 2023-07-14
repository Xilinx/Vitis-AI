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

#include <chrono>

using Clock = std::chrono::steady_clock;
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/performance_test.hpp>
#include <xir/tensor/tensor.hpp>
DEF_ENV_PARAM(DEBUG_TEST, "0");
DEF_ENV_PARAM(NUM_OF_REF, "4");
DEF_ENV_PARAM(THREAD_ADD_LOCK, "0");
DEF_ENV_PARAM(SAME_INPUT, "0");
DEF_ENV_PARAM(SAVE_INPUT_TO_FILE, "0");
DEF_ENV_PARAM(SAVE_ERROR_OUTPUT_TO_FILE, "0");
DEF_ENV_PARAM(COPY_INPUT, "1");
DEF_ENV_PARAM(COPY_OUTPUT, "1");
DEF_ENV_PARAM(ENABLE_MEMCMP, "1");

#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
using namespace std;

class MyPerformanceTestRunner : public vitis::ai::PerformanceTestRunner {
 public:
  explicit MyPerformanceTestRunner(const std::string& filename,  //
                                   const std::string& kernel);
  virtual ~MyPerformanceTestRunner();
  MyPerformanceTestRunner(const PerformanceTestRunner& other) = delete;
  MyPerformanceTestRunner& operator=(const PerformanceTestRunner& rhs) = delete;

 public:
  virtual void step(size_t idx, int thread_id) override;
  virtual size_t get_result() override;
  vart::Runner* get_runner() { return runner_.get(); };

 private:
  std::unique_ptr<vart::Runner> runner_;
  const vector<vector<vector<char>>> inputs_;
  const vector<vector<vector<char>>> ref_outputs_;
  std::vector<std::vector<char>> output_buffers_;
  size_t result_ = 0;
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
    static std::mt19937 rng(100);
    // MSVC NOTE: msvs does not support uniform_int_distribution<char>
    static std::uniform_int_distribution<int> dist;
    for (auto i = 0u; i < sz; ++i) {
      ret[i] = (char)dist(rng);
    }
  }
  return ret;
}

static vector<vector<char>> allocate_buffer(
    std::vector<const xir::Tensor*> tensors) {
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

// static vector<vector<vector<vector<char>>>> generate_inputs(
//     const vector<std::unique_ptr<vitis::ai::Runner>>& runners) {
//   auto sz = runners.size();
//   auto ret = vector<vector<vector<vector<char>>>>(runners.size());
//   for (auto i = 0u; i < sz; ++i) {
//     ret[i] = generate_inputs(runners[i].get());
//   }
//   return ret;
// }

static void copy_input(const vector<char>& data, vart::TensorBuffer* tb) {
  size_t batch_size = tb->get_tensor()->get_shape()[0];
  for (auto i = 0u; i < batch_size; ++i) {
    uint64_t input_data = 0u;
    auto input_size = 0u;
    auto dims = std::vector<int>(tb->get_tensor()->get_shape().size(), 0);
    dims[0] = (int)i;
    std::tie(input_data, input_size) = tb->data(dims);
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
    auto dims = std::vector<int>(tb->get_tensor()->get_shape().size(), 0);
    dims[0] = (int)i;
    std::tie(output_data, output_size) = tb->data(dims);
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

static void write_to_file(const char* buf, size_t size,
                          const std::string& file) {
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  CHECK(std::ofstream(file, mode).write(buf, size).good())
      << " faild to write to " << file;
}
static void write_tensors(size_t batch_size, const vector<char>& tensor_buffers,
                          const std::string& file) {
  for (auto batch = 0u; batch < tensor_buffers.size() / batch_size; ++batch) {
    write_to_file(&tensor_buffers[batch * batch_size], batch_size,
                  file + "_batch_" + std::to_string(batch) + ".bin");
  }
}

static void write_tensors(size_t batch_size,
                          const vector<vector<vector<char>>>& tensor_buffers,
                          const std::string& file) {
  int c = 0;
  for (auto& t : tensor_buffers) {
    write_tensors(batch_size, t[0], file + "_c_" + std::to_string(c++));
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
    write_tensors(input_batch, inputs, std::string("ref_input"));
    write_tensors(output_batch, ret, std::string("ref_ouput"));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "references are generated" << endl;
  return ret;
}

// static vector<vector<vector<vector<char>>>> generate_outputs(
//     vector<std::unique_ptr<vart::Runner>>& runners,
//     const vector<vector<vector<vector<char>>>>& inputs) {
//   CHECK_EQ(runners.size(), inputs.size());
//   auto sz = runners.size();
//   auto ret = vector<vector<vector<vector<char>>>>(sz);
//   for (auto i = 0u; i < sz; ++i) {
//     ret[i] = generate_outputs(runners[i].get(), inputs[i]);
//   }
//   return ret;
// }

// static std::string md5sum(const vector<char>& val) {
//   std::vector<unsigned char> result((size_t)MD5_DIGEST_LENGTH, '0');
//   std::ostringstream str;
//   MD5((const unsigned char*)&val[0], val.size(), (unsigned char*)&result[0]);
//   for (const auto x : result) {
//     str << std::hex << std::setfill('0') << std::setw(2);
//     str << ((unsigned int)x);
//   }
//   return str.str();
// }

// static std::string print_md5(const vector<char>& data) {
//   std::ostringstream str;
//   str << "[" << data.size() << ", " << md5sum(data) << "]";
//   return str.str();
// }

// static std::string print_md5(const vector<vector<char>>& data) {
//   std::ostringstream str;
//   for (auto i = 0u; i < data.size(); ++i) {
//     str << "\ti = " << i << print_md5(data[i]) << "\n";
//   }
//   return str.str();
// }

// static string print_md5(const vector<vector<vector<char>>>& data) {
//   std::ostringstream str;
//   for (auto i = 0u; i < data.size(); ++i) {
//     str << "i = " << i << "\n" << print_md5(data[i]) << endl;
//   }
//   return str.str();
// }

// static void print_md5(const vector<vector<vector<vector<char>>>>& data) {
//   for (auto i = 0u; i < data.size(); ++i) {
//     LOG(INFO) << "i = " << i << "\n" << print_md5(data[i]) << endl;
//   }
//   return;
// }

MyPerformanceTestRunner::MyPerformanceTestRunner(
    const std::string& filename,  //
    const std::string& kernel     //
    )
    : runner_{vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel)},
      inputs_{generate_inputs(runner_.get())},
      ref_outputs_{generate_outputs(runner_.get(), inputs_)},
      output_buffers_{allocate_buffer(runner_->get_output_tensors())} {  //
}
thread_local int error_counter = 0;
thread_local int ok_counter = 0;
int64_t errors_total = 0;
MyPerformanceTestRunner::~MyPerformanceTestRunner() {
  errors_total += error_counter;
  LOG_IF(INFO, error_counter) << "error_counter = " << error_counter
                              << ",errors_total = " << errors_total;
}
void MyPerformanceTestRunner::step(size_t idx, int thread_id) {
  if (ENV_PARAM(THREAD_ADD_LOCK)) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
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
              << " error_counter " << error_counter + 1 << " ok_counter "
              << (ok_counter + 1);
          if (r != 0 && ENV_PARAM(SAVE_ERROR_OUTPUT_TO_FILE)) {
            auto ref_file_name = std::string("ref_t") +
                                 std::to_string(thread_id) + std::string("_") +
                                 std::to_string(idx) + std::string("_batch_") +
                                 std::to_string(batch_idx) + std::string("_") +
                                 std::to_string(error_counter + 1) +
                                 std::string(".bin");
            auto out_file_name = std::string("out_t") +
                                 std::to_string(thread_id) + std::string("_") +
                                 std::to_string(idx) + std::string("_batch_") +
                                 std::to_string(batch_idx) + std::string("_") +
                                 std::to_string(error_counter + 1) +
                                 std::string(".bin");

            write_to_file(&ref_outputs_[idx][i][batch_idx * size_per_batch],
                          size_per_batch, ref_file_name);
            write_to_file(&output_buffers_[i][batch_idx * size_per_batch],
                          size_per_batch, out_file_name);
          }
        }
        if (ok) {
          ok_counter++;
        } else {
          error_counter++;
        }
        LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
            << "checking ok idx =" << idx << ",i=" << i;
      }
    }
  }
  result_ = result_ + runner_->get_input_tensors()[0]->get_shape()[0];
  return;
}

size_t MyPerformanceTestRunner::get_result() { return result_; }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "usage: " << argv[0] << " <xmodel> <subgraph_name> <num_of_threads>"
         << "\n"
         << "env variables:\n"  //
         << "\tCOPY_INPUT=1 : enable copying input\n"
         << "\tCOPY_OUTPUT=1 : enable copying output\n"
         << "\tENABLE_MEMCMP=1 : enable comparing\n"
         << "\tSLEEP_MS=60000 : sleep for 60s before stopping\n"
         << "\tNUM_OF_REF=4 : num of reference results per runner\n"
         << endl;
    return 1;
  }
  auto filename = argv[1];
  auto kernel = argv[2];
  auto runner_num = std::stoi(std::string(argv[3]));
  CHECK_GT(runner_num, 0);
  {
    auto runners = vector<std::unique_ptr<vitis::ai::PerformanceTestRunner>>();
    for (auto i = 0; i < runner_num; ++i) {
      LOG(INFO) << "create runner ... " << i << "/" << runner_num;
      runners.emplace_back(
          std::make_unique<MyPerformanceTestRunner>(filename, kernel));
    }
    std::make_unique<vitis::ai::PerformanceTest>()->main(argc, argv,
                                                         std::move(runners));
  }
  return errors_total == 0 ? 0 : -1;
}
