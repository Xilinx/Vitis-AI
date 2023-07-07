/*
 * Copyright 2021 Xilinx Inc.
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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#include <vart/runner.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/thread_pool.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include "common.h"

using namespace std;

#define MAX_THREADS 16
#define NUM_THREADS 1
#define CORES 1

void run(std::unique_ptr<vart::Runner> runner, const std::string test_dir,
         int num_sequences, int thread_idx, int loop_for) {
  auto prologue = chrono::steady_clock::now();
  int batch_size = runner->get_input_tensors()[0]->get_shape().at(0);

  int aligned_input_seq_dim = runner->get_input_tensors()[0]->get_shape().at(2);
  int aligned_output_seq_dim =
      runner->get_output_tensors()[0]->get_shape().at(2);

  auto vector_bin = test_dir + "/vector_bin";
  auto bench_bin = test_dir + "/bench_bin";

  auto curr_input_tensor = xir::Tensor::create(
      "iv", {batch_size, num_sequences, aligned_input_seq_dim},
      xir::DataType{xir::DataType::XINT, 16});
  auto curr_output_tensor = xir::Tensor::create(
      "ov", {batch_size, num_sequences, aligned_output_seq_dim},
      xir::DataType{xir::DataType::XINT, 16});

  size_t input_size = curr_input_tensor->get_element_num();
  size_t output_size = curr_output_tensor->get_element_num();

  char *vector_data_, *bench_data_;
  size_t vector_size_, bench_size_;
  std::tie(vector_data_, vector_size_) = read_binary_file(vector_bin);
  std::tie(bench_data_, bench_size_) = read_binary_file(bench_bin);

  std::vector<int16_t> input_vector(input_size);
  int output_batch_len = num_sequences * aligned_output_seq_dim;
  int batch_len = num_sequences * aligned_input_seq_dim;
  for (int b = 0; b < batch_size; b++) {
    memcpy(input_vector.data() + b * batch_len, vector_data_,
           batch_len * sizeof(int16_t));
  }

  std::vector<int16_t> output_vector(output_size, 0);

  auto input_tb = std::make_unique<CpuFlatTensorBuffer>(
      (void*)(input_vector.data()), std::move(curr_input_tensor.get()));
  auto output_tb = std::make_unique<CpuFlatTensorBuffer>(
      (void*)(output_vector.data()), std::move(curr_output_tensor.get()));

  std::vector<vart::TensorBuffer*> inputsPtr{input_tb.get()};
  std::vector<vart::TensorBuffer*> outputsPtr{output_tb.get()};

  auto loop_start = chrono::steady_clock::now();
  for (auto loop_idx = 0; loop_idx < loop_for; loop_idx++) {
    runner->execute_async(inputsPtr, outputsPtr);
    for (int b = 0; b < batch_size; ++b) {
      auto cmpret = std::memcmp((char*)bench_data_,
                                output_vector.data() + b * output_batch_len,
                                output_batch_len * sizeof(int16_t));
      if (cmpret != 0) {
        LOG(ERROR) << "[ERROR] Thread: " << thread_idx
                   << " | Batch: " << loop_idx << " | Sentence: " << b
                   << " | compare result: " << cmpret;
        int outsum = std::accumulate(
            output_vector.begin() + b * output_batch_len,
            output_vector.begin() + (b + 1) * output_batch_len, 0);
        int16_t* bd = reinterpret_cast<int16_t*>(bench_data_);
        int refsum = std::accumulate(bd, bd + output_batch_len, 0);
        LOG(INFO) << "Thread: " << thread_idx << " | Batch: " << loop_idx
                  << " | Sentence: " << b << " | Ref Sum: " << refsum
                  << " | Out Sum: " << outsum;
      }
    }
  }
  auto loop_end = chrono::steady_clock::now();
  auto time_taken =
      chrono::duration<float, std::milli>{loop_end - loop_start}.count();
  LOG(INFO)
      << "Thread(" << thread_idx << ") execute_async took [ms]: " << time_taken
      << " | Avg. time per batch [ms]: " << time_taken / loop_for
      << " | Init Over-head [ms]: "
      << chrono::duration<float, std::milli>{loop_start - prologue}.count();
}

int main(int argc, char* argv[]) {
  CHECK(argc > 5)
      << "please input the correct parameters in console: "
      << "./test_rnn_runner <xmodel_dir> <test_dir> <num_sequences> "
      << "                  <num_threads> <num_batches>\n";
  std::string xmodel_dir = argv[1];
  std::string test_dir = argv[2];
  int num_sequences = atoi(argv[3]);
  int num_threads = atoi(argv[4]);
  int num_batches = atoi(argv[5]);

  CHECK(num_threads <= MAX_THREADS)
      << "num_threads should be less than " << MAX_THREADS;
  if (num_threads == 0) num_threads = NUM_THREADS;
  LOG(INFO) << "\n  xmodel_dir: " << xmodel_dir << "\n  test_dir: " << test_dir
            << "\n  num_sequences: " << num_sequences
            << "\n  num_threads: " << num_threads
            << "\n  num_batches: " << num_batches;

  std::vector<std::string> xmodel_files;
  xmodel_files.push_back(xmodel_dir + "/compiled_batch_1.xmodel");

  std::vector<std::unique_ptr<vart::Runner>> runners;
  runners.reserve(num_threads);

  for (int i = 0; i < num_threads; i++) {
    std::string xmodel_file = xmodel_files.at(i % CORES);
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto rs = graph->get_root_subgraph();

    LOG(INFO) << "Creating Runner(" << i << ") with " << xmodel_file;
    runners.push_back(vart::Runner::create_runner(rs, "run"));
  }

  auto start = chrono::steady_clock::now();
  array<thread, MAX_THREADS> threads_list;
  for (int i = 0; i < num_threads; i++) {
    threads_list[i] = thread(run, std::move(runners.at(i)), test_dir,
                             num_sequences, i, num_batches);
  }
  for (int i = 0; i < num_threads; i++) {
    threads_list[i].join();
  }
  auto end = chrono::steady_clock::now();

  auto time_taken = chrono::duration<float, std::milli>{end - start}.count();
  LOG(INFO) << "Time taken by all the threads [ms]: " << time_taken;
  LOG(INFO) << "Throughput (batch/sec) : "
            << num_threads * num_batches * 1000.0f / time_taken;
  return 0;
}
