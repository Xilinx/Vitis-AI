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

#include <iostream>
using namespace std;
#include <chrono>
#include <thread>
#include <cstring>
#include <getopt.h>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include "xir/tensor/tensor.hpp"

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/thread_pool.hpp"

#include "common.h"

#define MAX_THREADS 16
#define NUM_THREADS 2


void run(std::vector<std::unique_ptr<vart::Runner>>* runners, int batch_size,
         const std::string input_bin, const std::string bench_bin,
         int num_sequences, int input_seq_dim, int output_seq_dim,
         int thread_idx, int loop_for) {
  std::unique_ptr<vart::Runner> runner = std::move(runners->at(thread_idx));
  int input_32seq_dim = runner->get_input_tensors()[0]->get_shape().at(2);
  int output_32seq_dim = runner->get_output_tensors()[0]->get_shape().at(2);
  LOG(INFO) << "Creating input/output tensors...";
  auto current_input_tensor = xir::Tensor::create("iv",
      {batch_size, num_sequences, input_32seq_dim}, xir::DataType{xir::DataType::XINT, 16});
  auto current_output_tensor = xir::Tensor::create("ov",
      {batch_size, num_sequences, output_32seq_dim}, xir::DataType{xir::DataType::XINT, 16});

  size_t input_size = batch_size*num_sequences*input_32seq_dim;
  size_t output_size = batch_size*num_sequences*output_32seq_dim;
  char* vector_data_, *bench_data_;
  size_t vector_size_, bench_size_;
  std::tie(vector_data_, vector_size_) = read_binary_file(input_bin);
  std::tie(bench_data_, bench_size_) = read_binary_file(bench_bin);

  std::vector<int16_t> input_vector(input_size, 0);
  int batch_len = num_sequences * input_seq_dim;
  int output_batch_len = num_sequences * output_seq_dim;
  for(int b=0; b<batch_size; b++){
    memcpy(input_vector.data()+b*batch_len,
           vector_data_, batch_len*sizeof(int16_t));
  }

  std::vector<int16_t> output_vector;
  output_vector.reserve(output_size);

  auto input_tb = std::make_unique<CpuFlatTensorBuffer>(
      (void*)(input_vector.data()), std::move(current_input_tensor.get()));
  auto output_tb = std::make_unique<CpuFlatTensorBuffer>(
      (void*)(output_vector.data()), std::move(current_output_tensor.get()));

  LOG(INFO) << "Creating input/output tensorBuffers...";
  std::vector<vart::TensorBuffer*> inputsPtr { input_tb.get() };
  std::vector<vart::TensorBuffer*> outputsPtr { output_tb.get() };;

  auto loop_start = chrono::steady_clock::now();
  for (auto loop_idx=0; loop_idx<loop_for; loop_idx++){
    runner->execute_async(inputsPtr, outputsPtr);
    for(int b=0; b<batch_size; ++b) {
      auto cmpret = std::memcmp((char*)bench_data_,
                                output_vector.data()+b*output_batch_len,
                                output_batch_len * sizeof(int16_t));
      if (cmpret != 0)
        LOG(ERROR) << "[ERROR] Thread: " << thread_idx << " | Batch: " << loop_idx
                  << " | Sentence: " << b << " | compare result: " << cmpret;
    }
  }
  auto loop_end = chrono::steady_clock::now();
  auto time_taken = chrono::duration_cast<chrono::nanoseconds>(loop_end - loop_start).count()/1000000.0f;
  LOG(INFO) << "Thread(" << thread_idx << ") execute_async took [ms]: " << time_taken
    << " | Avg. time per batch [ms]: " << time_taken/loop_for;
}

std::unique_ptr<xir::Graph> create_graph(
            const std::string device_name,
            const std::string model_dir, const std::string model_name,
            const int num_sequences, const int input_seq_dim,
            const int output_seq_dim) {
  /*
  model_dir:
      Should contain all the files specific to the model (Output from compiler)
  model_name:
      Should be one of [sentiment, satisfaction, openie]. To be used by xrnnController
  num_sequences:
      Maximum number of tokens with which the model is trained
  input_dim:
      Dimension of the vector of a token
  output_dim:
      Dimension of the vector returned by the model after processing a token
  */
  std::unique_ptr<xir::Graph> fakegraph = xir::Graph::create("lstm");
  auto rs = fakegraph->get_root_subgraph();

  std::map<std::string, std::string> subg_attr = {{"run", "libvart-rnn-runner.so"}};
  rs->set_attr<std::map<std::string, std::string>> ("runner", subg_attr);
  rs->set_attr<std::string>("device_name", device_name);
  rs->set_attr<int>("device_core_id", 0);
  rs->set_attr<std::string>("model_dir", model_dir);
  rs->set_attr<std::string>("model_name", model_name);
  rs->set_attr<int>("num_sequences", num_sequences);
  rs->set_attr<int>("input_seq_dim", input_seq_dim);
  rs->set_attr<int>("output_seq_dim", output_seq_dim);
  return fakegraph;
}


int main(int argc, char* argv[]) {
  CHECK(argc > 6) << "please input the correct parameters in console: "
    << "./test_rnn_runner <device_name> <model_dir> <model_name> "
    << "                  <test_dir> <num_sequences> <num_threads> <num_batches>\n";
  std::string device_name = argv[1];
  std::string model_dir = argv[2];
  std::string model_name = argv[3];
  std::string test_dir = argv[4];
  int num_sequences = atoi(argv[5]);
  int num_threads = atoi(argv[6]);
  int num_batches = atoi(argv[7]);

  CHECK(num_threads <= MAX_THREADS) << "num_threads should be less than " << MAX_THREADS;
  if (num_threads == 0)
    num_threads = NUM_THREADS;
  LOG(INFO) <<
    "device_name: " << device_name <<
    " | model_dir: " << model_dir <<
    " | model_name: " << model_name <<
    " | test_dir: " << test_dir << "\n";

  int batch_size = 1;
  std::string input_bin = test_dir + "/vector_bin";
  std::string bench_bin = test_dir + "/bench_bin";
  int input_seq_dim = 0;
  int output_seq_dim = 0;

  if(model_name == "sentiment") {
    input_seq_dim = 32;
    output_seq_dim = 128;
  }

  else if(model_name == "satisfaction") {
    input_seq_dim = 32;
    output_seq_dim = 128;
  }

  else if(model_name == "openie") {
    input_seq_dim = 224;
    output_seq_dim = 320;
  }

  std::vector<std::unique_ptr<vart::Runner>> runners(num_threads);

  for(int i=0; i<num_threads; i++){
    std::unique_ptr<xir::Graph> graph = create_graph(
      device_name, model_dir, model_name, num_sequences,
      input_seq_dim, output_seq_dim);
    auto rs = graph->get_root_subgraph();

    LOG(INFO) << "Creating Runner(" << i << ")";
    runners[i] = vart::Runner::create_runner(rs, "run");
    LOG(INFO) << "Done creating Runner(" << i << ")";
  }


  auto start = chrono::steady_clock::now();
  array<thread, MAX_THREADS>threads_list;
  for (int i=0; i<num_threads ; i++){
    threads_list[i] = thread(run, &runners, batch_size, input_bin, bench_bin, num_sequences,
        input_seq_dim, output_seq_dim, i, num_batches);
  }
  for (int i=0; i<num_threads; i++){
    threads_list[i].join();
  }
  auto end = chrono::steady_clock::now();

  LOG(INFO) << "Time taken by all the threads [ms]: "
    << chrono::duration_cast<chrono::nanoseconds>(end - start).count()/1000000.0f;

  return 0;

}
