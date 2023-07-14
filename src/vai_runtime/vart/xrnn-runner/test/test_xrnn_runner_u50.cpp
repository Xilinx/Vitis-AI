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


#include <iostream>
using namespace std;
#include <chrono>
#include <thread>
#include <cstring>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/thread_pool.hpp"
#include "runner_helper.hpp"

#include "common.h"
#include "model_param.hpp"

#define MAX_THREADS 16
#define CORES 2
DEF_ENV_PARAM(DEBUG, "0")
DEF_ENV_PARAM(NUM_OF_CORES, "2");
DEF_ENV_PARAM(NUM_OF_REQUESTS, "2");
DEF_ENV_PARAM_2(MODEL_PATH, "/scratch/yili/vart/xrnn-runner/", std::string);

void run(std::vector<std::unique_ptr<vart::Runner>>* runners, 
         std::vector<ModelParameters*>* vmp, 
         std::vector<ModelTestData*>* vtd, 
         int times, int index)
{
  std::unique_ptr<vart::Runner> runner = std::move(runners->at(index));
  ModelParameters* mp = vmp->at(index%CORES); 
  ModelTestData* td = vtd->at(index%CORES); 

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

  auto input_tensor  = std::unique_ptr<xir::Tensor>(xir::Tensor::create("iv", \
     mp->get_input_dims(), xir::DataType{"INT8"}));
  auto output_tensor = std::unique_ptr<xir::Tensor>(xir::Tensor::create("ov", \
     mp->get_output_dims(), xir::DataType{"INT8"}));

  char * input_file_data = td->get_data("vector");
  //size_t input_file_size = td.get_size("vector");

  auto batch = input_tensor->get_shape().at(0);

  size_t input_size = input_tensor->get_data_size();
  size_t batch_len = mp->get_batch_len(mp->get_input_dims());
  char * Vector = new char[input_size];

  LOG_IF(INFO, ENV_PARAM(DEBUG)) << "batch " << batch 
    << " input size " << input_size
    << " batch len" << batch_len;
  
  for(int i=0; i<batch; i++){
    memcpy(Vector+i*batch_len, input_file_data, batch_len); 
  }
  
  inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
    (void*)Vector, input_tensor.get()));

  size_t output_size = output_tensor->get_data_size();
  char * Result = new char[output_size];

  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
    (void*)Result, output_tensor.get()));

  inputsPtr.push_back(inputs[0].get());
  outputsPtr.push_back(outputs[0].get());

  char * bench_file_data = td->get_data("bench");
  size_t bench_file_size = td->get_size("bench");

  batch_len = mp->get_batch_len(mp->get_output_dims());

  LOG_IF(INFO, ENV_PARAM(DEBUG))
    << "bench length " << bench_file_size
    << " - outputs length " << output_size;

  auto loop_start=chrono::high_resolution_clock::now();  
  for (auto i=0; i<times; i++){
    runner->execute_async(inputsPtr, outputsPtr);
    LOG_IF(INFO, ENV_PARAM(DEBUG)) << " times " << i ;
    for(int i=0; i<batch; i++){
    //auto cmpret = std::memcmp((char*) bench_file_data, bench_file_data, output_tensor_size);
      auto cmpret = std::memcmp((char*) bench_file_data, Result+i*batch_len, batch_len);
      LOG_IF(INFO, ENV_PARAM(DEBUG))  
        << "thread " << index << " batch " << i << " compare result " << cmpret;
      if(cmpret !=0){
        LOG(INFO) << "Result Wrong";
        std::ofstream stream("./result_bin");
        stream.write(Result, output_size);
      }
    }
  }
  auto loop_end=chrono::high_resolution_clock::now();  
  LOG(INFO) << "thread: " << index << " " 
    << chrono::duration_cast<chrono::nanoseconds>(loop_end - loop_start).count() << " ns"
    << "  avg: " 
    << chrono::duration_cast<chrono::nanoseconds>(loop_end - loop_start).count()/times << " ns";

  delete [] Vector;
  delete [] Result;
}

int main(int argc, char* argv[]) {
  CHECK(argc > 3) << "please input the correct parameters in console"
    << "./test_xrnn_runner 0 100 1 \n"
    <<  "0 for model type, 100 for frame numbers, 1 for repeat times";

  auto type = atoi(argv[1]);
  auto frame_str = std::string(argv[2]);
  auto frames=atoi(frame_str.c_str());
  auto thread_num = atoi(argv[3]);
  
  CHECK(type >= 0 && type < 3 ) << "wrong model type";
  CHECK(frames != 0) << "wrong frame number";
  CHECK(thread_num <= MAX_THREADS) << "wrong thread number";

  auto model_dir = std::string("");
  if (!ENV_PARAM(MODEL_PATH).empty()) {
    model_dir = ENV_PARAM(MODEL_PATH);
  }  
  else{
    LOG(INFO) << "please provide the path to the modle !!!";
    return 0;
  }
  
  if (thread_num == 0){
    thread_num = ENV_PARAM(NUM_OF_REQUESTS);
  }

  //ModelParameters* model_params[CORES];
  //ModelTestData* model_test_data[CORES];
  std::vector<ModelParameters*> model_params(CORES);
  std::vector<ModelTestData*> model_test_data(CORES);

  if(type == 0){
    LOG(INFO) << "test sentiment, frame is ignored, use 500 instead";
    model_params[0] = new ModelParameters(model_dir, "sent", ""); 
    model_params[0]->update_vector_name("vector_bin_batch3");
    model_params[0]->update_bench_name("bench_bin_batch3");
    model_params[0]->update_ddr_name("ddr_bin_3k");
    model_params[0]->set(3, 500, 32, 128);
    model_test_data[0] = new ModelTestData(model_params[0]);

    model_params[1] = new ModelParameters(model_dir, "sent", ""); 
    model_params[1]->update_vector_name("vector_bin_batch4");
    model_params[1]->update_bench_name("bench_bin_batch4");
    model_params[1]->update_ddr_name("ddr_bin_4k");
    model_params[1]->set(4, 500, 32, 128);
    model_test_data[1] = new ModelTestData(model_params[1]);
  }
  else if(type==1){
    LOG(INFO) << "test satisfaction, frame is ignored, use 25 instead";

    model_params[0] = new ModelParameters(model_dir, "satis", "");
    model_params[0]->update_vector_name("vector_bin_batch3");
    model_params[0]->update_bench_name("bench_bin_batch3");
    model_params[0]->update_ddr_name("ddr_bin_3k");
    model_params[0]->set(3, 25, 32, 128);
    model_test_data[0] = new ModelTestData(model_params[0]);

    model_params[1] = new ModelParameters(model_dir, "satis", "");
    model_params[1]->update_vector_name("vector_bin_batch4");
    model_params[1]->update_bench_name("bench_bin_batch4");
    model_params[1]->update_ddr_name("ddr_bin_4k");
    model_params[1]->set(4, 25, 32, 128);
    model_test_data[1] = new ModelTestData(model_params[1]);
  }
  else if(type==2){
    LOG(INFO) << "Test Openie, frame is " << frames
      << ", test data is in " << "/"+frame_str;
    model_params[0] = new ModelParameters(model_dir, "oie", "/"+frame_str);
    model_params[0]->update_vector_name("vector_bin_batch3");
    model_params[0]->update_bench_name("bench_bin_batch3");
    model_params[0]->update_ddr_name("ddr_bin_3k");
    model_params[0]->set(3, frames, 224, 320);
    model_test_data[0] = new ModelTestData(model_params[0]);

    model_params[1] = new ModelParameters(model_dir, "oie", "/"+frame_str);
    model_params[1]->update_vector_name("vector_bin_batch4");
    model_params[1]->update_bench_name("bench_bin_batch4");
    model_params[1]->update_ddr_name("ddr_bin_4k");
    model_params[1]->set(4, frames, 224, 320);
    model_test_data[1] = new ModelTestData(model_params[1]);
  }
  
  std::vector<std::unique_ptr<vart::Runner>> runners(thread_num);
  std::vector<std::shared_ptr<xir::Graph>> fakegraphs(thread_num);

  for(int i=0; i<thread_num; i++){
    fakegraphs[i] = xir::Graph::create("lstm");
    auto rs = fakegraphs[i]->get_root_subgraph();

    //must keep like this
    std::map<std::string, std::string> subg_attr={{"run", "libvart-xrnn-runner.so"}};
    rs->set_attr<std::map<std::string, std::string>> ("runner", subg_attr);
    rs->set_attr<std::string>("device", "xrnn");
    // custom define
    rs->set_attr<unsigned>("device_core_id", i%CORES);   
    rs->set_attr<std::string>("xclbin",  model_params[i%CORES]->get_xclbin_file());
    rs->set_attr<std::string>("model_type",  model_params[i%CORES]->get_model_type());
    rs->set_attr<std::string>("model_init",  model_params[i%CORES]->get_ddr_file());
    rs->set_attr<std::string>("model_path",  model_params[i%CORES]->get_model_root());
    runners[i] = vart::Runner::create_runner(rs, "run");
  }

  auto start=chrono::high_resolution_clock::now();  
  array<thread, MAX_THREADS>threads_list;
  for (int i=0; i<thread_num ; i++){
    threads_list[i] = thread(run, &runners, &model_params, &model_test_data, 1, i);
  }
  for (int i=0; i<thread_num; i++){
    threads_list[i].join();
  }
  auto end=chrono::high_resolution_clock::now();  
  
  LOG(INFO) << "chrono::duration [ns]: " 
    << chrono::duration_cast<chrono::nanoseconds>(end - start).count();

  for(int i=0; i<CORES; i++){
    delete model_params[i];
    delete model_test_data[i];
  }

  return 0;
}
