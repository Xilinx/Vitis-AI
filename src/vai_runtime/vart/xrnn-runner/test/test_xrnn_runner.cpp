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
#include <getopt.h>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/thread_pool.hpp"

#include "common.h"
#include "model_param.hpp"

DEF_ENV_PARAM_2(MODEL_PATH, "/scratch/yili/vart/xrnn-runner/", std::string);

int main(int argc, char* argv[]) {
  cout << "what?" << endl;
  CHECK(argc > 3) << "please input the correct parameters in console"
    << "./test_xrnn_runner 0 100 1 \n"
    <<  "0 for model type, 100 for frame numbers, 1 for repeat times";
  auto type = atoi(argv[1]);
  auto frame_str = std::string(argv[2]);
  auto frames=atoi(frame_str.c_str());
  auto times = atoi(argv[3]);
  
  CHECK(type >= 0 && type < 3 ) << "wrong model type";
  CHECK(frames != 0) << "wrong frame number";
  CHECK(times != 0) << "wrong test times";

  auto model_dir = std::string("");
  if (!ENV_PARAM(MODEL_PATH).empty()) {
    model_dir = ENV_PARAM(MODEL_PATH);
  }
  else{
    LOG(INFO) << "please provide the path to the modle !!!";
    return 0;
  }

  ModelParameters* mp_ptr = nullptr;
  if(type == 0){
    LOG(INFO) << "test sentiment, frame is ignored, use 500 instead";
    mp_ptr = new ModelParameters(model_dir, "sent", "");
    mp_ptr->set(1, 500, 32, 128);
  }
  else if(type==1){
    LOG(INFO) << "test satisfaction, frame is ignored, use 25 instead";
    mp_ptr = new ModelParameters(model_dir, "satis", "");
    mp_ptr->set(1, 25, 32, 128);
  }
  else if(type==2){
    LOG(INFO) << "Test Openie, frame is " << frames
      << ", test data is in " << "/"+frame_str;
    
    mp_ptr = new ModelParameters(model_dir, "oie", "/"+frame_str);
    mp_ptr->set(1, frames, 224, 320);
  }
  
  std::shared_ptr<xir::Graph> fakegraph = xir::Graph::create("lstm");
  auto rs = fakegraph->get_root_subgraph();

  //must keep like this
  std::map<std::string, std::string> subg_attr={{"run", "libvart-xrnn-runner.so"}};
  rs->set_attr<std::map<std::string, std::string>> ("runner", subg_attr);
  rs->set_attr<std::string>("device", "xrnn");
  // custom define
  rs->set_attr<unsigned>("device_core_id", 0);   
  rs->set_attr<std::string>("xclbin",  mp_ptr->get_xclbin_file());
  rs->set_attr<std::string>("model_type",  mp_ptr->get_model_type());
  rs->set_attr<std::string>("model_init",  mp_ptr->get_ddr_file());
  rs->set_attr<std::string>("model_path",  mp_ptr->get_model_root());

  auto runner = vart::Runner::create_runner(rs, "run");

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

  auto input_tensor  = std::unique_ptr<xir::Tensor>(xir::Tensor::create("iv", \
     mp_ptr->get_input_dims(), xir::DataType{"INT8"}));
  auto output_tensor = std::unique_ptr<xir::Tensor>(xir::Tensor::create("ov", \
     mp_ptr->get_output_dims(), xir::DataType{"INT8"}));

  char * input_file_data = 0u;
  size_t input_file_size = 0u;
  std::tie(input_file_data, input_file_size) = read_binary_file(mp_ptr->get_vector_file());

  auto batch = input_tensor->get_shape().at(0);

  size_t input_size = input_tensor->get_data_size();
  size_t batch_len = mp_ptr->get_batch_len(mp_ptr->get_input_dims());
  char * Vector = new char[input_size];

  LOG(INFO) << "batch " << batch 
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

  char * bench_file_data = 0u;
  size_t bench_file_size = 0u;
  std::tie(bench_file_data, bench_file_size) = read_binary_file(mp_ptr->get_bench_file());

  batch_len = mp_ptr->get_batch_len(mp_ptr->get_output_dims());

  LOG(INFO) << "bench length " << bench_file_size
    << " - outputs length " << output_size;
   // << " - outputs length " << output_tensor_size;
  auto start=chrono::high_resolution_clock::now();  
  for (auto i=0; i<times; i++){
    runner->execute_async(inputsPtr, outputsPtr);

    LOG(INFO)  << " times " << i ;
    for(int i=0; i<batch; i++){
    //auto cmpret = std::memcmp((char*) bench_file_data, bench_file_data, output_tensor_size);
      auto cmpret = std::memcmp((char*) bench_file_data, Result+i*batch_len, batch_len);
      LOG(INFO)  << "batch " << i << " compare result " << cmpret;
      if(cmpret !=0){
        std::ofstream stream("./result_bin");
        stream.write(Result, output_size);
        break;
      }
    }
  }
  auto end=chrono::high_resolution_clock::now();  
  LOG(INFO) << "chrono::duration [ns]: " 
    << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
    << "  avg: " 
    << chrono::duration_cast<chrono::nanoseconds>(end - start).count()/times << " ns";

  delete [] input_file_data;
  delete [] bench_file_data;
  delete [] Vector;
  delete [] Result;

  LOG(INFO) << "DONE";

  return 0;

}
