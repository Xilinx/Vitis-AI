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

#include <thread>
#include <xrt.h>

#include "vart/runner.hpp"
#include "vitis/ai/env_config.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/tensor/tensor.hpp"
#include "xir/buffer_object.hpp"


#include "xrnn_controller.hpp"

DEF_ENV_PARAM(XRNN_RUNNER_MAX_THREADS, "16");
DEF_ENV_PARAM(XRNN_RUNNER_MAX_CUS, "32");
DEF_ENV_PARAM(XRNN_RUNNER_MAX_MONITOR, "16");
DEF_ENV_PARAM(DEBUG_XRNN_RUNNER, "0")

#define MAX_CUS 32

namespace {
class xrnnRunner : public vart::Runner {
 public:
  explicit xrnnRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  explicit xrnnRunner(const xir::Subgraph* subgraph, const std::string& mode);

  xrnnRunner(const xrnnRunner& other) = delete;

  virtual ~xrnnRunner();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;

 private:
  std::pair<char*, size_t>  read_binary_file(const std::string &file_name);
  void init_with_subgraph_attr(const xir::Subgraph* subgraph);

 private:
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;

  std::unique_ptr<vart::xrnn::XrnnController> xrnn_;

  std::string xclbin_;
  std::string device_;
  std::string model_;
  std::string init_;
  std::string path_;
  unsigned device_core_id_;

  size_t index_;
  int cu_parallel_; 

  char *config_ptr_;
  size_t config_size_;
  ModelConfig *model_config_;

  static size_t count_ [MAX_CUS];
  static bool initialized_[MAX_CUS];
  static std::mutex mtx_;
  static std::mutex cu_mtx_[MAX_CUS];
};

size_t xrnnRunner::count_[] = {0};
bool xrnnRunner::initialized_[] = {false};
std::mutex xrnnRunner::mtx_;
std::mutex xrnnRunner::cu_mtx_[MAX_CUS];


std::pair<char*, size_t>  xrnnRunner::read_binary_file(const std::string &file_name)
{
  CHECK(file_name.empty()==0);

  std::ifstream stream(file_name.c_str());
  stream.seekg(0, stream.end);
  size_t size = stream.tellg();
  stream.seekg(0, stream.beg);
  
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
    << file_name<< ", size " << size;
  
  char *file_ptr = new char[size];
  stream.read(file_ptr, size);

  return std::make_pair(file_ptr, size) ;
}

void xrnnRunner::init_with_subgraph_attr(const xir::Subgraph* subgraph)
{
  xclbin_ = subgraph->get_attr<std::string>("xclbin");
  device_ = subgraph->get_attr<std::string>("device");
  model_ = subgraph->get_attr<std::string>("model_type");
  init_ = subgraph->get_attr<std::string>("model_init");
  path_ = subgraph->get_attr<std::string>("model_path");
  device_core_id_ = subgraph->get_attr<unsigned>("device_core_id");
  
  CHECK(device_core_id_ < MAX_CUS) << "Too Much CUs: " << device_core_id_;

  xrnn_ = std::make_unique<vart::xrnn::XrnnController> (device_core_id_, model_, 
                                std::make_unique<xir::XrtCu>(device_));
 
  index_ = count_[device_core_id_]++;
  cu_parallel_ = xrnn_->get_board_name() == "u50"?1:16;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
    << cu_parallel_ << " support in a single cu";
   
  auto batch = xrnn_->get_batch_size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
    << batch << " batch support in the cu";
  inputs_.reserve(1);
  auto x = xir::Tensor::create("iv", {batch, -1, -1, -1}, xir::DataType{"INT8"});
  inputs_.emplace_back(std::move(x));
  
  outputs_.reserve(1);
  auto y = xir::Tensor::create("ov", {batch, -1, -1, -1}, xir::DataType{"INT8"});
  outputs_.emplace_back(std::move(y));
  
//  for(int i=0; i<MAX_CUS; i++){
//    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
//      << "is_init " << initialized_[i];
//  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "ddr init file" << init_;

  if(cu_parallel_ != 1)
    mtx_.lock();

  if (!initialized_[device_core_id_]){
    initialized_[device_core_id_] = true;
    size_t ddr_size;
    char* ddr_file;
    
    std::tie(ddr_file, ddr_size) = read_binary_file(init_);
    xrnn_->init(ddr_file, ddr_size);
    delete [] ddr_file;
  }

  if (xrnn_->get_board_name() == "u50"){// && model_ == "openie"){
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "update file" << path_+ "is_" + std::to_string(batch);
    std::tie(config_ptr_, config_size_) = read_binary_file(path_+ "is_" + std::to_string(batch));
    model_config_ = new ModelConfig(path_);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "update " << "ptr @" << (void *)config_ptr_ << " size " << config_size_;
  }
  else{
    config_ptr_ = nullptr;
    config_size_ = 0;
    model_config_ = nullptr;
  }

  if(cu_parallel_ != 1)
   mtx_.unlock();
}

xrnnRunner::xrnnRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : inputs_{}, outputs_{} {
  
  init_with_subgraph_attr(subgraph);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "@" << (void*)this << " xrnn runner " << index_ << " is created ";
}

xrnnRunner::xrnnRunner(const xir::Subgraph* subgraph, const std::string& mode)
    : inputs_{}, outputs_{} {
  
  init_with_subgraph_attr(subgraph);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "@" << (void*)this << " xrnn runner " << index_ << " is created ";
}

xrnnRunner::~xrnnRunner() {
  if (xrnn_->get_board_name() == "u50" && model_ == "openie"){
    delete config_ptr_;
  }
}

std::pair<uint32_t, int> xrnnRunner::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
/*  
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
    << " input shape: " << input[0]->get_tensor()->get_shape().at(0) 
    << " " << input[0]->get_tensor()->get_shape().at(1)
    << " " << input[0]->get_tensor()->get_shape().at(2)
    << " " << input[0]->get_tensor()->get_shape().at(3) ;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
    << "output shape: " << output[0]->get_tensor()->get_shape().at(0)
    << " " << output[0]->get_tensor()->get_shape().at(1)
    << " " << output[0]->get_tensor()->get_shape().at(2)
    << " " << output[0]->get_tensor()->get_shape().at(3);
*/
  auto batch = input[0]->get_tensor()->get_shape().at(0);
  auto frames = input[0]->get_tensor()->get_shape().at(1);

  uint64_t input_addr = 0u;
  uint64_t last_addr = 0u;
  size_t input_size = 0;
  std::tie(input_addr, input_size) = input[0]->data(std::vector({0,0,0,0}));
  std::tie(last_addr, input_size) = input[0]->data(std::vector({batch-1,0,0,0}));

  uint64_t output_addr = 0u;
  size_t output_size = 0;
  std::tie(output_addr, output_size) = output[0]->data(std::vector({0,0,0,0}));
  std::tie(last_addr, output_size) = output[0]->data(std::vector({batch-1,0,0,0}));

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << index_ << "(map to " << index_%cu_parallel_  << ") is runing ";

  if (xrnn_->get_board_name() == "u50"){ // && model_ == "openie"){
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << "update " << "ptr @" << (void *) config_ptr_ << " size " << config_size_;
    xrnn_->update(frames, model_config_, (uint32_t *)config_ptr_, config_size_);
  }
  
  unsigned imutex;
  if( cu_parallel_ == 1 )
    imutex = device_core_id_;
  else
    imutex = index_%cu_parallel_;

  std::mutex &m = cu_mtx_[imutex];
  m.lock();

  xrnn_->run((char*)input_addr, input_size, 
    (char*)output_addr, output_size,
    batch, frames, index_%cu_parallel_);

  m.unlock();

  return std::make_pair(index_, 0);
}

int xrnnRunner::wait(int jobid, int timeout) { return 0; }

static std::vector<const xir::Tensor*> copy(
    std::vector<std::unique_ptr<xir::Tensor>>& from) {
  auto ret = std::vector<const xir::Tensor*>();
  ret.reserve(from.size());
  for (auto& b : from) {
    ret.push_back(const_cast<const xir::Tensor*>(b.get()));
  }
  return ret;
}

std::vector<const xir::Tensor*> xrnnRunner::get_input_tensors() {
  return copy(inputs_);
}

std::vector<const xir::Tensor*> xrnnRunner::get_output_tensors() {
  return copy(outputs_);
}

}  // namespace
extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
	                                          xir::Attrs* attrs) {
  return new xrnnRunner(subgraph, attrs);
}

extern "C" vart::Runner* create_runner(const xir::Subgraph* subgraph,
				       const std::string& mode){
  return new xrnnRunner(subgraph, mode);
}


