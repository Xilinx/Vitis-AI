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

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "dpu4rnn_imp.hpp"
#include "vitis/ai/env_config.hpp"
#include <vart/runner.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

DEF_ENV_PARAM(DEBUG_XRNN, "0");

const std::string dpu4rnnImp::xclbin_path_ = "/usr/lib/dpu.xclbin";
const std::vector<std::string> dpu4rnnImp::model_type_ = {"sentiment", "satisfaction", "openie"};

void read_file(std::string filename, std::vector<char>& data) {
  CHECK(filename.empty()==0);

  std::ifstream stream(filename);
  stream.seekg(0, stream.end);
  size_t size = stream.tellg();
  stream.seekg(0, stream.beg);

  data.resize(size);
  stream.read(data.data(), size);
  stream.close();
}

class CpuFlatTensorBuffer : public vart::TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : TensorBuffer{tensor}, data_{data} {}
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
  if (idx.size() == 0) {
    return {reinterpret_cast<uint64_t>(data_), tensor_->get_data_size()};
  }
  auto dims = tensor_->get_shape();
  auto offset = 0;
  for (auto k = 0u; k < dims.size(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < dims.size(); m++) {
      stride *= dims[m];
    }
    offset += idx[k] * stride;
  }

  auto dtype_size = tensor_->get_data_type().bit_width / 8;
  auto elem_num = tensor_->get_element_num();

  return std::make_pair(reinterpret_cast<uint64_t>(data_) + offset * dtype_size,
                        (elem_num - offset) * dtype_size);
  }
 private:
  void* data_;
};

static bool match_str(const std::vector<std::string>& strs, const std::string& s) {
  bool flag = false;
  for(size_t i = 0; i < strs.size(); ++i) {
    if(strs[i].compare(s)==0) {
      flag = true;
      break;
    }
  }
  return flag;
}

dpu4rnnImp::~dpu4rnnImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
        << "Deconstruct" ;
}

dpu4rnnImp::dpu4rnnImp(const std::string model_name, const int device_id)
	:model_name_(model_name), device_id_(device_id) {
  if(!match_str(model_type_, model_name_)) {
    LOG(INFO) << "The model didn't support: " << model_name_;
    exit(0);
  }
  std::string files_path("./model/");
  std::string model_path = files_path + model_name_ + "_" + std::to_string(device_id) + ".bin";
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
        << "The xclbin path: " << xclbin_path_
        << " The model path: " << model_path
	<< " model name: " << model_name_;
    
  read_file(model_path, model_);
  // Allocate a block of memory for store output, it is better to set it a little big.

  fakegraph = xir::Graph::create("lstm");
  rs = fakegraph->get_root_subgraph();
  // Must keep like this
  std::map<std::string, std::string> subg_attr={{"run", "libvart-xrnn-runner.so"}};
  rs->set_attr<std::map<std::string, std::string>> ("runner", subg_attr);
  rs->set_attr<std::string>("device", "xrnn");

  // Custom could define it.
  rs->set_attr<unsigned>("device_core_id", device_id_);
  rs->set_attr<std::string>("xclbin", xclbin_path_);
  rs->set_attr<std::string>("model_type", model_name_);
  rs->set_attr<std::string>("model_init", model_path);
  rs->set_attr<std::string>("model_path", files_path);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
	  << "device_core_id is : " << device_id_
	  << "; xclbin is : "  << xclbin_path_
	  << "; model_type is : " << model_name_
	  << "; model_path is : " << model_path;

  runner_ = vart::Runner::create_runner(rs, "run");
  out_.resize(224*320*100);
  batch_ = runner_->get_input_tensors()[0]->get_dims()[0];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
       << "batch size = " << batch_;
}

// run the lstm part
void dpu4rnnImp::run(const char* input, int in_size, char* output, int frame_num, int batch) {
  if (model_name_.compare("sentiment")==0) {
    this->set(batch_, frame_num, 32, 128);
  } else if (model_name_.compare("satisfaction")==0) {
    this->set(batch_, frame_num, 32, 128);
  } else if (model_name_.compare("openie")==0) {
    this->set(batch_, frame_num, 224, 320);
  } else {
    LOG(INFO) << "This model doesn't exist! " << model_name_;
    exit(0);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
            << "input info: " << get_input_dims()[0]
            << " " << get_input_dims()[1]
            << " " << get_input_dims()[2]
            << " " << get_input_dims()[3];
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
            << "output info: " << get_output_dims()[0]
            << " " << get_output_dims()[1]
            << " " << get_output_dims()[2]
            << " " << get_output_dims()[3];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

  auto input_tensor = std::unique_ptr<xir::Tensor>(
         xir::Tensor::create("iv", get_input_dims(), xir::DataType(xir::DataType::Type::INT, 8)));
  auto output_tensor = std::unique_ptr<xir::Tensor>(
         xir::Tensor::create("ov", get_output_dims(), xir::DataType(xir::DataType::Type::INT, 8)));
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN))
        << "tensor info: " << input_tensor->get_shape().at(0)
        << " " << input_tensor->get_data_size();;

  // create the input and output tensor-buffer for the dpu run.
  inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
	  (void*)input, input_tensor.get()));
  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
	  (void*)(out_.data()), output_tensor.get()));
  inputsPtr.push_back(inputs[0].get());
  outputsPtr.push_back(outputs[0].get());
  runner_->execute_async(inputsPtr, outputsPtr);

  if (model_name_.compare("sentiment")==0) {
    for(int b = 0; b < batch; ++b) {
      int osize_per_batch = 100 * sizeof(int16_t);
      memcpy((char*)output + b*osize_per_batch,
           // we only need the last 128 numbers' first 100 numbers per batch.
           (char*)out_.data() + (b*500+499)*128*sizeof(int16_t),
           osize_per_batch);
    }
  } else if (model_name_.compare("satisfaction")==0) {
    for(int b = 0; b < batch; ++b)
      for(int i = 0; i < frame_num; ++i) {
        int osize_per_batch = 100 * sizeof(int16_t);
        memcpy((char*)output + (b*frame_num + i)*osize_per_batch,
               // This model's result we only need the first 100 numbers of per frame
               (char*)out_.data() + (b*frame_num+i)*128*sizeof(int16_t),
               osize_per_batch);
      }

  } else if (model_name_.compare("openie")==0) {
    for(int b = 0; b < batch; ++b)
      for(int i = 0; i < frame_num; ++i) {
        memcpy((char*)output + (b*frame_num+i)*300*sizeof(int16_t),
               (char*)out_.data() + (b*frame_num + frame_num -i -1)*320*sizeof(int16_t),
               300*sizeof(int16_t));
      }

  } else {
    LOG(INFO) << "This model doesn't exist! " << model_name_;
  }
}

int dpu4rnnImp::getBatch() {
  return batch_;
}
