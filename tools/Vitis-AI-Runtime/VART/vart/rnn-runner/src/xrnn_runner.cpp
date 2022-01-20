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

#include <xrt.h>

#include <sstream>
#include <thread>
#include <vart/runner.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/buffer_object.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>

#include "rnn_xmodel_parser.hpp"
#include "xrnn_controller.hpp"

DEF_ENV_PARAM(XRNN_RUNNER_MAX_THREADS, "16");
DEF_ENV_PARAM(XRNN_RUNNER_MAX_CUS, "32");
DEF_ENV_PARAM(XRNN_RUNNER_MAX_MONITOR, "16");
DEF_ENV_PARAM(DEBUG_XRNN_RUNNER, "0")

#define MAX_CUS 32

template <typename T>
static std::string vector_to_str(const std::vector<T>& v,
                                 std::string delim = ", ") {
  if (v.empty()) {
    return {"[]"};
  }
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << delim;
  }
  ss << v.back() << "]";
  auto s = ss.str();
  return s;
}

// [B, #F, L] --> [<=B, -1, L]
template <typename T>
static std::string expected_shape_to_str(const std::vector<T>& v,
                                         std::string delim = ", ") {
  std::ostringstream ss;
  ss << "[";
  ss << "<=" << v[0] << delim;
  ss << "-1" << delim;
  ss << v[2] << "]";
  auto s = ss.str();
  return s;
}

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
  std::pair<char*, size_t> read_binary_file(const std::string& file_name);
  void init_with_subgraph_attr(const xir::Subgraph* subgraph);

 private:
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;

  std::unique_ptr<vart::xrnn::XrnnController> xrnn_;

  unsigned device_core_id_;
  int batch_size_;
  int num_sequences_;

  size_t index_;
  int cu_parallel_;
  std::string init_;
  std::string path_;

  char* config_ptr_;
  size_t config_size_;
  std::unique_ptr<ModelConfig> model_config_;
  std::unique_ptr<vart::xrnn::RNNModelParser> model_parser_;

  static std::string device_;
  static size_t count_[MAX_CUS];
  static bool initialized_[MAX_CUS];
  static std::mutex mtx_;
  static std::mutex cu_mtx_[MAX_CUS];
};

std::string xrnnRunner::device_ =
    ENV_PARAM_XRNN_KERNEL_NAME::get_default_value();  // "xrnn";
size_t xrnnRunner::count_[] = {0};
bool xrnnRunner::initialized_[] = {false};
std::mutex xrnnRunner::mtx_;
std::mutex xrnnRunner::cu_mtx_[MAX_CUS];

// std::pair<char*, size_t> xrnnRunner::read_binary_file(
//     const std::string& file_name) {
//   CHECK(file_name.empty() == 0);
//
//   std::ifstream stream(file_name.c_str());
//   stream.seekg(0, stream.end);
//   size_t size = stream.tellg();
//   stream.seekg(0, stream.beg);
//
//   LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER)) << file_name << ", size " <<
//   size;
//
//   char* file_ptr = new char[size];
//   stream.read(file_ptr, size);
//
//   return std::make_pair(file_ptr, size);
// }

void xrnnRunner::init_with_subgraph_attr(const xir::Subgraph* subgraph) {
  DLOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER)) << __DATE__ << " : " << __TIME__;

  model_parser_ = std::make_unique<vart::xrnn::RnnXmodelParser>(subgraph);
  std::string device_name = model_parser_->get_target_device();
  batch_size_ = model_parser_->get_batch_size();

  device_core_id_ = 0;
  if (device_name == "U50") {
    if (batch_size_ == 3)
      device_core_id_ = 0;
    else
      device_core_id_ = 1;
  }

  num_sequences_ = -1;
  int input_seq_dim = model_parser_->get_model_input_seq_dim(false) / 2;
  int output_seq_dim = model_parser_->get_model_output_seq_dim(false) / 2;

  CHECK(device_core_id_ < MAX_CUS) << "Too Many CUs: " << device_core_id_;
  xrnn_ = vart::xrnn::RnnControllerCreate::Create(device_name, device_core_id_,
                                                  device_);
  index_ = count_[device_core_id_]++;

  cu_parallel_ = device_name == "U50" ? 1 : 16;
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << cu_parallel_ << " support in a single cu";

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << batch_size_ << " batch support in the cu";

  // Buffer dim : [BatchSize, #frames, frame_len], dtype=int16_t
  inputs_.reserve(1);
  int input_32seq_dim = ((input_seq_dim - 1) | 31) + 1;
  auto x =
      xir::Tensor::create("iv", {batch_size_, num_sequences_, input_32seq_dim},
                          xir::DataType{xir::DataType::XINT, 16});
  inputs_.emplace_back(std::move(x));

  outputs_.reserve(1);
  int output_32seq_dim = ((output_seq_dim - 1) | 31) + 1;
  auto y =
      xir::Tensor::create("ov", {batch_size_, num_sequences_, output_32seq_dim},
                          xir::DataType{xir::DataType::XINT, 16});
  outputs_.emplace_back(std::move(y));

  for (int i = 0; i < MAX_CUS; i++) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
        << "is_init: " << initialized_[i];
  }

  if (cu_parallel_ != 1) mtx_.lock();

  if (!initialized_[device_core_id_]) {
    initialized_[device_core_id_] = true;
    size_t ddr_size = model_parser_->get_weights().size();
    const char* ddr_file = model_parser_->get_weights().data();
    xrnn_->init(const_cast<char*>(ddr_file), ddr_size);
  }

  model_config_ = std::make_unique<ModelConfig>(model_parser_.get());
  xrnn_->set_model_config(model_config_.get());
  std::tie(config_ptr_, config_size_) = model_config_->get_instructions();

  if (cu_parallel_ != 1) mtx_.unlock();
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

xrnnRunner::~xrnnRunner() = default;

std::pair<uint32_t, int> xrnnRunner::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  __TIC__(EXECUTE_ASYNC)
  static int iter = 0;
  // Check if #inputs & #outputs is 1
  // TODO : abidk : Any models with multiple inputs/outputs
  CHECK_EQ(input.size(), 1) << "Expected only one input. Got " << input.size();
  CHECK_EQ(output.size(), 1)
      << "Expected only one output. Got " << output.size();

  // Check if buffer shapes complies with Runner specifications
  // Expected buffer shape : [Batch, #frames, frame_len], dtype=int16_t
  const auto& inbufshape = input[0]->get_tensor()->get_shape();
  const auto& outbufshape = output[0]->get_tensor()->get_shape();
  const auto& inshape = inputs_[0]->get_shape();
  const auto& outshape = outputs_[0]->get_shape();
  CHECK(inbufshape.size() == inshape.size() && inbufshape[0] <= inshape[0] &&
        inbufshape[2] == inshape[2])
      << std::endl
      << "Expected Input Shape : " << expected_shape_to_str(inshape) << "; "
      << "Obtained Input Shape : " << vector_to_str(inbufshape);
  CHECK(outbufshape.size() == inshape.size() && outbufshape[0] <= outshape[0] &&
        outbufshape[2] == outshape[2])
      << std::endl
      << "Expected Output Shape : " << expected_shape_to_str(outshape) << "; "
      << "Obtained Output Shape : " << vector_to_str(outbufshape);
  CHECK(inputs_[0]->get_data_type() == input[0]->get_tensor()->get_data_type());
  CHECK(outputs_[0]->get_data_type() ==
        output[0]->get_tensor()->get_data_type());

  auto batch_size = input[0]->get_tensor()->get_shape().at(0);
  auto num_sequences = input[0]->get_tensor()->get_shape().at(1);

  uint64_t input_addr = 0u;
  uint64_t last_addr = 0u;
  size_t input_size = 0;
  std::tie(input_addr, std::ignore) = input[0]->data(std::vector({0, 0, 0}));
  std::tie(last_addr, input_size) =
      input[0]->data(std::vector({batch_size - 1, 0, 0}));

  uint64_t output_addr = 0u;
  size_t output_size = 0;
  std::tie(output_addr, std::ignore) = output[0]->data(std::vector({0, 0, 0}));
  std::tie(last_addr, output_size) =
      output[0]->data(std::vector({batch_size - 1, 0, 0}));

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_RUNNER))
      << index_ << "(map to " << index_ % cu_parallel_ << ") is runing ";

  unsigned imutex;
  if (cu_parallel_ == 1)
    imutex = device_core_id_;
  else
    imutex = device_core_id_;  // index_ % cu_parallel_;

  std::mutex& m = cu_mtx_[imutex];
  m.lock();

  if (true) {  // (xrnn_->get_board_name() == "u50") {  // && model_ ==
               // "openie"){
    xrnn_->update(num_sequences, model_config_.get(), (uint32_t*)config_ptr_,
                  config_size_);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
        << "update "
        << "ptr @" << (void*)config_ptr_ << " size " << config_size_
        << model_config_->dump_instructions("instr_" + std::to_string(iter++) +
                                            "_bs" + std::to_string(batch_size) +
                                            "_t" + std::to_string(index_) +
                                            ".txt");
  }

  xrnn_->run((char*)input_addr, input_size, (char*)output_addr, output_size,
             batch_size, num_sequences, index_ % cu_parallel_);
  m.unlock();

  __TOC__(EXECUTE_ASYNC)
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
                                       const std::string& mode) {
  return new xrnnRunner(subgraph, mode);
}
