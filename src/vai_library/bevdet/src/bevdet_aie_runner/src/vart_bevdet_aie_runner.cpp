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

#include <math.h>

#include <filesystem>
#include <vitis/ai/path_util.hpp>
#include <vitis/ai/profiling.hpp>

#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/tensor/tensor.hpp"
DEF_ENV_PARAM(DEBUG_DUMMY_RUNNER, "0")
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0")
DEF_ENV_PARAM(DEBUG_AIE, "0")

#include "param.h"
#include "vai_aie_task_handler.hpp"

using namespace std;
namespace {

bool is_exist_path(const std::string& filename) {
  return std::filesystem::exists(filename);
}
static std::string get_parent_path(const std::string& path) {
  return path.substr(
      0, path.find_last_of(std::filesystem::path::preferred_separator));
}
static void mkdir_minus_p(const std::string& dirname) {
  CHECK(std::filesystem::create_directories(dirname))
      << "cannot create directories: " << dirname;
}
static void create_parent_path(const std::string& path) {
  if (is_exist_path(path)) {
    return;
  }
  auto parent_path = get_parent_path(path);
  if (!is_exist_path(parent_path)) {
    create_parent_path(parent_path);
  }
  mkdir_minus_p(path);
}

static std::string to_valid_file_name(const std::string& filename) {
  const std::string pat = "/():[]{}\\?%*|\"'><;=";
  std::ostringstream str;
  for (auto c : filename) {
    if (pat.find(c) != std::string::npos) {
      str << "_";  // << std::hex << (int)c << "_";
    } else {
      str << c;
    }
  }
  return str.str();
};
static void dump_tensor_buffer_(const std::string& dir0,
                                vart::TensorBuffer* tensor_buffer) {
  auto maybe_remove_trail_slah = [](const std::string& s) {
    if (s.back() == '/') {
      return s.substr(0, s.size() - 1);
    }
    return s;
  };
  std::string dir = maybe_remove_trail_slah(dir0);
  create_parent_path(dir);
  CHECK(is_exist_path(dir)) << "cannot create directory: dir=" << dir;
  auto tensor_name = tensor_buffer->get_tensor()->get_name();
  auto tensor_name_remove_fix = xir::remove_xfix(tensor_name);
  auto filename0 = to_valid_file_name(tensor_name_remove_fix);
  dir += "/" + filename0;
  std::vector<int> idx(tensor_buffer->get_tensor()->get_shape().size(), 0);
  uint64_t data = tensor_buffer->data(idx).first;
  auto filename = dir + ".bin";
  CHECK(std::ofstream(filename)
            .write((char*)data, tensor_buffer->get_tensor()->get_data_size())
            .good())
      << "failed to write: " << filename;

  return;
}
static void maybe_dump_tensor_buffers(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  if (!ENV_PARAM(XLNX_ENABLE_DUMP)) {
    return;
  }
  std::string dir = "dump/bevdet_aie/";
  for (auto&& i : input) {
    auto dirname = dir + "i";
    dump_tensor_buffer_(dirname, i);
  }
  for (auto&& i : output) {
    auto dirname = dir + "o";
    dump_tensor_buffer_(dirname, i);
  }
}

class BEVdetAIERunner : public vart::RunnerExt {
 public:
  explicit BEVdetAIERunner(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  BEVdetAIERunner(const BEVdetAIERunner& other) = delete;

  virtual ~BEVdetAIERunner();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 private:
  std::vector<std::string> input_names;
  std::string output_name;
  int batch;
  std::vector<std::vector<int32_t>> input_shapes;
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;
  std::vector<size_t> data_lens;
  size_t dataout_len;
  vai_aie_task_handler* graph_handler;
  xrtGraphHandle g_bevdet_ls0;
  xrtGraphHandle g_bevdet_ls1;
  xrtGraphHandle g_bevdet_ls2;

  std::vector<xrtBufferHandle> data_buffer;

  xrtBufferHandle out_buff;
};

BEVdetAIERunner::BEVdetAIERunner(const xir::Subgraph* subgraph,
                                 xir::Attrs* attrs)
    : input_names{"bevdetls0graph.datain[0]", "bevdetls0graph.datain[1]",
                  "bevdetls1graph.datain[0]", "bevdetls1graph.datain[1]",
                  "bevdetls1graph.datain[2]", "bevdetls1graph.datain[3]"},
      output_name{"bevdettransgraph.dataout[0]"},
      batch{1},
      input_shapes{{batch, 6, 704, 64},  {batch, 6, 704, 80},
                   {batch, 6, 36608, 3}, {batch, 6, 128, 3},
                   {batch, 6, 36608, 3}, {batch, 6, 16, 3}},
      inputs_{},
      outputs_{},
      data_lens{} {
  // VART Part
  for (size_t i = 0; i < input_names.size(); i++) {
    inputs_.emplace_back(xir::Tensor::create(input_names[i], input_shapes[i],
                                             xir::DataType{"INT8"}));
    data_lens.push_back(inputs_[i]->get_data_size());
  }

  outputs_.emplace_back(xir::Tensor::create(output_name, {2, 128, 128, 40},
                                            xir::DataType{"INT8"}));
  dataout_len = outputs_[0]->get_data_size();

  // AIE Part
  auto xclbin = attrs->get_attr<const char*>("xclbin");
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMMY_RUNNER)) << "xclbin:" << xclbin;

  graph_handler = new vai_aie_task_handler(xclbin);
  g_bevdet_ls0 =
      xrtGraphOpen(graph_handler->dhdl, graph_handler->uuid, "bevdetls0graph");
  g_bevdet_ls1 =
      xrtGraphOpen(graph_handler->dhdl, graph_handler->uuid, "bevdetls1graph");
  g_bevdet_ls2 = xrtGraphOpen(graph_handler->dhdl, graph_handler->uuid,
                              "bevdettransgraph");

  float exp_lut[256] = {0.0};
  for (int i = 0; i < 256; i++) {
    const int offset = -128;
    const float scale = 0.5;
    exp_lut[i] = expf((i + offset) * scale);
  }
  xrtGraphUpdateRTP(g_bevdet_ls0, "bevdetls0graph.bevdetls0kernel[0].in[1]",
                    (char*)(exp_lut), 256 * sizeof(float));
  for (auto i : data_lens) {
    if (batch == 1) i *= 2;
    data_buffer.push_back(
        xrtBOAlloc(graph_handler->dhdl, i, XCL_BO_FLAGS_CACHEABLE, 0));
  }
  out_buff = xrtBOAlloc(graph_handler->dhdl, dataout_len, 0, 0);
  LOG(INFO) << "create BEVdetAIERunner end";
}

BEVdetAIERunner::~BEVdetAIERunner() {
  for (auto&& i : data_buffer) {
    xrtBOFree(i);
  }
  xrtBOFree(out_buff);
  delete graph_handler;
}

std::pair<uint32_t, int> BEVdetAIERunner::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMMY_RUNNER))
      << "@" << (void*)this << " start to run: "
      << " inputs= " << to_string(input) << " "  //
      << " outputs= " << to_string(output) << " ";

  __TIC__(BEVdetAIERunner_execute_async)
  xrtGraphRun(g_bevdet_ls0, ITER0);
  xrtGraphRun(g_bevdet_ls1, ITER1);
  xrtGraphRun(g_bevdet_ls2, ITER2);
  __TIC__(BEVdetAIERunner_copy_input)
  for (size_t i = 0; i < data_buffer.size(); i++) {
    auto src = (void*)input[i]->data({0, 0, 0, 0}).first;
    auto dst = (char*)xrtBOMap(data_buffer[i]);
    memcpy(dst, src, data_lens[i]);
    if (batch == 1) {  // batch == 1 , Augmented data
      memcpy(dst + data_lens[i], src, data_lens[i]);
    }
  }
  __TOC__(BEVdetAIERunner_copy_input)
  __TIC__(BEVdetAIERunner_run)
  for (size_t i = 0; i < data_buffer.size(); i++) {
    if (batch == 1) {
      xrtBOSync(data_buffer[i], XCL_BO_SYNC_BO_TO_DEVICE, data_lens[i] * 2, 0);
      xrtSyncBOAIENB(graph_handler->dhdl, data_buffer[i],
                     input_names[i].c_str(), XCL_BO_SYNC_BO_GMIO_TO_AIE,
                     data_lens[i] * 2, 0);
    } else {
      xrtBOSync(data_buffer[i], XCL_BO_SYNC_BO_TO_DEVICE, data_lens[i], 0);
      xrtSyncBOAIENB(graph_handler->dhdl, data_buffer[i],
                     input_names[i].c_str(), XCL_BO_SYNC_BO_GMIO_TO_AIE,
                     data_lens[i], 0);
    }
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_AIE))
      << "xrtSyncBOAIENB XCL_BO_SYNC_BO_GMIO_TO_AIE end";
  xrtSyncBOAIENB(graph_handler->dhdl, out_buff, output_name.c_str(),
                 XCL_BO_SYNC_BO_AIE_TO_GMIO, dataout_len, 0);
  LOG_IF(INFO, ENV_PARAM(DEBUG_AIE))
      << "xrtSyncBOAIENB XCL_BO_SYNC_BO_AIE_TO_GMIO end";
  xrtGMIOWait(graph_handler->dhdl, output_name.c_str());
  LOG_IF(INFO, ENV_PARAM(DEBUG_AIE)) << "xrtGMIOWait end";
  xrtBOSync(out_buff, XCL_BO_SYNC_BO_FROM_DEVICE, dataout_len, 0);
  __TOC__(BEVdetAIERunner_run)
  __TIC__(BEVdetAIERunner_copy_output)
  memcpy((void*)output[0]->data({0, 0, 0, 0}).first, (char*)xrtBOMap(out_buff),
         dataout_len);
  __TOC__(BEVdetAIERunner_copy_output)
  __TOC__(BEVdetAIERunner_execute_async)
  maybe_dump_tensor_buffers(input, output);
  xrtGraphClose(g_bevdet_ls0);
  xrtGraphClose(g_bevdet_ls1);
  xrtGraphClose(g_bevdet_ls2);
  return std::make_pair(0u, 0);
}

int BEVdetAIERunner::wait(int jobid, int timeout) { return 0; }

static std::vector<const xir::Tensor*> copy(
    std::vector<std::unique_ptr<xir::Tensor>>& from) {
  auto ret = std::vector<const xir::Tensor*>();
  ret.reserve(from.size());
  for (auto& b : from) {
    ret.push_back(const_cast<const xir::Tensor*>(b.get()));
  }
  return ret;
}
std::vector<const xir::Tensor*> BEVdetAIERunner::get_input_tensors() {
  return copy(inputs_);
}
std::vector<const xir::Tensor*> BEVdetAIERunner::get_output_tensors() {
  return copy(outputs_);
}
std::vector<vart::TensorBuffer*> BEVdetAIERunner::get_inputs() { return {}; }

std::vector<vart::TensorBuffer*> BEVdetAIERunner::get_outputs() { return {}; }

}  // namespace
extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return new BEVdetAIERunner(subgraph, attrs);
}
