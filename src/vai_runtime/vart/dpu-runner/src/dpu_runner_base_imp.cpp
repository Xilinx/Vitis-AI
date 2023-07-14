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
#include "dpu_runner_base_imp.hpp"

#include <sys/stat.h>

#include <UniLog/UniLog.hpp>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>  // std::numeric_limits
#include <vitis/ai/dim_calc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/trace.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/util/tool_function.hpp>

#include "../../runner/src/runner_helper.hpp"
#include "./my_openssl_md5.hpp"
#include "dpu_kernel.hpp"
#include "my_tensor.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(XLNX_ENABLE_UPLOAD, "0");
DEF_ENV_PARAM(XLNX_ENABLE_CLEAR, "0");
DEF_ENV_PARAM(XLNX_SHOW_DPU_COUNTER, "0");
DEF_ENV_PARAM_2(XLNX_GOLDEN_DIR, "", std::string);
DEF_ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK, "1");
DEF_ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN, "0");

static bool xlnx_enable_compare_mode() {
  return !ENV_PARAM(XLNX_GOLDEN_DIR).empty();
}

static bool xlnx_enable_debug_dpu_data_mode() {
  return ENV_PARAM(XLNX_ENABLE_DUMP) || ENV_PARAM(XLNX_ENABLE_UPLOAD) ||
         ENV_PARAM(XLNX_ENABLE_CLEAR) || !ENV_PARAM(XLNX_GOLDEN_DIR).empty();
}
namespace vart {
namespace dpu {

DpuRunnerBaseImp::DpuRunnerBaseImp(  // TODO clear input_tensors
                                     // output_tensors my_input_tensors
                                     // my_output_tensors my_all_tensors
    const std::vector<const xir::Tensor*> input_tensors,
    const std::vector<const xir::Tensor*> output_tensors,
    DpuSessionBaseImp* session)
    : vart::Runner(),  //
      input_tensors_{input_tensors},
      output_tensors_{output_tensors},
      session_{session} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "create  dpu runner " << (void*)this                         //
      << " device_core_id " << session_->get_device_core_id() << " "  //
      ;
}  //

DpuRunnerBaseImp::~DpuRunnerBaseImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "destroying dpu runner " << (void*)this                      //
      << " device_core_id " << session_->get_device_core_id() << " "  //
      ;
}
std::vector<const xir::Tensor*> DpuRunnerBaseImp::get_input_tensors() {
  return input_tensors_;
}
std::vector<const xir::Tensor*> DpuRunnerBaseImp::get_output_tensors() {
  return output_tensors_;
}

std::vector<const xir::Tensor*> DpuRunnerBaseImp::get_input_tensor(
    const xir::Subgraph* subgraph) {
  auto xir_input_tensors = subgraph->get_sorted_input_tensors();
  auto input_tensors = std::vector<const xir::Tensor*>();
  for (auto tensor : xir_input_tensors) {
    input_tensors.push_back(find_tensor(tensor->get_name()).get_tensor());
  }
  return input_tensors;
}

std::vector<const xir::Tensor*> DpuRunnerBaseImp::get_output_tensor(
    const xir::Subgraph* subgraph) {
  auto xir_output_tensors = subgraph->get_sorted_output_tensors();
  auto output_tensors = std::vector<const xir::Tensor*>();
  for (auto tensor : xir_output_tensors) {
    output_tensors.push_back(find_tensor(tensor->get_name()).get_tensor());
  }
  return output_tensors;
}

static int get_reg_index(const std::string& reg_id) {
  UNI_LOG_CHECK(reg_id.size() >= 5 &&    //
                    reg_id[0] == 'R' &&  //
                    reg_id[1] == 'E' &&  //
                    reg_id[2] == 'G' &&  //
                    reg_id[3] == '_' &&  //
                    reg_id[4] >= '0' && reg_id[4] <= '9',
                VART_DPU_INFO_ERROR)
      << "reg id is not support! reg_id = " << reg_id;
  return reg_id[4] - '0';
}

static std::vector<uint64_t> build_gen_reg(
    const std::map<std::string, uint64_t>& pp, size_t num_of_batch,
    size_t NUM_OF_REGS) {
  // key: "REG_0", "REG_1", or "REG_2" etc
  // for pp
  auto ret = std::vector<uint64_t>(num_of_batch * NUM_OF_REGS,
                                   std::numeric_limits<uint64_t>::max());
  for (const auto& reg_value : pp) {
    auto idx = get_reg_index(reg_value.first);
    auto value = reg_value.second;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "build_gen_reg idx " << idx << " "              //
        << "reg_value.first " << reg_value.first << " "    //
        << "reg_value.second " << reg_value.second << " "  //
        ;
    for (auto i = 0u; i < num_of_batch; ++i) {
      ret[i * NUM_OF_REGS + idx] = value;
    }
  }
  return ret;
}

static std::string layer_name(const std::string& name) {
  (void)layer_name;
  auto name_remove_xfix = xir::remove_xfix(name);
  std::string ret;
  ret.reserve(name_remove_xfix.size());
  std::transform(name_remove_xfix.begin(), name_remove_xfix.end(),
                 std::back_inserter(ret), [](char c) {
                   bool ok = c >= '0' && c <= '9';
                   ok = ok || (c >= 'a' && c <= 'z');
                   ok = ok || (c >= 'A' && c <= 'Z');
                   // ok = ok || (c ==
                   // std::filesystem::path::preferred_separator);
                   ok = ok || (c == '_');
                   return ok ? c : '_';
                 });
  return ret;
}
static bool is_exist_file(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}

static std::string get_dump_filename(const std::string& subgraph_name,
                                     const std::string& tensor_type_str,
                                     const int engine_id,
                                     const std::string& tensor_layer_name) {
  const auto dump = std::filesystem::path("dump");
  const auto filename = std::filesystem::path(std::to_string(engine_id) + "." +
                                              tensor_layer_name + ".bin");
  return (dump / subgraph_name / tensor_type_str / filename).string();
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& out,
                                       const std::vector<T>& v) {
  int c = 0;
  out << "[";
  for (const auto x : v) {
    if (c++ != 0) {
      out << ",";
    }
    out << x;
  }
  out << "]";
  return out;
}
static std::vector<size_t> get_shape(const xir::Tensor* tensor) {
  auto dims = tensor->get_shape();
  std::vector<size_t> shape;
  std::transform(dims.begin(), dims.end(), std::back_inserter(shape),
                 [](std::int32_t n) { return static_cast<size_t>(n); });
  return shape;
}
static std::vector<size_t> get_strides(const xir::Tensor* tensor) {
  auto strides = tensor->get_attr<std::vector<std::int32_t>>("stride");
  std::vector<size_t> shape;
  std::transform(strides.begin(), strides.end(), std::back_inserter(shape),
                 [](std::int32_t n) { return static_cast<size_t>(n); });
  return shape;
}

static std::unique_ptr<vitis::ai::DimCalc> create_dim_calc(
    const xir::Tensor* tensor) {
  auto dims = get_shape(tensor);
  if (tensor->has_attr("stride")) {
    auto strides = get_strides(tensor);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER) >= 2)
        << tensor->get_name() << " dims :" << dims << " strides :" << strides;
    return std::make_unique<vitis::ai::DimCalc>(dims, strides);
  } else {
    return std::make_unique<vitis::ai::DimCalc>(dims);
  }
}

static void mkdir_minus_p(const std::string& dirname) {
  UNI_LOG_CHECK(std::filesystem::create_directories(dirname), VART_SYSTEM_ERROR)
      << "cannot create directories: " << dirname;
}

bool is_exist_path(const std::string& filename) {
  return std::filesystem::exists(filename);
}

static std::string get_full_filename(const std::string& filename) {
  if (filename[0] == std::filesystem::path::preferred_separator) {
    return filename;
  }
  return (std::filesystem::current_path() / filename).string();
}

static std::string get_parent_path(const std::string& path) {
  return path.substr(
      0, path.find_last_of(std::filesystem::path::preferred_separator));
}

static void create_parent_path(const std::string& path) {
  if (is_exist_path(path)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << path << " is exist!" << std::endl;
    return;
  }
  auto parent_path = get_parent_path(path);
  if (!is_exist_path(parent_path)) {
    create_parent_path(parent_path);
  }

  mkdir_minus_p(path);
}

bool DpuRunnerBaseImp::update_tensor_data_by_stride(std::vector<char>& buf,
                                                    const xir::Tensor* tensor,
                                                    const size_t offset) {
  auto dim_calc = create_dim_calc(tensor);
  auto dims_size = tensor->get_shape().size();
  auto idx = std::vector<size_t>(dims_size, 0u);
  auto next_idx = std::vector<size_t>(dims_size, 0u);
  auto sz = 0u;
  auto buf_idx = 0u;
  auto ok = true;
  for (std::tie(next_idx, sz) = dim_calc->next(idx); sz > 0 && ok;
       idx = next_idx, std::tie(next_idx, sz) = dim_calc->next(idx)) {
    ok = device_memory_->upload(&buf[buf_idx], offset + dim_calc->offset(idx),
                                sz);
    buf_idx += sz;
  }

  return true;
}
bool DpuRunnerBaseImp::download_tensor_data_by_stride(std::vector<char>& buf,
                                                      const xir::Tensor* tensor,
                                                      const size_t offset) {
  auto dim_calc = create_dim_calc(tensor);
  auto dims_size = tensor->get_shape().size();
  auto idx = std::vector<size_t>(dims_size, 0u);
  auto next_idx = std::vector<size_t>(dims_size, 0u);
  auto sz = 0u;
  auto buf_idx = 0u;
  auto ok = true;
  for (std::tie(next_idx, sz) = dim_calc->next(idx); sz > 0 && ok;
       idx = next_idx, std::tie(next_idx, sz) = dim_calc->next(idx)) {
    ok = device_memory_->download(&buf[buf_idx], offset + dim_calc->offset(idx),
                                  sz);
    buf_idx += sz;
  }

  return ok;
}
void DpuRunnerBaseImp::dump_tensor(const my_tensor_t& tensor) {
  if (tensor.get_location() != 1u) {
    return;
  }
  auto reg_id = tensor.get_reg_id();
  // auto is_parameter = reg_id != 1;  // dirty hack
  /*if (is_parameter) {
    // TODO: dump parameters;
    return;
    }*/
  auto subgraph_name = layer_name(subgraph_->get_name());
  auto tensor_offset = tensor.get_ddr_addr();
  auto tensor_layer_name = layer_name(tensor.get_name());

  auto num_of_engines = tensor.get_batch_size();
  //  const size_t tensor_size = tensor.size / num_of_engines;
  auto tensor_size = tensor.get_feature_map_size();
  auto device_core_id = session_->get_device_core_id();
  auto NUM_OF_DPU_REGS =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);

  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
    auto offset = base + tensor_offset;
    UNI_LOG_CHECK(base != std::numeric_limits<uint64_t>::max(),
                  VART_DPU_INFO_ERROR)
        << "NUM_OF_DPU_REGS " << NUM_OF_DPU_REGS << " "  //
        << "engine_id " << engine_id << " "              //
        << "reg_id " << reg_id << " "                    //
        ;
    auto filename = get_dump_filename(subgraph_name, tensor_output_dir_,
                                      engine_id, tensor_layer_name);

    auto buf = std::vector<char>(tensor_size);
    UNI_LOG_CHECK(buf.size() == (unsigned)tensor_size, VART_SIZE_MISMATCH);
    auto ok =
        download_tensor_data_by_stride(buf, tensor.get_xir_tensor(), offset);

    auto full_filename = get_full_filename(filename);
    auto parent_path = get_parent_path(full_filename);
    create_parent_path(parent_path);
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream(full_filename, mode).write(&buf[0], tensor_size).good())
        << " faild to write to " << filename;

    // auto ok = device_memory_->save(filename, offset, tensor_size);
    auto dump_ok = ok ? "success" : "fail";
    LOG(INFO) << "dump "
              << "  to " << filename                                          //
              << " device_core_id " << session_->get_device_core_id() << " "  //
              << "batch_idx " << engine_id << " "                             //
              << "reg_id " << reg_id << " "                                   //
              << "base " << base << " "                                       //
              << "tensor_offset " << tensor_offset << " "                     //
              << "offset " << offset << " "                                   //
              << "tensor_size " << tensor_size << " "                         //
              << "dump_ok " << dump_ok << " "                                 //
        ;
  }
}

void DpuRunnerBaseImp::upload_tensor(const my_tensor_t& tensor) {
  if (tensor.get_location() != 1) {
    return;
  }
  // auto subgraph_name = layer_name(subgraph_->get_name());
  auto reg_id = tensor.get_reg_id();
  auto is_parameter = reg_id == 0;  // dirty hack
  if (is_parameter) {
    return;
  }
  auto tensor_layer_name = layer_name(tensor.get_name());
  auto tensor_offset = tensor.get_ddr_addr();
  auto tensor_size = tensor.get_feature_map_size();
  //  constexpr auto NUM_OF_DPU_REGS = 8;
  //  auto engine_id = 0u;  // TODO: update for other tensors?
  //  auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
  //  auto offset = base + tensor_offset;

  auto golden_dirname = ENV_PARAM(XLNX_GOLDEN_DIR);
  std::string golden_filename =
      (std::filesystem::path(golden_dirname) / (tensor_layer_name + ".bin"))
          .string();
  if (!is_exist_file(golden_filename)) {
    LOG(INFO)
        << "XLNX_GOLDEN_DIR: upload data failed ! golden file is not exist : "
        << "layer_name " << tensor_layer_name << " ";
    return;
  }
  auto input_data = std::vector<char>(tensor_size);
  CHECK(std::ifstream(golden_filename).read(&input_data[0], tensor_size).good())
      << "fail to read! filename=" << golden_filename;

  auto num_of_engines = tensor.get_batch_size();
  auto device_core_id = session_->get_device_core_id();
  auto NUM_OF_DPU_REGS =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);

  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
    auto offset = base + tensor_offset;
    UNI_LOG_CHECK(base != std::numeric_limits<uint64_t>::max(),
                  VART_DPU_INFO_ERROR)
        << "NUM_OF_DPU_REGS " << NUM_OF_DPU_REGS << " "  //
        << "engine_id " << engine_id << " "              //
        << "reg_id " << reg_id << " "                    //
        ;
    // auto ok = device_memory_->upload(&input_data[0], offset, tensor_size);
    auto ok = update_tensor_data_by_stride(input_data, tensor.get_xir_tensor(),
                                           offset);

    auto upload_ok = ok ? "success" : "fail";
    LOG(INFO) << "XLNX_GOLDEN_DIR: upload data " << upload_ok << " ! "
              << " layer_name " << tensor_layer_name << " "
              << "device_core_id " << session_->get_device_core_id() << " "  //
              << "batch_idx " << engine_id << " "                            //
              << "reg_id " << reg_id << " "                                  //
              << "base " << base << " "                                      //
              << "tensor_offset " << tensor_offset << " "                    //
              << "offset " << offset << " "                                  //
              << "tensor_size " << tensor_size << " "                        //
        ;
  }
}

void DpuRunnerBaseImp::clear_tensor(const my_tensor_t& tensor) {
  if (tensor.get_location() != 1) {
    return;
  }
  auto reg_id = tensor.get_reg_id();
  auto is_parameter = reg_id == 0;  // dirty hack
  if (is_parameter) {
    return;
  }
  auto tensor_offset = tensor.get_ddr_addr();
  auto tensor_layer_name = layer_name(tensor.get_name());
  auto num_of_engines = tensor.get_batch_size();
  //  const size_t tensor_size = tensor.size / num_of_engines;
  auto tensor_size = tensor.get_feature_map_size();
  auto device_core_id = session_->get_device_core_id();
  auto NUM_OF_DPU_REGS =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);
  // constexpr auto NUM_OF_DPU_REGS = 8;
  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
    auto offset = base + tensor_offset;
    auto buf = std::vector<char>(tensor_size);
    UNI_LOG_CHECK(buf.size() == (unsigned)tensor_size, VART_SIZE_MISMATCH);
    for (auto i = 0u; i < tensor_size; i++) {
      buf[i] = (char)(i & 0xff);
    }
    //    auto ok = device_memory_->upload(&buf[0], offset, tensor_size);
    auto ok =
        update_tensor_data_by_stride(buf, tensor.get_xir_tensor(), offset);
    auto clear_ok = ok ? "success" : "fail";
    LOG(INFO) << "clear featuremap  to  layer_name" << tensor_layer_name << " "
              << "device_core_id " << session_->get_device_core_id() << " "  //
              << "batch_idx " << engine_id << " "                            //
              << "reg_id " << reg_id << " "                                  //
              << "base " << base << " "                                      //
              << "tensor_offset " << tensor_offset << " "                    //
              << "offset " << offset << " "                                  //
              << "tensor_size " << tensor_size << " "                        //
              << "clear_ok " << clear_ok << " "                              //
        ;
  }
}  // namespace dpu

void DpuRunnerBaseImp::compare_tensor(const my_tensor_t& tensor) {
  if (tensor.get_location() != 1) {
    return;
  }
  auto reg_id = tensor.get_reg_id();
  auto is_parameter = reg_id == 0;  // dirty hack
  if (is_parameter) {
    return;
  }

  auto tensor_offset = tensor.get_ddr_addr();
  auto tensor_layer_name = layer_name(tensor.get_name());
  //  const size_t tensor_size = tensor.size / num_of_engines;
  auto tensor_size = tensor.get_feature_map_size();
  //  constexpr auto NUM_OF_DPU_REGS = 8;
  //  auto engine_id = 0u;  // TODO : compare other engine data
  //  auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
  //  auto offset = base + tensor_offset;

  auto num_of_engines = tensor.get_batch_size();
  auto device_core_id = session_->get_device_core_id();
  auto NUM_OF_DPU_REGS =
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id);

  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    auto base = regs_[engine_id * NUM_OF_DPU_REGS + reg_id];
    auto offset = base + tensor_offset;
    UNI_LOG_CHECK(base != std::numeric_limits<uint64_t>::max(),
                  VART_SIZE_MISMATCH)
        << "NUM_OF_DPU_REGS " << NUM_OF_DPU_REGS << " "  //
        << "engine_id " << engine_id << " "              //
        << "reg_id " << reg_id << " "                    //
        ;
    auto buf = std::vector<char>(tensor_size);
    UNI_LOG_CHECK(buf.size() == (unsigned)tensor_size, VART_SIZE_MISMATCH);
    auto ok =
        download_tensor_data_by_stride(buf, tensor.get_xir_tensor(), offset);

    // auto ok = device_memory_->download(&buf[0], offset, tensor_size);
    if (ok) {
      auto dump_md5 = md5sum((const char*)&buf[0], tensor_size);
      auto golden_dirname = ENV_PARAM(XLNX_GOLDEN_DIR);
      std::string golden_filename =
          (std::filesystem::path(golden_dirname) /
           std::filesystem::path(tensor_layer_name + ".bin"))
              .string();

      if (!is_exist_file(golden_filename)) {
        LOG(INFO) << "XLNX_GOLDEN_DIR: compare data failed ! golden file is "
                     "not exist : "
                  << "layer_name " << tensor_layer_name << " " << dump_md5
                  << " ";
        return;
      }
      auto gloden_md5 = xir::get_md5_of_file(golden_filename);
      bool md5_ok = dump_md5 == gloden_md5;
      if (md5_ok) {
        LOG(INFO) << "XLNX_GOLDEN_DIR: compare data success !"
                  << "layer_name " << tensor_layer_name << " "  //
                  << "batch_idx " << engine_id << " "           //
                  << "dump_md5 " << dump_md5 << " "             //
            ;
      } else {
        LOG(INFO) << "XLNX_GOLDEN_DIR: compare data failed ! "
                  << "layer_name " << tensor_layer_name << " "  //
                  << "batch_idx " << engine_id << " "           //
                  << "dump tensor data : " << dump_md5 << " "   //
                  << "golden file : " << golden_filename << " " << gloden_md5
                  << " ";
      }

    } else {
      LOG(INFO) << "XLNX_GOLDEN_DIR: download data failed ! "
                << "layer_name " << tensor_layer_name << " "  //
                << "device_core_id " << session_->get_device_core_id()
                << " "                                       //
                << "batch_idx " << engine_id << " "          //
                << "reg_id " << reg_id << " "                //
                << "base " << base << " "                    //
                << "tensor_offset " << tensor_offset << " "  //
                << "offset " << offset << " "                //
                << "tensor_size " << tensor_size << " "      //
                << " ";
    }
  }
}

const my_tensor_t& DpuRunnerBaseImp::find_tensor(const std::string& name) {
  auto it = std::find_if(
      session_->my_all_tensors_.begin(), session_->my_all_tensors_.end(),
      [&name](const auto& tensor) { return tensor.get_name() == name; });
  UNI_LOG_CHECK(it != session_->my_all_tensors_.end(), VART_TENSOR_INFO_ERROR)
      << "cannot find tensor: tensor name=" << name;
  return *it;
}

static std::string to_string(const std::vector<uint64_t>& regs,
                             size_t NUM_OF_DPU_REGS) {
  std::ostringstream out;
  out << std::hex;
  for (auto i = 0u; i < regs.size() / NUM_OF_DPU_REGS; ++i) {
    out << "\n";
    for (auto j = 0u; j < NUM_OF_DPU_REGS; ++j) {
      out << "\t0x" << regs[i * NUM_OF_DPU_REGS + j];
    }
  }
  return out.str();
}

void DpuRunnerBaseImp::for_each_tensor(
    const std::vector<const xir::Tensor*> tensors, tensor_fun_t f) {
  for (auto* t : tensors) {
    (this->*f)(find_tensor(t->get_name()));
  }
}

std::vector<const xir::Tensor*> DpuRunnerBaseImp::get_internal_tensor(
    const xir::Subgraph* subgraph) {
  auto output_tensors = get_output_tensor(subgraph);
  auto is_output_tensor = [&output_tensors](const std::string& name) {
    bool ret = false;
    for (auto t : output_tensors) {
      if (t->get_name() == name) {
        ret = true;
        break;
      }
    }
    return ret;
  };
  auto internal_ops = subgraph->get_ops();
  auto ret = std::vector<const xir::Tensor*>();
  for (const auto* op : internal_ops) {
    auto tensor_name = op->get_output_tensor()->get_name();
    if (is_output_tensor(tensor_name)) {
      continue;
    }
    if (op->get_type() == "const-fix" || op->get_type() == "const") {
      continue;
    }
    ret.push_back(find_tensor(tensor_name).get_tensor());
  }
  return ret;
}

void DpuRunnerBaseImp::before_run_dpu() {
  if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
    tensor_output_dir_ = "input";
    for_each_tensor(get_input_tensor(subgraph_),
                    &DpuRunnerBaseImp::dump_tensor);
  }

  if (xlnx_enable_compare_mode()) {
    for_each_tensor(get_input_tensor(subgraph_),
                    &DpuRunnerBaseImp::compare_tensor);
  }
  if (xlnx_enable_compare_mode() && ENV_PARAM(XLNX_ENABLE_UPLOAD)) {
    for_each_tensor(get_input_tensor(subgraph_),
                    &DpuRunnerBaseImp::upload_tensor);
  }
  if (ENV_PARAM(XLNX_ENABLE_CLEAR)) {
    for_each_tensor(get_internal_tensor(subgraph_),
                    &DpuRunnerBaseImp::clear_tensor);
    for_each_tensor(get_output_tensor(subgraph_),
                    &DpuRunnerBaseImp::clear_tensor);
  }
}

void DpuRunnerBaseImp::after_run_dpu() {
  if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
    tensor_output_dir_ = "internal";
    for_each_tensor(get_internal_tensor(subgraph_),
                    &DpuRunnerBaseImp::dump_tensor);
    tensor_output_dir_ = "output";
    for_each_tensor(get_output_tensor(subgraph_),
                    &DpuRunnerBaseImp::dump_tensor);
  }
  if (xlnx_enable_compare_mode()) {
    for_each_tensor(get_internal_tensor(subgraph_),
                    &DpuRunnerBaseImp::compare_tensor);
    for_each_tensor(get_output_tensor(subgraph_),
                    &DpuRunnerBaseImp::compare_tensor);
  }
}

void DpuRunnerBaseImp::start_dpu2(size_t device_core_id) {
  if (ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN) >= 3) {
    LOG(INFO) << "DEBUG_DPU_RUNNER_DRY_RUN = "
              << ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN) << ", ignore running dpu";
    return;
  }
  auto kernel = session_->kernel_.get();
  auto sg_and_code = kernel->get_code(device_core_id);
  auto gen_reg = build_gen_reg(
      kernel->get_parameter(device_core_id), session_->get_num_of_engines(),
      const_cast<const xir::DpuController*>(session_->get_dpu_controller())
          ->get_size_of_gen_regs(device_core_id));
  fill_gen_reg(device_core_id, gen_reg);
  if (ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN) >= 2) {
    LOG(INFO) << "DEBUG_DPU_RUNNER_DRY_RUN = "
              << ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN) << ", ignore running dpu";
    return;
  }
  for (auto idx = 0u; idx < sg_and_code.size(); ++idx) {
    auto code = sg_and_code[idx].code_addr;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "@" << (void*)this << " device_core_id=" << device_core_id  //
        << " DPU: "
        << session_->get_dpu_controller()->get_full_name(device_core_id)  //
        << ":"
        << session_->get_dpu_controller()->get_device_id(device_core_id)  //
        << " running dpu code " << std::hex << " 0x" << code << std::dec << " "
        << "gen_reg.size() " << gen_reg.size() << " "  //
        << "gen_reg "
        << to_string(gen_reg,
                     session_->get_dpu_controller()->get_size_of_gen_regs(
                         device_core_id))
        << " "  //
        ;
    if (xlnx_enable_debug_dpu_data_mode()) {
      prepare_envirnment(sg_and_code[idx], gen_reg, device_core_id);
      before_run_dpu();
    }

    LOG_IF(INFO, ENV_PARAM(XLNX_SHOW_DPU_COUNTER))
        << "subgraph name : "
        << layer_name(sg_and_code[idx].subgraph->get_name());
    if (vitis::ai::trace::is_enabled()) {
      auto workload =
          sg_and_code[idx].subgraph->get_attr<std::uint64_t>("workload");
      auto depth = sg_and_code[idx].subgraph->get_depth();
      auto name = sg_and_code[idx].subgraph->get_name();
      auto batch = session_->get_num_of_engines();
      // MSVC NOTE: it is not safe to call template function across DLL.
#if !_WIN32
      vitis::ai::trace::add_trace("dpu-runner", name, batch, workload, depth);
#endif
    }
    LOG_IF(FATAL, ENV_PARAM(XLNX_ENABLE_FINGERPRINT_CHECK) &&
                      !check_fingerprint(session_->get_device_core_id()))
        << "fingerprint check failure.";
    if (!ENV_PARAM(DEBUG_DPU_RUNNER_DRY_RUN)) {
      if (ENV_PARAM(XLNX_SHOW_DPU_COUNTER)) {
        auto& subgraph = sg_and_code[idx].subgraph;
        auto workload = subgraph->get_attr<std::uint64_t>("workload");
        std::string log = "workload " + std::to_string(workload);
        if (subgraph->has_attr("workload_on_arch")) {
          auto arch = subgraph->get_attr<std::uint64_t>("workload_on_arch");
          log += " workload_on_arch " + std::to_string(arch);
        }
        std::cout << log << std::endl;
      }
      session_->get_dpu_controller()->run(device_core_id, code, gen_reg);
    }
    if (xlnx_enable_debug_dpu_data_mode()) {
      after_run_dpu();
      clear_environment();
    }
  }
}
bool DpuRunnerBaseImp::check_fingerprint(size_t device_core_id) {
  auto model_fingerprint = session_->kernel_->get_fingerprint();
  auto dpu_fingerprint =
      session_->get_dpu_controller()->get_fingerprint(device_core_id);
  auto ret = model_fingerprint == dpu_fingerprint;
  if (model_fingerprint == 0u) {
    // for elf file or debugging purpuse, if xmodel does not contain
    // a fingerprint, it is zero, and we ignore fingerprint
    // checkout.
    ret = true;
  }
  if (dpu_fingerprint == 0u) {
    // for vivado flow, we do not support finger print checking so disable it.
    return true;
  }
  if (!ret) {
    LOG_IF(WARNING, model_fingerprint != 0u && dpu_fingerprint != 0u)
        << "CHECK fingerprint fail! model_fingerprint 0x"      //
        << std::hex << model_fingerprint                       //
        << " is un-matched with actual dpu_fingerprint 0x"     //
        << dpu_fingerprint << ". "                             //
        << "Please re-compile xmodel with dpu_fingerprint 0x"  //
        << dpu_fingerprint << std::dec << " and try again.";
  }
  return ret;
}
void DpuRunnerBaseImp::prepare_envirnment(
    const DpuKernel::SubgraphCode& sg_and_code,
    const std::vector<uint64_t>& gen_reg, size_t device_core_id) {
  subgraph_ = sg_and_code.subgraph;
  regs_ = gen_reg;
  get_device_memory();
}

void DpuRunnerBaseImp::clear_environment() { device_memory_ = nullptr; }

xir::DeviceMemory* DpuRunnerBaseImp::get_device_memory() {
  if (!device_memory_) {
    auto device_id = session_->get_dpu_controller()->get_device_id(
        session_->get_device_core_id());
    device_memory_ = vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
        device_id, device_id);
  }
  return device_memory_.get();
}

}  // namespace dpu
}  // namespace vart
