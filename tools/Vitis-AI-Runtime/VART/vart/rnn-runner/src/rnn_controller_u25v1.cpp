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

#include "rnn_controller_u25v1.hpp"

#include <memory>
#include <utility>
#include <vitis/ai/weak.hpp>
#include <xir/xrt_device_handle.hpp>

#include "regmap_rnn_u25v1.hpp"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename T>
static std::string dump_binary_data(std::string filename, T* data,
                                    size_t size) {
  std::ofstream of(filename);
  for (size_t i = 0; i < size; ++i) {
    of << std::hex << i << " " << data[i] << '\n';
  }
  return " Done";
}

namespace vart {
namespace xrnn {

// Number of instructions in each set should be aligned
static constexpr uint32_t INSTR_ALIGN = 16;

static std::map<const std::string, std::vector<XRNN_REG_T>*> reg_init_map{
    {"u25_sentiment_cu0", &U25_SENTIMENT_REGS_CU0},
    {"u25_satisfaction_cu0", &U25_SATISFACTION_REGS_CU0},
    {"u25_openie_cu0", &U25_OPENIE_REGS_CU0}};

static std::map<const std::string, std::vector<size_t>*> batch_addr_map{
    {"u25_cu0", &U25_DDR_BASE_CU0}};

static std::map<const std::string, std::vector<size_t>*> init_addr_map{
    {"u25_cu0", &U25_DDR_INIT_ADDR_CU0}};

template <uint32_t N>
static uint32_t multiple_of(uint32_t x) {
  return ((x - 1) / N + 1) * N;
}

std::mutex RnnControllerU25::mutex_;

std::vector<uint32_t> RnnControllerU25::get_reg_data(int frame,
                                                     int thread_index) {
  std::vector<uint32_t> regs_array(MAX_REG_ADDR / 4, 0);

  std::string board = get_board_name();
  // for (unsigned i = 0; i < reg_init_data.size(); i++) {
  //   regs_array[reg_init_data[i].addr / 4] = reg_init_data[i].value;
  // }

  if (board == "u25") {
    regs_array[REG_INSTR_LOW_ADDR / 4] = LSB((U25_DEV_ADDR) + ADDR(INSTR));
    regs_array[REG_INSTR_HIGH_ADDR / 4] = HSB((U25_DEV_ADDR) + ADDR(INSTR));
    regs_array[REG_VECTOR_LOW_ADDR / 4] = LSB((U25_DEV_ADDR) + ADDR(VECTOR));
    regs_array[REG_VECTOR_HIGH_ADDR / 4] = HSB((U25_DEV_ADDR) + ADDR(VECTOR));
    regs_array[REG_RESULT_LOW_ADDR / 4] = LSB((U25_DEV_ADDR) + ADDR(RESL));
    regs_array[REG_RESULT_HIGH_ADDR / 4] = HSB((U25_DEV_ADDR) + ADDR(RESL));
    regs_array[REG_PROF_ADDR / 4] = LSB((U25_DEV_ADDR) + ADDR(PROF));
    regs_array[REG_CFG_DONE / 4] = 0x00000001;

    // Fill model registers
    regs_array[REG_CONF_LAYER_NUM / 4] = nlayers_;
    regs_array[REG_CONF_FRAME_LEN / 4] = frame;

    // Fill layer-wise first & loop instruction counts
    // and their address offsets
    const auto& instr_count_vec = model_config_->get_instr_count_vector(batch_);
    uint32_t instr_offset = ADDR(INSTR);
    for (int i = 0; i < nlayers_; ++i) {
      auto first_instr_count = instr_count_vec[i * 2];
      auto loop_instr_count = instr_count_vec[i * 2 + 1];
      auto first_instr_count_16 = multiple_of<INSTR_ALIGN>(first_instr_count);
      auto loop_instr_count_16 = multiple_of<INSTR_ALIGN>(loop_instr_count);
      regs_array[REG_CONF_INSTR_FIRST_LEN_0 / 4 + i] = first_instr_count;
      regs_array[REG_CONF_INSTR_LOOP_LEN_0 / 4 + i] = loop_instr_count;
      regs_array[REG_CONF_INSTR_FIRST_ADDR_0 / 4 + i] = instr_offset;
      regs_array[REG_CONF_INSTR_LOOP_ADDR_0 / 4 + i] =
          instr_offset + first_instr_count_16 * 16;
      instr_offset += (first_instr_count_16 + loop_instr_count_16) * 16;
    }
    regs_array[REG_CONF_INSTR_END_ADDR / 4] = instr_offset;

    // Fill Dynamic arrays
    // inputX, inputH and output feature map size
    // and their corresponding addresses (0x234+, 0x254+, 0x274+)
    // and address calculation direction for SAVE
    const auto THREAD_OFFSET = thread_index * THREAD_STEP;
    // auto actH_offset = 0;
    for (int i = 0; i < nlayers_; ++i) {
      auto actX_size = model_config_->get_reg_size(i, CONFIG_NAME::LOAD0) * 2;
      auto actH_size = model_config_->get_reg_size(i, CONFIG_NAME::LOAD1) * 2;
      auto load0_reg_dir = model_config_->get_reg_dir(i, CONFIG_NAME::LOAD0);
      auto save0_reg_dir = model_config_->get_reg_dir(i, CONFIG_NAME::SAVE0);

      regs_array[REG_CONF_ACTX_SIZE_0 / 4 + i] = actX_size;
      regs_array[REG_CONF_ACTH_SIZE_0 / 4 + i] = actH_size;
      int offset_load0 =
          (1 - load0_reg_dir) * (frame - 1) * actX_size + THREAD_OFFSET;
      int offset_save0 =
          (1 - save0_reg_dir) * (frame - 1) * actH_size + THREAD_OFFSET;

      if (i % 2 == 0) {
        regs_array[REG_CONF_ACTX_ADDR_0 / 4 + i] = ADDR(VECTOR) + offset_load0;
        regs_array[REG_CONF_ACTH_ADDR_0 / 4 + i] = ADDR(RESL) + offset_save0;
        regs_array[REG_CONF_SAVE_ADDR_0 / 4 + i] = ADDR(RESL) + offset_save0;
      } else {
        regs_array[REG_CONF_ACTX_ADDR_0 / 4 + i] = ADDR(RESL) + offset_load0;
        regs_array[REG_CONF_ACTH_ADDR_0 / 4 + i] = ADDR(VECTOR) + offset_save0;
        regs_array[REG_CONF_SAVE_ADDR_0 / 4 + i] = ADDR(VECTOR) + offset_save0;
      }

      regs_array[REG_CONF_SAVE_FORWARD_0 / 4 + i] = save0_reg_dir;

      // actH_offset = (frame - 1) * actH_size + THREAD_OFFSET;
    }

    // TODO : abidk: Why are these hardcoded?
    regs_array[REG_CONF_ACTH_ADDR_7 / 4] = ADDR(VECTOR) + THREAD_OFFSET;
    regs_array[REG_CONF_SAVE_ADDR_7 / 4] = ADDR(VECTOR) + THREAD_OFFSET;
  }

  return regs_array;
}

std::string RnnControllerU25::get_board_name() {
  // auto kernel_name = xrt_cu_->get_kernel_name(idx_);
  // return kernel_name.find("slr") != kernel_name.npos ? "u50" : "u25";
  return "u25";
}

std::string RnnControllerU25::get_addr_name() {
  auto core_id = xrt_cu_->get_core_id(idx_);
  std::string board = get_board_name();
  return board + "_cu" + std::to_string(core_id);
}

size_t RnnControllerU25::get_base_addr(unsigned batch_num) {
  std::string addr_name = get_addr_name();
  CHECK(batch_addr_map.count(addr_name) == 1)
      << "Can Not Find Addr For " << addr_name;
  std::vector<size_t>& base_addrs = *batch_addr_map[addr_name];
  CHECK(batch_num < base_addrs.size()) << "Invalid batch_num: " << batch_num;

  return base_addrs[batch_num];
}

int RnnControllerU25::get_batch_size() { return batch_; }

RnnControllerU25::RnnControllerU25(size_t device_core_id,
                                   std::unique_ptr<xir::XrtCu>&& xrt_cu)
    : idx_{device_core_id},
      xrt_cu_{std::move(xrt_cu)},
      memory_{vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
          xrt_cu_->get_device_id(idx_), xrt_cu_->get_device_id(idx_))} {
  batch_ = 1;
}

void RnnControllerU25::run(char* in, uint64_t isize, char* out, uint64_t osize,
                           int batch, int frame, int thread_index) {
  CHECK(batch <= batch_) << "Invalid Batch Size";

  auto core_id = xrt_cu_->get_core_id(idx_);

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "device_core_id " << idx_ << " core_id " << core_id;

  std::vector<uint32_t> reg_data = get_reg_data(frame, thread_index);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
      << "Dumping u25 register to u25_reg_v2.log ..."
      << dump_binary_data("u25_reg_v2.log", reg_data.data(), reg_data.size());

  for (auto i = 0; i < batch; i++) {
    auto base_addr = get_base_addr(i);

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "vetor input addr: " << std::hex
        << (base_addr + ADDR(VECTOR) + thread_index * THREAD_STEP);

    memory_->upload(
        (void*)(in + i * isize),
        (size_t)(base_addr + ADDR(VECTOR) + thread_index * THREAD_STEP),
        (size_t)isize);
  }

  // mutex_.lock();
  auto func = [=](ert_start_kernel_cmd* ecmd) -> void {
    auto rsz = (0x2e0 / 4 + 1) + 1;  // regmap array size
    ecmd->count = 1 + rsz;

    ecmd->data[0x00] = 0x00;
    for (unsigned i = 1; i < reg_data.size(); i++) {
      ecmd->data[i] = reg_data[i];
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "reg size" << reg_data.size();
  };

  auto out_addr = nlayers_ % 2 ? ADDR(RESL) : ADDR(VECTOR);

  xrt_cu_->run(
      idx_, func,
      // on_success
      [=](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "xrnn excute done! "
            << "core_id = " << core_id << " thread = " << thread_index;

        LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "get result len: " << osize << "(0x" << std::hex << osize << ")";

        for (auto i = 0; i < batch; i++) {
          auto base_addr = get_base_addr(i);
          LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
              << "result output addr: " << std::hex
              << (base_addr + out_addr + thread_index * THREAD_STEP);
          memory_->download(
              (void*)(out + i * osize),
              (size_t)(base_addr + out_addr + thread_index * THREAD_STEP),
              (size_t)osize);
          LOG_IF(INFO, ENV_PARAM(DEBUG_DUMP_DATA))
              << "Writing output to out_v2_b" << i << ".log... " << std::hex
              << dump_binary_data("out_v2_b" + std::to_string(i) + ".log",
                                  out + i * osize, osize);
        }
      },
      // on failure
      [core_id](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG(INFO) << "xrnn controller timeout! "
                  << "core_id = " << core_id << "\n";
      });

  // mutex_.unlock();
}

RnnControllerU25::~RnnControllerU25() {}

void RnnControllerU25::init(char* ddr, uint64_t size) {
  std::string addr_name = get_addr_name();
  CHECK(init_addr_map.count(addr_name) == 1)
      << "Can Not Find DDR Init Addr For " << addr_name;
  std::vector<size_t>& init_addrs = *init_addr_map[addr_name];

  for (unsigned i = 0; i < init_addrs.size(); i++) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "ddr_init at: " << std::hex << init_addrs[i];
    memory_->upload((void*)ddr, (size_t)init_addrs[i], (size_t)size);
  }
}

void RnnControllerU25::update(int frame, ModelConfig* mc, uint32_t* p_ddr,
                              size_t size) {
  nlayers_ = mc->get_layer_num();
}

// REGISTER_RNN_CONTROLLER("U50LV", RnnControllerU25)
struct RegisterU25 {
  RegisterU25() {
    RnnControllerCreate::Register(
        "U25v1", [](unsigned int device_core_id, std::string device) {
          return std::make_unique<RnnControllerU25>(
              device_core_id, std::make_unique<xir::XrtCu>(device));
        });
  }
};
static RegisterU25 dummyRegU25;

}  // namespace xrnn
}  // namespace vart
