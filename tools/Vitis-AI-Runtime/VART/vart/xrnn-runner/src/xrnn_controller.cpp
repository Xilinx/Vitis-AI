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
#include "xrnn_controller.hpp"

#include <map>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sys/mman.h>
#include <chrono>
#include <thread>

#include <glog/logging.h>
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
#include "xir/xrt_device_handle.hpp"

DEF_ENV_PARAM(DEBUG_XRNN_CONTROLLER, "0");
DEF_ENV_PARAM(XRNN_MAX_THREAD, "16");
DEF_ENV_PARAM_2(XRNN_KERNEL_NAME, "xrnn", std::string);
DEF_ENV_PARAM_2(XRNN_INSTANCE_NAME, "xrnn_1", std::string);

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace vart {
namespace xrnn {

static std::map<const std::string, MODEL_TYPE> model_type_map{
  {"sentiment", SENTIMENT},
  {"satisfaction", SATISFACTION},
  {"openie", OPENIE}
};  

static std::map<const std::string, std::vector<XRNN_REG_T>* > reg_init_map{
  {"u50_sentiment_cu0", &U50_SENTIMENT_REGS_CU0},
  {"u50_sentiment_cu1", &U50_SENTIMENT_REGS_CU1},
  {"u50_satisfaction_cu0", &U50_SATISFACTION_REGS_CU0},
  {"u50_satisfaction_cu1", &U50_SATISFACTION_REGS_CU1},
  {"u50_openie_cu0", &U50_OPENIE_REGS_CU0},
  {"u50_openie_cu1", &U50_OPENIE_REGS_CU1},
  {"u25_sentiment_cu0", &U25_SENTIMENT_REGS_CU0},
  {"u25_satisfaction_cu0", &U25_SATISFACTION_REGS_CU0},
  {"u25_openie_cu0", &U25_OPENIE_REGS_CU0}
};

static std::map<const std::string, std::vector<size_t>* > batch_addr_map{
  {"u50_cu0", &U50_HBM_BATCH3_CU0},
  {"u50_cu1", &U50_HBM_BATCH4_CU1},
  {"u25_cu0", &U25_DDR_BASE_CU0}
};

static std::map<const std::string, std::vector<size_t>* > init_addr_map{
  {"u50_cu0", &U50_DDR_INIT_ADDR_CU0},
  {"u50_cu1", &U50_DDR_INIT_ADDR_CU1},
  {"u25_cu0", &U25_DDR_INIT_ADDR_CU0}
};

std::mutex XrnnController::mutex_;

MODEL_TYPE XrnnController::get_model_type(const std::string& model_name){
  return model_type_map.count(model_name)==1?model_type_map[model_name]:UNKOWN;
}

std::string XrnnController::get_model_name(MODEL_TYPE model_type){
  std::map<const std::string, MODEL_TYPE>::iterator iter;
  for(iter=model_type_map.begin();iter!=model_type_map.end(); iter++){
    if(iter->second == model_type)
        return iter->first;
  }
  return std::string("");
}

std::vector<uint32_t> XrnnController::get_reg_data(int frame, int thread_index){
  std::vector<uint32_t> regs_array(MAX_REG_ADDR/4, 0);

  auto core_id = xrt_cu_->get_core_id(idx_);
  
  std::string board = get_board_name();
  std::string model = get_model_name(model_type_);
  std::string reg_name = board+"_"+model +"_cu"+std::to_string(core_id); 
  CHECK(reg_init_map.count(reg_name) == 1) << "Can Not Find Reg Init data " << reg_name;

  std::vector<XRNN_REG_T>& reg_init_data = *reg_init_map[reg_name];
  for(unsigned i=0; i<reg_init_data.size(); i++){
    regs_array[reg_init_data[i].addr/4] = reg_init_data[i].value;
  }

  if(board == "u50"){
    regs_array[reg_init_data[0].addr/4] = frame;
    regs_array[reg_init_data[reg_init_data.size()-2].addr/4] = \
        reg_init_data[reg_init_data.size()-2].value+(thread_index*THREAD_STEP);
    regs_array[reg_init_data[reg_init_data.size()-1].addr/4] = \
        reg_init_data[reg_init_data.size()-1].value+(thread_index*THREAD_STEP);
  }
  else if(board == "u25"){
    regs_array[reg_init_data[1].addr/4] = frame;
    regs_array[REG_INSTR_LOW_ADDR/4]=LSB((U25_DEV_ADDR)+ADDR(INSTR));
    regs_array[REG_INSTR_HIGH_ADDR/4]=HSB((U25_DEV_ADDR)+ADDR(INSTR));
    regs_array[REG_BIAS_LOW_ADDR/4]=LSB((U25_DEV_ADDR)+ADDR(BIAS));
    regs_array[REG_BIAS_HIGH_ADDR/4]=HSB((U25_DEV_ADDR)+ADDR(BIAS));
    regs_array[REG_PROF_ADDR/4]=LSB((U25_DEV_ADDR)+ADDR(PROF));

    regs_array[REG_RESULT_LOW_ADDR/4]=LSB((U25_DEV_ADDR)+ADDR(RESL));
    regs_array[REG_RESULT_HIGH_ADDR/4]=HSB((U25_DEV_ADDR)+ADDR(RESL));
    regs_array[REG_VECTOR_LOW_ADDR/4]=LSB((U25_DEV_ADDR)+ADDR(VECTOR));
    regs_array[REG_VECTOR_HIGH_ADDR/4]=HSB((U25_DEV_ADDR)+ADDR(VECTOR));
    regs_array[REG_CFG_DONE/4]=0x00000001;
    
    if(model_type_ == OPENIE){
      unsigned start = reg_init_data.size()-16;
      for(int i=0; i<16; i++){
        if(i%2==0)
          regs_array[reg_init_data[start+i].addr/4] = ADDR(RESL)+(frame-1)*reg_init_data[43].value;
        else
          regs_array[reg_init_data[start+i].addr/4] = ADDR(VECTOR)+(frame-1)*reg_init_data[43].value;
      }
      regs_array[reg_init_data[start+7].addr/4] = ADDR(VECTOR);
      regs_array[reg_init_data[start+15].addr/4] = ADDR(VECTOR);
    }

    unsigned i = model_type_==OPENIE?reg_init_data.size()-24:reg_init_data.size()-3;
    
    for (;i<reg_init_data.size();i++){
      regs_array[reg_init_data[i].addr/4] += thread_index*THREAD_STEP;
    }
  }

  return regs_array;
}

std::string XrnnController::get_board_name(){
  auto kernel_name = xrt_cu_->get_kernel_name(idx_);
  return kernel_name.find("slr")!=kernel_name.npos?"u50":"u25";
}

std::string XrnnController::get_addr_name(){
  auto core_id = xrt_cu_->get_core_id(idx_);
  std::string board = get_board_name();
  return board+"_cu"+std::to_string(core_id); 
}

size_t XrnnController::get_base_addr(unsigned batch_num)
{
  std::string addr_name = get_addr_name(); 
  CHECK(batch_addr_map.count(addr_name) == 1) << "Can Not Find Addr For " << addr_name;
  std::vector<size_t>& base_addrs = *batch_addr_map[addr_name];
  CHECK(batch_num < base_addrs.size()) << "Invalid batch_num: " << batch_num;

  return base_addrs[batch_num];
}

int XrnnController::get_batch_size(){
  return batch_;
}

XrnnController::XrnnController(size_t device_core_id,
                               const std::string& model_type,
                               std::unique_ptr<xir::XrtCu>&& xrt_cu)
  : idx_{device_core_id},
  xrt_cu_{std::move(xrt_cu)},
  memory_{vitis::ai::WeakStore<size_t, xir::DeviceMemory>::create(
        xrt_cu_->get_device_id(idx_), xrt_cu_->get_device_id(idx_))}{
  
  CHECK(model_type.empty()==0);
  model_type_ = get_model_type(model_type);
  CHECK(model_type_ != UNKOWN);

  auto core_id = xrt_cu_->get_core_id(idx_);
  auto kernel_name = xrt_cu_->get_kernel_name(idx_);
  if (kernel_name.find("slr")!=kernel_name.npos)
    batch_ = core_id==0?3:4;
  else
    batch_ = 1;
}

void XrnnController::run(
          char* in, uint64_t isize,
          char* out, uint64_t osize,
          int batch, int frame, int thread_index) {
     
  CHECK(batch <= batch_) << "Invalid Batch Size";

  auto core_id = xrt_cu_->get_core_id(idx_);
  
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "device_core_id " << idx_<< " core_id " << core_id;   

  std::vector<uint32_t> reg_data = get_reg_data(frame, thread_index);

  //for(unsigned i=0; i<reg_data.size(); i++){
  //  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
  //    << std::hex << i*4 << " --  "
  //    << std::hex << reg_data[i];
  //}

  for(auto i=0; i<batch; i++){
    auto base_addr = get_base_addr(i);

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "vetor input addr: "  << std::hex << (base_addr+ADDR(VECTOR)+thread_index*THREAD_STEP);

    memory_->upload((void*)(in+i*isize),\
      (size_t) (base_addr+ADDR(VECTOR)+thread_index*THREAD_STEP), \
      (size_t) isize);
  }

  //mutex_.lock();
  auto func = [=](ert_start_kernel_cmd* ecmd) -> void {
    auto rsz = (0x2e0/4+1) + 1; // regmap array size
    ecmd->count = 1 + rsz;

    ecmd->data[0x00] = 0x00;
    for(unsigned i=1; i < reg_data.size(); i++){
      ecmd->data[i] = reg_data[i];
    }

    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "reg size" << reg_data.size();
  };

  xrt_cu_->run(
      idx_, func,
      // on_success
      [=](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
          << "xrnn excute done! "
          << "core_id = " << core_id  << " thread = " << thread_index;

        if(model_type_ == OPENIE){
          LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "get result len: " << osize << "(0x" << std::hex << osize 
            <<") for type " << model_type_;

          for(auto i=0; i<batch; i++){
            auto base_addr = get_base_addr(i);
            LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
              << "result output addr: "  << std::hex << (base_addr+ADDR(VECTOR) + thread_index*THREAD_STEP);
              memory_->download((void*)(out+i*osize),\
                (size_t) (base_addr+ADDR(VECTOR) + thread_index*THREAD_STEP),\
                (size_t) osize);
          }
        }
        else{
          LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
            << "get result len: " << osize << "(0x" << std::hex << osize 
            <<") for type " << model_type_;

          for (auto i=0; i<batch; i++){
            auto base_addr = get_base_addr(i);
            LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
              << "result output addr: "  << std::hex << (base_addr+ADDR(RESL) + thread_index*THREAD_STEP);
            memory_->download((void*)(out+i*osize),\
              (size_t) (base_addr+ADDR(RESL) + thread_index*THREAD_STEP),\
              (size_t) osize);
          }
        }
      },
      // on failure
      [core_id](xclDeviceHandle handle, uint64_t cu_addr) -> void {
        LOG(FATAL) << "xrnn controller timeout! "
                   << "core_id = " << core_id << "\n";
      });

  //mutex_.unlock();
}

XrnnController::~XrnnController() {
}

void XrnnController::init(char *ddr, uint64_t size)
{
  std::string addr_name = get_addr_name(); 
  CHECK(init_addr_map.count(addr_name) == 1) << "Can Not Find DDR Init Addr For " << addr_name;
  std::vector<size_t>& init_addrs = *init_addr_map[addr_name];

  for(unsigned i=0; i<init_addrs.size(); i++){
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "ddr_init at: "<< std::hex << init_addrs[i];
    memory_->upload((void*)ddr, (size_t) init_addrs[i], (size_t) size);
  }
  
  //ModelConfig * mc = new ModelConfig("/scratch/yili/vart/xrnn-runner");
}

void XrnnController::update(int frame, ModelConfig *mc, 
  uint32_t *p_ddr, size_t size){

  int layers = mc->get_layer_num();
  int reg_step = 0x30;
  int reg_load0 = 0x30;
  int reg_load1 = 0x34;
  int reg_save0 = 0x50;

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "layers: "<< layers;

  int ddr_regs_ptr = 0;
  for(int i=0; i<layers; i++){
    ddr_regs_ptr += (i==0)?0:(mc->get_layer_instr_len(i-1, batch_));

    int dir_val = (mc->get_reg_dir(i, CONFIG_NAME::LOAD0)==1)?0:1;
    int offset_load0 = dir_val*(frame-1)*mc->get_reg_size(i, CONFIG_NAME::LOAD0)*2;

    //LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
    //  << "i:" << i << " layer_step: "<< layer_step;
    for(int b=0; b<batch_; b++){
      uint32_t  batch_base = get_base_addr(b)&0xFFFFFFFF;
      //LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      //  <<  "b:" << b << "  batch_base:  "<< std::hex << batch_base;
      if(i%2 == 0){
        p_ddr[(ddr_regs_ptr+(reg_load0+reg_step*b))/4] = batch_base + ADDR(VECTOR) + offset_load0;
        p_ddr[(ddr_regs_ptr+(reg_load1+reg_step*b))/4] = batch_base + ADDR(RESL);
        p_ddr[(ddr_regs_ptr+(reg_save0+reg_step*b))/4] = batch_base + ADDR(RESL);
      }
      else{
        p_ddr[(ddr_regs_ptr+(reg_load0+reg_step*b))/4] = batch_base + ADDR(RESL) + offset_load0;
        p_ddr[(ddr_regs_ptr+(reg_load1+reg_step*b))/4] = batch_base + ADDR(VECTOR);
        p_ddr[(ddr_regs_ptr+(reg_save0+reg_step*b))/4] = batch_base + ADDR(VECTOR);
      }
    }
  }

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
    << "update ddr";
  std::string addr_name = get_addr_name(); 
  CHECK(init_addr_map.count(addr_name) == 1) << "Can Not Find DDR Init Addr For " << addr_name;
  std::vector<size_t>& init_addrs = *init_addr_map[addr_name];

  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
    << "instruction: " << std::hex <<init_addrs[0]+ADDR(INSTR);
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
    << "p_ddr @" << p_ddr << "size: " << size;

  memory_->upload((void*)p_ddr, (size_t)(init_addrs[0]+ADDR(INSTR)), (size_t) size);
}

} // namespace xrnn
} // namespace vart
