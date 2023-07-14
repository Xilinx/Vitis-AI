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
#include "./dpu_control_uio.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits>
#include <vitis/ai/weak.hpp>

// TODO: make it configable
static size_t dpu_reg_base_addr() { return 0x8F000000; }

static size_t dpu_reg_base_size() { return 0x1000; }

std::shared_ptr<vitis::ai::DpuController> DpuControllerUio::getInstance() {
  return std::shared_ptr<vitis::ai::DpuController>(
      vitis::ai::WeakSingleton<DpuControllerUio>::create());
}

DpuControllerUio::DpuControllerUio()
    : vitis::ai::DpuController{},
      dpu_driver_{UioDriver<DpuDriver>::create()},
      dpu_reg_map_{vitis::ai::buffer_object_map::create(
          vitis::ai::buffer_object_fd::create("/dev/mem", O_RDWR | O_SYNC),
          dpu_reg_base_addr(), dpu_reg_base_size())} {}

DpuControllerUio::~DpuControllerUio() {}

constexpr int MAX_TRIES = 1000;
typedef struct __DPUReg {
  /*dpu pmu registers*/
  struct __regs_dpu_pmu {
    volatile uint32_t version;
    volatile uint32_t reset;
    volatile uint32_t _rsv[62];
  } pmu;

  /*dpu rgbout registers*/
  struct __regs_dpu_rgbout {
    volatile uint32_t display;
    volatile uint32_t _rsv[63];
  } rgbout;

  /*dpu control registers struct*/
  struct __regs_dpu_ctrl {
    volatile uint32_t hp_ctrl;
    volatile uint32_t addr_io;
    volatile uint32_t addr_weight;
    volatile uint32_t addr_code;
    volatile uint32_t addr_prof;

    volatile uint32_t prof_value;
    volatile uint32_t prof_num;
    volatile uint32_t prof_en;
    volatile uint32_t start;
    volatile uint32_t com_addr[16];  //< extension for DPUv1.3.0
    volatile uint32_t pend_cnt;
    volatile uint32_t cend_cnt;
    volatile uint32_t send_cnt;
    volatile uint32_t lend_cnt;
    volatile uint32_t pstart_cnt;
    volatile uint32_t cstart_cnt;
    volatile uint32_t sstart_cnt;
    volatile uint32_t lstart_cnt;
    volatile uint32_t _rsv[31];

  } ctlreg[DpuDriver::MAX_NUM_OF_DEVICES];

  /*dpu interrupt registers struct*/
  struct __regs_dpu_intr {
    volatile uint32_t isr;
    volatile uint32_t imr;
    volatile uint32_t irsr;
    volatile uint32_t icr;
    volatile uint32_t _rsv[60];
  } intreg;
} DPUReg;

int DpuControllerUio::RunDpu(int dpu_index, const DpuRegs& regs) {
  auto reg = dpu_reg_map_->get<DPUReg>();
  auto ctlreg = &reg->ctlreg[dpu_index];
  CHECK_EQ(regs.code & 0xFFF, 0)
      << " regs.code must be 4096 aligned. code = 0x" << std::hex << regs.code;
  LOG(INFO) << "running code at " << std::hex << "0x" << regs.code << " 0x"
            << (regs.code >> 12) << std::dec;
  ctlreg->addr_code = (regs.code >> 12);
  ctlreg->com_addr[0] = regs.regs[0];
  ctlreg->com_addr[1] = regs.regs[8];
  ctlreg->com_addr[2] = regs.regs[1];
  ctlreg->com_addr[3] = regs.regs[9];
  ctlreg->com_addr[4] = regs.regs[2];
  ctlreg->com_addr[5] = regs.regs[10];
  ctlreg->com_addr[6] = regs.regs[3];
  ctlreg->com_addr[7] = regs.regs[11];
  ctlreg->com_addr[8] = regs.regs[4];
  ctlreg->com_addr[9] = regs.regs[12];
  ctlreg->com_addr[10] = regs.regs[5];
  ctlreg->com_addr[11] = regs.regs[13];
  ctlreg->com_addr[12] = regs.regs[6];
  ctlreg->com_addr[13] = regs.regs[14];
  ctlreg->com_addr[14] = regs.regs[7];
  ctlreg->com_addr[15] = regs.regs[15];
  if (0) {
    volatile uint32_t pend_cnt = ctlreg->pend_cnt;
    volatile uint32_t cend_cnt = ctlreg->cend_cnt;
    volatile uint32_t send_cnt = ctlreg->send_cnt;
    volatile uint32_t lend_cnt = ctlreg->lend_cnt;
    volatile uint32_t pstart_cnt = ctlreg->pstart_cnt;
    volatile uint32_t cstart_cnt = ctlreg->cstart_cnt;
    volatile uint32_t sstart_cnt = ctlreg->sstart_cnt;
    volatile uint32_t lstart_cnt = ctlreg->lstart_cnt;
    volatile uint32_t irsr = reg->intreg.irsr;

    LOG(INFO) << "before start "
              << "irsr 0x" << irsr << " "
              << "lstart_cnt 0x" << lstart_cnt << " "
              << "lend_cnt 0x" << lend_cnt << " "
              << "cstart_cnt 0x" << cstart_cnt << " "
              << "cend_cnt 0x" << cend_cnt << " "
              << "sstart_cnt 0x" << sstart_cnt << " "
              << "send_cnt 0x" << send_cnt << " "
              << "pstart_cnt 0x" << pstart_cnt << " "
              << "pend_cnt 0x" << pend_cnt << " ";
  }
  dpu_driver_->EnableIrq(dpu_index);
  ctlreg->prof_en = 0;
  ctlreg->start = 1;
  auto r = dpu_driver_->WaitForIrq(dpu_index);
  LOG_IF(WARNING, r != 0) << " dpu[" << dpu_index
                          << "] cannot read irq from uio, err="
                          << strerror(errno);
  auto irsr = reg->intreg.irsr;
  auto finish = irsr & (1 << dpu_index);
  LOG_IF(WARNING, !finish)
      << " dpu[" << dpu_index << "] "                             //
      << " is not finished properly, hardware bug or dnnc bug? "  //
      << std::hex                                                 //
      << " finish=0x" << finish                                   //
      << " irsr=0x" << irsr <<                                    //
      std::dec;
  if (!finish) {
    for (int tries = 0; tries < MAX_TRIES; ++tries) {
      irsr = reg->intreg.irsr;
      finish = irsr & (1 << dpu_index);
      if (finish) {
        LOG(INFO) << " GOOD FINISHED = 1 irsr = " << irsr;
        break;
      }
      usleep(5 * 1000);
    }
  }
  if (!finish) {
    volatile uint32_t pend_cnt = ctlreg->pend_cnt;
    volatile uint32_t cend_cnt = ctlreg->cend_cnt;
    volatile uint32_t send_cnt = ctlreg->send_cnt;
    volatile uint32_t lend_cnt = ctlreg->lend_cnt;
    volatile uint32_t pstart_cnt = ctlreg->pstart_cnt;
    volatile uint32_t cstart_cnt = ctlreg->cstart_cnt;
    volatile uint32_t sstart_cnt = ctlreg->sstart_cnt;
    volatile uint32_t lstart_cnt = ctlreg->lstart_cnt;
    volatile uint32_t irsr = reg->intreg.irsr;
    LOG(WARNING) << "after start " << std::hex  //

                 << "irsr 0x" << irsr << " "
                 << "lstart_cnt 0x" << lstart_cnt << " "
                 << "lend_cnt 0x" << lend_cnt << " "
                 << "cstart_cnt 0x" << cstart_cnt << " "
                 << "cend_cnt 0x" << cend_cnt << " "
                 << "sstart_cnt 0x" << sstart_cnt << " "
                 << "send_cnt 0x" << send_cnt << " "
                 << "pstart_cnt 0x" << pstart_cnt << " "
                 << "pend_cnt 0x" << pend_cnt << " " << std::dec;
  }

  // must disable start register;
  ctlreg->start = 0;
  // clear interrupt results
  reg->intreg.icr = (1 << dpu_index);
  for (int tries = 0; tries < MAX_TRIES; ++tries) {
    uint32_t irsr = reg->intreg.irsr;
    finish = irsr & (1 << dpu_index);
    if (!finish) {
      break;
    }
  }
  irsr = reg->intreg.irsr;
  LOG_IF(WARNING, finish) << " dpu[" << dpu_index << "]"
                          << "cannot clear finish state properly, hardware bug?"
                          << std::hex << "0x" << irsr << std::dec;
  ;
  reg->intreg.icr = 0;
  irsr = reg->intreg.irsr;
  LOG_IF(INFO, false) << "after clean up "
                      << " dpu[" << dpu_index << "]" << std::hex << " irsr=0x"
                      << irsr << std::dec;

  return 0;
}

void DpuControllerUio::run(const uint64_t workspace, const uint64_t code,
                           const uint64_t parameter) {
  LOG(INFO) << std::hex                            //
            << "workspace 0x" << workspace << " "  //
            << "code 0x" << code << " "            //
            << "parameter 0x" << parameter << " "  //
            << std::dec                            //
      ;
  // if(1) {
  //   return;
  // }
  auto dpu_index = dpu_driver_->AllocDevice();
  CHECK_LT(code, std::numeric_limits<uint64_t>::max());
  CHECK_EQ(code & 0xFFF, 0)
      << " code must be 4096 aligned. code = 0x" << std::hex << code;
  CHECK_LT(parameter, std::numeric_limits<uint64_t>::max());
  CHECK_LT(workspace, std::numeric_limits<uint64_t>::max());
  auto regs = DpuRegs{(uint64_t)code, (uint64_t)parameter, (uint64_t)workspace,
                      (uint64_t)code};
  if (1) {
    LOG(INFO) << "dpu " << dpu_index << " is allocated, and start dpu task, "
              << std::hex                  //
              << "0x" << workspace << " "  //
              << "0x" << code << " "       //
              << "0x" << parameter << " "  //
              << std::dec;
  }
  auto r = RunDpu(dpu_index, regs);
  CHECK_EQ(r, 0);
  dpu_driver_->FreeDevice(dpu_index);
  this->
}
