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
// ==============================================================
// Copyright (C) 2016 Xilinx Inc. All rights reserved.
//
// ==============================================================

//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved

// clang-format off

#pragma once

#include <cstdint>
#include <vector>

// #pragma GCC diagnostic ignored "-Wunused-variable"

#define MAX_SIZE(type)      type##_MAX_SIZE
#define ADDR(type)          DDR_##type

#define LSB(value)           ((value) & 0xFFFFFFFF)
#define HSB(value)           (((value)>>32) & 0xFFFFFFFF)
#define IND(value)           ((value) / 4)

#define DEV_BASE_ADDR         (0x800000000)

#define REG_AP_CTRL               (0x0000)
#define REG_VERSION               (0x0010)
#define REG_SOFT_RESET            (0x0014)
#define REG_FRAME_LEN             (0x0018)
#define REG_INSTR_BASE_ADDR_H     (0x001C)
#define REG_INSTR_BASE_ADDR_L     (0x0020)
#define REG_MODEL_BASE_ADDR_L     (0x0024)
#define REG_INPUT_BASE_ADDR_H     (0x0028)
#define REG_INPUT_BASE_ADDR_L     (0x002C)
#define REG_OUTPUT_BASE_ADDR_L    (0x0030)
#define REG_PROF_START_ADDR_L     (0x0034)
#define REG_PROF_START_ADDR_H     (0x0038)
#define REG_PROF_EN               (0x003C)
#define REG_STATUS_0              (0x0040)
#define REG_STATUS_1              (0x0044)
#define REG_STATUS_2              (0x0048)
#define REG_STATUS_3              (0x004C)
#define REG_STATUS_4              (0x0150)
#define REG_STATUS_5              (0x0154)
#define REG_STATUS_6              (0x0158)
#define REG_STATUS_7              (0x015C)

#define DDR_WEIGHT             (0x0000'0000)
#define DDR_TANH               (0x0200'0000)
#define DDR_SGMD               (0x0210'0000)
#define DDR_VECTOR             (0x0300'0000)
#define DDR_BIAS               (0x0400'0000)
#define DDR_CELL               (0x0500'0000)
#define DDR_RESL               (0x0600'0000)
#define DDR_INSTR              (0x0700'0000)
#define DDR_PROF               (0x0800'0000)

#define BIAS_MAX_SIZE          (0x0100'0000)  // 16 MB
#define CELL_MAX_SIZE          (0x0100'0000)  // 16 MB
#define INSTR_MAX_SIZE         (0x0100'0000)  // 16 MB
#define PROF_MAX_SIZE          (0x0000'1000)  //  4 KB
#define RESL_MAX_SIZE          (0x0100'0000)  // 16 MB
#define SGMD_MAX_SIZE          (0x0010'0000)  //  1 MB
#define TANH_MAX_SIZE          (0x0010'0000)  //  1 MB
#define VECTOR_MAX_SIZE        (0x0100'0000)  // 16 MB
#define WEIGHT_MAX_SIZE        (0x0200'0000)  // 32 MB

#define THREAD_STEP            (0x0010'0000)  //  1 MB

#define MAX_REG_ADDR           (0x60)         // 96 B


struct XRNN_REG_T
{
  uint32_t addr;
  uint32_t value;
};


//Batch address for U50
static std::vector<size_t> U50_HBM_BATCH3_CU0{
  23 * 0x1000'0000UL,
  25 * 0x1000'0000UL,
  27 * 0x1000'0000UL,
  //  29 * 0x1000'0000UL,
};

static std::vector<size_t> U50_HBM_BATCH4_CU1{
  8  * 0x1000'0000UL,
  10 * 0x1000'0000UL,
  12 * 0x1000'0000UL,
  14 * 0x1000'0000UL,
};

static std::vector<size_t> U50_DDR_INIT_ADDR_CU0{
  0x1'0000'0000,
  0x1'1000'0000,
  0x1'2000'0000,
  0x1'3000'0000,
  0x1'5000'0000,
  0x1'7000'0000,
  0x1'9000'0000,
  0x1'b000'0000,
  0x1'd000'0000,
};

static std::vector<size_t> U50_DDR_INIT_ADDR_CU1{
  0x0000'0000,
  0x1000'0000,
  0x2000'0000,
  0x3000'0000,
  0x6000'0000,
  0x8000'0000,
  0xa000'0000,
  0xc000'0000,
  0xe000'0000,
};

//Registers Fot U50

static std::vector<XRNN_REG_T> U50_SENTIMENT_REGS_CU0{
  {REG_FRAME_LEN,          0x0000'01f4}, //frame_len
  {REG_INSTR_BASE_ADDR_H,  0x0000'0001}, //instr_addr_h
  {REG_INSTR_BASE_ADDR_L,  0x0700'0000}, //instr_addr_l
  {REG_MODEL_BASE_ADDR_L,  0x0000'0000}, //model_addr_l
  {REG_INPUT_BASE_ADDR_H,  0x0000'0001}, //input_addr_h
  {REG_PROF_START_ADDR_L,  0x0800'0000}, //prof_addr_l
  {REG_PROF_START_ADDR_H,  0x0000'0001}, //prof_addr_h
  {REG_PROF_EN,            0x0000'0001}, //prof_en
  {REG_INPUT_BASE_ADDR_L,  0x0000'0000}, //input_addr_l
  {REG_OUTPUT_BASE_ADDR_L, 0x0000'0000}, //output_addr_l
};

static std::vector<XRNN_REG_T> U50_SENTIMENT_REGS_CU1{
  {REG_FRAME_LEN,          0x0000'01f4},
  {REG_INSTR_BASE_ADDR_H,  0x0000'0000},
  {REG_INSTR_BASE_ADDR_L,  0x0700'0000},
  {REG_MODEL_BASE_ADDR_L,  0x0000'0000},
  {REG_INPUT_BASE_ADDR_H,  0x0000'0000},
  {REG_PROF_START_ADDR_L,  0x0800'0000},
  {REG_PROF_START_ADDR_H,  0x0000'0000},
  {REG_PROF_EN,            0x0000'0001},
  {REG_INPUT_BASE_ADDR_L,  0x0000'0000}, //input_addr_l
  {REG_OUTPUT_BASE_ADDR_L, 0x0000'0000},  //output_addr_l
};

static std::vector<XRNN_REG_T> U50_SATISFACTION_REGS_CU0{

  {0x00000018, 0x0000'0019}, //frame_len
  {0x0000001c, 0x0000'0001}, //instr_addr_h
  {0x00000020, 0x0700'0000}, //instr_addr_l
  {0x00000024, 0x0000'0000}, //model_addr_l
  {0x00000028, 0x0000'0001}, //input_addr_h
  {0x00000034, 0x0800'0000}, //prof_addr_l
  {0x00000038, 0x0000'0001}, //prof_addr_h
  {0x0000003c, 0x0000'0001}, //prof_en
  {0x0000002c, 0x0000'0000}, //input_addr_l
  {0x00000030, 0x0000'0000}  //output_addr_l
};

static std::vector<XRNN_REG_T> U50_SATISFACTION_REGS_CU1{
  {0x00000018, 0x0000'0019},
  {0x0000001c, 0x0000'0000},
  {0x00000020, 0x0700'0000},
  {0x00000024, 0x0000'0000},
  {0x00000028, 0x0000'0000},
  {0x00000034, 0x0800'0000},
  {0x00000038, 0x0000'0000},
  {0x0000003c, 0x0000'0001},
  {0x0000002c, 0x0000'0000}, //input_addr_l
  {0x00000030, 0x0000'0000}  //output_addr_l
};

static std::vector<XRNN_REG_T> U50_OPENIE_REGS_CU0{
  {0x00000018, 0x0000'003b}, //frame_len
  {0x0000001c, 0x0000'0001}, //instr_addr_h
  {0x00000020, 0x0700'0000}, //instr_addr_l
  {0x00000024, 0x0000'0000}, //model_addr_l
  {0x00000028, 0x0000'0001}, //input_addr_h
  {0x00000034, 0x0800'0000}, //prof_addr_l
  {0x00000038, 0x0000'0001}, //prof_addr_h
  {0x0000003c, 0x0000'0001}, //prof_en
  {0x0000002c, 0x0000'0000}, //input_addr_l
  {0x00000030, 0x0000'0000}  //output_addr_l
};
static std::vector<XRNN_REG_T> U50_OPENIE_REGS_CU1{
  {0x00000018, 0x0000'003b},
  {0x0000001c, 0x0000'0000},
  {0x00000020, 0x0700'0000},
  {0x00000024, 0x0000'0000},
  {0x00000028, 0x0000'0000},
  {0x00000034, 0x0800'0000},
  {0x00000038, 0x0000'0000},
  {0x0000003c, 0x0000'0001},
  {0x0000002c, 0x0000'0000}, //input_addr_l
  {0x00000030, 0x0000'0000}  //output_addr_l
};

// clang-format on
