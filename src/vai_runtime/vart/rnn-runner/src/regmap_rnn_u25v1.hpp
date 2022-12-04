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

#define LSB(value)           ((value) & 0xFFFF'FFFF)
#define HSB(value)           (((value)>>32) & 0xFFFF'FFFF)

#define U25_DEV_ADDR         (0x8'0000'0000)

#define REG_AP_CTRL            (0x0000)
#define REG_SOFT_RESET         (0x0010)
#define REG_INSTR_LOW_ADDR     (0x0014)
#define REG_INSTR_HIGH_ADDR    (0x0018)
#define REG_VECTOR_LOW_ADDR    (0x011C)
#define REG_VECTOR_HIGH_ADDR   (0x0120)
#define REG_BIAS_LOW_ADDR      (0x0124)   /* Not Used */
#define REG_BIAS_HIGH_ADDR     (0x0128)   /* Not Used */
#define REG_RESULT_LOW_ADDR    (0x012C)
#define REG_RESULT_HIGH_ADDR   (0x0130)
#define REG_PROF_ADDR          (0x0134)
#define REG_PROF_EN            (0x0138)
#define REG_CFG_DONE           (0x013C)
#define REG_VERSION            (0x0140)   /* Reserved */
#define REG_STATUS_0           (0x0148)   /* RO */
#define REG_STATUS_1           (0x014C)   /* RO */
#define REG_STATUS_2           (0x0150)   /* RO */
#define REG_STATUS_3           (0x0154)   /* RO */
#define REG_STATUS_4           (0x0158)   /* RO */
#define REG_STATUS_5           (0x015C)   /* RO */
#define REG_STATUS_6           (0x0160)   /* RO */
#define REG_STATUS_7           (0x0164)   /* RO */

#define DDR_WEIGHT             (0x0000'0000)
#define DDR_TANH               (0x0200'0000)
#define DDR_SGMD               (0x0210'0000)
#define DDR_VECTOR             (0x0300'0000)
#define DDR_BIAS               (0x0400'0000)
#define DDR_CELL               (0x0500'0000)
#define DDR_RESL               (0x0600'0000)
#define DDR_INSTR              (0x0700'0000)
#define DDR_PROF               (0x0800'0000)

#define WEIGHT_MAX_SIZE        (0x0200'0000)  // 32 MB
#define TANH_MAX_SIZE          (0x0010'0000)  //  1 MB
#define SGMD_MAX_SIZE          (0x0010'0000)  //  1 MB
#define VECTOR_MAX_SIZE        (0x0100'0000)  // 16 MB
#define BIAS_MAX_SIZE          (0x0100'0000)  // 16 MB
#define CELL_MAX_SIZE          (0x0100'0000)  // 16 MB
#define RESL_MAX_SIZE          (0x0100'0000)  // 16 MB
#define INSTR_MAX_SIZE         (0x0100'0000)  // 16 MB
#define PROF_MAX_SIZE          (0x0000'1000)  //  4 KB

#define THREAD_STEP            (0x0010'0000)  //  1 MB

#define MAX_REG_ADDR           (0x02C0)       // 704 B

// MODEL REGISTERS
#define REG_CONF_LAYER_NUM             (0x0168)
#define REG_CONF_FRAME_LEN             (0x016C)
#define REG_CONF_INSTR_FIRST_ADDR_0    (0x0170)
#define REG_CONF_INSTR_FIRST_ADDR_1    (0x0174)
#define REG_CONF_INSTR_FIRST_ADDR_2    (0x0178)
#define REG_CONF_INSTR_FIRST_ADDR_3    (0x017C)
#define REG_CONF_INSTR_FIRST_ADDR_4    (0x0180)
#define REG_CONF_INSTR_FIRST_ADDR_5    (0x0184)
#define REG_CONF_INSTR_FIRST_ADDR_6    (0x0188)
#define REG_CONF_INSTR_FIRST_ADDR_7    (0x018C)
#define REG_CONF_INSTR_FIRST_LEN_0     (0x0190)
#define REG_CONF_INSTR_FIRST_LEN_1     (0x0194)
#define REG_CONF_INSTR_FIRST_LEN_2     (0x0198)
#define REG_CONF_INSTR_FIRST_LEN_3     (0x019C)
#define REG_CONF_INSTR_FIRST_LEN_4     (0x01A0)
#define REG_CONF_INSTR_FIRST_LEN_5     (0x01A4)
#define REG_CONF_INSTR_FIRST_LEN_6     (0x01A8)
#define REG_CONF_INSTR_FIRST_LEN_7     (0x01AC)
#define REG_CONF_INSTR_LOOP_ADDR_0     (0x01B0)
#define REG_CONF_INSTR_LOOP_ADDR_1     (0x01B4)
#define REG_CONF_INSTR_LOOP_ADDR_2     (0x01B8)
#define REG_CONF_INSTR_LOOP_ADDR_3     (0x01BC)
#define REG_CONF_INSTR_LOOP_ADDR_4     (0x01C0)
#define REG_CONF_INSTR_LOOP_ADDR_5     (0x01C4)
#define REG_CONF_INSTR_LOOP_ADDR_6     (0x01C8)
#define REG_CONF_INSTR_LOOP_ADDR_7     (0x01CC)
#define REG_CONF_INSTR_LOOP_LEN_0      (0x01D0)
#define REG_CONF_INSTR_LOOP_LEN_1      (0x01D4)
#define REG_CONF_INSTR_LOOP_LEN_2      (0x01D8)
#define REG_CONF_INSTR_LOOP_LEN_3      (0x01DC)
#define REG_CONF_INSTR_LOOP_LEN_4      (0x01E0)
#define REG_CONF_INSTR_LOOP_LEN_5      (0x01E4)
#define REG_CONF_INSTR_LOOP_LEN_6      (0x01E8)
#define REG_CONF_INSTR_LOOP_LEN_7      (0x01EC)
#define REG_CONF_INSTR_END_ADDR        (0x01F0)
#define REG_CONF_ACTX_SIZE_0           (0x01F4)
#define REG_CONF_ACTX_SIZE_1           (0x01F8)
#define REG_CONF_ACTX_SIZE_2           (0x01FC)
#define REG_CONF_ACTX_SIZE_3           (0x0200)
#define REG_CONF_ACTX_SIZE_4           (0x0204)
#define REG_CONF_ACTX_SIZE_5           (0x0208)
#define REG_CONF_ACTX_SIZE_6           (0x020C)
#define REG_CONF_ACTX_SIZE_7           (0x0210)
#define REG_CONF_ACTH_SIZE_0           (0x0214)
#define REG_CONF_ACTH_SIZE_1           (0x0218)
#define REG_CONF_ACTH_SIZE_2           (0x021C)
#define REG_CONF_ACTH_SIZE_3           (0x0220)
#define REG_CONF_ACTH_SIZE_4           (0x0224)
#define REG_CONF_ACTH_SIZE_5           (0x0228)
#define REG_CONF_ACTH_SIZE_6           (0x022C)
#define REG_CONF_ACTH_SIZE_7           (0x0230)
#define REG_CONF_ACTX_ADDR_0           (0x0234)
#define REG_CONF_ACTX_ADDR_1           (0x0238)
#define REG_CONF_ACTX_ADDR_2           (0x023C)
#define REG_CONF_ACTX_ADDR_3           (0x0240)
#define REG_CONF_ACTX_ADDR_4           (0x0244)
#define REG_CONF_ACTX_ADDR_5           (0x0248)
#define REG_CONF_ACTX_ADDR_6           (0x024C)
#define REG_CONF_ACTX_ADDR_7           (0x0250)
#define REG_CONF_ACTH_ADDR_0           (0x0254)
#define REG_CONF_ACTH_ADDR_1           (0x0258)
#define REG_CONF_ACTH_ADDR_2           (0x025C)
#define REG_CONF_ACTH_ADDR_3           (0x0260)
#define REG_CONF_ACTH_ADDR_4           (0x0264)
#define REG_CONF_ACTH_ADDR_5           (0x0268)
#define REG_CONF_ACTH_ADDR_6           (0x026C)
#define REG_CONF_ACTH_ADDR_7           (0x0270)
#define REG_CONF_SAVE_ADDR_0           (0x0274)
#define REG_CONF_SAVE_ADDR_1           (0x0278)
#define REG_CONF_SAVE_ADDR_2           (0x027C)
#define REG_CONF_SAVE_ADDR_3           (0x0280)
#define REG_CONF_SAVE_ADDR_4           (0x0284)
#define REG_CONF_SAVE_ADDR_5           (0x0288)
#define REG_CONF_SAVE_ADDR_6           (0x028C)
#define REG_CONF_SAVE_ADDR_7           (0x0290)
#define REG_CONF_SAVE_FORWARD_0        (0x0294)
#define REG_CONF_SAVE_FORWARD_1        (0x0298)
#define REG_CONF_SAVE_FORWARD_2        (0x029C)
#define REG_CONF_SAVE_FORWARD_3        (0x02A0)
#define REG_CONF_SAVE_FORWARD_4        (0x02A4)
#define REG_CONF_SAVE_FORWARD_5        (0x02A8)
#define REG_CONF_SAVE_FORWARD_6        (0x02AC)
#define REG_CONF_SAVE_FORWARD_7        (0x02B0)


struct XRNN_REG_T
{
  uint32_t addr;
  uint32_t value;
};

static std::vector<size_t> U25_DDR_BASE_CU0{
  0x800000000,
};

static std::vector<size_t> U25_DDR_INIT_ADDR_CU0{
  0x800000000,
};
//Registers Fot U25
static std::vector<XRNN_REG_T> U25_SENTIMENT_REGS_CU0{
  {0x00000168, 0x00000001},
  {0x0000016c, 0x000001f4},
  {0x00000170, 0x07000000},
  {0x00000190, 0x0000013b},
  {0x000001b0, 0x07001400},
  {0x000001d0, 0x00000034},
  {0x000001f0, 0x07001800},
  {0x000001f4, 0x00000040},
  {0x00000214, 0x00000100},
  {0x00000294, 0x00000001},
  // dynamic
  {0x00000234, 0x03000000},
  {0x00000254, 0x06000000},
  {0x00000274, 0x06000000},
};

static std::vector<XRNN_REG_T> U25_SATISFACTION_REGS_CU0{
  {0x00000168, 0x00000001},
  {0x0000016c, 0x00000019},
  {0x00000170, 0x07000000},
  {0x00000190, 0x0000013b},
  {0x000001b0, 0x07001400},
  {0x000001d0, 0x00000034},
  {0x000001f0, 0x07001800},
  {0x000001f4, 0x00000040},
  {0x00000214, 0x00000100},
  {0x00000294, 0x00000001},
  // dynamic
  {0x00000234, 0x03000000},
  {0x00000254, 0x06000000},
  {0x00000274, 0x06000000},
};

static std::vector<XRNN_REG_T> U25_OPENIE_REGS_CU0{
  {0x00000168, 0x00000008},
  {0x0000016c, 0x00000000},
  {0x00000170, 0x07000000},
  {0x00000174, 0x07002900},
  {0x00000178, 0x07005200},
  {0x0000017c, 0x07007b00},
  {0x00000180, 0x0700a400},
  {0x00000184, 0x0700cd00},
  {0x00000188, 0x0700f600},
  {0x0000018c, 0x07011f00},
  {0x00000190, 0x000001fd},
  {0x00000194, 0x000001f9},
  {0x00000198, 0x000001f9},
  {0x0000019c, 0x000001f9},
  {0x000001a0, 0x000001f9},
  {0x000001a4, 0x000001f9},
  {0x000001a8, 0x000001f9},
  {0x000001ac, 0x000001f8},
  {0x000001b0, 0x07002000},
  {0x000001b4, 0x07004900},
  {0x000001b8, 0x07007200},
  {0x000001bc, 0x07009b00},
  {0x000001c0, 0x0700c400},
  {0x000001c4, 0x0700ed00},
  {0x000001c8, 0x07011600},
  {0x000001cc, 0x07013f00},
  {0x000001d0, 0x0000008e},
  {0x000001d4, 0x0000008e},
  {0x000001d8, 0x0000008e},
  {0x000001dc, 0x0000008e},
  {0x000001e0, 0x0000008e},
  {0x000001e4, 0x0000008e},
  {0x000001e8, 0x0000008e},
  {0x000001ec, 0x0000008e},
  {0x000001f0, 0x07014800},
  {0x000001f4, 0x000001c0},
  {0x000001f8, 0x00000280},
  {0x000001fc, 0x00000280},
  {0x00000200, 0x00000280},
  {0x00000204, 0x00000280},
  {0x00000208, 0x00000280},
  {0x0000020c, 0x00000280},
  {0x00000210, 0x00000280},
  {0x00000214, 0x00000280},
  {0x00000218, 0x00000280},
  {0x0000021c, 0x00000280},
  {0x00000220, 0x00000280},
  {0x00000224, 0x00000280},
  {0x00000228, 0x00000280},
  {0x0000022c, 0x00000280},
  {0x00000230, 0x00000280},
  {0x0000029c, 0x00000000},
  {0x000002a0, 0x00000000},
  {0x000002a4, 0x00000000},
  {0x000002a8, 0x00000000},
  {0x000002ac, 0x00000000},
  {0x000002b0, 0x00000001},
  // dynamic
  {0x00000234, 0x03000000},
  {0x00000238, 0x06000000},
  {0x0000023c, 0x03000000},
  {0x00000240, 0x06000000},
  {0x00000244, 0x03000000},
  {0x00000248, 0x06000000},
  {0x0000024c, 0x03000000},
  {0x00000250, 0x06000000},
  {0x00000254, 0x06000000},  // index 59:  cal + (frm_num-1)*value in 0x214
  {0x00000258, 0x03000000},
  {0x0000025c, 0x06000000},
  {0x00000260, 0x03000000},
  {0x00000264, 0x06000000},
  {0x00000268, 0x03000000},
  {0x0000026c, 0x06000000},
  {0x00000270, 0x03000000},
  {0x00000274, 0x06000000},
  {0x00000278, 0x03000000},
  {0x0000027c, 0x06000000},
  {0x00000280, 0x03000000},
  {0x00000284, 0x06000000},
  {0x00000288, 0x03000000},
  {0x0000028c, 0x06000000},
  {0x00000290, 0x03000000}, // index 74
};

// clang-format on
