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
// ==============================================================
// Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once
#define BIT(nr) (1 << (nr))

#define XDPU_CONTROL_AP 0x00
#define XDPU_CONTROL_AP_START 0x00
#define XDPU_CONTROL_GIE 0x04
#define XDPU_GLOBAL_INT_ENABLE BIT(0)
#define XDPU_CONTROL_IER 0x08
#define XDPU_IP_INT_ENABLE BIT(0)
#define XDPU_CONTROL_ISR 0x0C
#define XDPU_CONTROL_START 0x10  /* write 1 to enable DPU start */
#define XDPU_CONTROL_STATUS 0x18 /* write 1 to clear the finish status */
#define XDPU_CONTROL_RESET 0x1C  /* reset DPU active low */
#define XDPU_CONTROL_HP                                                        \
  0x20 /* [0:7]  outstanding of AW channel [15:8] outstanding of AR channel */
#define XDPU_CONTROL_MEAN0 0x24
#define XDPU_CONTROL_MEAN4 0x28
#define XDPU_CONTROL_MEAN8 0x2C
#define XDPU_CONTROL_MEAN12 0x30
#define XDPU_CONTROL_MEAN16 0x34
#define XDPU_CONTROL_MEAN20 0x38
#define XDPU_CONTROL_MEAN24 0x3C
#define XDPU_CONTROL_MEAN28 0x40
#define XDPU_CONTROL_DONE 0x80 /* 0 - dpu finish */
#define XDPU_DONE 0x0
#define XDPU_CONTROL_PEND_CNT 0x84
#define XDPU_CONTROL_CEND_CNT 0x88
#define XDPU_CONTROL_SEND_CNT 0x8C
#define XDPU_CONTROL_LEND_CNT 0x90
#define XDPU_CONTROL_PSTART_CNT 0x94
#define XDPU_CONTROL_CSTART_CNT 0x98
#define XDPU_CONTROL_SSTART_CNT 0x9C
#define XDPU_CONTROL_LSTART_CNT 0xA0
#define XDPU_CONTROL_CYCLE_CNT 0xA8
#define XDPU_CONTROL_ADDR_0_L 0x100
#define XDPU_CONTROL_ADDR_0_H 0x104
#define XDPU_CONTROL_ADDR_1_L 0x108
#define XDPU_CONTROL_ADDR_1_H 0x10C
#define XDPU_CONTROL_ADDR_2_L 0x110
#define XDPU_CONTROL_ADDR_2_H 0x114
#define XDPU_CONTROL_ADDR_3_L 0x118
#define XDPU_CONTROL_ADDR_3_H 0x11C
#define XDPU_CONTROL_ADDR_4_L 0x120
#define XDPU_CONTROL_ADDR_4_H 0x124
#define XDPU_CONTROL_ADDR_5_L 0x128
#define XDPU_CONTROL_ADDR_5_H 0x12C
#define XDPU_CONTROL_ADDR_6_L 0x130
#define XDPU_CONTROL_ADDR_6_H 0x134
#define XDPU_CONTROL_ADDR_7_L 0x138
#define XDPU_CONTROL_ADDR_7_H 0x13C
#define XDPU_CONTROL_INSTR_L 0x140
#define XDPU_CONTROL_INSTR_H 0x144
#define XDPU_HW_VER 0x1F0
#define XDPU_BIST_0 0x1F8
#define XDPU_BIST_1 0x1FC
#define XDPU_LAST 0x63c /* end of reg map */
#include <array>

constexpr std::array<int, 64> GEN_REG_OFFSET = {
    XDPU_CONTROL_ADDR_0_L / 4,  //
    XDPU_CONTROL_ADDR_1_L / 4,  //
    XDPU_CONTROL_ADDR_2_L / 4,  //
    XDPU_CONTROL_ADDR_3_L / 4,  //
    XDPU_CONTROL_ADDR_4_L / 4,  //
    XDPU_CONTROL_ADDR_5_L / 4,  //
    XDPU_CONTROL_ADDR_6_L / 4,  //
    XDPU_CONTROL_ADDR_7_L / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x100) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x100) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x200) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x200) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x300) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x300) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x400) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x400) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x500) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x500) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x600) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x600) / 4,  //

    (XDPU_CONTROL_ADDR_0_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_1_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_2_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_3_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_4_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_5_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_6_L + 0x700) / 4,  //
    (XDPU_CONTROL_ADDR_7_L + 0x700) / 4,  //
};

enum class DPU_CLOUD_TYPE { V3E, V4E, V3ME };
