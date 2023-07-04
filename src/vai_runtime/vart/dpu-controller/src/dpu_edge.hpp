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

#define BIT(nr) (1 << (nr))

#define XDPU_CONTROL_AP 0x00
#define XDPU_CONTROL_AP_START 0x00
#define XDPU_CONTROL_GIE 0x04 /*GIE : GLOBAL INT ENABLE*/
#define XDPU_GLOBAL_INT_ENABLE BIT(0)

#define XDPU_SYS 0x20
#define XDPU_SYS_TIMESTAMP 0x24
#define XDPU_SYS_FREQ 0x28

#define XDPU_CONTROL_IER 0x40
#define XDPU_IP_INT_ENABLE BIT(0)
#define XDPU_CONTROL_PROF_ENA 0x44

#define XDPU_CONTROL_HP 0x48
#define XDPU_CONTROL_ADDR_INSTR_L 0x50
#define XDPU_CONTROL_ADDR_INSTR_H 0x54
#define XDPU_CONTROL_ADDR_PROF_L 0x58
#define XDPU_CONTROL_ADDR_PROF_H 0x5C
#define XDPU_CONTROL_ADDR_0_L 0x60
#define XDPU_CONTROL_ADDR_0_H 0x64
#define XDPU_CONTROL_ADDR_1_L 0x68
#define XDPU_CONTROL_ADDR_1_H 0x6C
#define XDPU_CONTROL_ADDR_2_L 0x70
#define XDPU_CONTROL_ADDR_2_H 0x74
#define XDPU_CONTROL_ADDR_3_L 0x78
#define XDPU_CONTROL_ADDR_3_H 0x7C
#define XDPU_CONTROL_ADDR_4_L 0x80
#define XDPU_CONTROL_ADDR_4_H 0x84
#define XDPU_CONTROL_ADDR_5_L 0x88
#define XDPU_CONTROL_ADDR_5_H 0x8C
#define XDPU_CONTROL_ADDR_6_L 0x90
#define XDPU_CONTROL_ADDR_6_H 0x94
#define XDPU_CONTROL_ADDR_7_L 0x98
#define XDPU_CONTROL_ADDR_7_H 0x9C

#define XDPU_GIT_COMMIT_ID 0x100
#define XDPU_GIT_COMMIT_TIME 0x104
#define XDPU_SUB_VERSION 0x108
#define XDPU_TIMER 0x108
#define XDPU_ARCH 0x110
#define XDPU_RAM 0x114
#define XDPU_LOAD 0x118
#define XDPU_CONV 0x11C
#define XDPU_SAVE 0x120
#define XDPU_POOL 0x124
#define XDPU_DWCV 0x12C
#define XDPU_MISC 0x130
#define XDPU_RSVD 0x134

#define XDPU_DBG_LCNT_START 0x180
#define XDPU_DBG_LCNT_END 0x184
#define XDPU_DBG_CCNT_START 0x188
#define XDPU_DBG_CCNT_END 0x18C
#define XDPU_DBG_SCNT_START 0x190
#define XDPU_DBG_SCNT_END 0x194
#define XDPU_DBG_MCNT_START 0x198
#define XDPU_DBG_MCNT_END 0x19C
#define XDPU_EXEC_CYCLE_L 0x1A0
#define XDPU_EXEC_CYCLE_H 0x1A4
#define XDPU_DPU_STS 0x1A8
#define XDPU_DBG_AXI_STS 0x1AC

/*
#define XDPU_CONTROL_GIE 0x04
#define XDPU_GLOBAL_INT_ENABLE BIT(0)
#define XDPU_CONTROL_IER 0x08
#define XDPU_IP_INT_ENABLE BIT(0)
#define XDPU_CONTROL_ISR 0x0C
#define XDPU_CONTROL_START 0x10
#define XDPU_CONTROL_STATUS 0x18
#define XDPU_CONTROL_RESET 0x1C
#define XDPU_CONTROL_HP    0x20
#define XDPU_CONTROL_MEAN0 0x24
#define XDPU_CONTROL_MEAN4 0x28
#define XDPU_CONTROL_MEAN8 0x2C
#define XDPU_CONTROL_MEAN12 0x30
#define XDPU_CONTROL_MEAN16 0x34
#define XDPU_CONTROL_MEAN20 0x38
#define XDPU_CONTROL_MEAN24 0x3C
#define XDPU_CONTROL_MEAN28 0x40
#define XDPU_CONTROL_DONE 0x80
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
#define XDPU_CONTROL_LENGTH_DATA 0x148
#define XDPU_HW_VER 0x1F0
#define XDPU_BIST_0 0x1F8
#define XDPU_BIST_1 0x1FC
*/
