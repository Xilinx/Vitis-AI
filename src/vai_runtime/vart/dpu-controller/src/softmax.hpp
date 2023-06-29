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

#define XSFM_CONTROL_AP 0x00
#define XSFM_CONTROL_AP_START 0x00
#define XSFM_CONTROL_GIE 0x04 /*GIE : GLOBAL INT ENABLE*/
#define XSFM_GLOBAL_INT_ENABLE BIT(0)

#define XSFM_SYS 0x20
#define XSFM_SYS_TIMESTAMP 0x24
#define XSFM_SYS_FREQ 0x28

#define XSFM_CONTROL_IER 0x40
#define XSFM_IP_INT_ENABLE BIT(0)
#define XSFM_X_LEN 0x44
#define XSFM_Y_LEN 0x48

#define XSFM_SRC_ADDR_L 0x4C
#define XSFM_SRC_ADDR_H 0x50
#define XSFM_DST_ADDR_L 0x54
#define XSFM_DST_ADDR_H 0x58

#define XSFM_SCALE 0x5C

#define XSFM_OFFSET 0x60
#define XFC_INPUT_CHANNEL_DIM 0x64
#define XFC_OUTPUT_CHANNEL_DIM 0x68
#define XFC_BATCH_DIM 0x6C
#define XFC_WEIGHT_START_ADDR 0x70
#define XFC_WEIGHT_END_ADDR 0x74

#define XSFM_CALC_MOD 0x78  /* 0-softmax, 1-fc */
#define XSFM_DST_ADDR_SEL 0x7C  /* 1-DDR, it should be 1 now */

#define XFC_RELU_EN 0x80
