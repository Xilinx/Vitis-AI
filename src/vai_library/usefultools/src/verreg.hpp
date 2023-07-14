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
#pragma once
#include <cstdint>
struct VerReg {
  // 0x0
  volatile union {
    struct {
      unsigned int m : 16;
      unsigned int ver_fmt : 8;
      unsigned int n_of_reg : 8;
    };
    uint32_t u32;
  } x0;
  // 0x4
  volatile union {
    struct {
      unsigned int encryption : 1;
      unsigned int frequency : 10;
      unsigned int year : 5;
      unsigned int month : 4;
      unsigned int day : 5;
      unsigned int hour : 5;
      unsigned int bit_ver : 2;
    };
    uint32_t u32;
  } x4;

  // 0x8
  volatile union {
    struct {
      unsigned int interrupt_base0 : 8;
      unsigned int interrupt_base1 : 8;
      unsigned int reserved : 16;
    };
    uint32_t u32;
  } x8;

  // 0xc
  volatile union {
    struct {
      unsigned int dpu_core_number : 4;
      unsigned int dpu_hp_interact : 4;
      unsigned int dpu_target_version : 8;
      unsigned int arch_type : 4;
      unsigned int bank_group : 4;
      unsigned int data_width : 4;
      unsigned int hp_width : 4;
    };
    uint32_t u32;
  } xc;

  // 0x10
  volatile union {
    struct {
      unsigned int dpu0_interrupt_number : 4;
      unsigned int dpu1_interrupt_number : 4;
      unsigned int dpu2_interrupt_number : 4;
      unsigned int dpu3_interrupt_number : 4;
      unsigned int dpu4_interrupt_number : 4;
      unsigned int dpu5_interrupt_number : 4;
      unsigned int dpu6_interrupt_number : 4;
      unsigned int dpu7_interrupt_number : 4;
    };
    uint32_t u32;
  } x10;

  // 0x14
  volatile union {
    struct {
      unsigned int dpu8_interrupt_number : 4;
      unsigned int dpu9_interrupt_number : 4;
      unsigned int dpu10_interrupt_number : 4;
      unsigned int dpu11_interrupt_number : 4;
      unsigned int dpu12_interrupt_number : 4;
      unsigned int dpu13_interrupt_number : 4;
      unsigned int dpu14_interrupt_number : 4;
      unsigned int dpu15_interrupt_number : 4;
    };
    uint32_t u32;
  } x14;

  // 0x18
  volatile union {
    struct {
      unsigned int pre_relu : 4;
      unsigned int relu_addon : 4;
      unsigned int depthwise_conv : 8;
      unsigned int average_pool : 16;
    };
    uint32_t u32;
  } x18;

  // 0x1c
  volatile union {
    struct {
      unsigned int nl_ratio_index : 4;
      unsigned int load_img_mean : 4;
      unsigned int load_augm : 4;
      unsigned int pool_parallel : 4;
      unsigned int reserved : 16;
    };
    uint32_t u32;
  } x1c;

  // 0x20
  volatile union {
    struct {
      unsigned int f7 : 4;
      unsigned int f6 : 4;
      unsigned int f5 : 4;
      unsigned int f4 : 4;
      unsigned int f3 : 4;
      unsigned int f2 : 4;
      unsigned int f1 : 4;
      unsigned int f0 : 4;
    };
    uint32_t u32;
  } x20;
  // 0x24
  volatile union {
    struct {
      unsigned int hdmi_valid : 1;
      unsigned int hdmi_version : 3;
      unsigned int hdmi_interrupt_number : 4;
      unsigned int bt1120_valid : 1;
      unsigned int bt1120_version : 3;
      unsigned int bt1120_interrupt_number : 4;
      unsigned int fc_valid : 1;
      unsigned int fc_version : 3;
      unsigned int fc_interrupt_number : 4;
      unsigned int softmax_valid : 1;
      unsigned int softmax_version : 3;
      unsigned int softmax_interrupt_number : 4;
    };
    uint32_t u32;
  } x24;

  // 0x28
  volatile union {
    struct {
      unsigned int bgr2yuv_valid : 1;
      unsigned int bgr2yuv_version : 3;
      unsigned int bgr2yuv_interrupt_number : 4;
      unsigned int yuv2rgb_valid : 1;
      unsigned int yuv2rgb_version : 3;
      unsigned int yuv2rgb_interrupt_number : 4;
      unsigned int sigmoid_valid : 1;
      unsigned int sigmoid_version : 3;
      unsigned int sigmoid_interrupt_number : 4;
      unsigned int resize_valid : 1;
      unsigned int resize_version : 3;
      unsigned int resize_interrupt_number : 4;
    };
    uint32_t u32;
  } x28;

  // 0x2c
  volatile union {
    struct {
      unsigned int board_number : 16;
      unsigned int board_type : 4;
      unsigned int board_hw_version : 4;
      unsigned int chip_part : 4;
      unsigned int auth_en : 1;
      unsigned int reserved : 3;
    };
    uint32_t u32;
  } x2c;
};
