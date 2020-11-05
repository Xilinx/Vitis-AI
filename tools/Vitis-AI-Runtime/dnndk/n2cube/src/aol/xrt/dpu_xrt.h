/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _DPUHAL_XRT_H_
#define _DPUHAL_XRT_H_

#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>

#define ENV_DPUCORE_MASK "DPU_COREMASK"
#define ENV_XCLBIN_PATH "DPU_XCLBIN_PATH"
#define ENV_TIMEOUT "DPU_TIMEOUT"

#define MAX_REG_NUM 32

// DPU Registers
#define DPU_CONFIG_REG_LEN 0x100

#define DPUREG_CTRL 0x00
#define DPUREG_GLBL_IRQ 0x04

#define DPUREG_VERSION 0x20
#define DPUREG_TIMESTAMP 0x24

#define DPUREG_GIT_COMMIT_ID 0x100

#define DPUREG_LOAD_START 0x180

#define DPU_CONFIGS_SHM_KEY 0x180717

typedef struct {
  volatile uint32_t AP_CTRL;     // 0x00;
  volatile uint32_t GIE;         // 0x04;
  volatile uint32_t IER;         // 0x08;
  volatile uint32_t ISR;         // 0x0c;
  volatile uint32_t rsv[12];     // rsvd
  volatile uint32_t DONECLR;     // 0x40;
  volatile uint32_t X_LEN;       // 0x44;
  volatile uint32_t Y_LEN;       // 0x48;
  volatile uint32_t ADDR_SRC_L;  // 0x4C
  volatile uint32_t ADDR_SRC_H;  // 0x50
  volatile uint32_t ADDR_DST_L;  // 0x54
  volatile uint32_t ADDR_DST_H;  // 0x58
  volatile uint32_t SCALE;       // 0x5C
  volatile uint32_t OFFSET;      // 0x60
  volatile uint32_t rsv2[3];

} SOFTMAX_REG;

uint32_t _get_dpu_config(int index);

void _set_dpu_config(int index, uint32_t val);

#define dprint(level, fmt, args...) \
  do {                              \
    printf("[DPU]" fmt, ##args);    \
  } while (0)

#define log_dpu(fmt, args...)         \
  do {                                \
    printf("[DPU_XRT] " fmt, ##args); \
  } while (0)

#define log_err(fmt, args...) printf("[DNNDK_XRT] " fmt, ##args);

#ifdef DPUDRV_DBG
#define log_dbg(fmt, args...) printf("[DNNDK_XRT] " fmt, ##args);
#else
#define log_dbg(fmt, args...)
#endif

#ifdef DPUDRV_INFO
#define log_info(fmt, args...) printf("[DNNDK_XRT] " fmt, ##args);
#else
#define log_info(fmt, args...)
#endif

#define DPU_CORE_MAX (16)

// dpu features structure
typedef struct {
  union {
    uint32_t git_id : 28;
    uint32_t rsv0 : 4;
  };
  uint32_t git_time;
  union {
    uint32_t version;
    struct {
      uint32_t ver_target : 12;
      uint32_t ver_ip : 8;
      uint32_t rsv1 : 12;
    };
  };
  union {
    uint32_t timer;
    struct {
      uint32_t timer_val : 12;  // hour * freq / 10
      uint32_t timer_enable : 1;
      uint32_t rsv2 : 19;
    };
  };
  union {
    uint32_t arch;
    struct {
      uint32_t arch_icp : 8;
      uint32_t arch_ocp : 8;
      uint32_t arch_pp : 4;
      uint32_t arch_img_bg : 4;
      uint32_t arch_data_width : 4;
      uint32_t arch_hp_width : 4;
    };
  };
  union {
    uint32_t ram;
    struct {
      uint32_t ram_depth_bias : 4;
      uint32_t ram_depth_weight : 4;
      uint32_t ram_depth_img : 4;
      uint32_t ram_depth_mean : 4;
      uint32_t rsv3 : 16;
    };
  };
  union {
    uint32_t load;
    struct {
      uint32_t load_parallel : 4;
      uint32_t load_mean_en : 4;
      uint32_t load_augm_en : 4;
      uint32_t rsv4 : 20;
    };
  };
  union {
    uint32_t conv;
    struct {
      uint32_t conv_wr_parallel : 4;
      uint32_t conv_relu6_en : 4;
      uint32_t conv_leakyrelu_en : 4;
      uint32_t rsv5 : 20;
    };
  };
  union {
    struct {
      uint32_t save_parallel : 4;
      uint32_t rsv6 : 28;
    };
  };
  union {
    uint32_t pool;
    struct {
      uint32_t pool_avg_en : 1;
      uint32_t rsv7 : 31;
    };
  };
  union {
    uint32_t elew;
    struct {
      uint32_t elew_parallel : 4;
      uint32_t rsv8 : 28;
    };
  };
  union {
    uint32_t dwcv;
    struct {
      uint32_t dwcv_parallel : 4;
      uint32_t dwcv_relu6_en : 4;
      uint32_t dwcv_alu_en : 4;
      uint32_t rsv9 : 20;
    };
  };
  union {
    uint32_t misc;
    struct {
      uint32_t misc_wr_parallel : 4;
      uint32_t rsv10 : 28;
    };
  };
} dpu_feature_t;

// dpu configuration
typedef struct _dpu_config {
  uint64_t base_addr;
  uint32_t version;
  uint32_t arch;
  uint32_t freq;
  uint32_t cu_index;
  dpu_feature_t feature;
} dpu_conf_t;

// softmax configuration
typedef struct _softmax_config {
  uint64_t base_addr;
  uint32_t version;
  uint32_t arch;
  uint32_t freq;
  uint32_t cu_index;
} sm_conf_t;

#define MAX_CORE_NUM 16
typedef struct _sys_config {
  uint32_t rt_mode;
  uint32_t dpu_core_num;
  uint32_t dpu_core_mask;
  dpu_conf_t dpu_conf[MAX_CORE_NUM];
  uint32_t sm_core_num;
  uint32_t sm_core_mask;
  sm_conf_t sm_conf[MAX_CORE_NUM];
} sys_conf_t;

typedef struct {
  dpu_aol_dev_mem_t aol_mem;
  unsigned bo;
} xrt_mem_t;

#endif /*_DPU_H_*/
