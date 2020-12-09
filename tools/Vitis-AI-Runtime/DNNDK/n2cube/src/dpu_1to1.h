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

#ifndef __DPU_1to1_H__
#define __DPU_1to1_H__

typedef struct {
    struct {
        uint32_t AP_START : 1;
        uint32_t AP_DONE : 1;
        uint32_t AP_IDLE : 1;
        uint32_t AP_READY : 1;
        uint32_t Reserved0 : 1;
        uint32_t AP_RESET : 1;
        uint32_t AP_RESET_DONE : 1;
        uint32_t Reserved1 : 25;
    }CTRL; //0x00
    struct {
        uint32_t GLBL_IRQ_ENA : 1;
        uint32_t Reserved : 31;
    }GLBL_IRQ; //0x04
    uint32_t IER; //?
    uint32_t ISR; //?
}dpu_1to1_ap_reg_t;

typedef struct {
    struct {
        uint32_t SYS_REGMAP_VER : 8;
        uint32_t SYS_REGMAP_SIZE : 8;
        uint32_t SYS_IP_TYPE : 8;
        uint32_t SYS_IP_VER : 8;
    }SYS;
    struct {
        uint32_t TIME_QUARTER : 4;
        uint32_t TIME_HOUR : 8;
        uint32_t TIME_DAY : 8;
        uint32_t TIME_MONTH : 4;
        uint32_t TIME_YEAR : 8;
    }TIMESTAMP;
    struct {
        uint32_t M_AXI_FREQ_MHZ : 12;
        uint32_t S_AXI_FREQ_MHZ : 12;
    }FREQ;
} dpu_1to1_sys_reg_t;

typedef struct {
    struct {
        uint32_t IRQ_CLR : 1;
        uint32_t Reserved : 31;
    }IRQ_CLR; //0x40
    struct {
        uint32_t PROF_ENA : 1;
        uint32_t Reserved : 31;
    }PROF_ENA; //0x44
    struct {
        uint32_t HP_ARLEN : 8;
        uint32_t HP_AWLEN : 8;
        uint32_t HP_ARCOUNT_MAX : 8;
        uint32_t HP_AWCOUNT_MAX : 8;
    }HP_BUS; // 0x48
    struct {
        uint32_t Reserved;
    }RSVD; // 0x4C
    uint32_t INSTR_ADDR_L; // 0x50
    uint32_t INSTR_ADDR_H; // 0x54
    uint32_t PROF_ADDR_L; // 0x58
    uint32_t PROF_ADDR_H; // 0x5C
    uint32_t BASE_ADDR_0_L; // 0x60
    uint32_t BASE_ADDR_0_H; // 0x64
    uint32_t BASE_ADDR_1_L; // 0x68
    uint32_t BASE_ADDR_1_H; // 0x6C
    uint32_t BASE_ADDR_2_L; // 0x70
    uint32_t BASE_ADDR_2_H; // 0x74
    uint32_t BASE_ADDR_3_L; // 0x78
    uint32_t BASE_ADDR_3_H; // 0x7C
    uint32_t BASE_ADDR_4_L; // 0x80
    uint32_t BASE_ADDR_4_H; // 0x84
    uint32_t BASE_ADDR_5_L; // 0x88
    uint32_t BASE_ADDR_5_H; // 0x8C
    uint32_t BASE_ADDR_6_L; // 0x90
    uint32_t BASE_ADDR_6_H; // 0x94
    uint32_t BASE_ADDR_7_L; // 0x98
    uint32_t BASE_ADDR_7_H; // 0x9C
} dpu_1to1_configurable_reg_t; // length is 96 bytes

// dpu features structure
typedef struct {
    struct {
        uint32_t GIT_COMMIT_ID : 28;
        uint32_t Reserved : 4;
    }GIT_COMMIT_ID; //0x100
    uint32_t GIT_COMMIT_TIME; //0x104
    struct {
        uint32_t VER_TARGET : 12;
        uint32_t VER_IP_REV : 8;
        uint32_t Reserved : 12;
    }SUB_VERSION; //0x108
    struct {
        uint32_t TIMER_HOUR : 12;
        uint32_t TIMER_ENA : 1;
        uint32_t Reserved : 19;
    }TIMER; //0x10C
    struct {
        uint32_t ARCH_OCP : 8;
        uint32_t ARCH_ICP : 8;
        uint32_t ARCH_PP : 4;
        uint32_t ARCH_IMG_BKGRP : 4;
        uint32_t ARCH_DATA_BW : 4;
        uint32_t ARCH_HP_BW : 4;
    }ARCH; //0x110
    struct {
        uint32_t RAM_DEPTH_IMG : 4;
        uint32_t RAM_DEPTH_WGT : 4;
        uint32_t RAM_DEPTH_BIAS : 4;
        uint32_t RAM_DEPTH_MEAN : 4;
        uint32_t Reserved : 16;
    }RAM; //0x114
    struct {
        uint32_t LOAD_PARALLEL : 4;
        uint32_t LOAD_IMG_MEAN : 4;
        uint32_t LOAD_AUGM : 4;
        uint32_t Reserved : 20;
    }LOAD; //0x118
    struct {
        uint32_t CONV_WR_PARALLEL : 4;
        uint32_t CONV_RELU6 : 4;
        uint32_t CONV_LEAKYRELU : 4;
        uint32_t Reserved : 20;
    }CONV; // 0x11C
    struct {
        uint32_t SAVE_PARALLEL : 4;
        uint32_t Reserved : 28;
    }SAVE; //0x120
    struct {
        uint32_t POOL_AVERAGE : 1;
        uint32_t Reserved : 31;
    }POOL; //0x124
    struct {
        uint32_t ELEW_PARALLEL : 4;
        uint32_t Reserved : 28;
    }ELEW; //0x128
    struct {
        uint32_t DWCV_PARALLEL : 4;
        uint32_t DWCV_RELU6 : 4;
        uint32_t DWCV_ALU_MODE : 4;
        uint32_t Reserved : 20;
    }DWCV; //0x12C
    struct {
      uint32_t MISC_WR_PARALLEL : 4;
      uint32_t Reserved : 28;
    }MISC; //0x130
} dpu_1to1_feature_reg_t;

typedef struct _dpureg {
  uint32_t APCTL;  // 0x00
  uint32_t GIE;
  uint32_t IER;
  uint32_t ISR;
  uint32_t rsv[4];
  uint32_t version;    // 0x20
  uint32_t timestamp;  // 0x24
  uint32_t rsv1[6];
  uint32_t irq_clr;  // 0x40;
  uint32_t prof_en;  // 0x44
  uint32_t hp_bus;   // 0x48
  uint32_t rsv2;

  uint64_t addr_code;  // 0x50
  uint64_t addr_prof;  // 0x58
  uint64_t addr0;
  uint64_t addr1;
  uint64_t addr2;
  uint64_t addr3;
  uint64_t addr4;
  uint64_t addr5;
  uint64_t addr6;
  uint64_t addr7;

  uint32_t rsv3[24];

  uint32_t GIT_COMMIT_ID;    // 0x100
  uint32_t GIT_COMMIT_TIME;  // 0x104
  uint32_t SUB_VERSION;      // 0x108
  uint32_t TIMER;            // 0x10c
  uint32_t ARCH;             // 0x110
  uint32_t CONF_RAM;         // 0x114
  uint32_t CONF_LOAD;        // 0x118
  uint32_t CONF_CONV;        // 0x11c
  uint32_t CONF_SAVE;        // 0x120
  uint32_t CONF_POOL;        // 0x124
  uint32_t CONF_ELEW;        // 0x128
  uint32_t CONF_DWCV;        // 0x12c
  uint32_t CONF_MISC;        // 0x130

  uint32_t rsv4[19];

  uint32_t load_start;  // 0x180
  uint32_t load_end;
  uint32_t conv_start;
  uint32_t conv_end;
  uint32_t save_start;
  uint32_t save_end;
  uint32_t misc_start;
  uint32_t misc_end;
  uint32_t cycle_h;
  uint32_t cycle_l;
  uint32_t dpu_status;
  uint32_t axi_status;  // 0x1ac

} DPU_Reg;

#define OFFSET_1t01_DPU_CTRL 0x00
#define OFFSET_1t01_DPU_GLBL_IRQ 0x04
#define OFFSET_1t01_DPU_IER 0x08
#define OFFSET_1t01_DPU_IRQ_CLR 0x40
#define OFFSET_1t01_DPU_HP_BUS 0x48
#define OFFSET_1t01_DPU_INSTR_ADDR_L 0x50
#define OFFSET_1to1_DPU_BASE_ADDR_0_L 0x60
#define OFFSET_1to1_DPU_BASE_ADDR_0_H 0x64

#define OFFSET_SMFC_CRTL 0x00
#define OFFSET_SMFC_GLBL_IRQ 0x04
#define OFFSET_SMFC_IER 0x8
#define OFFSET_SMFC_INT_CLR 0x40
#define OFFSET_SMFC_CMD_X_LEN 0x44
#define OFFSET_SMFC_CMD_Y_LEN 0x48
#define OFFSET_SMFC_SM_SRC_ADDR_L 0x4C
#define OFFSET_SMFC_SM_SRC_ADDR_H 0x50
#define OFFSET_SMFC_SM_DST_ADDR_L 0x54
#define OFFSET_SMFC_SM_DST_ADDR_H 0x58
#define OFFSET_SMFC_SM_CMD_SCALE 0x5C
#define OFFSET_SMFC_SM_CMD_OFFSET 0x60
#define OFFSET_SMFC_CALC_MOD 0x78

int dpu_1to1_get_caps(uint32_t *pBaseAddr, dpu_caps_t *pCaps);
int get_dpu_info_1to1(dpu_aol_dev_handle_t *p_signature, dpu_configurable_t *p_info, uint32_t count);
void show_dpu_regs_1to1(void);
void show_ext_regs_1to1(void);

#endif
