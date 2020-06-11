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

#ifndef __DPU_CAPS_H__
#define __DPU_CAPS_H__

#define DPU_EXT_HDMI         (1<<1)
#define DPU_EXT_BT1120       (1<<2)
#define DPU_EXT_FULLCONNECT  (1<<3)
#define DPU_EXT_SOFTMAX      (1<<4)
#define DPU_EXT_RESIZE       (1<<5)

#define SIG_BASE_NULL           0X00000000
#ifdef SIG_BASE_ADDR
#define SIG_BASE                SIG_BASE_ADDR
#else
#define SIG_BASE                SIG_BASE_NULL
#endif

#define SIG_BASE_MASK           0XFF000000
#define DPU_BASE(signature)     (((signature)&SIG_BASE_MASK)+0x0000)
#define DPU_SIZE                0X00000700
#define DPU_EXT_SOFTMAX_BASE(signature)   (((signature) & SIG_BASE_MASK) + 0x0700)
#define DPU_EXT_SOFTMAX_SIZE       0X00000041

/*dpu signature magic number*/
#define SIG_MAGIC 0X4450

#define SIG_SIZE_MASK   0XFF000000
#define SIG_VER_MASK    0X00FF0000
#define SIG_MAGIC_MASK  0X0000FFFF

#define DPU_CORENUM_MASK 0X0000000F
#define SOFTMAX_VLD_MASK 0X01000000
#define FC_VLD_MASK      0X00010000
#define RESIZE_VLD_MASK  0X00000001

#define BIT_VER_MASK            0XC0000000
#define HOUR_MASK               0X3E000000
#define DATE_MASK               0X01F00000
#define MONTH_MASK              0X000F0000
#define YEAR_MASK               0X0000F800
#define FREQ_MASK               0X000007FE
#define ENCRYPT_MASK            0X00000001

#define PS_INTBASE1_MASK        0X0000FF00
#define PS_INTBASE0_MASK        0X000000FF

#define HP_WIDTH_MASK           0XF0000000
#define DATA_WIDTH_MASK         0X0F000000
#define BANK_GROUP_MASK         0X00F00000
#define DPU_ARCH_MASK           0X000F0000
#define DPU_TARGET_MASK         0X0000FF00
#define DPU_HP_INTERACT_MASK    0X000000F0
#define DPU_CORENUM_MASK        0X0000000F

#define AVGPOOL_MASK            0XFFFF0000
#define CONV_DEPTHWISE_MASK     0X0000FF00
#define RELU_LEAKY_MASK         0X000000F0
#define RELU_P_MASK             0X0000000F

#define LOAD_AUG_MASK         0X00000F00
#define LOAD_IMG_MEAN_MASK    0X000000F0
#define SERDES_NONLINEAR_MASK   0X0000000F

#define RESERVED_FUNC7_MASK     0XF0000000
#define RESERVED_FUNC6_MASK     0X0F000000
#define RESERVED_FUNC5_MASK     0X00F00000
#define RESERVED_FUNC4_MASK     0X000F0000
#define RESERVED_FUNC3_MASK     0X0000F000
#define RESERVED_FUNC2_MASK     0X00000F00
#define RESERVED_FUNC1_MASK     0X000000F0
#define RESERVED_FUNC0_MASK     0X0000000F

#define SOFTMAX_IRQ_MASK        0XF0000000
#define SOFTMAX_VER_MASK        0X0E000000
#define SOFTMAX_VLD_MASK        0X01000000
#define FC_IRQ_MASK             0X00F00000
#define FC_VER_MASK             0X000E0000
#define FC_VLD_MASK             0X00010000
#define BT1120_IRQ_MASK         0X0000F000
#define BT1120_VER_MASK         0X00000E00
#define BT1120_VLD_MASK         0X00000100
#define HDMI_IRQ_MASK           0X000000F0
#define HDMI_VER_MASK           0X0000000E
#define HDMI_VLD_MASK           0X00000001

#define RESIZE_IRQ_MASK         0X000000F0
#define RESIZE_VER_MASK         0X0000000E
#define RESIZE_VLD_MASK         0X00000001

//according to ver_reg release @2019072422
#define VER_MAX_ENTRY           (12)
const static uint32_t VER_RESERVERD[VER_MAX_ENTRY] = {
    0x00000000, //0x00
    0x00000000, //0x04
    0xFFFF0000, //0x08
    0x00000000, //0x0c
    0x00000000, //0x10
    0x00000000, //0x14
    0x00000000, //0x18
    0xFFFFFFF0, //0x1c
    0xFFFFFFFF, //0x20
    0x00000000, //0x24
    0xFFFF0000, //0x28
    0xE0000000  //0x2c
};

/*dpu registers*/
#define MAX_CORE_NUM 4
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
		volatile uint32_t com_addr[16]; //< extension for DPUv1.3.0
		//volatile uint32_t _rsv[39];
		volatile uint32_t pend_cnt;
		volatile uint32_t cend_cnt;
		volatile uint32_t send_cnt;
		volatile uint32_t lend_cnt;
		volatile uint32_t pstart_cnt;
		volatile uint32_t cstart_cnt;
		volatile uint32_t sstart_cnt;
		volatile uint32_t lstart_cnt;
		volatile uint32_t axi_status;
		volatile uint32_t _rsv[10];
		volatile uint32_t timer_status;
		volatile uint32_t _rsv2[19];
	} ctlreg[MAX_CORE_NUM];

	/*dpu interrupt registers struct*/
	struct __regs_dpu_intr {
		volatile uint32_t isr;
		volatile uint32_t imr;
		volatile uint32_t irsr;
		volatile uint32_t icr;
		volatile uint32_t _rsv[60];

	} intreg;

} DPUReg;

typedef struct {
	volatile uint32_t done; //< 0x000	command done reg (1：done，0：not）
	volatile uint32_t sm_len_x; //< 0x004	vector length（unit:float）
	volatile uint32_t sm_len_y; //< 0x008	vector count
	volatile uint32_t src; //< 0x00c	source address, require 256 byte alignment
	volatile uint32_t dst; //< 0x010	destination address, require 256 byte alignment
	volatile uint32_t scale; //< 0x014	fix point
	volatile uint32_t sm_offset; //< 0x018	offset
	volatile uint32_t clr; //< 0x01c	clear interrupt	reg （1:clear，0：not）
	volatile uint32_t start; //< 0x020	start reg: valid on rising_edge,
	volatile uint32_t fc_input_channel; //< 0x024	fc input channel, maxinum 4096B
	volatile uint32_t fc_output_channel; //< 0x028	fc output channel,maxinum 4096B
	volatile uint32_t fc_batch; //< 0x02c	fc batch,
	volatile uint32_t fc_weight_start; //< 0x030	fc weight and bias start addr, 256B alignment
	volatile uint32_t fc_weight_end; //< 0x034	fc weight and bias end addr, 256B alignment
	volatile uint32_t calc_mod; //< 0x038	0: softmax; 1: fc
	volatile uint32_t dst_addr_sel; //< 0x03c	fix to 1: ddr,
	volatile uint32_t fc_relu_en; //< 0x040	fc relu,

} smfc_reg_t;

extern const DPUReg g_dpu_reg;
extern const smfc_reg_t g_smfc_reg;

int get_dpu_info_v0(dpu_aol_dev_handle_t *p_signature, dpu_info_t *p_info, uint32_t count);
int get_dpu_info_v1(dpu_aol_dev_handle_t *p_signature, dpu_configurable_t *p_info, uint32_t count); // configurable IP
int check_signature_default_v0(dpu_aol_dev_handle_t *p_signature);
int get_dpu_caps(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps);
int get_dpu_info(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps);
void show_dpu_regs(dpu_aol_dev_handle_t *p_signature, int dpu_count);
void show_ext_regs(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps);

#endif
