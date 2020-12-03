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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

#include "../common/dpu_types.h"
#include "aol/dpu_aol.h"
#include "dpu_caps.h"
#include "dpu_err.h"
#include "dpu_scheduler.h"
#include "dpu_1to1.h"

extern dpu_aol_dev_handle_t *gp_dpu_aol_handle;

int dpu_1to1_get_caps(uint32_t *pBaseAddr, dpu_caps_t *pCaps) {
    uint32_t buffer[16];
    dpu_1to1_sys_reg_t *sys_reg = (dpu_1to1_sys_reg_t *)buffer;
    dpu_1to1_feature_reg_t *feature_reg = (dpu_1to1_feature_reg_t *)buffer;

    pCaps->dpu_cnt = gp_dpu_aol_handle->core_count[IP_ID_DPU];

	dpu_aol_read_regs(gp_dpu_aol_handle, (uint64_t)(pBaseAddr + (0x20 / 4)), (uint32_t *)sys_reg, sizeof(dpu_1to1_sys_reg_t));
	sprintf(pCaps->hw_timestamp, "20%02d-%02d-%02d %02d:%02d:00",
		sys_reg->TIMESTAMP.TIME_YEAR,
		sys_reg->TIMESTAMP.TIME_MONTH,
		sys_reg->TIMESTAMP.TIME_DAY,
		sys_reg->TIMESTAMP.TIME_HOUR,
		sys_reg->TIMESTAMP.TIME_QUARTER * 15);
    //printf("sizeof: %d, res:%s\n", sizeof(dpu_1to1_sys_reg_t), pCaps->hw_timestamp);

    dpu_aol_read_regs(gp_dpu_aol_handle, (uint64_t)(pBaseAddr + (0x100 / 4)), (uint32_t *)feature_reg, sizeof(dpu_1to1_feature_reg_t));
	pCaps->irq_base0 = 0; //1to1 no this field
	pCaps->irq_base1 = 0; //1to1 no this field
	pCaps->hp_width = feature_reg->ARCH.ARCH_HP_BW;
	pCaps->data_width = feature_reg->ARCH.ARCH_DATA_BW;
	pCaps->bank_group = feature_reg->ARCH.ARCH_IMG_BKGRP;

	if ((pCaps->hp_width != 2) && (pCaps->hp_width != 3)) {
		DPU_LOG_MSG("Invalid hp width '%d' found in DPU signature.\n",
				pCaps->hp_width);
		return -1;
	}
	if ((pCaps->data_width != 1) && (pCaps->data_width != 2)) {
		DPU_LOG_MSG("Invalid data width '%d' found in DPU signature.\n",
				pCaps->data_width);
		return -1;
	}
	if ((pCaps->bank_group != 2) && (pCaps->bank_group != 3)) {
		DPU_LOG_MSG("Invalid bank group '%d' found in DPU signature.\n",
				pCaps->bank_group);
		return -1;
	}

	pCaps->avgpool.version = feature_reg->POOL.POOL_AVERAGE;
	pCaps->conv_depthwise.version = feature_reg->DWCV.DWCV_PARALLEL;

    // Bit0 CONV_LEAKYRELU, Bit1 CONV_RELU6, Bit2, DWCV_RELU6
    pCaps->relu_leaky.version = 0;
    if (feature_reg->CONV.CONV_LEAKYRELU == 1) {
        pCaps->relu_leaky.version |= 0x1;
    } else {
        if (feature_reg->CONV.CONV_LEAKYRELU != 0) {
            DPU_LOG_MSG("Invalid CONV_LEAKYRELU '%d' found in DPU signature.\n", feature_reg->CONV.CONV_LEAKYRELU);
            return -1;
        }
    }
    if (feature_reg->CONV.CONV_RELU6 == 1) {
        pCaps->relu_leaky.version |= 0x2;
    } else {
        if (feature_reg->CONV.CONV_RELU6 != 0) {
            DPU_LOG_MSG("Invalid CONV_RELU6 '%d' found in DPU signature.\n", feature_reg->CONV.CONV_RELU6);
            return -1;
        }
    }
    if (feature_reg->DWCV.DWCV_RELU6 == 1) {
        pCaps->relu_leaky.version |= 0x04;
    } else {
        if (feature_reg->DWCV.DWCV_RELU6 != 0) {
            DPU_LOG_MSG("Invalid DWCV_RELU6 '%d' found in DPU signature.\n", feature_reg->CONV.CONV_RELU6);
            return -1;
        }
    }

	pCaps->relu_p.version = 0; //1to1 no this field
	pCaps->serdes_nonlinear.version = 0; //1to1 no this field

	// 1to1 no this field
	pCaps->hdmi.enable = 0;
	pCaps->hdmi.version = 0;
	pCaps->hdmi.valid = 0;
	pCaps->hdmi.irq = 0;

    // 1to1 no this field
	pCaps->bt1120.enable = 0;
	pCaps->bt1120.version = 0;
	pCaps->bt1120.valid = 0;
	pCaps->bt1120.irq = 0;

    // 1to1 no this field
	pCaps->fullconnect.enable = 0;
	pCaps->fullconnect.version = 0;
	pCaps->fullconnect.valid = 0;
	pCaps->fullconnect.irq = 0;

    // Softmax
    if (gp_dpu_aol_handle->core_count[IP_ID_SOFTMAX] > 0) {
        pCaps->softmax.enable = 1;
        pCaps->softmax.version = 0; // 1to1 no this field
        pCaps->softmax.valid = 1;
        pCaps->softmax.irq = 0; // 1to1 no this field
    }

	// 1to1 no this field
	pCaps->resize.enable = 0;
	pCaps->resize.version = 0;
	pCaps->resize.valid = 0;
	pCaps->resize.irq = 0;

    return 0;
}

int get_dpu_info_1to1(dpu_aol_dev_handle_t *p_signature, dpu_configurable_t *p_info, uint32_t count) {
	int idx;
	uint32_t *pBaseAddr = (uint32_t *)p_signature->core_phy_addr[0];
    dpu_1to1_feature_reg_t feature;
    dpu_1to1_sys_reg_t sys_reg;

    dpu_aol_read_regs(gp_dpu_aol_handle, (uint64_t)(pBaseAddr + (0x20 / 4)), (uint32_t *)&sys_reg, sizeof(dpu_1to1_sys_reg_t));
    dpu_aol_read_regs(gp_dpu_aol_handle, (uint64_t)(pBaseAddr + (0x100 / 4)), (uint32_t *)&feature, sizeof(dpu_1to1_feature_reg_t));

	for (idx = 0; idx < count; idx++) {
		(*(p_info + idx)).base.dpu_arch = 2 * feature.ARCH.ARCH_PP * feature.ARCH.ARCH_ICP * feature.ARCH.ARCH_OCP;
		(*(p_info + idx)).base.dpu_freq = sys_reg.FREQ.M_AXI_FREQ_MHZ;

		if (sys_reg.SYS.SYS_IP_TYPE != 0x1) {
			DPU_LOG_MSG("Invalid SYS.SYS_IP_TYPE '%d' found in DPU signature.\n", sys_reg.SYS.SYS_IP_TYPE);
			return -1;
		}
		(*(p_info + idx)).sys.sys_ip_type = sys_reg.SYS.SYS_IP_TYPE;

		//if (sys_reg.SYS.SYS_REGMAP_VER != 2) {
		//	DPU_LOG_MSG("Invalid SYS.SYS_REGMAP_VER '%d' found in DPU signature.\n", sys_reg.SYS.SYS_REGMAP_VER);
		//	return -1;
		//}
		(*(p_info + idx)).sys.sys_regmap_ver = sys_reg.SYS.SYS_REGMAP_VER;

		(*(p_info + idx)).sub_version.ver_target = feature.SUB_VERSION.VER_TARGET;

		(*(p_info + idx)).arch.arch_hp_bw = pow(2, dpu_caps.hp_width + 4); // 16*(2^N)
		(*(p_info + idx)).arch.arch_data_bw = 8 * dpu_caps.data_width;
		(*(p_info + idx)).arch.arch_img_bkgrp = dpu_caps.bank_group;
		(*(p_info + idx)).arch.arch_pp = feature.ARCH.ARCH_PP;
		(*(p_info + idx)).arch.arch_icp = feature.ARCH.ARCH_ICP;
		(*(p_info + idx)).arch.arch_ocp = feature.ARCH.ARCH_OCP;

		(*(p_info + idx)).ram.ram_depth_mean = (0x1+1)*16;
		(*(p_info + idx)).ram.ram_depth_bias = (0x3+1)*512;
		(*(p_info + idx)).ram.ram_depth_wgt  = (0x3+1)*512;
		(*(p_info + idx)).ram.ram_depth_img  = (0x3+1)*512;

		(*(p_info + idx)).load.load_augm_enable = feature.LOAD.LOAD_AUGM;
		(*(p_info + idx)).load.load_img_mean_enable = feature.LOAD.LOAD_IMG_MEAN;

		(*(p_info + idx)).conv.conv_leakyrelu_enable = (dpu_caps.relu_leaky.version & 0x01);
		(*(p_info + idx)).conv.conv_relu6_enable = (dpu_caps.relu_leaky.version & 0x2) >> 1;

		(*(p_info + idx)).conv.conv_wr_parallel = feature.CONV.CONV_WR_PARALLEL;

		(*(p_info + idx)).pool.pool_average_enable = dpu_caps.avgpool.version;

		(*(p_info + idx)).elew.elew_parallel = feature.ELEW.ELEW_PARALLEL;

		(*(p_info + idx)).dwcv.dwcv_alu_mode_enable = feature.DWCV.DWCV_ALU_MODE;
		(*(p_info + idx)).dwcv.dwcv_relu6_enable = (dpu_caps.relu_leaky.version & 0x4) >> 2;
		(*(p_info + idx)).dwcv.dwcv_parallel = feature.DWCV.DWCV_PARALLEL;

		(*(p_info + idx)).misc.misc_wr_parallel = feature.MISC.MISC_WR_PARALLEL;
	}

    return 0;
}

void show_dpu_regs_1to1(void) {
	dpu_aol_dev_handle_t *p_signature = gp_dpu_aol_handle;
	int dpu_count = p_signature->core_count[IP_ID_DPU];

	for (int i = 0; i < dpu_count; i++) {
		uint64_t dpu_base_addr = p_signature->core_phy_addr[i];
		DPU_Reg dpu;
		dpu_aol_read_regs(p_signature, (uint64_t)dpu_base_addr, (uint32_t *)&dpu, sizeof(DPU_Reg));

		printf("[DPU Core %d Register]\n", i);

		printf("CTL       : 0x%.8x\n", dpu.APCTL);
		printf("GIE       : 0x%.8x\n", dpu.GIE);
		printf("IRQ       : 0x%.8x\n", dpu.ISR);
		printf("HP        : 0x%.8x\n", dpu.hp_bus);

		printf("CODE      : 0x%.16x\n", dpu.addr_code);
		printf("BASE0     : 0x%.16x\n", dpu.addr0);
		printf("BASE1     : 0x%.16x\n", dpu.addr1);
		printf("BASE2     : 0x%.16x\n", dpu.addr2);
		printf("BASE3     : 0x%.16x\n", dpu.addr3);
		printf("BASE4     : 0x%.16x\n", dpu.addr4);
		printf("BASE5     : 0x%.16x\n", dpu.addr5);
		printf("BASE6     : 0x%.16x\n", dpu.addr6);
		printf("BASE7     : 0x%.16x\n", dpu.addr7);

		printf("CYCLE_H   : 0x%.8x\n", dpu.cycle_h);
		printf("CYCLE_L   : 0x%.8x\n", dpu.cycle_l);

		printf("REGVER    : 0x%.8x\n", dpu.version);
		printf("TIMESTAMP : 0x%.8x\n", dpu.timestamp);
		printf("GITID     : 0x%.8x\n", dpu.GIT_COMMIT_ID);
		printf("GITTIME   : 0x%.8x\n", dpu.GIT_COMMIT_TIME);
		printf("VERSION   : 0x%.8x\n", dpu.SUB_VERSION);
		printf("TIMER     : 0x%.8x\n", dpu.TIMER);
		printf("ARCH      : 0x%.8x\n", dpu.ARCH);
		printf("RAM       : 0x%.8x\n", dpu.CONF_RAM);
		printf("LOAD      : 0x%.8x\n", dpu.CONF_LOAD);
		printf("CONV      : 0x%.8x\n", dpu.CONF_CONV);
		printf("SAVE      : 0x%.8x\n", dpu.CONF_SAVE);
		printf("POOL      : 0x%.8x\n", dpu.CONF_POOL);
		printf("ELEW      : 0x%.8x\n", dpu.CONF_ELEW);
		printf("DWCV      : 0x%.8x\n", dpu.CONF_DWCV);
		printf("MISC      : 0x%.8x\n", dpu.CONF_MISC);

		printf("DPU STATUS: 0x%.8x\n", dpu.dpu_status);
		printf("AXI STATUS: 0x%.8x\n", dpu.axi_status);
		printf("LOAD START: %d\n", dpu.load_start);
		printf("LOAD END  : %d\n", dpu.load_end);
		printf("SAVE START: %d\n", dpu.save_start);
		printf("SAVE END  : %d\n", dpu.save_end);
		printf("CONV START: %d\n", dpu.conv_start);
		printf("CONV END  : %d\n", dpu.conv_end);
		printf("MISC START: %d\n", dpu.misc_start);
		printf("MISC END  : %d\n", dpu.misc_end);

		printf("\n");
	}
}


/**
 * show_ext_regs - show dpu extension module registers
 * @mask  : extension module's mask
 */
void show_ext_regs_1to1(void) {
	uint32_t ip_handle_index;
	uint32_t *ip_base_address;
	uint32_t buffer[17];
	dpu_aol_dev_handle_t *p_signature = gp_dpu_aol_handle;
	dpu_caps_t *p_caps = &dpu_caps;

	ip_handle_index = p_signature->core_count[IP_ID_VER_REG];
	ip_handle_index += p_signature->core_count[IP_ID_DPU];
	ip_base_address = (uint32_t *)p_signature->core_phy_addr[ip_handle_index];
//	printf("softmax base addr: 0x%x\n", ip_base_address);

	if (p_caps->softmax.enable || p_caps->fullconnect.enable) {
		printf("[SMFC Registers]\n");
		dpu_aol_read_regs(p_signature, (uint64_t)(ip_base_address), buffer, 2 * 4);
		printf("%-8s\t: 0x%.8x\n", "CTRL", buffer[0]);
		printf("%-8s\t: 0x%.8x\n", "GIE", buffer[1]);

		dpu_aol_read_regs(p_signature, (uint64_t)(ip_base_address + (0x40 >> 2)), buffer, 1 * 4);
		printf("%-8s\t: 0x%.8x\n", "CLR", buffer[0]); // regs->clr

		dpu_aol_read_regs(p_signature, (uint64_t)(ip_base_address + (0x44 >> 2)), buffer, 8 * 4);
		printf("%-8s\t: 0x%.8x\n", "SM_LEN_X", buffer[0]); // regs->sm_len_x
		printf("%-8s\t: 0x%.8x\n", "SM_LEN_Y", buffer[1]); // regs->sm_len_y
		printf("%-8s\t: 0x%.8x\n", "SRC", buffer[2]); // regs->src
		printf("%-8s\t: 0x%.8x\n", "DST", buffer[4]); // regs->dst
		printf("%-8s\t: 0x%.8x\n", "SCALE", buffer[6]); // regs->scale
		printf("%-8s\t: 0x%.8x\n", "SM_OFFSET", buffer[7]); // regs->sm_offset

		dpu_aol_read_regs(p_signature, (uint64_t)(ip_base_address + (0x64 >> 2)), buffer, 8 * 4);
		printf("%-8s\t: 0x%.8x\n", "FC_INPUT_CHANNEL", buffer[0]); // regs->fc_input_channel
		printf("%-8s\t: 0x%.8x\n", "FC_OUTPUT_CHANNEL", buffer[1]); // regs->fc_output_channel
		printf("%-8s\t: 0x%.8x\n", "FC_BATCH", buffer[2]); // regs->fc_batch
		printf("%-8s\t: 0x%.8x\n", "FC_WEIGHT_START", buffer[3]); // regs->fc_weight_start
		printf("%-8s\t: 0x%.8x\n", "FC_WEIGHT_END", buffer[4]); // regs->fc_weight_end
		printf("%-8s\t: 0x%.8x\n", "CALC_MOD", buffer[5]); // regs->calc_mod
		printf("%-8s\t: 0x%.8x\n", "DST_ADDR_SEL", buffer[6]); // regs->dst_addr_sel
		printf("%-8s\t: 0x%.8x\n", "FC_RELU_EN", buffer[7]); // regs->fc_relu_en
	}
}
