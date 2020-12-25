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

static uint32_t field_mask_value(uint32_t val, uint32_t mask);

//control which extension is enabled(corresponding bit set to 1)
unsigned long extension = -1; // whether use cache; 0:no, 1:yes

const DPUReg g_dpu_reg;
const smfc_reg_t g_smfc_reg;

static uint32_t field_mask_value(uint32_t val, uint32_t mask)
{
	int i;
	int max_bit = sizeof(uint32_t) * 8;
	int lowest_set_bit = max_bit - 1;

	/* Iterate through each bit of mask */
	for (i = 0; i < max_bit; i++) {
		/* If current bit is set */
		if ((mask >> i) & 1) {
			lowest_set_bit = i;
			break;
		}
	}

	return (val & mask) >> lowest_set_bit;
};

#define DPU_CORE_MAX (16)
int get_dpu_info_v0(dpu_aol_dev_handle_t *p_signature, dpu_info_t *p_info, uint32_t count) {
    int i;
	uint32_t signature_field;
	uint32_t dpu_freq;
	uint32_t dpu_arch;
	uint32_t dpu_target;
	uint32_t irqs[DPU_CORE_MAX];
	uint32_t signature_temp;
	uint32_t irq_base0, irq_base1;
	uint32_t sig_index = 0;
	uint32_t *sig_base_address = (uint32_t *)(p_signature->core_phy_addr[0]);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 1), &signature_field, 4);
	dpu_freq = field_mask_value(signature_field, FREQ_MASK);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 3), &signature_field, 4);
	dpu_arch = field_mask_value(signature_field, DPU_ARCH_MASK);
	dpu_target = field_mask_value(signature_field, DPU_TARGET_MASK);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 2), &signature_field, 4);
	irq_base0 = field_mask_value(signature_field, PS_INTBASE0_MASK);
	irq_base1 = field_mask_value(signature_field, PS_INTBASE1_MASK);

	// offset 4
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 4), &signature_field, 4);
	for (i = 0; i < 8; i++) {
		signature_temp = field_mask_value(signature_field, 0xF << (4 * i));
		irqs[i] = signature_temp & 0x8 ? 
					(signature_temp & 0x7) + irq_base1 :
					(signature_temp & 0x7) + irq_base0;
	}

	// offset 5
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 5), &signature_field, 4);
	for (i = 0; i < 8; i++) {
		signature_temp = field_mask_value(signature_field, 0xF << (4 * i));
		irqs[8 + i] = signature_temp & 0x8 ?
						(signature_temp & 0x7) + irq_base1 :
						(signature_temp & 0x7) + irq_base0;
	}

	for (i = 0; i < count; i++) {
		(*(p_info + i)).base.dpu_arch = dpu_arch;
		(*(p_info + i)).base.dpu_freq = dpu_freq;
		(*(p_info + i)).dpu_target = dpu_target;
		(*(p_info + i)).irq = irqs[i];

		if ((*(p_info + i)).base.dpu_arch > DPU_ARCH_RESERVE) {
			DPU_LOG_MSG("Unknown DPU arch type '%d' found in DPU signature.\n", (*(p_info + i)).base.dpu_arch);
			DPU_LOG_MSG("Try to update DPU driver to the latest version to resolve this issue.\n");
			return -1;
		}
		if ((*(p_info + i)).dpu_target > DPU_TARGET_RESERVE) {
			DPU_LOG_MSG("Unknown DPU target type '%d' found in DPU signature.\n", (*(p_info + i)).dpu_target);
			DPU_LOG_MSG("Try to update DPU driver to the latest version to resolve this issue.\n");
			return -1;
		}
	}

    return 0;
}

int get_dpu_info_v1(dpu_aol_dev_handle_t *p_signature, dpu_configurable_t *p_info, uint32_t count) {
	int idx;
	uint32_t signature_field;
	uint32_t dpu_freq;
	uint32_t dpu_arch;
	uint32_t dpu_target;
	uint32_t hp_width, data_width, bank_group;
	uint32_t relu_leaky_version, avgpool_version, conv_depthwise_version;
	uint32_t sig_index = 0;
	uint32_t *sig_base_address = (uint32_t *)p_signature->core_phy_addr[0];

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 6), &signature_field, 4);
	relu_leaky_version = field_mask_value(signature_field, RELU_LEAKY_MASK);
	avgpool_version = field_mask_value(signature_field, AVGPOOL_MASK);
	conv_depthwise_version = field_mask_value(signature_field, CONV_DEPTHWISE_MASK);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 3), &signature_field, 4);
	hp_width = field_mask_value(signature_field, HP_WIDTH_MASK);
	data_width = field_mask_value(signature_field, DATA_WIDTH_MASK);
	bank_group = field_mask_value(signature_field, BANK_GROUP_MASK);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 1), &signature_field, 4);
	dpu_freq = field_mask_value(signature_field, FREQ_MASK);

	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 3), &signature_field, 4);
	dpu_arch = field_mask_value(signature_field, DPU_ARCH_MASK);
	dpu_target = field_mask_value(signature_field, DPU_TARGET_MASK);

	switch (dpu_arch) {
		case 1: dpu_arch = 1024; break;
		case 2: dpu_arch = 1152; break;
		case 3: dpu_arch = 4096; break;
		case 4: dpu_arch = 256; break;
		case 5: dpu_arch = 512; break;
		case 6: dpu_arch = 800; break;
		case 7: dpu_arch = 1600; break;
		case 8: dpu_arch = 2048; break;
		case 9: dpu_arch = 2304; break;
		case 10: dpu_arch = 8192; break;
		case 11: dpu_arch = 3136; break;
		case 12: dpu_arch = 288; break;
		case 13: dpu_arch = 144; break;
		case 14: dpu_arch = 5184; break;
		default:
			DPU_LOG_MSG("Unknown DPU arch type '%d' found in DPU signature.\n", dpu_arch);
			DPU_LOG_MSG("Try to update DPU driver to the latest version to resolve this issue.\n");
			return -1;
	}
	switch (dpu_target) {
		case 1: dpu_target = 0x113; break;
		case 2: dpu_target = 0x130; break;
		case 3: dpu_target = 0x131; break;
		case 4: dpu_target = 0x132; break;
		case 5: dpu_target = 0x133; break;
		case 6: dpu_target = 0x134; break;
		case 7: dpu_target = 0x135; break;
		case 8: dpu_target = 0x140; break;
		case 9: dpu_target = 0x141; break;
		case 10: dpu_target = 0x142; break;
		case 11: dpu_target = 0x136; break;
		case 12: dpu_target = 0x137; break;
		case 13: dpu_target = 0x138; break;
		default:
			DPU_LOG_MSG("Unknown DPU target type '%d' found in DPU signature.\n", dpu_target);
			DPU_LOG_MSG("Try to update DPU driver to the latest version to resolve this issue.\n");
			return -1;
	}

	for (idx = 0; idx < count; idx++) {
		(*(p_info + idx)).base.dpu_arch = dpu_arch;
		(*(p_info + idx)).base.dpu_freq = dpu_freq;

		(*(p_info + idx)).sys.sys_ip_type = 0x1;
		(*(p_info + idx)).sys.sys_regmap_ver = 0x1;

		(*(p_info + idx)).sub_version.ver_target = dpu_target;

		(*(p_info + idx)).arch.arch_hp_bw = 1<<(hp_width+4);
		(*(p_info + idx)).arch.arch_data_bw = 8*data_width;
		(*(p_info + idx)).arch.arch_img_bkgrp = bank_group;
		(*(p_info + idx)).arch.arch_pp = 0x4;
		(*(p_info + idx)).arch.arch_icp = 0xc;
		(*(p_info + idx)).arch.arch_ocp = 0xc;

		(*(p_info + idx)).ram.ram_depth_mean = (0x1+1)*16;
		(*(p_info + idx)).ram.ram_depth_bias = (0x3+1)*512;
		(*(p_info + idx)).ram.ram_depth_wgt  = (0x3+1)*512;
		(*(p_info + idx)).ram.ram_depth_img  = (0x3+1)*512;

		dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 7), &signature_field, 4);
		(*(p_info + idx)).load.load_augm_enable = field_mask_value(signature_field, LOAD_AUG_MASK);
		(*(p_info + idx)).load.load_img_mean_enable = field_mask_value(signature_field, LOAD_IMG_MEAN_MASK);

		(*(p_info + idx)).conv.conv_leakyrelu_enable = relu_leaky_version & 0x1;;
		(*(p_info + idx)).conv.conv_relu6_enable = (relu_leaky_version & 0x2) >> 1;
		(*(p_info + idx)).conv.conv_wr_parallel = 0x1; // no check

		(*(p_info + idx)).pool.pool_average_enable = avgpool_version;

		(*(p_info + idx)).elew.elew_parallel = 0x1; // no check

		(*(p_info + idx)).dwcv.dwcv_alu_mode_enable = 0x0; // no check
		(*(p_info + idx)).dwcv.dwcv_relu6_enable = (relu_leaky_version & 0x2) >> 1;
		(*(p_info + idx)).dwcv.dwcv_parallel = conv_depthwise_version;

		(*(p_info + idx)).misc.misc_wr_parallel = 0x1; // no check
	}

    return 0;
}

int check_signature_default_v0(dpu_aol_dev_handle_t *p_signature) {
	int i;
	uint32_t signature_field;
	uint32_t signature_temp;
	uint32_t *sig_base_address = (uint32_t *)p_signature->core_phy_addr[0];

	signature_temp = 0;
	for (i = 0; i < VER_MAX_ENTRY; i++) {
		dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + i), &signature_field, 4);
		signature_temp = field_mask_value(signature_field, VER_RESERVERD[i]);
		if (signature_temp != 0) {
			DPU_LOG_MSG("Unknown reserved field found in DPU signature at offset: %#X.\n", i * 4);
			DPU_LOG_MSG("Try to update DPU driver to the latest version to resolve this issue.\n");
			return -1;
		}
	}

	return 0;
}

int get_dpu_caps(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps) {
	int32_t signature_field;
	uint32_t signature_temp;
	uint32_t signatrue_ver;
	uint32_t *sig_base_address = (uint32_t *)p_signature->core_phy_addr[0];

	// Check signature valid
	if (p_signature->core_count[IP_ID_VER_REG] == 1) {
		p_caps->signature_valid = 1;
	} else {
		p_caps->signature_valid = 0;
	}

	// Get signature version
	if (p_caps->signature_valid == 0) { // for 1to1 DPU, no ver_reg page
		if(dpu_1to1_get_caps(sig_base_address, p_caps) != 0) {
			return -1;
		}

		// conver to configurable DPU
		p_caps->signature_version = 2;
		dpu_caps.signature_valid = 1;
		p_caps->magic = DPU_CONF_MAGIC;
		return 1;
	} else {
		dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address), &signature_field, 4);
		if ((signature_field & SIG_MAGIC_MASK) == SIG_MAGIC) {
			signatrue_ver = field_mask_value(signature_field, SIG_VER_MASK);
			if (signatrue_ver == 1) {
				p_caps->magic = DPU_CONF_MAGIC;
				p_caps->signature_version = 1;
			} else {
				p_caps->magic = 0;
				p_caps->signature_version = 0;
			}
		} else {
			DPU_LOG_MSG("Invalid signature address.\n");
			return -1;
		}
	}

	// offset 1
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 1), &signature_field, 4);
	sprintf(p_caps->hw_timestamp, "20%02d-%02d-%02d %02d:%02d:00",
		field_mask_value(signature_field, YEAR_MASK),
		field_mask_value(signature_field, MONTH_MASK),
		field_mask_value(signature_field, DATE_MASK),
		field_mask_value(signature_field, HOUR_MASK),
		field_mask_value(signature_field, BIT_VER_MASK) * 15);

	// offset 2
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 2), &signature_field, 4);
	p_caps->irq_base0 = field_mask_value(signature_field, PS_INTBASE0_MASK);
	p_caps->irq_base1 = field_mask_value(signature_field, PS_INTBASE1_MASK);

	// offset 3
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 3), &signature_field, 4);
	p_caps->hp_width = field_mask_value(signature_field, HP_WIDTH_MASK);
	p_caps->data_width = field_mask_value(signature_field, DATA_WIDTH_MASK);
	p_caps->bank_group = field_mask_value(signature_field, BANK_GROUP_MASK);
	p_caps->dpu_cnt = field_mask_value(signature_field, DPU_CORENUM_MASK);
	if (p_caps->hp_width >= DPU_HP_WIDTH_RESERVE) {
		DPU_LOG_MSG("Invalid hp width '%d' found in DPU signature.\n",
				p_caps->hp_width);
		return -1;
	}
	if (p_caps->data_width >= DPU_DATA_WIDTH_RESERVE) {
		DPU_LOG_MSG("Invalid data width '%d' found in DPU signature.\n",
				p_caps->data_width);
		return -1;
	}
	if (p_caps->bank_group >= DPU_BANK_GROUP_RESERVE ||
		DPU_BANK_GROUP_1 == p_caps->bank_group) {
		DPU_LOG_MSG("Invalid bank group '%d' found in DPU signature.\n",
				p_caps->bank_group);
		return -1;
	}

	// offset 6
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 6), &signature_field, 4);
	p_caps->avgpool.version = field_mask_value(signature_field, AVGPOOL_MASK);
	p_caps->conv_depthwise.version = field_mask_value(signature_field, CONV_DEPTHWISE_MASK);
	p_caps->relu_leaky.version = field_mask_value(signature_field, RELU_LEAKY_MASK);
	p_caps->relu_p.version = field_mask_value(signature_field, RELU_P_MASK);

	// offset 7
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 7), &signature_field, 4);
	p_caps->serdes_nonlinear.version = field_mask_value(signature_field, SERDES_NONLINEAR_MASK);

	// offset 9
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 9), &signature_field, 4);
	p_caps->hdmi.enable = field_mask_value(extension, DPU_EXT_HDMI);
	p_caps->hdmi.version = field_mask_value(signature_field, HDMI_VER_MASK);
	p_caps->hdmi.valid = field_mask_value(signature_field, HDMI_VLD_MASK);
	p_caps->hdmi.enable &= p_caps->hdmi.valid;
	signature_temp = field_mask_value(signature_field, HDMI_IRQ_MASK);
	p_caps->hdmi.irq = signature_temp & 0x8 ?
					(signature_temp & 0x7) + p_caps->irq_base1 :
					(signature_temp & 0x7) + p_caps->irq_base0;

	p_caps->bt1120.enable = field_mask_value(extension, DPU_EXT_BT1120);
	p_caps->bt1120.version = field_mask_value(signature_field, BT1120_VER_MASK);
	p_caps->bt1120.valid = field_mask_value(signature_field, BT1120_VLD_MASK);
	p_caps->bt1120.enable &= p_caps->bt1120.valid;
	signature_temp = field_mask_value(signature_field, BT1120_IRQ_MASK);
	p_caps->bt1120.irq = signature_temp & 0x8 ?
						(signature_temp & 0x7) + p_caps->irq_base1 :
						(signature_temp & 0x7) + p_caps->irq_base0;

	p_caps->fullconnect.enable = field_mask_value(extension, DPU_EXT_FULLCONNECT);
	p_caps->fullconnect.version = field_mask_value(signature_field, FC_VER_MASK);
	p_caps->fullconnect.valid = field_mask_value(signature_field, FC_VLD_MASK);
	p_caps->fullconnect.enable &= p_caps->fullconnect.valid;
	signature_temp = field_mask_value(signature_field, FC_IRQ_MASK);
	p_caps->fullconnect.irq = signature_temp & 0x8 ?
						(signature_temp & 0x7) + p_caps->irq_base1 :
						(signature_temp & 0x7) + p_caps->irq_base0;

	p_caps->softmax.enable = field_mask_value(extension, DPU_EXT_SOFTMAX);
	p_caps->softmax.version = field_mask_value(signature_field, SOFTMAX_VER_MASK);
	p_caps->softmax.valid = field_mask_value(signature_field, SOFTMAX_VLD_MASK);
	p_caps->softmax.enable &= p_caps->softmax.valid;
	signature_temp = field_mask_value(signature_field, SOFTMAX_IRQ_MASK);
	p_caps->softmax.irq = signature_temp & 0x8 ?
						(signature_temp & 0x7) + p_caps->irq_base1 :
						(signature_temp & 0x7) + p_caps->irq_base0;

	// offset 10
	dpu_aol_read_regs(p_signature, (uint64_t)(sig_base_address + 10), &signature_field, 4);
	p_caps->resize.enable = field_mask_value(extension, DPU_EXT_RESIZE);
	p_caps->resize.version = field_mask_value(signature_field, RESIZE_VER_MASK);
	p_caps->resize.valid = field_mask_value(signature_field, RESIZE_VLD_MASK);
	p_caps->resize.enable &= p_caps->resize.valid;
	signature_temp = field_mask_value(signature_field, RESIZE_IRQ_MASK);
	p_caps->resize.irq = signature_temp & 0x8 ?
						(signature_temp & 0x7) + p_caps->irq_base1 :
						(signature_temp & 0x7) + p_caps->irq_base0;

	return 0;
}

/**
 * _get_state_str - get dpu state description string
 */
char *_get_state_str(int state)
{
	switch (state) {
	case DPU_IDLE:
		return "Idle";
	case DPU_RUNNING:
		return "Running";
	case DPU_DISABLE:
		return "Disable";
	default:
		return "UNDEF";
	}
}

/**
 * proc_show_dpuinfo - dpu proc file show infomation function:
 *	                 type "cat /proc/dpu_info " to get dpu driver
 *information
 */
int get_dpu_info(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps) {
	uint32_t *sig_base_address = (uint32_t *)p_signature->core_phy_addr[0];

	int i = 0, j = 0;
	uint32_t MemInUse = 0;
	unsigned long flags;
	struct list_head *plist;
	struct memblk_node *p;
	dpu_status_t dpu_status;

	printf("[DPU Debug Info]\n");
	for (i = 0; i < p_caps->dpu_cnt; i++) {
		dpu_scheduler_get_status(i, &dpu_status);
		printf("Core %d schedule : %lu\n", i, dpu_status._run_counter);
		printf("Core %d interrupt: %lu\n", i, dpu_status._int_counter);
	}
	printf("\n");

	printf("[DPU Resource]\n");
	for (i = 0; i < p_caps->dpu_cnt; i++) {
		dpu_scheduler_get_status(i, &dpu_status);
		printf("%-10s\t: %d\n", "DPU Core", i);
		printf("%-10s\t: %s\n", "State", _get_state_str(dpu_status.status));
		printf("%-10s\t: %d\n", "PID", dpu_status.pid);
		printf("%-10s\t: %ld\n", "TaskID", dpu_status.task_id);
		printf("%-10s\t: %lld\n", "Start", dpu_status.time_start);
		printf("%-10s\t: %lld\n", "End", dpu_status.time_end);
		printf("\n");
	}

	if (p_caps->signature_version == 2) {
		show_dpu_regs_1to1();
		show_ext_regs_1to1();
	} else {
		show_dpu_regs(p_signature, p_caps->dpu_cnt);
		show_ext_regs(p_signature, p_caps);
	}

	return 0;
}

void show_dpu_regs(dpu_aol_dev_handle_t *p_signature, int dpu_count) {
	int i;
	uint32_t dpu_handle_index;
	uint32_t *dpu_base_address;
	uint32_t buffer[9];

	dpu_handle_index = p_signature->core_count[IP_ID_VER_REG];
	dpu_base_address = (uint32_t *)p_signature->core_phy_addr[dpu_handle_index];

	printf("[DPU Registers]\n");
	dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address), buffer, 8);
	printf("%-10s\t: 0x%.8x\n", "VER", buffer[0]); // pmu.version
	printf("%-10s\t: 0x%.8x\n", "RST", buffer[1]); // pmu.reset
	dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address + 64), buffer, 32);
	printf("%-10s\t: 0x%.8x\n", "ISR", buffer[0]); // intreg.isr
	printf("%-10s\t: 0x%.8x\n", "IMR", buffer[1]); // intreg.imr
	printf("%-10s\t: 0x%.8x\n", "IRSR", buffer[2]); // intreg.irsr
	printf("%-10s\t: 0x%.8x\n", "ICR", buffer[3]); // intreg.icr
	printf("\n");
	for (i = 0; i < dpu_count; i++) {
		printf("%-8s\t: %d\n", "DPU Core", i);
		dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address + (64 * (2 + i))), buffer, 36);
		printf("%-8s\t: 0x%.8x\n", "HP_CTL", buffer[0]); // ctlreg[i].hp_ctrl
		printf("%-8s\t: 0x%.8x\n", "ADDR_IO", buffer[1]); // ctlreg[i].addr_io)
		printf("%-8s\t: 0x%.8x\n", "ADDR_WEIGHT", buffer[2]); // ctlreg[i].addr_weight
		printf("%-8s\t: 0x%.8x\n", "ADDR_CODE", buffer[3]); // ctlreg[i].addr_code
		printf("%-8s\t: 0x%.8x\n", "ADDR_PROF", buffer[4]); // ctlreg[i].addr_prof
		printf("%-8s\t: 0x%.8x\n", "PROF_VALUE", buffer[5]); // ctlreg[i].prof_value
		printf("%-8s\t: 0x%.8x\n", "PROF_NUM", buffer[6]); // ctlreg[i].prof_num
		printf("%-8s\t: 0x%.8x\n", "PROF_EN", buffer[7]); // ctlreg[i].prof_en
		printf("%-8s\t: 0x%.8x\n", "START", buffer[8]); // ctlreg[i].start
//#if defined(CONFIG_DPU_v1_3_0)
		for (int j = 0; j < 8; j++) {
			dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address + (64 * (2 + i)) + (0x24/4) + (j * 2)), buffer, 8);
			printf("%-8s%d\t: 0x%.8x\n", "COM_ADDR_L", j, buffer[0]); // ctlreg[i].com_addr[j * 2]
			printf("%-8s%d\t: 0x%.8x\n", "COM_ADDR_H", j, buffer[1]); // ctlreg[i].com_addr[j * 2 + 1]
		}
		dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address + (64 * (2 + i)) + (0x64/4)), buffer, 36);
		printf("%-8s\t: %d\n", "LOAD START", buffer[7]); //lstart_cnt
		printf("%-8s\t: %d\n", "LOAD END", buffer[3]); //lend_cnt
		printf("%-8s\t: %d\n", "SAVE START", buffer[6]); //sstart_cnt
		printf("%-8s\t: %d\n", "SAVE END", buffer[2]); //send_cnt
		printf("%-8s\t: %d\n", "CONV START", buffer[5]); //cstart_cnt
		printf("%-8s\t: %d\n", "CONV END", buffer[1]);  //cend_cnt
		printf("%-8s\t: %d\n", "MISC START", buffer[4]); //pstart_cnt
		printf("%-8s\t: %d\n", "MISC END", buffer[0]); //pend_cnt
		printf("%-8s\t: 0x%.8x\n", "AXI_STATUS", buffer[8]); //axi_status
		dpu_aol_read_regs(p_signature, (uint64_t)(dpu_base_address + (64 * (2 + i)) + (0xB0/4)), buffer, 4);
		printf("%-8s\t: 0x%.8x\n", "TIMER_STATUS", buffer[0]); //timer_status
//#endif
		printf("\n");
	}
}


/**
 * show_ext_regs - show dpu extension module registers
 * @mask  : extension module's mask
 */
void show_ext_regs(dpu_aol_dev_handle_t *p_signature, dpu_caps_t *p_caps) {
	uint32_t ip_handle_index;
	uint32_t *ip_base_address;
	uint32_t buffer[17];

	ip_handle_index = p_signature->core_count[IP_ID_VER_REG];
	ip_handle_index += p_signature->core_count[IP_ID_DPU];
	ip_base_address = (uint32_t *)p_signature->core_phy_addr[ip_handle_index];
//	printf("softmax base addr: 0x%x\n", ip_base_address);

	if (p_caps->softmax.enable || p_caps->fullconnect.enable) {
		dpu_aol_read_regs(p_signature, (uint64_t)(ip_base_address), buffer, 17 * 4);
		printf("[SMFC Registers]\n");
		printf("%-8s\t: 0x%.8x\n", "DONE", buffer[0]); // regs->done
		printf("%-8s\t: 0x%.8x\n", "SM_LEN_X", buffer[1]); // regs->sm_len_x
		printf("%-8s\t: 0x%.8x\n", "SM_LEN_Y", buffer[2]); // regs->sm_len_y
		printf("%-8s\t: 0x%.8x\n", "SRC", buffer[3]); // regs->src
		printf("%-8s\t: 0x%.8x\n", "DST", buffer[4]); // regs->dst
		printf("%-8s\t: 0x%.8x\n", "SCALE", buffer[5]); // regs->scale
		printf("%-8s\t: 0x%.8x\n", "SM_OFFSET", buffer[6]); // regs->sm_offset
		printf("%-8s\t: 0x%.8x\n", "CLR", buffer[7]); // regs->clr
		printf("%-8s\t: 0x%.8x\n", "START", buffer[8]); // regs->start
		printf("%-8s\t: 0x%.8x\n", "FC_INPUT_CHANNEL", buffer[9]); // regs->fc_input_channel
		printf("%-8s\t: 0x%.8x\n", "FC_OUTPUT_CHANNEL", buffer[10]); // regs->fc_output_channel
		printf("%-8s\t: 0x%.8x\n", "FC_BATCH", buffer[11]); // regs->fc_batch
		printf("%-8s\t: 0x%.8x\n", "FC_WEIGHT_START", buffer[12]); // regs->fc_weight_start
		printf("%-8s\t: 0x%.8x\n", "FC_WEIGHT_END", buffer[13]); // regs->fc_weight_end
		printf("%-8s\t: 0x%.8x\n", "CALC_MOD", buffer[14]); // regs->calc_mod
		printf("%-8s\t: 0x%.8x\n", "DST_ADDR_SEL", buffer[15]); // regs->dst_addr_sel
		printf("%-8s\t: 0x%.8x\n", "FC_RELU_EN", buffer[16]); // regs->fc_relu_en
	}
}
