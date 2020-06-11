/*
 * Copyright (C) 2019 Xilinx, Inc.
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 */

#ifndef _DPUCORE_H_
#define _DPUCORE_H_

#include <asm/cacheflush.h>
#include <asm/delay.h>
#include <asm/io.h>
#include <asm/irq.h>
#include <asm/thread_info.h>
#include <asm/uaccess.h>
#include <linux/debugfs.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/dma-direction.h>
#include <linux/export.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/list.h>
#include <linux/mempolicy.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_irq.h>
#include <linux/platform_device.h>
#include <linux/proc_fs.h>
#include <linux/sched.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/stat.h>
#include <linux/version.h>
#include <linux/wait.h>

#include "dpudef.h"

#define DPU_DRIVER_VERSION "4.0.0"

#define DPU_EXT_HDMI (1 << 1)
#define DPU_EXT_BT1120 (1 << 2)
#define DPU_EXT_FULLCONNECT (1 << 3)
#define DPU_EXT_SOFTMAX (1 << 4)
#define DPU_EXT_RESIZE (1 << 5)

#define SIG_BASE_NULL 0X00000000
#ifdef SIG_BASE_ADDR
#define SIG_BASE SIG_BASE_ADDR
#else
#define SIG_BASE SIG_BASE_NULL
#endif

#define SIG_BASE_MASK 0XFF000000
#define DPU_BASE(signature)             (((signature) & SIG_BASE_MASK) + 0x0000)
#define DPU_SIZE                        0X00000700
#define DPU_EXT_SOFTMAX_BASE(signature) (((signature) & SIG_BASE_MASK) + 0x0700)
#define DPU_EXT_SOFTMAX_SIZE            0X00000041
#define MAX_REG_SIZE                    0X00001000

/*dpu signature magic number*/
#define SIG_MAGIC 0X4450

#define SIG_SIZE_MASK   0XFF000000
#define SIG_VER_MASK    0X00FF0000
#define SIG_MAGIC_MASK  0X0000FFFF

#define DPU_CORENUM_MASK 0X0000000F
#define SOFTMAX_VLD_MASK 0X01000000

#define FALSE 0
#define TRUE 1

#define dprint(fmt, args...)                          	\
	do {                                               	\
		printk(KERN_ERR "[DPU][%d]" fmt, current->pid,  \
		       ##args);                                 \
	} while (0)

#define dpr_init(fmt, args...) pr_alert("[DPU][%d]" fmt, current->pid, ##args);

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
		volatile uint32_t _rsv[39];

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
} softmax_reg_t;

typedef struct {
	wait_queue_head_t waitqueue;
	struct semaphore dpu_lock;
	int irq_no;
	int irq_flag;
}dpu_intrrupt_data_t;

/*memory block node struct*/
struct memblk_node {
	unsigned long size;
	unsigned long virt_addr;
	dma_addr_t phy_addr;
	struct list_head list;
};

#endif /*_DPU_H_*/
