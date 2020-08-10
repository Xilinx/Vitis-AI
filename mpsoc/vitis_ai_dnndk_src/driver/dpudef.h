/*
 * Copyright (C) 2019 Xilinx, Inc.
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 */

#ifndef _DPU_DEF_H_
#define _DPU_DEF_H_

#define DPU_IOCTL_MAGIC 'D'

/* allocate DPU memory */
#define DPU_IOCTL_MEM_ALLOC       _IOWR(DPU_IOCTL_MAGIC, 1, struct ioc_mem_alloc_t *)
/* free DPU memory */
#define DPU_IOCTL_MEM_FREE        _IOWR(DPU_IOCTL_MAGIC, 2, struct ioc_mem_free_t *)
/* run DPU */
#define DPU_IOCTL_RUN             _IOWR(DPU_IOCTL_MAGIC, 3, ioc_aol_run_t *)
/* init dpu registers */
#define DPU_IOCTL_INIT            _IOWR(DPU_IOCTL_MAGIC, 4, ioc_aol_init_t *)
/* Memory accessible from the CPU, synchronized to memory that the device can access */
#define DPU_IOCTL_SYNC_TO_DEV     _IOWR(DPU_IOCTL_MAGIC, 5, ioc_cache_ctrl_t *)
/* Memory accessible from the device, synchronized back to the memory that the CPU can access */
#define DPU_IOCTL_SYNC_FROM_DEV   _IOWR(DPU_IOCTL_MAGIC, 6, ioc_cache_ctrl_t *)
/* Get the cores physical address */
#define DPU_IOCTL_GET_DEV_HANDLE  _IOWR(DPU_IOCTL_MAGIC, 7, ioc_aol_device_handle_t *)
/* read the registers of the IPs */
#define DPU_IOCTL_READ_REGS       _IOWR(DPU_IOCTL_MAGIC, 8, ioc_aol_read_regs_t *)

#define SUPPORT_IP_MAX_COUNT 16
#define SUPPORT_CORE_MAX_COUNT 32

struct ioc_mem_alloc_t {
	unsigned long size; /* size of memory space to be allocated */
	unsigned long addr_phy; /* suv the start pyhsical address of allocated DPU memory (RETURNED) */
};

struct ioc_mem_free_t {
	unsigned long addr_phy; /* the start pyhsical address of allocated DPU memory */
};

typedef struct {
    unsigned long addr_phy; /* physical address of memory range */
    unsigned long size;     /* size of memory range */
}ioc_cache_ctrl_t;

#define READ_REG_DEFAULT_BUF_LEN 64
typedef struct {
	uint64_t phy_address;
	uint32_t byte_size;
	uint32_t out_buffer[READ_REG_DEFAULT_BUF_LEN];
} ioc_aol_read_regs_t;

typedef struct {
    uint32_t offset;
    uint32_t value;
}ioc_aol_reg_t;

/*
 * ID of each IP tha may be included in the system.
 */
typedef enum {
	IP_ID_VER_REG = 0,
	IP_ID_DPU,
	IP_ID_SOFTMAX,
	IP_ID_FULLCONNECT,
	IP_ID_RESIZE,
	IP_ID_SIGMOID,
	IP_MAX_COUNT,
}ioc_aol_ip_id_t;

#define DPU_AOL_REG_NUM 32
typedef struct {
	uint64_t time_start;   /*[Output] The start timestamp in nano-second */
    uint64_t time_end;     /*[Output] The end timestamp in nano-second */
    uint32_t timeout;      /*[Input] The timeout setting for IP computing in second */
    uint32_t core_mask;    /*[Input] Specify the core to be scheduled, each bit represents a core */
    uint32_t reg_count;    /*[Input] Specify the count of registers to be written. No more than DPU_AOL_REG_NUM. */
    ioc_aol_ip_id_t ip_id; /*[Input] Specify the ip_id to be scheduled */
    ioc_aol_reg_t regs[DPU_AOL_REG_NUM]; /*[Input] The registers data buffer to be written. The actual count is specified by reg_count. */
} ioc_aol_run_t;

typedef struct {
	uint32_t core_mask;    /*[Input] Specify the core to be scheduled, each bit represents a core */
    uint32_t reg_count;    /*[Input] Specify the count of registers to be written. No more than DPU_AOL_REG_NUM. */
    ioc_aol_ip_id_t ip_id; /*[Input] Specify the ip_id to be scheduled */
    ioc_aol_reg_t regs[DPU_AOL_REG_NUM];     /*[Input] The registers data buffer to be written. The actual count is specified by reg_count. */
    uint32_t regs_delay_us[DPU_AOL_REG_NUM]; /*[Input] The delay time array in microsecond after writing each register specified by regs. */
} ioc_aol_init_t;

typedef struct {
	uint32_t aol_version;                           /*[Output] The version of AOL interface, fixed to 0x0100 */
    uint8_t core_count[SUPPORT_IP_MAX_COUNT];       /*[Output] The core count of each related DPU IP. The order according dpu_aol_ip_id_t. */
    uint64_t core_phy_addr[SUPPORT_CORE_MAX_COUNT]; /*[Output] The physical address of each IP core. */
} ioc_aol_device_handle_t;

#endif

