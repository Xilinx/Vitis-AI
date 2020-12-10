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

#ifndef _DPU_HAL_H_
#define _DPU_HAL_H_

#define DPU_IOCTL_MAGIC 'D'

/* allocate DPU memory */
#define REQ_DPU_MEM_ALLOC           _IOWR(DPU_IOCTL_MAGIC, 1, struct req_mem_alloc_t *)
/* free DPU memory */
#define REQ_DPU_MEM_FREE            _IOWR(DPU_IOCTL_MAGIC, 2, struct req_mem_free_t *)
/* run DPU */
#define REQ_DPU_RUN                 _IOWR(DPU_IOCTL_MAGIC, 3, dpu_aol_run_t *)
/* init dpu registers */
#define REQ_DPU_INIT                _IOWR(DPU_IOCTL_MAGIC, 4, dpu_aol_init_t *)
/* Memory accessible from the CPU, synchronized to memory that the device can access */
#define REQ_SYNC_TO_DEV             _IOWR(DPU_IOCTL_MAGIC, 5, req_cache_ctrl_t*)
/* Memory accessible from the device, synchronized back to the memory that the CPU can access */
#define REQ_SYNC_FROM_DEV           _IOWR(DPU_IOCTL_MAGIC, 6, req_cache_ctrl_t*)
/* Get the cores physical address */
#define REQ_GET_DEV_HANDLE          _IOWR(DPU_IOCTL_MAGIC, 7, dpu_aol_dev_handle_t *)
/* read the registers of the IPs */
#define REQ_DPU_READ_REGS           _IOWR(DPU_IOCTL_MAGIC, 8, dpu_aol_read_regs_t*)

struct req_mem_alloc_t {
    unsigned long       size;        /* size of memory space to be allocated */
    unsigned long       addr_phy;    /* suv the start pyhsical address of allocated DPU memory (RETURNED) */
};

struct req_mem_free_t {
    unsigned long       addr_phy;    /* the start pyhsical address of allocated DPU memory */
};

typedef struct {
    unsigned long addr_phy;         /* physical address of memory range */
    unsigned long size;             /* size of memory range */
}req_cache_ctrl_t;

#define READ_REG_DEFAULT_BUF_LEN 64
typedef struct {
	uint64_t phy_address;
	uint32_t size;
	uint32_t out_buffer[READ_REG_DEFAULT_BUF_LEN];
} dpu_aol_read_regs_t;

#endif

