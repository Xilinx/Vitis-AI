/* SPDX-License-Identifier: GPL-2.0 OR Apache-2.0 */
/*
 *  Copyright 2022-2023 Advanced Micro Devices, Inc.
 *
 * This file is dual-licensed; you may select either the GNU General Public
 * License version 2 or Apache License, Version 2.0.
 *
 */

#ifndef _DPU_H_
#define _DPU_H_

typedef uint64_t u64;
typedef uint32_t u32;

#define DPU_NUM(x) (((x)&0x0f) << 0)
#define SFM_NUM(x) ((x >> 4) & 0x0f)

#define DPCMA_FROM_CPU_TO_DEVICE (0)
#define DPCMA_FROM_DEVICE_TO_CPU (1)

struct dpcma_req_free {
  u64 phy_addr;
  size_t capacity;
};

struct dpcma_req_alloc {
  size_t size;
  u64 phy_addr;
  size_t capacity;
};

struct dpcma_req_sync {
  u64 phy_addr;
  size_t size;
  int direction;
};

struct ioc_kernel_run_t {
  u64 addr_code;  /* the address for DPU code */
  u64 addr0;      /* address reg0 */
  u64 addr1;      /* address reg1 */
  u64 addr2;      /* address reg2 */
  u64 addr3;      /* address reg3 */
  u64 addr4;      /* address reg4 */
  u64 addr5;      /* address reg5 */
  u64 addr6;      /* address reg6 */
  u64 addr7;      /* address reg7 */
  u64 time_start; /* the start timestamp before running (RETURNED) */
  u64 time_end;   /* the end timestamp after running (RETURNED) */
  u64 counter;
  int core_id; /* the core id of the task*/
  u32 pend_cnt;
  u32 cend_cnt;
  u32 send_cnt;
  u32 lend_cnt;
  u32 pstart_cnt;
  u32 cstart_cnt;
  u32 sstart_cnt;
  u32 lstart_cnt;
};

struct ioc_softmax_t {
  u32 width;  /* width dimention of Tensor */
  u32 height; /* height dimention of Tensor */
  u64 input;  /* physical address of input Tensor */
  u64 output; /* physical address of output Tensor */
  u32 scale;  /* quantization info of input Tensor */
  u32 offset; /* offset value for input Tensor */
};

#define DPU_IOC_MAGIC 'D'

#define DPUIOC_CREATE_BO _IOWR(DPU_IOC_MAGIC, 1, struct dpcma_req_alloc*)
#define DPUIOC_FREE_BO _IOWR(DPU_IOC_MAGIC, 2, struct dpcma_req_free*)
#define DPUIOC_SYNC_BO _IOWR(DPU_IOC_MAGIC, 3, struct dpcma_req_sync*)
#define DPUIOC_G_INFO _IOR(DPU_IOC_MAGIC, 4, u32)
#define DPUIOC_G_TGTID _IOR(DPU_IOC_MAGIC, 5, u64)
#define DPUIOC_RUN _IOWR(DPU_IOC_MAGIC, 6, struct ioc_kernel_run_t*)
#define DPUIOC_RUN_SOFTMAX _IOWR(DPU_IOC_MAGIC, 7, struct ioc_softmax_t*)

#endif /*_DPU_H_*/
