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

//------------------------------------------------------
// The DPU AOL version 1.0.0
//------------------------------------------------------

#ifndef __DPU_AOL_H__
#define __DPU_AOL_H__

#ifdef __cplusplus
extern "C" {
#endif

/*
 * define DPU AOL status code
 */
#define DPU_AOL_OK      (0)
#define DPU_AOL_ERROR   (-1)

#define SUPPORT_IP_MAX_COUNT 16
#define SUPPORT_CORE_MAX_COUNT 32

typedef struct
{
    uint32_t size;
    uint64_t addr_phy;
    uint64_t addr_virt;
} dpu_aol_dev_mem_t;

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
}dpu_aol_ip_id_t;

typedef struct {
    uint32_t aol_version;                           /*[Output] The version of AOL interface, fixed to 0x0100 */
    uint8_t core_count[SUPPORT_IP_MAX_COUNT];       /*[Output] The core count of each related DPU IP. The order according dpu_aol_ip_id_t. */
    uint64_t core_phy_addr[SUPPORT_CORE_MAX_COUNT]; /*[Output] The physical address of each IP core. */
} dpu_aol_dev_handle_t;

typedef struct {
    uint32_t offset;
    uint32_t value;
}dpu_aol_reg_t;

#define DPU_AOL_REG_NUM 32
typedef struct {
    uint64_t time_start;   /*[Output] The start timestamp in nano-second */
    uint64_t time_end;     /*[Output] The end timestamp in nano-second */
    uint32_t timeout;      /*[Input] The timeout setting for IP computing in second */
    uint32_t core_mask;    /*[Input] Specify the core to be scheduled, each bit represents a core */
    uint32_t reg_count;    /*[Input] Specify the count of registers to be written. No more than DPU_AOL_REG_NUM. */
    dpu_aol_ip_id_t ip_id; /*[Input] Specify the ip_id to be scheduled */
    dpu_aol_reg_t regs[DPU_AOL_REG_NUM]; /*[Input] The registers data buffer to be written. The actual count is specified by reg_count. */
} dpu_aol_run_t;

typedef struct {
    uint32_t core_mask;    /*[Input] Specify the core to be scheduled, each bit represents a core */
    uint32_t reg_count;    /*[Input] Specify the count of registers to be written. No more than DPU_AOL_REG_NUM. */
    dpu_aol_ip_id_t ip_id; /*[Input] Specify the ip_id to be scheduled */
    dpu_aol_reg_t regs[DPU_AOL_REG_NUM];     /*[Input] The registers data buffer to be written. The actual count is specified by reg_count. */
    uint32_t regs_delay_us[DPU_AOL_REG_NUM]; /*[Input] The delay time array in microsecond after writing each register specified by regs. */
} dpu_aol_init_t;

/* Attach DPU device and other IPs.
 * Input:
 *     mode - Conventional the DPU scheduling mode. Only DPU_SCHEDULE_MODE_SINGLE used currently.
 * Return:
 *     The device handle that the subsequent function needs to usd.
 */
#define DPU_SCHEDULE_MODE_SINGLE 1
dpu_aol_dev_handle_t *dpu_aol_attach(uint32_t mode);

/* Detach DPU device and other IPs.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Return:
 *     The device handle that the subsequent function needs to usd.
 */
int dpu_aol_detach(dpu_aol_dev_handle_t *dev);

/* Read the DPU related IPs registers in word.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     phy_address - The physical address of the registers to be read.
 *     count - Byte lenght of the read register data.
 * Output:
 *     buf - The output buffer in word.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_read_regs(dpu_aol_dev_handle_t *dev, uint64_t phy_address, uint32_t *buf, uint32_t count);

/* Initialize DPU or other IPs. It may be called when the IP first starts or times out.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Input/Output:
 *     data - The data required for this scheduling. See dpu_aol_init_t for detail.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_init(dpu_aol_dev_handle_t *dev, dpu_aol_init_t *data);

/* Make a DPU or other IP schedule.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Input/Output:
 *     data - The data required for this scheduling. See dpu_aol_run_t for detail.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for timeout.
 */
int dpu_aol_run(dpu_aol_dev_handle_t *dev, dpu_aol_run_t *data);

/* Allocate physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     size - Byte length of the memory to be allocated.
 *     prot - Fxied to (DPU_AOL_MEM_PROT_READ|DPU_AOL_MEM_PROT_WRITE) in this version.
 * Return:
 *     The handle of the requedsted memory. NULL for failure.
 */
#define DPU_AOL_MEM_PROT_NONE 0
#define DPU_AOL_MEM_PROT_READ 1
#define DPU_AOL_MEM_PROT_WRITE 2
#define DPU_AOL_MEM_PROT_EXEC 4
dpu_aol_dev_mem_t *dpu_aol_alloc_dev_mem(dpu_aol_dev_handle_t *dev, uint64_t size, uint32_t prot);

/* Free physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_free_dev_mem(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem);

/* Memory accessible from the CPU, synchronized to memory that the device can access.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 *     offset - The byte offset of memory address from mem->addr_phy needs to be flushed.
 *     size - The byte length of memory needs to be flushed.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_sync_to_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset, uint32_t size);

/* Memory accessible from the device, synchronized back to the memory that the CPU can access.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 *     offset - The byte offset of memory address from mem->addr_phy needs to be flushed.
 *     size - The byte length of memory needs to be flushed.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_sync_from_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif
