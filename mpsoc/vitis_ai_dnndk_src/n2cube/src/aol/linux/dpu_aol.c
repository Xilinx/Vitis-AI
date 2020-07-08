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
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

#include "../dpu_aol.h"
#include "dpu_hal.h"

#define DPU_DEV_NAME "/dev/dpu"
#define PAGE_SIZE    sysconf(_SC_PAGE_SIZE)

typedef struct {
    dpu_aol_dev_handle_t device;
    int fd;
} dpu_aol_linux_t;

/*
 * Modify the size of memory segment for system page size alignment
 */
unsigned long align_for_page_size(unsigned long size)
{
    if (size % PAGE_SIZE) {
        return ((size/PAGE_SIZE) * PAGE_SIZE + PAGE_SIZE);
    }

    return size;
}

/* Attach DPU device and other IPs.
 * Input:
 *     mode - Conventional the DPU scheduling mode. Only DPU_SCHEDULE_MODE_SINGLE used currently.
 * Return:
 *     The device handle that the subsequent function needs to usd.
 */
dpu_aol_dev_handle_t *dpu_aol_attach(uint32_t mode){
    int fd;
    int ret;
    dpu_aol_linux_t *p = NULL;

    if (mode != DPU_SCHEDULE_MODE_SINGLE) {
        return NULL;
    }

    p = (dpu_aol_linux_t *)malloc(sizeof(dpu_aol_linux_t));
    memset(p, 0, sizeof(dpu_aol_linux_t));

    fd = open(DPU_DEV_NAME, O_RDWR | O_SYNC, 0);
    if (fd < 0) {
        return NULL;
    }
    p->fd = fd;

    // Get DPU handle
    ret = ioctl(p->fd, REQ_GET_DEV_HANDLE, (void *)&p->device);
    if (ret < 0) {
        return NULL;
    }

    p->device.aol_version = 0x0100;
    return &(p->device);
}

/* Detach DPU device and other IPs.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Return:
 *     The device handle that the subsequent function needs to usd.
 */
int dpu_aol_detach(dpu_aol_dev_handle_t *dev) {
    int ret;
    dpu_aol_linux_t *p;

    if (dev == NULL) {
        return DPU_AOL_ERROR;
    }

    p = (dpu_aol_linux_t *)dev;
    ret = close(p->fd);
    free(p);

    if (ret) {
        return DPU_AOL_ERROR;
    }

    return DPU_AOL_OK;
}

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
int dpu_aol_read_regs(dpu_aol_dev_handle_t *dev, uint64_t phy_address, uint32_t *buf, uint32_t count) {
    int i, ret;
    int counter = 0;
    dpu_aol_linux_t *p = (dpu_aol_linux_t *)dev;
    dpu_aol_read_regs_t read_reg;

    if ((dev == NULL) || ((count >> 2) > READ_REG_DEFAULT_BUF_LEN)) {
        return -1;
    }

    // Get DPU handle
    read_reg.size = count;
    read_reg.phy_address = phy_address;
    ret = ioctl(p->fd, REQ_DPU_READ_REGS, (void *)&read_reg);
    if (ret < 0) {
        return -1;
    }
    for(i = 0; i < (count >> 2); i++) {
        buf[i] = read_reg.out_buffer[i];
    }

    return 0;
}

/* Allocate physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     size - Byte length of the memory to be allocated.
 *     prot - Fixed to (DPU_AOL_MEM_PROT_READ|DPU_AOL_MEM_PROT_WRITE) in this version.
 * Return:
 *     The handle of the requedsted memory. NULL for failure.
 */
dpu_aol_dev_mem_t *dpu_aol_alloc_dev_mem(dpu_aol_dev_handle_t *dev, uint64_t size, uint32_t prot) {
    int ret;
    void *map;
    dpu_aol_linux_t *p;
    struct req_mem_alloc_t req_alloc;

    req_alloc.size = align_for_page_size(size);
    p = (dpu_aol_linux_t *)dev;
    ret = ioctl(p->fd, REQ_DPU_MEM_ALLOC, (void *)&req_alloc);
    if (ret < 0) {
        return NULL;
    }

    dpu_aol_dev_mem_t *mem = (dpu_aol_dev_mem_t *)malloc(sizeof(dpu_aol_dev_mem_t));
    mem->size = req_alloc.size;
    mem->addr_phy = req_alloc.addr_phy;
    map = mmap(NULL, mem->size, PROT_READ | PROT_WRITE, MAP_SHARED, p->fd, mem->addr_phy);
    if (map == MAP_FAILED) {
        free(mem);
        return NULL;
    }

    mem->addr_virt = (uint64_t)map;
    return mem;
}

/* Free physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_free_dev_mem(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem) {
    int ret;
    dpu_aol_linux_t *p;
    struct req_mem_free_t req_free;

    ret = munmap((void *)mem->addr_virt, mem->size);
    if (ret != 0) {
        return DPU_AOL_ERROR;
    }

    req_free.addr_phy = mem->addr_phy;
    p = (dpu_aol_linux_t *)dev;
    ret = ioctl(p->fd, REQ_DPU_MEM_FREE, (void *)&req_free);
    if (ret < 0) {
        return DPU_AOL_ERROR;
    }

    return DPU_AOL_OK;
}

/* Memory accessible from the CPU, synchronized to memory that the device can access.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 *     offset - The byte offset of memory address from mem->addr_phy needs to be flushed.
 *     size - The byte length of memory needs to be flushed.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_sync_to_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset, uint32_t size) {
    int ret;
    req_cache_ctrl_t cache_ctrl;
    dpu_aol_linux_t *p = (dpu_aol_linux_t *)dev;

    cache_ctrl.size = size;
    cache_ctrl.addr_phy = (unsigned long)(mem->addr_phy + offset);
    ret = ioctl(p->fd, REQ_SYNC_TO_DEV, (void *)&cache_ctrl);
    if (ret >= 0) {
        return DPU_AOL_OK;
    }

    return DPU_AOL_ERROR;
}

/* Memory accessible from the device, synchronized back to the memory that the CPU can access.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 *     offset - The byte offset of memory address from mem->addr_phy needs to be flushed.
 *     size - The byte length of memory needs to be flushed.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_sync_from_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset, uint32_t size) {
    int ret;
    req_cache_ctrl_t cache_ctrl;
    dpu_aol_linux_t *p = (dpu_aol_linux_t *)dev;

    cache_ctrl.size = size;
    cache_ctrl.addr_phy = (unsigned long)(mem->addr_phy + offset);
    ret = ioctl(p->fd, REQ_SYNC_FROM_DEV, (void *)&cache_ctrl);
    if (ret >= 0) {
        return DPU_AOL_OK;
    }

    return DPU_AOL_ERROR;
}

/* Make a DPU or other IP schedule.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Input/Output:
 *     data - The data required for this scheduling. See dpu_aol_run_t for detail.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for timeout.
 */
int dpu_aol_run(dpu_aol_dev_handle_t *dev, dpu_aol_run_t *data) {
    int ret;
    dpu_aol_linux_t *p = (dpu_aol_linux_t *)dev;

    ret = ioctl(p->fd, REQ_DPU_RUN, (void *)data);
    if (ret < 0) {
        return DPU_AOL_ERROR;
    }

    return DPU_AOL_OK;
}

/* Initialize DPU or other IPs. It may be called when the IP first starts or times out.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Input/Output:
 *     data - The data required for this scheduling. See dpu_aol_init_t for detail.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_init(dpu_aol_dev_handle_t *dev, dpu_aol_init_t *data) {
    int ret;
    dpu_aol_linux_t *p = (dpu_aol_linux_t *)dev;

    ret = ioctl(p->fd, REQ_DPU_INIT, (void *)data);
    if (ret < 0) {
        return ret;
    }

    return DPU_AOL_OK;
}
