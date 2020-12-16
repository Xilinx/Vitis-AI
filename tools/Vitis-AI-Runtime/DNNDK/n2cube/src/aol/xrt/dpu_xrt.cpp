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

#include <fcntl.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <map>
#include <thread>

#include <xrt/ert.h>
#include <xrt/xclbin.h>
#include <xrt/xclhal2.h>

#include "../dpu_aol.h"
#include "dpu_xrt.h"

using namespace std::chrono;
using namespace std;

//#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                          \
  {                                                                       \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "us" << endl;                 \
  }
#else
#define _T(func) func;
#endif

#define PAGESIZE 4096

int card_index = 0;
int timeout = 2;  // (s)

static xuid_t read_reg_uuid;

xclDeviceHandle mdev_handle = nullptr;
unsigned int user_core_mask = 0xFFFFFFFF;
char xclbin_path[200];
// Global System configuration pointer
sys_conf_t g_sys_config;
sys_conf_t *SYSCONF = nullptr;

inline void _check_dev_handle(xclDeviceHandle &handle) {
  if (handle == nullptr) {
    handle = xclOpen(card_index, NULL, XCL_ERROR);
    if (handle == nullptr) {
      log_err("[TID:%lu] xclOpen Error, ret = %p\n", this_thread::get_id(), handle);
      exit(-1);
    }
  }
}

/**
 * _get_cu_mask - convert the user's mask value to cu_mask in XRT context
 *
 **/
uint32_t _get_cu_mask(dpu_aol_run_t *run_data) {
  uint32_t mask = run_data->core_mask;
  uint32_t ret_mask = 0;
  int idx = 0;
  while (mask) {
    if (idx >= SYSCONF->dpu_core_num) {
      return ret_mask;
    }
    if (mask & 1) {
      switch (run_data->ip_id) {
        case IP_ID_DPU:
          ret_mask |= (1 << SYSCONF->dpu_conf[idx].cu_index);
          break;
        case IP_ID_SOFTMAX:
          ret_mask |= (1 << SYSCONF->sm_conf[idx].cu_index);
          break;
        default:
          break;
      }
    }
    mask >>= 1;
    idx++;
  }
  return ret_mask;
}
/**
 *  dpu_run -run dpu function
 * @prun : dpu run struct, contains the necessary address info
 *
 */

int run_dpu(dpu_aol_run_t *run_data) {
  static thread_local xclDeviceHandle dpu_handle = nullptr;
  static thread_local unsigned run_bo = 0;
  static thread_local void *bo_data = nullptr;

  _check_dev_handle(dpu_handle);

  if (run_bo == 0) {
    run_bo = xclAllocBO(dpu_handle, 4096, xclBOKind(0), XCL_BO_FLAGS_EXECBUF);
    bo_data = xclMapBO(dpu_handle, run_bo, true);
  }

  auto ecmd = reinterpret_cast<ert_start_kernel_cmd *>(bo_data);
  auto rsz = DPU_CONFIG_REG_LEN / 4 + 2;  // regmap array size
  memset(ecmd, 0, (sizeof *ecmd) + rsz);
  ecmd->state = ERT_CMD_STATE_NEW;
  ecmd->opcode = ERT_START_CU;
  ecmd->count = 1 + rsz;

  ecmd->cu_mask = _get_cu_mask(run_data);

  log_dbg("Configure DPU Registers\n");
  // for (int i = 0; i < run_data->reg_count; i++) {
  //   dpu_aol_reg_t reg = run_data->regs[i];
  //   ecmd->data[reg.offset / 4] = reg.value;
  //   log_dbg("offset: 0x%.4x, value: 0x%.8x\n", reg.offset, reg.value);
  // }
  // todo for debug only
  log_dbg("code:%x, %x, %x\n", run_data->regs[8].value, run_data->regs[0].value,
          run_data->regs[1].value);
  uint32_t *pReg = (uint32_t *)ecmd->data;

  for (uint32_t i = 0; i < run_data->reg_count; i++) {
    pReg[run_data->regs[i].offset >> 2] = run_data->regs[i].value;
  }

  int ret;
  if ((ret = xclExecBuf(dpu_handle, run_bo)) != 0) {
//    log_err("Unable to triggle DPU, error = %d\n", ret);
    return ret;
  }

  do {
    ret = xclExecWait(dpu_handle, run_data->timeout * 1000);
    if (ret == 0) {
//      log_err("DPU Task Time out, state = %d, cu_mask = 0x%x\n", ecmd->state, ecmd->cu_mask);
      return -1;

    } else if (ecmd->state == ERT_CMD_STATE_COMPLETED) {
      run_data->time_end = ((uint64_t)ecmd->data[2] * (uint64_t)1000000 + (uint64_t)ecmd->data[3]) * (uint64_t)1000;
      run_data->time_start = ((uint64_t)ecmd->data[0] * (uint64_t)1000000 + (uint64_t)ecmd->data[1]) * (uint64_t)1000;
      log_info("DPU Task Done, core = 0x%x , time = %7ld us\n", ecmd->cu_mask,
               run_data->time_end - run_data->time_start);
      break;
    } else if (ecmd->state == ERT_CMD_STATE_TIMEOUT) {
//      log_err("DPU Task Time out, state = %d, cu_mask = 0x%x\n", ecmd->state, ecmd->cu_mask);
      return -1;
    }
  } while (ret);

  return 0;
}

int run_softmax(dpu_aol_run_t *run_data) {
  static thread_local xclDeviceHandle sm_handle = nullptr;
  static thread_local unsigned run_bo = 0;
  static thread_local void *bo_data = nullptr;
  uint32_t *pReg;

  _check_dev_handle(sm_handle);

  if (run_bo == 0) {
    run_bo = xclAllocBO(sm_handle, 4096, xclBOKind(0), XCL_BO_FLAGS_EXECBUF);
    bo_data = xclMapBO(sm_handle, run_bo, true);
  }

  auto ecmd = reinterpret_cast<ert_start_kernel_cmd *>(bo_data);
  memset(ecmd, 0, (sizeof *ecmd) + sizeof(SOFTMAX_REG) / 4);
  ecmd->state = ERT_CMD_STATE_NEW;
  ecmd->opcode = ERT_START_CU;
  ecmd->count = 1 + sizeof(SOFTMAX_REG) / 4;
  ecmd->cu_mask = _get_cu_mask(run_data);

  log_dbg("Configure Softmax Registers\n");
  pReg = (uint32_t *)ecmd->data;

  for (int i = 0; i < run_data->reg_count; i++) {
    pReg[run_data->regs[i].offset / 4] = run_data->regs[i].value;
    log_dbg("offset: 0x%.4x, value: 0x%.8x\n", run_data->regs[i].offset, run_data->regs[i].value);
  }

  int ret;
  if ((ret = xclExecBuf(sm_handle, run_bo)) != 0) {
//    log_err("Unable to triggle SOFTMAX IP, error = %d\n", ret);
    return ret;
  }

  do {
    ret = xclExecWait(sm_handle, run_data->timeout * 1000);
    if (ret == 0) {
//      log_err("SOFTMAX Task Time out, state = %d, cu_mask = 0x%x\n", ecmd->state, ecmd->cu_mask);
      return -1;

    } else if (ecmd->state == ERT_CMD_STATE_COMPLETED) {
      uint64_t time_end = (ecmd->data[2] * 1000000 + ecmd->data[3]) * 1000;
      uint64_t time_start = (ecmd->data[0] * 1000000 + ecmd->data[1]) * 1000;
      log_info("SOFTMAX Task Done, core = 0x%x , time = %7ld us\n", ecmd->cu_mask,
               time_end - time_start);
      break;
    } else if (ecmd->state == ERT_CMD_STATE_TIMEOUT) {
//      log_err("SOFTMAX Task Time out, state = %d, cu_mask = 0x%x\n", ecmd->state, ecmd->cu_mask);
      return -1;
    }
  } while (ret);

  return 0;
}

static int _init_xrt(const char *bit) {
  if (card_index >= xclProbe()) {
    log_err("Cannot find device, index: %d\n", card_index);
    exit(-1);
  }
  _check_dev_handle(mdev_handle);

  if (!bit || !strlen(bit)) {
    log_err("Invalid bitstream file namen");
    exit(-1);
  }
  char tempFileName[1024];
  strcpy(tempFileName, bit);
  ifstream stream(bit);
  if (!stream) {
    log_err("Invalid bitstream: %s\n", bit);
    exit(-1);
  }
  stream.seekg(0, stream.end);
  int size = stream.tellg();
  stream.seekg(0, stream.beg);

  char *header = new char[size];
  stream.read(header, size);

  if (strncmp(header, "xclbin2", 8)) {
    log_err("Invalid bitstream: %s\n", bit);
    exit(-1);
  }

  if (xclLoadXclBin(mdev_handle, (const xclBin *)header)) {
    delete[] header;
    log_err("Bitstream download failed\n");
    exit(-1);

  } else {
    log_info("Finished downloading bitstream\n");
  }

  const axlf *top = (const axlf *)header;
  auto ip = xclbin::get_axlf_section(top, IP_LAYOUT);
  struct ip_layout *layout = (ip_layout *)(header + ip->m_sectionOffset);

  memcpy(read_reg_uuid, top->m_header.uuid, sizeof(xuid_t));

  int dpu_core_num = 0, sm_core_num = 0;
  map<uint64_t, int> cu_addr_map;
  SYSCONF->dpu_core_mask = 0;
  SYSCONF->sm_core_mask = 0;
  for (int i = 0; i < layout->m_count; ++i) {
    if (layout->m_ip_data[i].m_type != IP_KERNEL) continue;
    uint64_t cu_base_addr = layout->m_ip_data[i].m_base_address;
    if (strncmp((const char *)(layout->m_ip_data[i].m_name), "dpu_xrt_top",
                strlen("dpu_xrt_top")) == 0) {
      cu_addr_map[cu_base_addr] = 1;

      SYSCONF->dpu_conf[dpu_core_num].base_addr = cu_base_addr;
      SYSCONF->dpu_conf[dpu_core_num].cu_index = xclIPName2Index(mdev_handle, (const char *)(layout->m_ip_data[i].m_name));
      SYSCONF->dpu_core_mask |= (1 << (SYSCONF->dpu_conf[dpu_core_num].cu_index));

      log_info("Kernel:%s , BaseAddr: 0x%x\n", layout->m_ip_data[i].m_name,
               layout->m_ip_data[i].m_base_address);
      log_info("ver :%x\n", SYSCONF->dpu_conf[dpu_core_num].version);
      log_info("arch:%x\n", SYSCONF->dpu_conf[dpu_core_num].arch);

      dpu_core_num++;
    } else if (strncmp((const char *)(layout->m_ip_data[i].m_name), "DPUCZDX8G",
                strlen("DPUCZDX8G")) == 0) {
      cu_addr_map[cu_base_addr] = 1;

      SYSCONF->dpu_conf[dpu_core_num].base_addr = cu_base_addr;
      SYSCONF->dpu_conf[dpu_core_num].cu_index = xclIPName2Index(mdev_handle, (const char *)(layout->m_ip_data[i].m_name));
      SYSCONF->dpu_core_mask |= (1 << (SYSCONF->dpu_conf[dpu_core_num].cu_index));

      log_info("Kernel:%s , BaseAddr: 0x%x\n", layout->m_ip_data[i].m_name,
               layout->m_ip_data[i].m_base_address);
      log_info("ver :%x\n", SYSCONF->dpu_conf[dpu_core_num].version);
      log_info("arch:%x\n", SYSCONF->dpu_conf[dpu_core_num].arch);

      dpu_core_num++;
    } else if (strncmp((const char *)(layout->m_ip_data[i].m_name), "sfm_xrt_top",
                       strlen("sfm_xrt_top")) == 0) {
      cu_addr_map[cu_base_addr] = 2;
      SYSCONF->sm_conf[sm_core_num].base_addr = cu_base_addr;
      SYSCONF->sm_conf[sm_core_num].cu_index = xclIPName2Index(mdev_handle, (const char *)(layout->m_ip_data[i].m_name));
      SYSCONF->sm_core_mask |= (1 << (SYSCONF->sm_conf[sm_core_num].cu_index));

      log_info("Kernel:%s , BaseAddr: 0x%x\n", layout->m_ip_data[i].m_name,
               layout->m_ip_data[i].m_base_address);
      sm_core_num++;
    }
  }

  SYSCONF->dpu_core_num = dpu_core_num;
  SYSCONF->sm_core_num = sm_core_num;

  auto ipx = xclbin::get_axlf_section(top, CLOCK_FREQ_TOPOLOGY);
  if (ipx != nullptr) {
    struct clock_freq_topology *layoutx = (clock_freq_topology *)(header + ipx->m_sectionOffset);

    for (int i = 0; i < layoutx->m_count; ++i) {
      log_info("Freq %d - %s: %d\n", i, layoutx->m_clock_freq[i].m_name,
               layoutx->m_clock_freq[i].m_freq_Mhz);
    }
  }

  delete[] header;

  return 0;
}

void _init_config_shm() {
    SYSCONF = &g_sys_config;
    SYSCONF->dpu_core_mask = 0xFF;
}

void _init_from_env() {
  char *c;

  c = getenv(ENV_DPUCORE_MASK);
  if (c != NULL)
    user_core_mask = strtoul(c, 0, 0);
  else
    user_core_mask = SYSCONF->dpu_core_mask;
  log_dbg(ENV_DPUCORE_MASK "(%s), user_core_mask = 0x%x\n", c, user_core_mask);

  c = getenv(ENV_XCLBIN_PATH);

  if (c == NULL)
    strncpy(xclbin_path, "/usr/lib/dpu.xclbin", 100);
  else {
    strncpy(xclbin_path, c, 100);
  }
  log_dbg(ENV_XCLBIN_PATH ", path = %s\n", xclbin_path);
}

void _init_dpu() {
  xclDeviceHandle handle = xclOpen(card_index, NULL, XCL_ERROR);
  if (handle == nullptr) {
    log_err("[TID:%lu] xclOpen Error, ret = %p\n", this_thread::get_id(), handle);
    exit(-1);
  }

  unsigned initBO = xclAllocBO(handle, 4096, xclBOKind(0), XCL_BO_FLAGS_EXECBUF);
  auto initData = xclMapBO(handle, initBO, true);

  auto ecmd = reinterpret_cast<ert_init_kernel_cmd *>(initData);
  auto rsz = DPU_CONFIG_REG_LEN / 4 + 2;  // regmap array size
  memset(ecmd, 0, (sizeof *ecmd) + rsz);
  ecmd->state = ERT_CMD_STATE_NEW;
  ecmd->opcode = ERT_INIT_CU;
  ecmd->count = 1 + rsz;

  ecmd->cu_run_timeout = timeout * 1000000;
  ecmd->cu_reset_timeout = 10000;  // 10ms
  ecmd->cu_mask = SYSCONF->dpu_core_mask;

  int ret;
  if ((ret = xclExecBuf(handle, initBO)) != 0) {
    log_err("Unable to initialize DPU, ret = %d\n", ret);
    return;
  }

  do {
    ret = xclExecWait(handle, timeout * 1000);
    if (ret == 0) {
      log_err("Timeout when initializing DPU, ret = %d\n", ret);
    } else if (ecmd->state == ERT_CMD_STATE_COMPLETED) {
      log_info("Initialize DPU sucessfully!\n");
      break;
    }
  } while (ret);

  xclClose(handle);
}

/******************************************************************
 * AOL Interfaces implement
 *******************************************************************/

dpu_aol_dev_handle_t *dpu_aol_attach(uint32_t mode) {  // todo
  dpu_aol_dev_handle_t *dev = nullptr;
  dev = (dpu_aol_dev_handle_t *)malloc(sizeof(dpu_aol_dev_handle_t));
  if (dev == nullptr) {
    log_err("malloc dpu_aol_dev_handle_t memory space failed!\n");
    return nullptr;
  }
  memset(dev, 0, sizeof(dpu_aol_dev_handle_t));

  _init_config_shm();

  _init_from_env();

  _init_xrt(xclbin_path);

  _init_dpu();

  dev->aol_version = 0x0100;
  int cu_index = 0;
  dev->core_count[IP_ID_DPU] = SYSCONF->dpu_core_num;
  for (int i = 0; i < SYSCONF->dpu_core_num; i++) {
    dev->core_phy_addr[cu_index++] = (((uint64_t)SYSCONF->dpu_conf[i].cu_index) << 32);
  }
  dev->core_count[IP_ID_SOFTMAX] = SYSCONF->sm_core_num;
  for (int i = 0; i < SYSCONF->sm_core_num; i++) {
    dev->core_phy_addr[cu_index++] = (((uint64_t)SYSCONF->sm_conf[i].cu_index) << 32);
  }
  return dev;
}

/* Detach DPU device and other IPs.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 * Return:
 *     The device handle that the subsequent function needs to usd.
 */
int dpu_aol_detach(dpu_aol_dev_handle_t *dev) {
  if (mdev_handle) {
    xclClose(mdev_handle);
  }
  free(dev);
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
    uint32_t cu_index = phy_address >> 32;
    _check_dev_handle(mdev_handle);

    xclOpenContext(mdev_handle, read_reg_uuid, cu_index, false);
    for (int i = 0; i < (count >> 2); i++) {
        xclRegRead(mdev_handle, cu_index, ((phy_address & 0xFFFFFFFF) + (i << 2)), &buf[i]);
    }
    xclCloseContext(mdev_handle, read_reg_uuid, cu_index);

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
  log_dbg("No init needed\n");
  return DPU_AOL_OK;
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
  switch (data->ip_id) {
    case IP_ID_DPU:
      if (run_dpu(data) != 0) {
        return DPU_AOL_ERROR;
      }
      break;
    case IP_ID_SOFTMAX:
      if (run_softmax(data) != 0) {
        return DPU_AOL_ERROR;
      }
      break;
    default:
      log_err("Unsupported hardware!\n");
      break;
  }
  return DPU_AOL_OK;
}

/* Allocate physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     size - Byte length of the memory to be allocated.
 *     prot - Fxied to (DPU_AOL_MEM_PROT_READ|DPU_AOL_MEM_PROT_WRITE) in this version.
 * Return:
 *     The handle of the requedsted memory. NULL for failure.
 */

dpu_aol_dev_mem_t *dpu_aol_alloc_dev_mem(dpu_aol_dev_handle_t *dev, uint64_t size, uint32_t prot) {
  xrt_mem_t *mem = (xrt_mem_t *)malloc(sizeof(xrt_mem_t));
  if (mem == nullptr) {
    log_err("malloc xrt_mem_t memory space failed!\n");
    return nullptr;
  }

  _check_dev_handle(mdev_handle);

  uint64_t size_align = (size + (PAGESIZE - 1)) & ~(PAGESIZE - 1);  // though not necessory, still keep it
  mem->bo = xclAllocBO(mdev_handle, size_align, XCL_BO_DEVICE_RAM, XCL_BO_FLAGS_CACHEABLE);
  if (mem->bo == unsigned(-1)) {
    log_err("Alloc BO Failed, size: 0x%x\n", size_align);
    return nullptr;
  }
  mem->aol_mem.addr_virt = (unsigned long)xclMapBO(mdev_handle, mem->bo, true);
  mem->aol_mem.size = size_align;

  // Get Device address
  struct xclBOProperties p;
  if (xclGetBOProperties(mdev_handle, mem->bo, &p) != 0) {
    return nullptr;
  }
  mem->aol_mem.addr_phy = p.paddr;

  log_dbg("[Alloc BO]size: 0x%x, BO_ID:%d vaddr: %p, paddr:%p \n", size_align, mem->bo,
          mem->aol_mem.addr_virt, mem->aol_mem.addr_phy);
  return reinterpret_cast<dpu_aol_dev_mem_t *>(mem);
}

/* Free physically contiguous DMA device memory.
 * Input:
 *     dev - The device handle obtained by dpu_aol_attach.
 *     mem - The memory handle obtained by dpu_aol_alloc_dev_mem.
 * Return:
 *     DPU_AOL_OK for success, DPU_AOL_ERROR for failure.
 */
int dpu_aol_free_dev_mem(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem) {
  xrt_mem_t *mem_free = reinterpret_cast<xrt_mem_t *>(mem);
  _check_dev_handle(mdev_handle);

  xclUnmapBO(mdev_handle, mem_free->bo, (void *)mem_free->aol_mem.addr_virt);
  xclFreeBO(mdev_handle, mem_free->bo);

  log_dbg("[Free BO]size:  BO_ID:%d\n", mem_free->bo);

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
int dpu_aol_sync_to_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset,
                        uint32_t size) {
  xrt_mem_t *mem_sync = reinterpret_cast<xrt_mem_t *>(mem);
  _check_dev_handle(mdev_handle);

  int ret = xclSyncBO(mdev_handle, mem_sync->bo, XCL_BO_SYNC_BO_TO_DEVICE, size, offset);
  if (ret == 0) {
    log_dbg("[Sync BO to Dev] BO_ID:%d, size: %d, offset: %d\n", mem_sync->bo, size, offset);
    return DPU_AOL_OK;
  } else {
    log_err("Error when sync BO to device, BO:%d, size: 0x%x, offset: %d\n", mem_sync->bo, size,
            offset);
    return DPU_AOL_ERROR;
  }
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
int dpu_aol_sync_from_dev(dpu_aol_dev_handle_t *dev, dpu_aol_dev_mem_t *mem, uint32_t offset,
                          uint32_t size) {
  xrt_mem_t *mem_sync = reinterpret_cast<xrt_mem_t *>(mem);
  _check_dev_handle(mdev_handle);

  int ret = xclSyncBO(mdev_handle, mem_sync->bo, XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
  if (ret == 0) {
    log_dbg("[Sync BO From Dev] BO_ID:%d, size: %d, offset: %d\n", mem_sync->bo, size, offset);
    return DPU_AOL_OK;
  } else {
    log_err("Error when sync BO from device, BO:%d, size: 0x%x, offset: %d\n", mem_sync->bo, size,
            offset);
    return DPU_AOL_ERROR;
  }
}
