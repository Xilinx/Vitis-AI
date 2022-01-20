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

#include <sys/time.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/xxd.hpp>

// host_src includes
#include "xclbin.h"
#include "xclhal2.h"

// lowlevel common include
#include "utils.h"
#include "xhls_dpupostproc_m_axi_hw.h"
using namespace std;

class PostHandle {
 public:
  xclDeviceHandle handle;
  uint64_t cu_base_addr;
  unsigned cu_index;
  unsigned bo_out_idx_Handle;
  unsigned char *out_idxptr;
  uint64_t bo_output_phy_addr;
  unsigned bo_out_max_Handle;
  unsigned char *out_maxptr;
  uint64_t bo_outmax_phy_addr;
  unsigned execHandle;
  size_t output_size;
  void *execData;
};

int postprocess(PostHandle *&posthandle, unsigned char *out_idx_data,
                uint64_t dpu_output_phy_addr) {
  xclDeviceHandle handle = posthandle->handle;
  uint64_t cu_base_addr = posthandle->cu_base_addr;
  unsigned bo_out_idx_Handle = posthandle->bo_out_idx_Handle;
  unsigned bo_out_max_Handle = posthandle->bo_out_max_Handle;
  unsigned char *out_idxptr = posthandle->out_idxptr;
  unsigned char *out_maxptr = posthandle->out_maxptr;
  unsigned execHandle = posthandle->execHandle;
  void *execData = posthandle->execData;

  const int output_size = posthandle->output_size;
  //# Send the input imageToDevice data to the device memory

  try {
    auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd *>(execData);
    // Program the register map
    ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_INP_DATA_DATA / 4] =
        dpu_output_phy_addr & 0xFFFFFFFF;
    ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_INP_DATA_DATA / 4 + 1] =
        (dpu_output_phy_addr >> 32) & 0xFFFFFFFF;
    // if (1)
    //  cout << "debug post ecmd: "
    //       << vitis::ai::xxd((unsigned char *)ecmd2,
    //                         (sizeof *ecmd2) + ecmd2->count * 4, 8, 1)
    //       << endl;
    //;
    __TIC__(hw_post);
    int ret;
    if ((ret = xclExecBuf(handle, execHandle)) != 0) {
      std::cout << "Unable to trigger Preprocess, error:" << ret << std::endl;
      return ret;
    }
    do {
      ret = xclExecWait(handle, 1000);
      if (ret == 0)
        std::cout << "Preprocess Task Time out, state =" << ecmd2->state
                  << "cu_mask = " << ecmd2->cu_mask << std::endl;
      else if (ecmd2->state == ERT_CMD_STATE_COMPLETED)
        break;
    } while (1);
    __TOC__(hw_post);
  } catch (std::exception const &e) {
    std::cout << "Exception: " << e.what() << "\n";
    std::cout << "FAILED TEST\n";
    return 1;
  }
  if (xclSyncBO(handle, bo_out_idx_Handle, XCL_BO_SYNC_BO_TO_DEVICE,
                output_size, 0))
    return 1;
  // std::memcpy(out_idx_data, out_idxptr, output_size);
  // if (xclSyncBO(handle, bo_out_max_Handle, XCL_BO_SYNC_BO_TO_DEVICE,
  //              output_size, 0))
  //  return 1;
  // std::memcpy(out_idx_data, out_maxptr, output_size);
  return 0;
}

void releaseBO(PostHandle *&posthandle) {
  xclDeviceHandle handle = posthandle->handle;
  xclFreeBO(handle, posthandle->bo_out_idx_Handle);
  xclClose(handle);
  return;
}

int post_kernel_init(PostHandle *&posthandle, char *xclbin, float scale_fact,
                     int16_t out_height, int16_t out_width) {
  PostHandle *my_handle = new PostHandle;
  posthandle = (PostHandle *)my_handle;

  unsigned index = 0;
  std::string halLogfile;
  unsigned cu_index = 1;

  xclDeviceHandle handle;
  uint64_t cu_base_addr = 0;
  uuid_t xclbinId;
  int first_mem = -1;
  bool ret_initXRT = 0;
  bool ret_firstmem = 0;
  bool ret_runkernel = 0;
  bool ret_checkDevMem = 0;
  if (initXRT(xclbin, index, halLogfile.c_str(), handle, cu_index, cu_base_addr,
              first_mem, xclbinId))
    ret_initXRT = 1;
  if (xclOpenContext(handle, xclbinId, cu_index, true))
    throw std::runtime_error("Cannot create context");

  //# creating memory for FHD image
  const int outToHost_size = out_height * out_width * 1 * sizeof(char);
  // Allocate the device memory
  unsigned bo_out_idx_Handle = xclAllocBO(handle, outToHost_size, 0, 0);
  unsigned bo_out_max_Handle = xclAllocBO(handle, outToHost_size, 0, 0);
  // Create the mapping to the host memory
  unsigned char *out_idxptr =
      (unsigned char *)xclMapBO(handle, bo_out_idx_Handle, true);
  unsigned char *out_maxptr =
      (unsigned char *)xclMapBO(handle, bo_out_max_Handle, true);
  if ((out_idxptr == NULL) || (out_maxptr == NULL))
    throw std::runtime_error("imageToDevice pointer is invalid\n");

  xclBOProperties p;
  uint64_t bo_output_phy_addr =
      !xclGetBOProperties(handle, bo_out_idx_Handle, &p) ? p.paddr : -1;
  if ((bo_output_phy_addr == (uint64_t)(-1))) ret_checkDevMem = 1;

  uint64_t bo_outmax_phy_addr =
      !xclGetBOProperties(handle, bo_out_max_Handle, &p) ? p.paddr : -1;
  if ((bo_outmax_phy_addr == (uint64_t)(-1))) ret_checkDevMem = 1;
  // thread_local static
  unsigned execHandle = 0;
  // thread_local static
  void *execData = nullptr;

  if (execHandle == 0)
    execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1 << 31));
  if (execData == nullptr) execData = xclMapBO(handle, execHandle, true);

  auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd *>(execData);
  // Clear the command in case it was recycled
  size_t regmap_size =
      XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_WIDTH_DATA / 4 + 1;  // regmap
  std::memset(ecmd2, 0, (sizeof *ecmd2) + regmap_size);

  // Program the command packet header
  ecmd2->state = ERT_CMD_STATE_NEW;
  ecmd2->opcode = ERT_START_CU;
  ecmd2->count = 1 + regmap_size;  // cu_mask + regmap
  // Program the CU mask. One CU at index 0
  ecmd2->cu_mask = 0x4;
  // Program the register map
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_AP_CTRL] = 0x0;  // ap_start
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_OUT_MAX_DATA / 4] =
      bo_outmax_phy_addr & 0xFFFFFFFF;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_OUT_MAX_DATA / 4 + 1] =
      (bo_outmax_phy_addr >> 32) & 0xFFFFFFFF;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_OUT_INDEX_DATA / 4] =
      bo_output_phy_addr & 0xFFFFFFFF;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_OUT_INDEX_DATA / 4 + 1] =
      (bo_output_phy_addr >> 32) & 0xFFFFFFFF;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_SCALE_FACT_DATA / 4] =
      *(int *)&scale_fact;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_HEIGHT_DATA / 4] = out_height;
  ecmd2->data[XHLS_DPUPOSTPROC_M_AXI_CONTROL_ADDR_WIDTH_DATA / 4] = out_width;

  my_handle->handle = handle;
  my_handle->cu_base_addr = cu_base_addr;
  my_handle->cu_index = cu_index;
  my_handle->bo_out_idx_Handle = bo_out_idx_Handle;
  my_handle->out_idxptr = out_idxptr;
  my_handle->bo_output_phy_addr = bo_output_phy_addr;
  my_handle->bo_out_max_Handle = bo_out_max_Handle;
  my_handle->out_maxptr = out_maxptr;
  my_handle->bo_outmax_phy_addr = bo_outmax_phy_addr;
  my_handle->execHandle = execHandle;
  my_handle->execData = execData;
  posthandle->output_size = outToHost_size;

  return 0;
}

