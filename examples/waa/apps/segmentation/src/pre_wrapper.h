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

#include <glog/logging.h>
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
#include "xhls_dpupreproc_m_axi_hw.h"
using namespace std;
class PreHandle {
	public:
		xclDeviceHandle handle;
		uint64_t cu_base_addr;
		unsigned cu_index;
		unsigned bo_img_inp_Handle;
		unsigned char *img_inptr;
		uint64_t bo_input_phy_addr;
		unsigned execHandle;
		void *execData;
};

static uint32_t get_reg(xclDeviceHandle xcl_handle, uint64_t cu_addr) {
	uint32_t value = 0;
	size_t size = sizeof(value);
	auto read_result =
		xclRead(xcl_handle, XCL_ADDR_KERNEL_CTRL, cu_addr, &value, size);
	CHECK_EQ(read_result, size)
		<< "xclRead has error!"                              //
		<< "read_result " << read_result << " "              //
		<< "cu_addr " << std::hex << "0x" << cu_addr << " "  //
		;
	return value;
}

int preprocess(PreHandle *&prehandle, unsigned char *inimg_data,
		uint64_t dpu_input_phy_addr, int img_ht, int img_wt) {
	xclDeviceHandle handle = prehandle->handle;
	uint64_t cu_base_addr = prehandle->cu_base_addr;
	unsigned bo_img_inp_Handle = prehandle->bo_img_inp_Handle;
	unsigned char *img_inptr = prehandle->img_inptr;
	unsigned execHandle = prehandle->execHandle;
	void *execData = prehandle->execData;

	const int imageToDevice_size = img_wt * img_ht * 3 * sizeof(char);
	//# Send the input imageToDevice data to the device memory
	std::memcpy(img_inptr, inimg_data, imageToDevice_size);
	if (xclSyncBO(handle, bo_img_inp_Handle, XCL_BO_SYNC_BO_TO_DEVICE,
				imageToDevice_size, 0))
		return 1;

	try {
		auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd *>(execData);
		// Program the register map
		ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_OUT_DATA / 4] =
			dpu_input_phy_addr & 0xFFFFFFFF;
		ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_OUT_DATA / 4 + 1] =
			(dpu_input_phy_addr >> 32) & 0xFFFFFFFF;
		// if (0)
		//  cout << "debug ecmd: "
		//       << vitis::ai::xxd((unsigned char *)ecmd2,
		//                         (sizeof *ecmd2) + ecmd2->count * 4, 8, 1)
		//       << endl;
		//;
		__TIC__(hw_pre);
		int ret;
		if ((ret = xclExecBuf(handle, execHandle)) != 0) {
			std::cout << "Unable to trigger Preprocess, error:" << ret << std::endl;
			return ret;
		}
		do {
			ret = xclExecWait(handle, 1000);
			if (ret == 0) {
				std::cout << "base_address " << std::hex << cu_base_addr + 0x38
					<< std::dec << " " << get_reg(handle, cu_base_addr + 0x38)
					<< " " << ecmd2->data[0x38 / 4] << std::endl;
				std::cout << "Preprocess Task Time out, state =" << ecmd2->state
					<< " count = " << ecmd2->count
					<< " opcode = " << ecmd2->opcode
					<< " cu_mask = " << ecmd2->cu_mask << std::endl;
			} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED)
				break;
		} while (1);
		__TOC__(hw_pre);
	} catch (std::exception const &e) {
		std::cout << "Exception: " << e.what() << "\n";
		std::cout << "FAILED TEST\n";
		return 1;
	}
	return 0;
}

void releaseBO(PreHandle *&prehandle) {
	xclDeviceHandle handle = prehandle->handle;
	xclFreeBO(handle, prehandle->bo_img_inp_Handle);
	xclClose(handle);
	return;
}

int pre_kernel_init(PreHandle *&prehandle, char *xclbin, float norm_fact,
		float shift_fact, float scale_fact, int16_t out_height,
		int16_t out_width) {
	PreHandle *my_handle = new PreHandle;
	prehandle = (PreHandle *)my_handle;

	unsigned index = 0;
	std::string halLogfile;
	unsigned cu_index = 2;

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
	// std::cout << "index: " << index << " base_address " << std::hex
	//          << cu_base_addr << std::dec << std::endl;
	// auto start_cycle = get_reg(handle, cu_base_addr + 0x1a0);
	// cout << "start cycle: " << start_cycle << endl;

	//# creating memory for FHD image
	const int imageToDevice_size = out_width * out_height * 3 * sizeof(char);
	// Allocate the device memory
	unsigned bo_img_inp_Handle = xclAllocBO(handle, imageToDevice_size, 0, 0);
	// Create the mapping to the host memory
	unsigned char *img_inptr =
		(unsigned char *)xclMapBO(handle, bo_img_inp_Handle, true);
	if ((img_inptr == NULL))
		throw std::runtime_error("imageToDevice pointer is invalid\n");

	xclBOProperties p;
	uint64_t bo_input_phy_addr =
		!xclGetBOProperties(handle, bo_img_inp_Handle, &p) ? p.paddr : -1;
	if ((bo_input_phy_addr == (uint64_t)(-1))) ret_checkDevMem = 1;

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
		XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_WIDTH_DATA / 4 + 1;  // regmap
	std::memset(ecmd2, 0, (sizeof *ecmd2) + regmap_size);

	// Program the command packet header
	ecmd2->state = ERT_CMD_STATE_NEW;
	ecmd2->opcode = ERT_START_CU;
	ecmd2->count = 1 + regmap_size;  // cu_mask + regmap
	// Program the CU mask. One CU at index 0
	ecmd2->cu_mask = 0x4;
	// Program the register map
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_AP_CTRL] = 0x0;  // ap_start
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_INP_DATA / 4] =
		bo_input_phy_addr & 0xFFFFFFFF;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_INP_DATA / 4 + 1] =
		(bo_input_phy_addr >> 32) & 0xFFFFFFFF;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_NORM_FACT_DATA / 4] =
		*(int *)&norm_fact;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_SHIFT_FACT_DATA / 4] =
		*(int *)&shift_fact;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_SCALE_FACT_DATA / 4] =
		*(int *)&scale_fact;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_HEIGHT_DATA / 4] = out_height;
	ecmd2->data[XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_WIDTH_DATA / 4] = out_width;

	my_handle->handle = handle;
	my_handle->cu_base_addr = cu_base_addr;
	my_handle->cu_index = cu_index;
	my_handle->bo_img_inp_Handle = bo_img_inp_Handle;
	my_handle->img_inptr = img_inptr;
	my_handle->bo_input_phy_addr = bo_input_phy_addr;
	my_handle->execHandle = execHandle;
	my_handle->execData = execData;

	return 0;
}
