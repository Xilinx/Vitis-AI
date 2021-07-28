/*
 * Copyright (C) 2020, Xilinx Inc - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#define EN_SORT_PROFILE 0

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <dirent.h>

// driver includes
#include "ert.h"

// host_src includes
#include "xclhal2.h"
#include "xclbin.h"

// lowlevel common include
#include "utils.h"

#define DSA64 1

#if defined(DSA64)
#include "xsort_hw_64.h"     
#include "xf_pppre_hw_64.h" 
#else
#include "xsort_hw.h"
#include "xf_pppre_hw.h" 
#endif

class PPHandle {
public:

	xclDeviceHandle handle;
	uint64_t cu_base_addr;
	unsigned cu_index;
	unsigned execHandle;
	void *execData;
	//# sort_nms vars
	unsigned bo_output;
	uint64_t bo_output_phy_addr;
	uint64_t dpu_conf_out_phy_addr;
	uint64_t dpu_box_out_phy_addr;
	signed char *bo_output_ptr;
	unsigned prbo_output;
	uint64_t prbo_output_phy_addr;
	short *prbo_output_ptr;
	int out_size;	
	//# pre proc vars
	unsigned paramsbuf_bo;
	uint64_t bo_paramsbuf_devAddr;
	unsigned img_inp_bo;
	unsigned char *img_inptr;
	uint64_t bo_input_phy_addr;
	float *paramsbuf_ptr;
	
};

extern PPHandle* pphandle;

static std::vector<std::string> get_xclbins_in_dir(std::string path)
{
	if (path.find(".xclbin") != std::string::npos)
		return {path};

	std::vector<std::string> xclbinPaths;

	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(path.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL)
		{
			std::string name(ent->d_name);
			if (name.find(".xclbin") != std::string::npos)
				xclbinPaths.push_back(path + "/" + name);
		}
		closedir(dir);
	}
	return xclbinPaths;
}

const int NUM_CLASS = 91;
const int MAX_BOX_PER_CLASS = 32;
const int ELE_PER_BOX = 8;
const int NMS_FX = 9830;
const int CLASS_SIZE = 1917;


static int runHWSort(PPHandle * &pphandle, short *&nms_out)
{	
	xclDeviceHandle handle = pphandle->handle;
	uint64_t cu_base_addr = pphandle->cu_base_addr;
	unsigned bo_output = pphandle->bo_output;
	uint64_t bo_output_phy_addr = pphandle->bo_output_phy_addr;
	signed char *bo_output_ptr = pphandle->bo_output_ptr;
	unsigned prbo_output = pphandle->prbo_output;
	uint64_t prbo_output_phy_addr = pphandle->prbo_output_phy_addr;
	short *prbo_output_ptr = pphandle->prbo_output_ptr;
	unsigned execHandle = pphandle->execHandle;
	void *execData =  pphandle->execData;		
	long int outSize_bytes = pphandle->out_size;

	try {
		auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);

		// Clear the command in case it was recycled
		size_t regmap_size = XSORT_CONTROL_ADDR_SCALAR_DATA/4 + 1; // regmap
		std::memset(ecmd2,0,(sizeof *ecmd2) + regmap_size);

		// Program the command packet header
		ecmd2->state = ERT_CMD_STATE_NEW;
		ecmd2->opcode = ERT_START_CU;
		ecmd2->count = 1 + regmap_size;  // cu_mask + regmap

		// Program the CU mask. One CU at index 0
		ecmd2->cu_mask = 0x4;

		// Program the register map
		ecmd2->data[XSORT_CONTROL_ADDR_AP_CTRL] = 0x0; // ap_start

		uint64_t bo1devAddr_conf = pphandle->dpu_conf_out_phy_addr;
		uint64_t bo1devAddr_box = pphandle->dpu_box_out_phy_addr;		

		ecmd2->data[XSORT_CONTROL_ADDR_inConf_DATA/4] = bo1devAddr_conf & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inConf_DATA/4 + 1] = (bo1devAddr_conf >> 32) & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox_DATA/4] = bo1devAddr_box & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox_DATA/4 + 1] = (bo1devAddr_box >> 32) & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox1_DATA/4] = bo1devAddr_box & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox1_DATA/4 + 1] = (bo1devAddr_box >> 32) & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox2_DATA/4] = bo1devAddr_box & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox2_DATA/4 + 1] = (bo1devAddr_box >> 32) & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox3_DATA/4] = bo1devAddr_box & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_inBox3_DATA/4 + 1] = (bo1devAddr_box >> 32) & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_priors_DATA/4] = prbo_output_phy_addr & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_priors_DATA/4 + 1] = (prbo_output_phy_addr >> 32) & 0xFFFFFFFF; // input
		//# output data
		ecmd2->data[XSORT_CONTROL_ADDR_outBoxes_DATA/4] = bo_output_phy_addr & 0xFFFFFFFF; // input
		ecmd2->data[XSORT_CONTROL_ADDR_outBoxes_DATA/4 + 1] = (bo_output_phy_addr >> 32) & 0xFFFFFFFF; // input
		//# scalar params
		ecmd2->data[XSORT_CONTROL_ADDR_SCALAR_DATA1/4] = CLASS_SIZE;
		ecmd2->data[XSORT_CONTROL_ADDR_SCALAR_DATA/4] = NMS_FX;

#if EN_SORT_PROFILE
		auto start_k2= std::chrono::system_clock::now();
#endif

		int ret;
		if ((ret = xclExecBuf(handle, execHandle)) != 0) {
			std::cout << "Unable to trigger SORT, error:" << ret << std::endl;
			return ret;
		}
		do {
			ret = xclExecWait(handle, 1000);
			if (ret == 0) {
				std::cout << "SORT Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;

			} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {

				break;
			}
		} while (1);

#if EN_SORT_PROFILE
		auto end_k2= std::chrono::system_clock::now();
		auto difft_k2= end_k2-start_k2;
		auto value_k2 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k2);
		std::cout << "Kernel Execution " << value_k2.count() << " us\n";

		auto start_k3= std::chrono::system_clock::now();
#endif
		//Get the output;
		if(xclSyncBO(handle, bo_output, XCL_BO_SYNC_BO_FROM_DEVICE, outSize_bytes, 0)) {
			return 1;
		}

#if EN_SORT_PROFILE
		auto end_k3= std::chrono::system_clock::now();
		auto difft_k3= end_k3-start_k3;
		auto value_k3 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k3);
		std::cout << "synbo Out " << value_k3.count() << " us\n";
#endif
		nms_out = (short *)bo_output_ptr;
	}
	catch (std::exception const& e)
	{
		std::cout << "Exception: " << e.what() << "\n";
		std::cout << "FAILED TEST\n";
		return 1;
	}
	return 0;
}


//# Free xrt bo
static void postproc_dealloc(PPHandle * &pphandle) {
	xclDeviceHandle handle = pphandle->handle;	
	//xclUnmapBO(handle, pphandle->paramsbuf_bo, pphandle->paramsbuf_ptr);
	//xclUnmapBO(handle, pphandle->img_inp_bo, pphandle->img_inptr);
	xclFreeBO(handle, pphandle->prbo_output);
	xclFreeBO(handle, pphandle->bo_output);
	xclClose(handle);
	return;
}

static int hw_sort_init(
		PPHandle * &pphandle,
		const short *&fx_pror)
{
	// get xclbin path and acquire handle
	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");

	if (xclbinPath == nullptr)
		throw std::runtime_error("Error: xclbinPath is not set, please consider setting XLNX_VART_FIRMWARE.");

	// get available xclbins
	auto xclbins = get_xclbins_in_dir(xclbinPath);

	PPHandle *my_handle = new PPHandle;
	pphandle = my_handle = (PPHandle *)my_handle;

	unsigned index = 0;
	std::string halLogfile;
	unsigned cu_index = 2;

	xclDeviceHandle handle;
	uint64_t cu_base_addr = 0;
	uuid_t xclbinId;
	int first_mem = -1;
	bool ret_initXRT=0;
	bool ret_firstmem=0;
	bool ret_runkernel=0;
	bool ret_checkDevMem=0;

	if (initXRT(xclbins[0].c_str(), index, halLogfile.c_str(), handle, cu_index, cu_base_addr, first_mem, xclbinId))
		ret_initXRT=1;

	if(xclOpenContext(handle, xclbinId, cu_index, true))
		throw std::runtime_error("Cannot create context");

	// Allocate the device memory
	long int outSize_bytes = (NUM_CLASS-1)*MAX_BOX_PER_CLASS*ELE_PER_BOX*sizeof(short);//+  (classNum_SizeBuff_align)*sizeof(short) + BoxBuff_align*sizeof(signed char);
	unsigned bo_output = xclAllocBO(handle, outSize_bytes, XCL_BO_DEVICE_RAM, 11);   // output score

	const int prior_sz = 1917*4*sizeof(short); 
	//# prfx ptr
	unsigned prior_bo_output = xclAllocBO(handle, prior_sz, XCL_BO_DEVICE_RAM, 10);   // output score
	// Create the mapping to the host memory
	signed char *bo_output_ptr = (signed char*)xclMapBO(handle, bo_output, false);
	short *pr_bo_output_ptr = (short*)xclMapBO(handle, prior_bo_output, true);
	if((pr_bo_output_ptr == NULL))
		throw std::runtime_error("prior ptr invalid\n");
	
	//std::cout << "device write prior ....\n";
	std::memcpy(pr_bo_output_ptr, fx_pror, prior_sz);
	//std::cout << "device write prior success\n";

	// Send the input data to the device memory
	//std::cout << "Send the input data to the device memory.\n";
	if(xclSyncBO(handle, prior_bo_output, XCL_BO_SYNC_BO_TO_DEVICE , prior_sz, 0)) {
		return 1;
	}

	// Get & check the device memory address
	xclBOProperties p;
	uint64_t bo_output_phy_addr = !xclGetBOProperties(handle, bo_output, &p) ? p.paddr : -1;
	if( (bo_output_phy_addr == (uint64_t)(-1)) ){
		ret_checkDevMem=1;
	}
	uint64_t prbo_output_phy_addr = !xclGetBOProperties(handle, prior_bo_output, &p) ? p.paddr : -1;
	if( (prbo_output_phy_addr == (uint64_t)(-1)) ){
		ret_checkDevMem=1;
	}

	//thread_local static 
	unsigned execHandle = 0;
	//thread_local static 
	void *execData = nullptr;
	if(execHandle == 0) execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1<<31));
	if(execData == nullptr) execData = xclMapBO(handle, execHandle, true);

	my_handle->handle = handle;
	my_handle->cu_base_addr = cu_base_addr;
	my_handle->cu_index = cu_index;
	my_handle->bo_output = bo_output;
	my_handle->bo_output_phy_addr = bo_output_phy_addr;
	my_handle->bo_output_ptr = bo_output_ptr;
	my_handle->prbo_output = prior_bo_output;
	my_handle->prbo_output_phy_addr = prbo_output_phy_addr;
	my_handle->prbo_output_ptr = pr_bo_output_ptr;
	my_handle->execHandle = execHandle;
	my_handle->execData = execData;
	my_handle->out_size = outSize_bytes;

	//std::cout << "sort nms inti done ......\n";
	return 0;
}


static int preprocess(PPHandle * &pphandle,
		unsigned char* inimg_data,
		int img_ht,
		int img_wt,
		int org_ht,
		int org_wt,
		uint64_t dpu_input_phy_addr)
{
	xclDeviceHandle handle = pphandle->handle;
	uint64_t cu_base_addr = pphandle->cu_base_addr;
	unsigned paramsbuf_bo = pphandle-> paramsbuf_bo;
	uint64_t bo_paramsbuf_devAddr = pphandle-> bo_paramsbuf_devAddr;
	float *paramsbuf_ptr = pphandle-> paramsbuf_ptr;
	unsigned img_inp_bo = pphandle->img_inp_bo;
	unsigned char* img_inptr = pphandle->img_inptr;
	uint64_t bo_input_phy_addr = pphandle->bo_input_phy_addr;	
	unsigned execHandle = pphandle->execHandle;
	void *execData =  pphandle->execData;
	int th1 = 127, th2 = 128;	
	const int imageToDevice_size = img_wt*img_ht*3*sizeof(char);

	std::memcpy(img_inptr, inimg_data, imageToDevice_size);

	// Send the input imageToDevice data to the device memory
	//std::cout << "Send the imageToDevice input data to the device memory.\n";
	if(xclSyncBO(handle, img_inp_bo, XCL_BO_SYNC_BO_TO_DEVICE , imageToDevice_size, 0)) {
		return 1;
	}

	try {
		auto ecmd2 = reinterpret_cast<ert_start_kernel_cmd*>(execData);
		// Clear the command in case it was recycled
		size_t regmap_size = XFPPPre_CONTROL_ADDR_roi_posy/4 + 1; // regmap
		std::memset(ecmd2,0,(sizeof *ecmd2) + regmap_size);
		// Program the command packet header
		ecmd2->state = ERT_CMD_STATE_NEW;
		ecmd2->opcode = ERT_START_CU;
		ecmd2->count = 1 + regmap_size;  // cu_mask + regmap
		// Program the CU mask. One CU at index 0
		ecmd2->cu_mask = 0x2;
		// Program the register map
		ecmd2->data[XFPPPre_CONTROL_ADDR_AP_CTRL] = 0x0; // ap_start

#if defined(DSA64)
		//std::cout << "64-bit address\n";
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp/4] = bo_input_phy_addr & 0xFFFFFFFF;
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp/4 + 1] = (bo_input_phy_addr >> 32) & 0xFFFFFFFF;
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_out/4] = dpu_input_phy_addr & 0xFFFFFFFF;
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_out/4 + 1] = (dpu_input_phy_addr >> 32) & 0xFFFFFFFF;
		ecmd2->data[XFPPPre_CONTROL_ADDR_params/4] = bo_paramsbuf_devAddr & 0xFFFFFFFF;
		ecmd2->data[XFPPPre_CONTROL_ADDR_params/4 + 1] = (bo_paramsbuf_devAddr >> 32) & 0xFFFFFFFF; 
#else
		std::cout << "32-bit address\n";
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_inp/4] = bo_input_phy_addr;
		ecmd2->data[XFPPPre_CONTROL_ADDR_img_out/4] = dpu_input_phy_addr;	
		ecmd2->data[XFPPPre_CONTROL_ADDR_params/4] = bo_paramsbuf_devAddr;
#endif
		
		ecmd2->data[XFPPPre_CONTROL_ADDR_in_img_width/4] = img_wt;
		ecmd2->data[XFPPPre_CONTROL_ADDR_in_img_height/4] = img_ht;
		ecmd2->data[XFPPPre_CONTROL_ADDR_in_img_linestride/4] = img_wt;
		ecmd2->data[XFPPPre_CONTROL_ADDR_resize_width/4] = org_wt;	
		ecmd2->data[XFPPPre_CONTROL_ADDR_resize_height/4] = org_ht;
		ecmd2->data[XFPPPre_CONTROL_ADDR_out_img_width/4] = org_wt;
		ecmd2->data[XFPPPre_CONTROL_ADDR_out_img_height/4] = org_ht;
		ecmd2->data[XFPPPre_CONTROL_ADDR_out_img_linestride/4] = org_wt;
		ecmd2->data[XFPPPre_CONTROL_ADDR_roi_posx/4] = 0;
		ecmd2->data[XFPPPre_CONTROL_ADDR_roi_posy/4] = 0;

		int ret;
		if ((ret = xclExecBuf(handle, execHandle)) != 0) {
			std::cout << "Unable to trigger preprocess, error:" << ret << std::endl;
			return ret;
		}
		do {
			ret = xclExecWait(handle, 1000);
			if (ret == 0) {
				std::cout << "Preprocess Task Time out, state =" << ecmd2->state << "cu_mask = " << ecmd2->cu_mask << std::endl;
			} else if (ecmd2->state == ERT_CMD_STATE_COMPLETED) {
				break;
			}
		} while (1);
	}
	catch (std::exception const& e)
	{
		std::cout << "Exception: " << e.what() << "\n";
		std::cout << "FAILED TEST\n";
		return 1;
	}
	
	return 0;

}


//# Free xrt bo
static void pp_dealloc(PPHandle * &pphandle) {
	xclDeviceHandle handle = pphandle->handle;	
	//xclUnmapBO(handle, pphandle->paramsbuf_bo, pphandle->paramsbuf_ptr);
	//xclUnmapBO(handle, pphandle->img_inp_bo, pphandle->img_inptr);
	xclFreeBO(handle, pphandle->paramsbuf_bo);
	xclFreeBO(handle, pphandle->img_inp_bo);
	xclClose(handle);
	return;
}

static int pp_kernel_init(PPHandle * &pphandle,
		float input_scale)
{
	// get xclbin path and acquire handle
	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");

	if (xclbinPath == nullptr)
		throw std::runtime_error("Error: xclbinPath is not set, please consider setting XLNX_VART_FIRMWARE.");

	// get available xclbins
	auto xclbins = get_xclbins_in_dir(xclbinPath);
	
	PPHandle *my_handle = new PPHandle;
	pphandle = my_handle = (PPHandle *)my_handle;

	unsigned index = 0;
	std::string halLogfile;
	unsigned cu_index = 1;

	//xclbinutil --dump-section BITSTREAM:RAW:bitstream.bit --input binary_container_1.xclbin
	//std::cout <<"preproc xclbin: " << xclbin << std::endl;

	xclDeviceHandle handle;
	uint64_t cu_base_addr = 0;
	uuid_t xclbinId;
	int first_mem = -1;
	bool ret_initXRT=0;
	bool ret_firstmem=0;
	bool ret_runkernel=0;
	bool ret_checkDevMem=0;

	if (initXRT(xclbins[0].c_str(), index, halLogfile.c_str(), handle, cu_index, cu_base_addr, first_mem, xclbinId))
		ret_initXRT=1;
	//printf("cu_base_addr in pp_kernel_init: %x\n",cu_base_addr);
	if(xclOpenContext(handle, xclbinId, cu_index, true))
		throw std::runtime_error("Cannot create context");

	float params[9];
	//# Mean params
	params[0] = 0.0;
	params[1] = 0.0;
	params[2] = 0.0;
	//# Input scale
	params[3] = params[4] = params[5] = 0.502;	
	//# Set to default zero
	params[6] = params[7] = params[8] = -64;

	const int paramsbuf_size = 9*4;//*sizeof(float);
	unsigned paramsbuf_bo = xclAllocBO(handle, paramsbuf_size, XCL_BO_DEVICE_RAM, 17);
	float *paramsbuf_ptr = (float*)xclMapBO(handle, paramsbuf_bo, true);
	if((paramsbuf_ptr == NULL))
		throw std::runtime_error("paramsbuf pointer is invalid\n");

	std::memcpy(paramsbuf_ptr, params, paramsbuf_size);
	
	// Get & check the device memory address
	xclBOProperties p;
	uint64_t bo_paramsbuf_devAddr = !xclGetBOProperties(handle, paramsbuf_bo, &p) ? p.paddr : -1;
	if( (bo_paramsbuf_devAddr == (uint64_t)(-1)) ){
		ret_checkDevMem=1;
	}

	if(xclSyncBO(handle, paramsbuf_bo, XCL_BO_SYNC_BO_TO_DEVICE , paramsbuf_size, 0)) {
		return 1;
	}

	//# creating memory for HD image
	const int imageToDevice_size = 1920*1080*3*sizeof(char);	
	// Allocate the device memory	
	unsigned img_inp_bo = xclAllocBO(handle, imageToDevice_size, XCL_BO_DEVICE_RAM, 15);
	// Create the mapping to the host memory
	unsigned char *img_inptr = (unsigned char*)xclMapBO(handle, img_inp_bo, true);
	if((img_inptr == NULL))
		throw std::runtime_error("imageToDevice pointer is invalid\n");
	uint64_t bo_input_phy_addr = !xclGetBOProperties(handle, img_inp_bo, &p) ? p.paddr : -1;
	if( (bo_input_phy_addr == (uint64_t)(-1)) ){
		ret_checkDevMem=1;
	}
		 
	//thread_local static 
	unsigned execHandle = 0;
	//thread_local static 
	void *execData = nullptr;

	if(execHandle == 0) execHandle = xclAllocBO(handle, 4096, xclBOKind(0), (1<<31));
	if(execData == nullptr) execData = xclMapBO(handle, execHandle, true);

	my_handle->handle = handle;
	my_handle->cu_base_addr = cu_base_addr;
	my_handle->cu_index = cu_index;
	my_handle->paramsbuf_bo = paramsbuf_bo;
	my_handle->bo_paramsbuf_devAddr = bo_paramsbuf_devAddr;
	my_handle->paramsbuf_ptr = paramsbuf_ptr;
	my_handle->img_inp_bo = img_inp_bo;
	my_handle->img_inptr = img_inptr;
	my_handle->bo_input_phy_addr = bo_input_phy_addr;
	my_handle->execHandle = execHandle;
	my_handle->execData = execData;

	//printf("\n xclbin:%s kernelName:%s deviceIdx:%d\n",*xclbin,*kernelName,deviceIdx);
	return 0;
}
