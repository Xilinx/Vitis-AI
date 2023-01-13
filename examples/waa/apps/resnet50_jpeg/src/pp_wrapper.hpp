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
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "jfif.h"


#define MAX_LENGTH        16777216 // (16 MB)

int img_num;

typedef struct AcceleratorHandle_t
{
	xrt::kernel jpeg_kernel;
	xrt::kernel blob_kernel;
	xrt::device device;
	xrt::run blob_runner;
	xrt::run jpeg_runner;
	xrt::bo FileDataBuff;
	xrt::bo YDataBuff;
	xrt::bo UDataBuff;
	xrt::bo VDataBuff;
	xrt::bo output;
	xrt::bo params;
	void *FileData;
	void *YData;
	void *UData;
	void *VData;
	void *output_m;
	void *params_m;
} AcceleratorHandle;


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

int preprocess(
		AcceleratorHandle *accelerator, const char *img_name,int out_ht, int out_wt,uint64_t dpu_input_buf_addr, int no_zcpy)
{

	uint8_t *FileData = (uint8_t*)accelerator->FileData;

	std::string jpeg_fname = img_name;

	// Initialize context
	auto jfif_inst = jfif_init(FileData);

	if (jfif_inst == NULL) 
	{
		std::cout << "[  ERROR] Problem initializing JFIF context." << std::endl;
		return EXIT_FAILURE;
	}

	// Read file
	auto file_size = jfif_file_read(jfif_inst, jpeg_fname.c_str());

	if (file_size == 0) 
	{
		std::cout << "[  ERROR] Problem reading JPEG file : \'" << jpeg_fname.c_str() << "'" << std::endl;
		return EXIT_FAILURE;
	} 

	// Run parser
	auto status = jfif_parse(jfif_inst);    

	if (status) 
	{
		std::cout << "[  ERROR] Problem parsing the JPEG file : \'" << jpeg_fname.c_str() << "'" << std::endl;
		return EXIT_FAILURE;
	}

	uint32_t image_height  = jfif_get_frame_y(jfif_inst);
	uint32_t image_width   = jfif_get_frame_x(jfif_inst);
	uint32_t luma_stride   = image_width;//4096;
	uint32_t chroma_stride = image_width;//4096;
	uint32_t d_stride=0;

	if(image_height > 1080 || image_width > 1920)
	{
		std::cout<<"Image resolution of "<<img_name<<" exceeds the maximum permitted (SKIPPING)"<< std::endl ;
		return 0;
	}

	img_num++;

	// Making stride multiple of 8
	
	luma_stride   = (luma_stride   + 7) & 0xFFFFFFF8;
	chroma_stride = (chroma_stride + 7) & 0xFFFFFFF8;
	d_stride = (chroma_stride << 16) | (luma_stride & 0x0000FFFF);
	
	//image_width = luma_stride;

	accelerator->FileDataBuff.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE,file_size,0);

	uint32_t d_mode          = 0,
		 d_size          = 0;


	// Launch JPEG kernel
	accelerator->jpeg_runner(d_mode,d_size,d_stride,file_size,accelerator->FileDataBuff,accelerator->YDataBuff,accelerator->UDataBuff,accelerator->VDataBuff);

	//// Wait for execution to finish
	accelerator->jpeg_runner.wait();

	//Launch BlobfromImage kernel
	if (!no_zcpy)
		accelerator->blob_runner(accelerator->YDataBuff,accelerator->UDataBuff,accelerator->VDataBuff,dpu_input_buf_addr,accelerator->params,image_width ,image_height,luma_stride,out_wt,out_ht, out_wt, out_ht, out_wt,0,0);
	else
		accelerator->blob_runner(accelerator->YDataBuff,accelerator->UDataBuff,accelerator->VDataBuff,accelerator->output,accelerator->params,image_width ,image_height,luma_stride,out_wt,out_ht, out_wt, out_ht, out_wt,0,0);

	//// Wait for execution to finish
	accelerator->blob_runner.wait();

	if (no_zcpy)
	{     
		// Copy the output data from device to host memory
		const int output_size = out_ht * out_wt * 3 * sizeof(char);

		accelerator->output.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);

		int8_t *out_data = (int8_t *)dpu_input_buf_addr;   
		std::memcpy(out_data,accelerator->output_m, output_size);
	}

	return 0;
}

	AcceleratorHandle *
pp_kernel_init(int out_ht, int out_wt, int no_zcpy,float input_scale,float* mean)

{

	// get xclbin dir path and acquire handle
	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");

	if (xclbinPath == nullptr)
		throw std::runtime_error("Error: xclbinPath is not set, please consider setting XLNX_VART_FIRMWARE.");

	// get available xclbins
	auto xclbins = get_xclbins_in_dir(xclbinPath);
	const char *xclbin = xclbins[0].c_str();

	// Device/Card Index on system
	unsigned device_index = 0;

	// Check for devices on the system
	if (device_index >= xclProbe())
	{
		throw std::runtime_error("Cannot find device index specified");
		return nullptr;
	} 

	// Acquire Device by index
	auto device = xrt::device(device_index);    
	// Load XCLBIN
	auto uuid = device.load_xclbin(xclbin);    
	// Get Kernel/Pre-Proc CU
	auto preproc_accelerator = xrt::kernel(device, uuid.get(), "blobfromimage_accel");
	auto jpeg_decoder = xrt::kernel(device, uuid.get(), "jpeg_decoder");

	// Get runner instance from xrt
	auto blob_runner = xrt::run(preproc_accelerator); 
	auto jpeg_runner = xrt::run(jpeg_decoder);

	// Create BO for input/output/params

	auto FileDataBuff_mem_grp = jpeg_decoder.group_id(4);

	auto YDataBuff_mem_grp = preproc_accelerator.group_id(0);
	auto UDataBuff_mem_grp = preproc_accelerator.group_id(1);
	auto VDataBuff_mem_grp = preproc_accelerator.group_id(2);
	auto output_mem_grp = preproc_accelerator.group_id(3);
	auto params_mem_grp = preproc_accelerator.group_id(4);

	// Creating memory for 4K image

	long MaxDataSize = (MAX_LENGTH/8 * sizeof(uint64_t));

	auto FileDataBuff = xrt::bo(device,MaxDataSize, FileDataBuff_mem_grp);
	auto YDataBuff = xrt::bo(device,MaxDataSize, YDataBuff_mem_grp);
	auto UDataBuff = xrt::bo(device, MaxDataSize, UDataBuff_mem_grp);
	auto VDataBuff = xrt::bo(device,MaxDataSize, VDataBuff_mem_grp);

	void *FileData = FileDataBuff.map();
	if (FileData == nullptr)
		throw std::runtime_error("[ERRR] FileData pointer is invalid\n");

	void *YData = YDataBuff.map();
	if (YData == nullptr)
		throw std::runtime_error("[ERRR] YData pointer is invalid\n");


	void *UData = UDataBuff.map();
	if (UData == nullptr)
		throw std::runtime_error("[ERRR] UData pointer is invalid\n");


	void *VData = VDataBuff.map();
	if (VData == nullptr)
		throw std::runtime_error("[ERRR] VData pointer is invalid\n");


	// Create memory for params
	const int params_size = 9 * sizeof(float);
	auto params = xrt::bo(device, params_size, params_mem_grp);
	void *params_m = params.map();

	if (params_m == nullptr)    
		throw std::runtime_error("[ERRR] Params pointer is invalid\n");    

	float params_local[9];
	//# Mean params
	params_local[0] = mean[0];
	params_local[1] = mean[1];
	params_local[2] = mean[2];
	//# Input scale
	params_local[3] = params_local[4] = params_local[5]= input_scale;
	//# Set to default zero
	params_local[6] = params_local[7] = params_local[8] = 0.0;

	// Copy to params BO
	std::memcpy(params_m, params_local, params_size);
	// Send the params data to device memory
	params.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
	// Return accelerator handle
	auto accel_handle = new AcceleratorHandle();

	if (no_zcpy)
	{
		// Create memory for output image
		const int out_image_size = out_ht * out_wt * 3 * sizeof(char);

		auto output = xrt::bo(device, out_image_size, output_mem_grp);
		void *output_m = output.map();

		if (output_m == nullptr)
			throw std::runtime_error("[ERRR] Output pointer is invalid\n");

		accel_handle->output = std::move(output);
		accel_handle->output_m = output_m;
	}


	accel_handle->blob_kernel = std::move(preproc_accelerator);
	accel_handle->device = std::move(device);
	accel_handle->jpeg_kernel = std::move(jpeg_decoder);
	accel_handle->jpeg_runner = std::move(jpeg_runner);
	accel_handle->blob_runner = std::move(blob_runner);
	accel_handle->FileDataBuff = std::move(FileDataBuff);
	accel_handle->YDataBuff = std::move(YDataBuff);
	accel_handle->UDataBuff = std::move(UDataBuff);
	accel_handle->VDataBuff = std::move(VDataBuff);

	accel_handle->params = std::move(params);
	accel_handle->FileData = FileData;
	accel_handle->YData = YData;
	accel_handle->UData = UData;
	accel_handle->VData = VData;

	accel_handle->params_m = params_m;

	// Return accelerator handle
	return accel_handle;
}
