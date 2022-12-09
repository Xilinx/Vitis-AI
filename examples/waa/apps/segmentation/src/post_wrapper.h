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
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// lowlevel common include
#include "utils.h"
#include "xhls_dpupostproc_m_axi_hw.h"
using namespace std;

class PostHandle {
	public:

		xrt::kernel kernel;
		xrt::device device;
		xrt::run runner;
		xrt::bo post_input_bo;
		xrt::bo out_max_bo;
		xrt::bo out_idx_bo;
		void *post_input_m;
		void *out_max_m;
		void *out_idx_m;
		int out_size;
};

#define POST_MAX_HEIGHT     832/2   //((1080+8)/2)  // number of lines per image
#define POST_MAX_WIDTH     1920/2   //(1920/2)      // number of pixels per line

int postprocess(PostHandle* posthandle, int8_t* out_idx_data, uint64_t dpu_output_phy_addr, float scale_fact, int out_height, int out_width) 
{

	posthandle->runner(dpu_output_phy_addr, posthandle->out_max_bo, posthandle->out_idx_bo, scale_fact, out_height, out_width);

	posthandle->runner.wait();

	posthandle->out_idx_bo.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);


	return 0;
}

PostHandle* post_kernel_init(char *xclbin, int out_height, int out_width)
{
	unsigned device_index = 0;

	// Check for devices on the system
	if (device_index >= xclProbe()) {
		throw std::runtime_error("Cannot find device index specified");
		return nullptr;
	}

	// Acquire Device by index
	auto device = xrt::device(device_index);
	// Load XCLBIN
	auto uuid = device.load_xclbin(xclbin);
	// Get Kernel/Pre-Proc CU
	auto postproc_accelerator = xrt::kernel(device, uuid.get(), "hls_dpupostproc_m_axi");

	// Get runner instance from xrt
	auto runner = xrt::run(postproc_accelerator);
	// Create BO for input/output/params

	auto input_mem_grp = postproc_accelerator.group_id(0);
	auto out_max_grp = postproc_accelerator.group_id(1);
	auto out_idx_grp = postproc_accelerator.group_id(2);

	// Creating memory for 4K image
	const int outToHost_size = out_height * out_width * 1 * sizeof(char);
	// const int outToHost_size = POST_MAX_HEIGHT * POST_MAX_WIDTH * 1 * sizeof(char);

	// Create BO
	auto post_input_bo = xrt::bo(device, outToHost_size * 28, input_mem_grp);
	auto out_max_bo = xrt::bo(device, outToHost_size, out_max_grp);
	auto out_idx_bo = xrt::bo(device, outToHost_size, out_idx_grp);

	void *out_max_m = out_max_bo.map();
	if (out_max_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");

	void *out_idx_m = out_idx_bo.map();
	if (out_idx_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");

	void *post_input_m = post_input_bo.map();
	if (post_input_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");  

	auto post_accel = new PostHandle;  

	post_accel->kernel = std::move(postproc_accelerator);
	post_accel->device = std::move(device);
	post_accel->runner = std::move(runner);
	post_accel->post_input_bo = std::move(post_input_bo);
	post_accel->out_max_bo = std::move(out_max_bo);
	post_accel->out_idx_bo = std::move(out_idx_bo);
	post_accel->post_input_m = post_input_m;
	post_accel->out_max_m = out_max_m;
	post_accel->out_idx_m = out_idx_m;
	post_accel->out_size = outToHost_size;

	// Return accelerator handle
	return post_accel;
}


