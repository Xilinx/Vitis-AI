/*
 * Copyright 2021 Xilinx Inc.
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

#include <mutex>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/weak.hpp>

#include "./cost_volume.hpp"


CostVolumeAccel::CostVolumeAccel(std::string xclbin, unsigned device_index) {
	// Acquire Device by index
	// std::cout << "[INFO] Creating device: " << device_index << std::endl;
	device = xrt::device(device_index);
	// Load XCLBIN
	//std::cout << "[INFO] Loading xclbin: " << xclbin << std::endl;
	auto uuid = device.load_xclbin(xclbin);
	// Get Kernel/Pre-Proc CU
	//std::cout << "[INFO] Creating CostVolume kernel: " << KERNEL_NAME << std::endl;
	krnl = xrt::kernel(device, uuid, KERNEL_NAME);
        
        auto input_l_mem_grp = krnl.group_id(0);
        auto input_r_mem_grp = krnl.group_id(1);
        auto output_mem_grp = krnl.group_id(2);


	// // Get runner instance from xrt
	// std::cout << "[INFO] Creating runner" << std::endl;
	runner = xrt::run(krnl);

	// std::cout << "[INFO] Creating buffer objects" << std::endl;
	left_input = xrt::bo(device, IN_DATA_BYTES, input_l_mem_grp);
	right_input = xrt::bo(device, IN_DATA_BYTES, input_r_mem_grp);
	output = xrt::bo(device, OUT_DATA_BYTES, output_mem_grp);

	// // Create input memory maps
	// std::cout << "[INFO] Creating memory maps" << std::endl;
	left_input_m = left_input.map();
	if (left_input_m == nullptr)
		throw std::runtime_error("[ERRR] Input left pointer is invalid\n");

	right_input_m = right_input.map();
	if (right_input_m == nullptr)
		throw std::runtime_error("[ERRR] Input right pointer is invalid\n");

	// // Create memory for output image
	output_m = output.map();
	if (output_m == nullptr)
		throw std::runtime_error("[ERR] Output pointer is invalid\n");
}


void CostVolumeAccel::run(int8_t *left_input_data, int8_t *right_input_data,
						  int8_t *output_data)
{
	static std::shared_ptr<std::mutex> mtx =
    	vitis::ai::WeakStore<std::string, std::mutex>::create("cost-volume");
  	std::lock_guard<std::mutex> lock(*mtx);
	// Copy to input BO
	 //std::cout << "[INFO] Copy input data to memory maps" << std::endl;
	__TIC__(CVOL_COPY_INPUT_TO_MEM_MAPS)
        memcpy(left_input_m, left_input_data, IN_DATA_BYTES);
	memcpy(right_input_m, right_input_data, IN_DATA_BYTES);
	__TOC__(CVOL_COPY_INPUT_TO_MEM_MAPS)

	// // Send the input imageToDevice data to the device memory
	 //std::cout << "[INFO] Syncing BO to device" << std::endl;
	__TIC__(CVOL_SYNC_INPUT_TO_DEVICE)
	left_input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
	right_input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);

	
	runner.set_arg(0, left_input);

	runner.set_arg(1, right_input);

	runner.set_arg(2, output);

	runner.set_arg(3, output);
	__TOC__(CVOL_SYNC_INPUT_TO_DEVICE)

	 //std::cout << "[INFO] Executing runner" << std::endl;
	__TIC__(CVOL_RUNNER)
	runner.start();
	// std::cout << "[INFO] Waiting on runner" << std::endl;
	runner.wait();  // Wait for execution to finish
	__TOC__(CVOL_RUNNER)

	 //std::cout << "[INFO] Syncing output from device to host" << std::endl;

	__TIC__(CVOL_SYNC_OUTPUT_FROM_DEVICE)
	output.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
	__TOC__(CVOL_SYNC_OUTPUT_FROM_DEVICE)
	// // Copy to output buffer

	__TIC__(CVOL_COPY_OUTPUT_FROM_MEM_MAPS)
	memcpy(output_data, output_m, OUT_DATA_BYTES);
	__TOC__(CVOL_COPY_OUTPUT_FROM_MEM_MAPS)
}

