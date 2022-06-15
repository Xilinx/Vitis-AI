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




#include "sort_wrapper.h"



const int NUM_CLASS = 91;
const int MAX_BOX_PER_CLASS = 32;
const int ELE_PER_BOX = 8;
const int NMS_FX = 9830;
const int CLASS_SIZE = 1917;


using namespace std;

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



AcceleratorHandle* pp_kernel_init()

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
    AcceleratorHandle* preproc_handle = new AcceleratorHandle();

    // Get runner instance from xrt
    auto runner = xrt::run(preproc_accelerator);
    // Create BO for input/output/params

    auto input_mem_grp = preproc_accelerator.group_id(0);
    auto dummy1_grp = preproc_accelerator.group_id(1); 
	auto dummy2_grp = preproc_accelerator.group_id(2);
    auto output_mem_grp = preproc_accelerator.group_id(3);
    auto params_mem_grp = preproc_accelerator.group_id(4);
     // std::cout << "[INFO] Mem Grp Input : " << input_mem_grp << std::endl;
    //std::cout << "[INFO] Mem Grp Output: " << output_mem_grp << std::endl;
    //std::cout << "[INFO] Mem Grp Params: " << params_mem_grp << std::endl;

    // Creating memory for 4K image
    const int in_image_size = 3840 *2160 * 3 * sizeof(char);
    
    auto input = xrt::bo(device, in_image_size, input_mem_grp);
    void *input_m = input.map();
    
    if (input_m == nullptr)
        throw std::runtime_error("[ERRR] Input pointer is invalid\n");

    auto dummy1 = xrt::bo(device, sizeof(float), dummy1_grp);
	
	void *dummy1_m = dummy1.map();
	if (dummy1_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
	
	auto dummy2 = xrt::bo(device, sizeof(float), dummy2_grp);
	
	void *dummy2_m = dummy2.map();
	if (dummy2_m == nullptr)
		throw std::runtime_error("[ERRR] Input pointer is invalid\n");
    
    // Create memory for params
    const int params_size = 9 * sizeof(float);
    auto params = xrt::bo(device, params_size, params_mem_grp);
    void *params_m = params.map();
    if (params_m == nullptr)
        throw std::runtime_error("[ERRR] Params pointer is invalid\n");
    float params_local[9];
    //# Mean params
    params_local[0] = 0.0;
    params_local[1] = 0.0;
    params_local[2] = 0.0;
    //# Input scale
    params_local[3] = params_local[4] = params_local[5]=0.502;
    //# Set to default zero
    params_local[6] = params_local[7] = params_local[8] = -64;
    // Copy to params BO
    std::memcpy(params_m, params_local, params_size);
    // Send the params data to device memory
    params.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
    // Return accelerator handle
    
    float xx = 1;
	float* yy = &xx;

	std::memcpy(dummy1_m, yy,sizeof(float));
	std::memcpy(dummy2_m, yy,sizeof(float));

    dummy1.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE,sizeof(float), 0);    
	dummy2.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE,sizeof(float), 0);
 


    preproc_handle->kernel = std::move(preproc_accelerator);
    preproc_handle->device = std::move(device);
    preproc_handle->runner = std::move(runner);
    preproc_handle->input = std::move(input);
    preproc_handle->dummy1 = std::move(dummy1);
	preproc_handle->dummy2 = std::move(dummy2);
    preproc_handle->params = std::move(params);
    preproc_handle->input_m = input_m;
    preproc_handle->dummy1_m = dummy1_m;
	preproc_handle->dummy2_m = dummy2_m; 
    preproc_handle->params_m = params_m;
    
    // Return accelerator handle
    return preproc_handle;
}

int preprocess(unsigned char *in_image_data,
    int img_ht, int img_wt, int out_ht, int out_wt, uint64_t dpu_input_buf_addr,AcceleratorHandle* preproc_handle)
{

    // Input size to transfer
    const int imageToDevice_size = img_wt * img_ht * 3 * sizeof(char);
    // Copy to input BO
	
    std::memcpy(preproc_handle->input_m, in_image_data, imageToDevice_size);
    // Send the input imageToDevice data to the device memory
    preproc_handle->input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE, imageToDevice_size, 0);    
    // threshold values
    int th1 = 127, th2 = 128;
     
    // Invoke accelerator
    
    preproc_handle->runner(preproc_handle->input, preproc_handle->dummy1, preproc_handle->dummy2, dpu_input_buf_addr, preproc_handle->params, img_wt, img_ht,img_wt, out_wt, out_ht,out_wt,out_ht, out_wt, 0, 0);
      
    //// Wait for execution to finish
	
    preproc_handle->runner.wait();
   
   
   
      return 0;
}

AcceleratorHandle* hw_sort_init(const short *&fx_pror)
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
    AcceleratorHandle* postproc_handle = new AcceleratorHandle();
    // Acquire Device by index
    auto device = xrt::device(device_index);
    // Load XCLBIN
    auto uuid = device.load_xclbin(xclbin);
    auto postproc_accelerator = xrt::kernel(device, uuid.get(), "sort_nms_accel");
    
    // Get runner instance from xrt
    auto runner = xrt::run(postproc_accelerator);
    // Create BO for input/output/params

    auto input_mem_grp = postproc_accelerator.group_id(5);
    auto output_mem_grp = postproc_accelerator.group_id(6);
    //  std::cout << "[INFO] Mem Grp Input : " << input_mem_grp << std::endl;
    //std::cout << "[INFO] Mem Grp Output: " << output_mem_grp << std::endl;
    //std::cout << "[INFO] Mem Grp Params: " << params_mem_grp << std::endl;
    // Creating memory
    long int outSize_bytes = (NUM_CLASS-1)*MAX_BOX_PER_CLASS*ELE_PER_BOX*sizeof(short);
    const int prior_sz = 1917*4*sizeof(short);
    auto input = xrt::bo(device, prior_sz, input_mem_grp);
    void *input_m = input.map();
    if (input_m == nullptr)
        throw std::runtime_error("[ERRR] Input pointer is invalid\n");
      

	// Create memory for params
    auto output = xrt::bo(device, outSize_bytes, output_mem_grp);
    
    void *output_m = output.map();
    if (output_m == nullptr)
    {
        throw std::runtime_error("[ERRR] Params pointer is invalid\n");
        return nullptr;
    }

    std::memcpy(input_m, fx_pror, prior_sz);   
    input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
    

    postproc_handle->kernel = std::move(postproc_accelerator);
    postproc_handle->device = std::move(device);
    postproc_handle->runner = std::move(runner);
    postproc_handle->input = std::move(input);
    postproc_handle->output = std::move(output);
    postproc_handle->input_m = input_m;
    postproc_handle->output_m = output_m;
   
    return postproc_handle;
}

int runHWSort(AcceleratorHandle* postproc_handle, short *&nms_out)
{	
postproc_handle->runner(postproc_handle->dpu_conf_out_phy_addr, postproc_handle->dpu_box_out_phy_addr, postproc_handle->dpu_box_out_phy_addr, postproc_handle->dpu_box_out_phy_addr, postproc_handle->dpu_box_out_phy_addr, postproc_handle->input, postproc_handle->output, CLASS_SIZE, NMS_FX);
postproc_handle->runner.wait();

#if EN_SORT_PROFILE
		auto start_k2= std::chrono::system_clock::now();

		auto end_k2= std::chrono::system_clock::now();
		auto difft_k2= end_k2-start_k2;
		auto value_k2 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k2);
		std::cout << "Kernel Execution " << value_k2.count() << " us\n";

		auto start_k3= std::chrono::system_clock::now();

		auto end_k3= std::chrono::system_clock::now();
		auto difft_k3= end_k3-start_k3;
		auto value_k3 = std::chrono::duration_cast<std::chrono::microseconds>(difft_k3);
		std::cout << "synbo Out " << value_k3.count() << " us\n";
#endif
        postproc_handle->output.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
		nms_out = (short *)postproc_handle->output_m;

	
	return 0;
}



  
