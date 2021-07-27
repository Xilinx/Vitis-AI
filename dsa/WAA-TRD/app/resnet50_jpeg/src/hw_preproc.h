// Copyright 2021 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//Host code to run the pre-processing pipeline

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <sys/time.h>
#include <CL/cl.h>
#include "xcl2.hpp"
#include "jfif.h"
#define MAX_LENGTH        16777216 // (16 MB)

extern "C"{
//PP Handle class
class PPHandle {
	public:
		cl::Context contxt;
		cl::Kernel krnl_pp;
		cl::Kernel krnl_jpg;  
		cl_command_queue commands;
		cl_program program;
		cl_device_id device_id;
		cl::Device device;
		cl::Buffer FileDataBuff;
		cl::Buffer YDataBuff;
		cl::Buffer UDataBuff;
		cl::Buffer VDataBuff;
		cl::size_type MaxDataSize;
		uint8_t *FileData;
		uint8_t *YData;
		uint8_t *UData;
		uint8_t *VData;
		cl::CommandQueue queue_jpg;
		cl::CommandQueue queue_pp;
   
};

// Function to load/read XCLBIN file
cl::Program::Binaries import_binary_file(std::string xclbin_file_name) {
    std::cout << "[   INFO] Importing " << xclbin_file_name << std::endl;
    if (access(xclbin_file_name.c_str(), R_OK) != 0) {
        std::cout << "[  ERROR] '" << xclbin_file_name.c_str() << "' xclbin not available. Please build it first." << std::endl;
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    // Loading XCL Bin into char buffer
    std::cout << "[   INFO] Loading: '" << xclbin_file_name.c_str() << "'" << std::endl;
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);
    bin_file.close();
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    return bins;
}

// Function to wait for Command Q to be empty
void waitForQueueEmpty(cl::CommandQueue *queue, std::string extra_msg = "") {
  if (extra_msg != "") {
    extra_msg = "("+extra_msg+") ";
  }
  //std::cout << "[   INFO] Waiting for command Q to finsh " << extra_msg << "..." << std::endl;
  queue->finish();
  //std::cout << "[Info] Command Q is empty" << std::endl;
}

uint32_t out_height,out_width;
uint32_t out_height_resize,out_width_resize;

int op_ker_img;
//Init function to Load xclbin and get cl kernel and context
int pp_kernel_init(PPHandle * &handle, char *xclbin, const char *kernelName, int deviceIdx,float *mean,float input_scale)
{
	op_ker_img = 1;
	PPHandle *my_handle = new PPHandle;
	handle = my_handle = (PPHandle *)my_handle;
	cl_int err; // error code returned from api calls
	
	//Enable Multiprocess mode
    char mps_env[] = "XCL_MULTIPROCESS_MODE=1";
   	if (putenv(mps_env) != 0) {
		std::cout << "putenv failed" << std::endl;
		}
	
	//Find xilinx device and create clContext
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[deviceIdx];
	cl::Context context(device);

	cl::Program::Binaries bins = import_binary_file(xclbin);
	devices.resize(1);

	//Check for device name
	/*	std::string dev_name ("xilinx_u200_xdma_201830_1");
		if(dev_name.compare(devices[0].getInfo<CL_DEVICE_NAME>()) != 0){
			std::cout << "Device Not Supported" << std::endl;
			return -1;
		}
	*/

	//Create clKernel and clProgram
	cl_device_info info;
	OCL_CHECK(err,cl::Program program(context, devices, bins, NULL, &err));
	OCL_CHECK(err,cl::Kernel krnl_pp(program,kernelName,&err));
    OCL_CHECK(err, cl::Kernel krnl_jpg(program, "jpeg_decoder", &err));
    cl::size_type MaxDataSize = (cl::size_type)(MAX_LENGTH/8 * sizeof(uint64_t));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer  FileDataBuff(context, CL_MEM_READ_ONLY,       MaxDataSize, NULL, &err));
    OCL_CHECK(err, cl::Buffer     YDataBuff(context, CL_MEM_WRITE_ONLY,      MaxDataSize, NULL, &err));
    OCL_CHECK(err, cl::Buffer     UDataBuff(context, CL_MEM_WRITE_ONLY,      MaxDataSize, NULL, &err));
    OCL_CHECK(err, cl::Buffer     VDataBuff(context, CL_MEM_WRITE_ONLY,      MaxDataSize, NULL, &err));
  
    uint32_t d_mode          = 0,
             d_size          = 0;

  	//Set JPEG kernel arguments
	OCL_CHECK(err, err = krnl_jpg.setArg(5, YDataBuff));
	OCL_CHECK(err, err = krnl_jpg.setArg(6, UDataBuff));
	OCL_CHECK(err, err = krnl_jpg.setArg(7, VDataBuff));
	OCL_CHECK(err, err = krnl_jpg.setArg(4, FileDataBuff));
    OCL_CHECK(err, err = krnl_jpg.setArg(0, d_mode));
    OCL_CHECK(err, err = krnl_jpg.setArg(1, d_size)); // Dummy
	
	//Set PP kernel arguments
	OCL_CHECK(err, err = krnl_pp.setArg(0, YDataBuff));
    OCL_CHECK(err, err = krnl_pp.setArg(1, UDataBuff));
    OCL_CHECK(err, err = krnl_pp.setArg(2, VDataBuff));
	
	OCL_CHECK(err, cl::CommandQueue queue_jpg(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	uint8_t *FileData  = (uint8_t  *)queue_jpg.enqueueMapBuffer( FileDataBuff, CL_TRUE, CL_MAP_WRITE, 0, MaxDataSize);
    uint8_t *YData     = (uint8_t  *)queue_jpg.enqueueMapBuffer(    YDataBuff, CL_TRUE, CL_MAP_READ,  0, MaxDataSize);
    uint8_t *UData     = (uint8_t  *)queue_jpg.enqueueMapBuffer(    UDataBuff, CL_TRUE, CL_MAP_READ,  0, MaxDataSize);
    uint8_t *VData     = (uint8_t  *)queue_jpg.enqueueMapBuffer(    VDataBuff, CL_TRUE, CL_MAP_READ,  0, MaxDataSize);

	out_height = 224;
	out_width = 224;

	//Params for quantization kernel
	float params[9];
	//Mean values
	params[0] = 123.68;
	params[1] = 116.78f;
	params[2] = 103.94f;
	params[3] = params[4] = params[5] = 1.0;
	params[6] = params[7] = params[8] = 0.0;
	int th1=255,th2=255;
	cl::Buffer paramsbuf(context, CL_MEM_READ_ONLY,9*4);
	
	OCL_CHECK(err, err = krnl_pp.setArg(6, out_height));
	OCL_CHECK(err, err = krnl_pp.setArg(7, out_width));
	OCL_CHECK(err, err = krnl_pp.setArg(8, out_height));
	OCL_CHECK(err, err = krnl_pp.setArg(9, out_width));
	OCL_CHECK(err, err = krnl_pp.setArg(10, paramsbuf));
	OCL_CHECK(err, err = krnl_pp.setArg(11, th1));
	OCL_CHECK(err, err = krnl_pp.setArg(12, th2));
	
	cl::CommandQueue queue_pp(context, device,CL_QUEUE_PROFILING_ENABLE);
	queue_pp.enqueueWriteBuffer(paramsbuf,CL_TRUE,0,9*4,params);

	my_handle->krnl_pp = krnl_pp;
    my_handle->krnl_jpg = krnl_jpg;
	my_handle->contxt = context;
	my_handle->device = device;
    my_handle->FileDataBuff = FileDataBuff;
    my_handle->YDataBuff = YDataBuff;
    my_handle->UDataBuff = UDataBuff;
    my_handle->VDataBuff = VDataBuff;
    my_handle->MaxDataSize = MaxDataSize;
	my_handle->FileData = FileData;
	my_handle->YData = YData;
	my_handle->UData = UData;
	my_handle->VData = VData;
	my_handle->queue_jpg = queue_jpg;
	my_handle->queue_pp = queue_pp;

	if(err == 0)
		return 0;
	else
		return -1;
}

//pre-processing kernel execution
int preprocess(PPHandle * &handle, const char *img_name, int *org_ht, int *org_wt, unsigned char *data_ptr)
{
	cv::Mat img,result;
	cl::Context context = handle->contxt;
	cl::Kernel krnl_pp = handle->krnl_pp;
	cl::Device device = handle->device;
	cl::Kernel krnl_jpg = handle->krnl_jpg;
	cl::Buffer FileDataBuff = handle->FileDataBuff;
	cl::Buffer YDataBuff = handle->YDataBuff;
	cl::Buffer UDataBuff = handle->UDataBuff;
	cl::Buffer VDataBuff = handle->VDataBuff;
	cl::size_type MaxDataSize = handle->MaxDataSize;
	cl::CommandQueue queue_jpg = handle->queue_jpg;
	cl::CommandQueue queue_pp = handle->queue_pp;
	cl_int err;
	
	uint8_t *FileData = handle->FileData;
	uint8_t *YData    = handle->YData;
	uint8_t *UData    = handle->UData;
	uint8_t *VData    = handle->VData;

    std::string jpeg_fname = img_name;

	// Initialize context
    auto jfif_inst = jfif_init(FileData);
    if (jfif_inst == NULL) {
        std::cout << "[  ERROR] Problem initializing JFIF context." << std::endl;
        return EXIT_FAILURE;
    }

    // Read file
    auto file_size = jfif_file_read(jfif_inst, jpeg_fname.c_str());

    if (file_size == 0) {
        std::cout << "[  ERROR] Problem reading JPEG file : \'" << jpeg_fname.c_str() << "'" << std::endl;
        return EXIT_FAILURE;
    } else {
        //std::cout << "[   INFO] Successfully read file and loaded into host memory (" << file_size << " Bytes)." << std::endl;
    }

    // Run parser
    auto status = jfif_parse(jfif_inst);    
    if (status) {
        std::cout << "[  ERROR] Problem parsing the JPEG file : \'" << jpeg_fname.c_str() << "'" << std::endl;
        return EXIT_FAILURE;
    } else {
        //std::cout << "[   INFO] Successfully parsed the file." << std::endl;
    }

    uint32_t d_stride          = 0;
    uint32_t image_height  = jfif_get_frame_y(jfif_inst);
    uint32_t image_width   = jfif_get_frame_x(jfif_inst);
    uint32_t luma_stride   = image_width;//4096;
    uint32_t chroma_stride = image_width;//4096;

    // Making stride multiple of 8
    luma_stride   = (luma_stride   + 7) & 0xFFFFFFF8;
    chroma_stride = (chroma_stride + 7) & 0xFFFFFFF8;
    d_stride = (chroma_stride << 16) | (luma_stride & 0x0000FFFF);

    //Set JPEG kernel arguments
	OCL_CHECK(err, err = krnl_jpg.setArg(2, d_stride));
    OCL_CHECK(err, err = krnl_jpg.setArg(3, file_size));

    // Migrate input data to kernel space
    OCL_CHECK(err, err = queue_jpg.enqueueMigrateMemObjects({FileDataBuff}, 0)); // 0 means from host
    waitForQueueEmpty(&queue_jpg, "pre-process and MigrateMemObjects");

    cl::Event event_krnl;

	// Lanch JPEG kernel
	OCL_CHECK(err, err = queue_jpg.enqueueTask(krnl_jpg, NULL, &event_krnl));
	waitForQueueEmpty(&queue_jpg, "kernel execution");

	image_width = luma_stride;
	*org_ht = image_height;
	*org_wt = image_width;
	//result.create(cv::Size(224, 224),CV_32FC3);
	//result.create(cv::Size(224, 224),CV_32FC3);
	result.create(cv::Size(224, 224),CV_8UC3);
	std::vector<cl::Memory> inBufVec, outBufVec, paramasbufvec;
	//cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width*3*4);
	cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width*3);
	
	//Set PP kernel arguments
	OCL_CHECK(err, err = krnl_pp.setArg(3, imageFromDevice));
	OCL_CHECK(err, err = krnl_pp.setArg(4, image_height));
	OCL_CHECK(err, err = krnl_pp.setArg(5, image_width));

	cl::Event event_sp;
	
	// Launch PP kernel
	queue_pp.enqueueTask(krnl_pp,NULL,&event_sp);
	clWaitForEvents(1, (const cl_event*) &event_sp);

	//Copy data from device to Host
	queue_pp.enqueueReadBuffer(imageFromDevice,CL_TRUE,0,out_height*out_width*3,data_ptr);
	queue_pp.finish();
	return 0;
}
} // end of extern "C"