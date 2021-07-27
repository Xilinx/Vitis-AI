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

//Host code to run the pre-processing pipeline
#include "xf_headers.h"


#include <sys/time.h>

#include <CL/cl.h>
#include "xcl2.hpp"

//Init function to Load xclbin and get cl kernel and context
int pp_kernel_init(PPHandle * &handle,
					char *xclbin,
					const char *kernelName,
					int deviceIdx,
					float *mean)
{
	PPHandle *my_handle = new PPHandle;
	handle = my_handle = (PPHandle *)my_handle;

//Enable Multiprocess mode
    	char mps_env[] = "XCL_MULTIPROCESS_MODE=1";
   	 if (putenv(mps_env) != 0) {
        std::cout << "putenv failed" << std::endl;
    	} //else

//Find xilinx device and create clContext
	std::vector<cl::Device> devices = xcl::get_xil_devices();

	cl::Device device = devices[deviceIdx];

	cl::Context context(device);

//Load xclbin
	unsigned fileBufSize;
	std::string binaryFile = xclbin;
	auto fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    	cl::Program::Binaries bins{{fileBuf, fileBufSize}};

	devices.resize(1);

//Create clKernel and clProgram
//	cl_device_info info;
	cl_int errr;

	OCL_CHECK(errr,cl::Program program(context, devices, bins, NULL, &errr));

	std::string kernelName_s = kernelName;
	OCL_CHECK(errr,cl::Kernel krnl(program,kernelName,&errr));

	cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);


	float params[9];
	params[0] = mean[0];
	params[1] = mean[1];
	params[2] = mean[2];
	params[3] = params[4] = params[5] = 0.0;
	int th1=255,th2=255;

	cl::Buffer paramsbuf(context, CL_MEM_READ_ONLY,9*4);

	krnl.setArg(8, paramsbuf);
	krnl.setArg(9, th1);
	krnl.setArg(10, th2);

	q.enqueueWriteBuffer(
                         paramsbuf,
                         CL_TRUE,
                         0,
                         9*4,
                         params);




       
	my_handle->kernel = krnl;
	my_handle->contxt = context;
	my_handle->device = device;
	my_handle->q = q;

	if(errr == 0)
	return 0;
	else
	return -1;
}

//pre-processing kernel execution
int preprocess(PPHandle * &handle, cv::Mat img, int out_ht, int out_wt, float *mean, float *data_ptr)
{
	if(!img.data){
		fprintf(stderr,"\n input image not found");
		return -1;
	}
	int in_width,in_height;
	int out_width,out_height;

	in_width = img.cols;
	in_height = img.rows;
	
//output image dimensions 224x224
	out_height = out_ht;
	out_width = out_wt;
	
	


/////////////////////////////////////// CL ///////////////////////////////////////

	cl::Context context = handle->contxt;
	cl::Kernel krnl = handle->kernel;
	cl::Device device = handle->device;
	//Buffer creation
	cl::CommandQueue q;//(context, device,CL_QUEUE_PROFILING_ENABLE);
	q = handle->q;
	
	cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, in_height*in_width*3);
	cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width*3*4);
 	

//Set kernel arguments

	krnl.setArg(0, imageToDevice);
	krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, in_height);
	krnl.setArg(3, in_width);
	krnl.setArg(4, out_height);
	krnl.setArg(5, out_width);
	krnl.setArg(6, out_height);
	krnl.setArg(7, out_width);
	
//Copy data from host to FPGA
	q.enqueueWriteBuffer(
                         imageToDevice,
                         CL_TRUE,
                         0,
                         in_height*in_width*3,
                         img.data);
	

	cl::Event event_sp;

// Launch the kernel


	q.enqueueTask(krnl,NULL,&event_sp);
	clWaitForEvents(1, (const cl_event*) &event_sp);

//Copy data from device to Host
	q.enqueueReadBuffer(
                         imageFromDevice,
                         CL_TRUE,
                         0,
                         out_height*out_width*3*4,
                         data_ptr);
		     

	q.finish();


/////////////////////////////////////// end of CL ///////////////////////////////////////


	return 0;
}

