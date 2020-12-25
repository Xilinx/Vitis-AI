// Copyright 2019 Xilinx Inc.
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

#include "common/xf_headers.hpp"


#include <sys/time.h>

#include <CL/cl.h>
#include "xcl2.hpp"

//Creating a class containing cl objects
extern "C"{
//PP Handle class
class PPHandle {
public:

  cl::Context contxt;
  cl::Kernel kernel;
  cl_command_queue commands;
  cl_program program;
  cl_device_id device_id;
  cl::Device device;

};

//Init function to Load xclbin and get cl kernel and context
int pp_kernel_init(PPHandle * &handle,
					char *xclbin,
					const char *kernelName,
					int deviceIdx)
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
//Check for device name
/*	std::string dev_name ("xilinx_u200_xdma_201830_1");
	if(dev_name.compare(devices[0].getInfo<CL_DEVICE_NAME>()) != 0){
		std::cout << "Device Not Supported" << std::endl;
		return -1;
	}
*/

//Create clKernel and clProgram
	cl_device_info info;
	cl_int errr;

	OCL_CHECK(errr,cl::Program program(context, devices, bins, NULL, &errr));

	OCL_CHECK(errr,cl::Kernel krnl(program,kernelName,&errr));


	my_handle->kernel = krnl;
	my_handle->contxt = context;
	my_handle->device = device;


	if(errr == 0)
	return 0;
	else
	return -1;
}

//pre-processing kernel execution
int preprocess(PPHandle * &handle, char *img_name, int *org_ht, int *org_wt, float *data_ptr)
{
	//profiling Objects
//	struct timeval start_,end_;
//	struct timeval start_imread,end_imread;
//	struct timeval start_fx2fl,end_fx2fl;
//	double lat_ = 0.0f;
//	double lat_imread = 0.0f;
//	double lat_fx2fl = 0.0f;
//    gettimeofday(&start_, 0);
//CV Mat to store input image and output data
	cv::Mat img,result;
//Read input image
//    gettimeofday(&start_imread, 0);
	img = cv::imread(img_name, 1);
// gettimeofday(&end_imread, 0);
//    lat_imread = (end_imread.tv_sec * 1e6 + end_imread.tv_usec) - (start_imread.tv_sec * 1e6 + start_imread.tv_usec);
//    std::cout << "\n\n imread latency " << lat_imread / 1000 << "ms" << std::endl;

	if(!img.data){
		fprintf(stderr,"\n input image not found");
		return -1;
	}
	int in_width,in_height;
	int out_width,out_height;

	in_width = img.cols;
	in_height = img.rows;

	*org_ht = in_height;
	*org_wt = in_width;
//output image dimensions 224x224
	out_height = 224;
	out_width = 224;

	result.create(cv::Size(224, 224),CV_32FC3);

	//Params for quantization kernel
	float params[9];
	//Mean values
	params[0] = 123.68;
	params[1] = 116.78f;
	params[2] = 103.94f;
//	params[0] = 104.007f;
//	params[1] = 116.669f;
//	params[2] = 122.679f;
	params[3] = params[4] = params[5] = 0.0;
	int th1=255,th2=255;
	int act_img_h, act_img_w;

/////////////////////////////////////// CL ///////////////////////////////////////



	cl::Context context = handle->contxt;
	cl::Kernel krnl = handle->kernel;
	cl::Device device = handle->device;
	//Buffer creation
	cl::CommandQueue q(context, device,CL_QUEUE_PROFILING_ENABLE);
	std::vector<cl::Memory> inBufVec, outBufVec, paramasbufvec;
	cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, in_height*in_width*3);
	cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width*3*4);
	cl::Buffer paramsbuf(context, CL_MEM_READ_ONLY,9*4);

//Set kernel arguments

	krnl.setArg(0, imageToDevice);
	krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, in_height);
	krnl.setArg(3, in_width);
	krnl.setArg(4, out_height);
	krnl.setArg(5, out_width);
	krnl.setArg(6, paramsbuf);
	krnl.setArg(7, th1);
	krnl.setArg(8, th2);

//Copy data from host to FPGA
	q.enqueueWriteBuffer(
                         imageToDevice,
                         CL_TRUE,
                         0,
                         in_height*in_width*3,
                         img.data);

	q.enqueueWriteBuffer(
                         paramsbuf,
                         CL_TRUE,
                         0,
                         9*4,
                         params);



	// Profiling Objects
	cl_ulong start= 0;
	cl_ulong end = 0;
	double diff_prof = 0.0f;
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
                        // result.data);
//Profiling
//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
//	diff_prof = end-start;
//	std::cout<<"kernel latency = "<<(diff_prof/1000000)<<"ms"<<std::endl;

	q.finish();


/////////////////////////////////////// end of CL ///////////////////////////////////////

// gettimeofday(&end_, 0);

//    lat_ = (end_.tv_sec * 1e6 + end_.tv_usec) - (start_.tv_sec * 1e6 + start_.tv_usec);
//    std::cout << "\n\n Overall latency " << lat_ / 1000 << "ms" << std::endl;

	return 0;
}
}
