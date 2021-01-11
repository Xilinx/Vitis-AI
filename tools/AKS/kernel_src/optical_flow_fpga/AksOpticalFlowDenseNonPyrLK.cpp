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

// Kernel Functions Implementation

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

#include <cmath>
#include <iostream>
#include <stdint.h>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <sys/time.h> 
#include <opencv2/core/core.hpp>

#include <getopt.h>
#include <dirent.h>
#include <sys/types.h>

#include <CL/opencl.h>

#define STR_VALUE(arg)         #arg
#define GET_STRING(name)       STR_VALUE(name)
#define TARGET_DEVICE          GET_STRING(xilinx_u200_xdma_201830_2)
#define CL_EMIT_PROFILING_INFO 1

using namespace std;

cl_int load_file_to_memory(const char *filename, char **result)
{
  cl_uint size = 0;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    *result = NULL;
    return -1; // -1 means file opening fail
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size+1);
  if (size != fread(*result, sizeof(char), size, f)) {
    free(*result);
    return -2; // -2 means file reading fail
  }
  fclose(f);
  (*result)[size] = 0;
  return size;
}


class OpticalFlowDenseNonPyrLK: public AKS::KernelBase {

  public:
    bool isExecAsync() { return false; }
    void nodeInit(AKS::NodeParams*);
    int exec_async (
        std::vector<AKS::DataDescriptor *> &in, 
        std::vector<AKS::DataDescriptor *> &out, 
        AKS::NodeParams* params, 
        AKS::DynamicParamValues* dynParams);
    int getNumCUs(void);

  private:
    std::string lk_kName;
    int of_width;               // Equal to input image width
    int of_height;              // Equal to input image height
    int frame_id = 0;
    // allocate OF input and output on host
    cv::Mat flowx;              // EnqueueReadBuffer (host memory)
    cv::Mat flowy;              // EnqueueReadBuffer (host memory)
    cl_mem ocl_flowx;           // To CreateCLBuffer (device memory)
    cl_mem ocl_flowy;           // To CreateCLBuffer (device memory)
    // allocate OF input and output opencl buffers
    cl_mem ocl_of_in_prev_buf;
    cl_mem ocl_of_in_cur_buf;

    cl_kernel of_kernel;
    cl_command_queue command_queue;
};


extern "C" {
  AKS::KernelBase* getKernel(AKS::NodeParams* params) {
    // Create kernel object
    OpticalFlowDenseNonPyrLK * nonPyrOF = new OpticalFlowDenseNonPyrLK();
    return nonPyrOF;
  }
} // extern C


void OpticalFlowDenseNonPyrLK::nodeInit(AKS::NodeParams* params)
{
  // Optical flow ocl kernel and ocl I/O buffers allocation
  lk_kName = params->_stringParams.find("lk_kName") == params->_stringParams.end() ?
    "dense_non_pyr_of_accel" : params->_stringParams["lk_kName"];
  of_width = params->_intParams["of_w"];
  of_height = params->_intParams["of_h"];

  std::cout << "OF kernel" << ": " << lk_kName << std::endl;

  unsigned char *kernelbinary;
  cl_int err; cl_int status;          // error code returned from api calls
  cl_platform_id platform_id;         // platform id
  cl_device_id device_id;             // compute device id
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute programs
  cl_kernel kernel, pp_krnl;          // compute kernel

  char cl_platform_vendor[1001];
  char target_device_name[1001] = TARGET_DEVICE;

  // Get all platforms and then select Xilinx platform
  cl_platform_id platforms[16];       // platform id
  cl_uint platform_count;
  cl_uint platform_found = 0;
  err = clGetPlatformIDs(16, platforms, &platform_count);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to find an OpenCL platform!\n");
    printf("Test failed\n");
    return ;
  }
  printf("INFO: Found %d platforms\n", platform_count);

  // Find Xilinx Plaftorm
  for (cl_uint iplat=0; iplat<platform_count; iplat++) {
    err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS) {
      printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
      printf("Test failed\n");
      return ;
    }
    if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
      printf("INFO: Selected platform %d from %s\n", iplat, cl_platform_vendor);
      platform_id = platforms[iplat];
      platform_found = 1;
    }
  }
  if (!platform_found) {
    printf("ERROR: Platform Xilinx not found. Exit.\n");
    return ;
  }

  // Get Accelerator compute device
  cl_uint num_devices;
  cl_uint device_found = 0;
  cl_device_id devices[16];  // compute device id
  char cl_device_name[1001];
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &num_devices);
  printf("INFO: Found %d devices\n", num_devices);
  if (err != CL_SUCCESS) {
    printf("ERROR: Failed to create a device group!\n");
    printf("ERROR: Test failed\n");
    return ;
  }

  //iterate all devices to select the target device.
  for (cl_uint i=0; i<num_devices; i++) {
    err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to get device name for device %d!\n", i);
      printf("Test failed\n");
      return ;
    }
    printf("CL_DEVICE_NAME %s\n", cl_device_name);
    if(strcmp(cl_device_name, target_device_name) == 0) {
      device_id = devices[i];
      device_found = 1;
      printf("Selected %s as the target device\n", cl_device_name);
    }
  }

  printf("cl device name is %s \n", cl_device_name);
  if (!device_found) {
    printf("Target device %s not found. Exit.\n", target_device_name);
    return ;
  }


  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    printf("Error: Failed to create a compute context!\n");
    printf("Test failed\n");
    return ;
  }
  //------------------------------------------------------------------------------
  // xclbin
  //------------------------------------------------------------------------------
  std::string xclBinary = params->_stringParams["xclbin"];
  printf("INFO: loading xclbin %s\n", xclBinary.c_str());
  cl_uint n_i0 = load_file_to_memory(xclBinary.c_str(), (char **) &kernelbinary);
  if (n_i0 < 0) {
    printf("failed to load kernel from xclbin: %s\n", xclBinary.c_str());
    printf("Test failed\n");
    return ;
  }

  size_t n0 = n_i0;
  // Create the compute program from offline
  program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
      (const unsigned char **) &kernelbinary, &status, &err);
  free(kernelbinary);
  if ((!program) || (err!=CL_SUCCESS)) {
    printf("Error: Failed to create compute program from binary %d!\n", err);
    printf("Test failed\n");
    return ;
  }

  // Build the program executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    printf("Test failed\n");
    return ;
  }

  // Create the compute kernel in the program we wish to run
  of_kernel = clCreateKernel(program, lk_kName.c_str(), &err);
  if (!of_kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create compute kernel!\n");
    printf("Test failed\n");
    return ;
  } else {
    std::cout << "kernel dense_non_pyr_of_accel open" << std::endl;
  }

  int commandQueueProperties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (CL_EMIT_PROFILING_INFO)
    commandQueueProperties |= CL_QUEUE_PROFILING_ENABLE;
  command_queue = clCreateCommandQueue(
      context, device_id, commandQueueProperties, &err);
  if (!command_queue) {
    printf("ERROR: Failed to create command queue\n");
    printf("ERROR: code %i\n", err);
    return;
  }

  // allocate OF input and output opencl buffers
  err = CL_SUCCESS;

  ocl_of_in_prev_buf = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      (size_t)(of_width*of_height*sizeof(unsigned char)), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: Failed to create ocl_in_prev_buf %d=\n",err);
    return;
  }

  ocl_of_in_cur_buf = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      (size_t)(of_width*of_height *sizeof(unsigned char)), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: Failed to create ocl_in_cur_buf %d=\n",err);
    return;
  }

  ocl_flowx = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      (size_t)(of_width*of_height *sizeof(float)), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: Failed to create ocl_outx0 %d=\n",err);
    return;
  }

  ocl_flowy = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      (size_t)(of_width*of_height *sizeof(float)), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("ERROR: Failed to create ocl_outy0 %d=\n",err);
    return;
  }

  flowx.create(of_width, of_height, CV_32FC1);
  flowy.create(of_width, of_height, CV_32FC1);
}


int OpticalFlowDenseNonPyrLK::getNumCUs(void)
{
  return 1;
}

void dd2Mat(AKS::DataDescriptor* src, cv::Mat &dst) {
  // Gray DataDescriptor --> Gray cv::Mat
  auto shape = src->getShape();
  int channels = shape[1];
  assert(channels == 1);
  int rows = shape[2];
  int cols = shape[3];
  uint8_t* srcData = static_cast<uint8_t*>(src->data());
  for (int k=0; k<channels; k++) {
    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        dst.at<cv::Vec<uint8_t, 1>>(i,j)[k] = srcData[(k*rows*cols) + (i*cols) + j];
      }
    }
  }
}


int OpticalFlowDenseNonPyrLK::exec_async (
    vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out, 
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams) 
{
  std::vector<int> inShape = in[0]->getShape();
  cv::Mat in_frame(inShape[2], inShape[3], CV_8UC(inShape[1]));

  std::vector<int> prevShape = in[1]->getShape();
  cv::Mat prev_frame(prevShape[2], prevShape[3], CV_8UC(prevShape[1]));
  dd2Mat(in[0], in_frame);
  dd2Mat(in[1], prev_frame);

  int err = 0;
  // Setting up OF HW kernel arguments
  err |= clSetKernelArg(of_kernel, 0, sizeof(ocl_of_in_prev_buf), &ocl_of_in_prev_buf);
  err |= clSetKernelArg(of_kernel, 1, sizeof(ocl_of_in_cur_buf), &ocl_of_in_cur_buf);
  err |= clSetKernelArg(of_kernel, 2, sizeof(ocl_flowx), &ocl_flowx);
  err |= clSetKernelArg(of_kernel, 3, sizeof(ocl_flowy), &ocl_flowy);
  err |= clSetKernelArg(of_kernel, 4, sizeof(of_height), &of_height);
  err |= clSetKernelArg(of_kernel, 5, sizeof(of_width), &of_width);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to clSetKernelArg! = %d\n",err);
  }
  // Write prev & curr frame into DDR memory through OCL APIs
  err = clEnqueueWriteBuffer(
      command_queue, ocl_of_in_prev_buf, CL_TRUE, 0, (of_width*of_height)*sizeof(unsigned char),
      (long long int*)prev_frame.data, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array error = %d!\n",err);
  }
  err = clEnqueueWriteBuffer(
      command_queue, ocl_of_in_cur_buf, CL_TRUE, 0, (of_width*of_height)*sizeof(unsigned char),
      (long long int*)in_frame.data, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array! = %d\n",err);
  }
  // Launch OF kernel
  cl_event event;

  err = clEnqueueTask(command_queue, of_kernel, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to clEnqueueTask error %d!\n",err);
  }
  clFinish(command_queue);
  err = clWaitForEvents(1, (const cl_event*) &event);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to clEnqueueTask clWaitForEvents error %d!\n",err);
  }

  cl_event readevent, readevent1;
  // Read back the results x and y flow vectors from the device to host memory
  err = clEnqueueReadBuffer(
      command_queue, ocl_flowx, CL_TRUE, 0,
      (of_height*of_width)*sizeof(float), (float*)flowx.data, 0, NULL, &readevent);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! flowx0 %d\n", err);
  }
  err = clWaitForEvents(1, &readevent);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to clEnqueueReadBuffer flowx clWaitForEvents error %d!\n",err);
  }

  err = clEnqueueReadBuffer(
      command_queue, ocl_flowy, CL_TRUE, 0,
      (of_height*of_width)*sizeof(float), (float*)flowy.data, 0, NULL, &readevent1);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! flowy0  %d\n", err);
  }
  err = clWaitForEvents(1, &readevent1);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to clEnqueueReadBuffer flowy clWaitForEvents error %d!\n",err);
  }

  // std::cout << "[DBG] OpticalFlowDenseNonPyrLK: running now ... " << std::endl;

  AKS::DataDescriptor *xFlowDD = new AKS::DataDescriptor(
      { of_height, of_width }, AKS::DataType::FLOAT32);
  float* xFlowData = static_cast<float*>(xFlowDD->data());
  std::memcpy(xFlowData, flowx.data, of_height*of_width*sizeof(float));
  out.push_back(xFlowDD);

  AKS::DataDescriptor *yFlowDD = new AKS::DataDescriptor(
      { of_height, of_width }, AKS::DataType::FLOAT32);
  float* yFlowData = static_cast<float*>(yFlowDD->data());
  std::memcpy(yFlowData, flowy.data, of_height*of_width*sizeof(float));
  out.push_back(yFlowDD);
  return 0;
}
