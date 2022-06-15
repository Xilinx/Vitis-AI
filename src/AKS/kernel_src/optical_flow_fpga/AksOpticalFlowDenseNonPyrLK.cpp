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

// Kernel Functions Implementation

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksTensorBuffer.h>
#include <aks/AksLogger.h>

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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>

#define STR_VALUE(arg)         #arg
#define GET_STRING(name)       STR_VALUE(name)
#define TARGET_DEVICE          GET_STRING(xilinx_u200_gen3x16_xdma_base_1)
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
        std::vector<vart::TensorBuffer *> &in,
        std::vector<vart::TensorBuffer *> &out,
        AKS::NodeParams* params, 
        AKS::DynamicParamValues* dynParams);
    int getNumCUs(void);

  private:
    std::string xclBinary;
    std::string lk_kName;
    int of_width;               // Equal to input image width
    int of_height;              // Equal to input image height
    int frame_id = 0;
    // allocate OF input and output on host
    std::vector<float> flowx;   // EnqueueReadBuffer (host memory)
    std::vector<float> flowy;   // EnqueueReadBuffer (host memory)
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
  lk_kName = params->_stringParams.find("lk_kName") == params->_stringParams.end() ?
    "dense_non_pyr_of_accel" : params->getValue<std::string>("lk_kName");
  of_width = params->getValue<int>("of_w");
  of_height = params->getValue<int>("of_h");

  if (params->_stringParams.find("xclbin") != params->_stringParams.end()) {
    xclBinary = params->getValue<std::string>("xclbin");
  }
  else if ((std::getenv("XLNX_VART_FIRMWARE") != NULL)) {
    xclBinary = std::getenv("XLNX_VART_FIRMWARE");
  }
  else {
    std::cout << "[ERROR] Either pass xclbin parameter to the graph node or set XLNX_VART_FIRMWARE";
    return;
  }

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
    cout << "[ERROR] Failed to find an OpenCL platform!";
    std::cout << "Test failed" << std::endl;
    return ;
  }
  std::cout << "[INFO] Found " << platform_count << " platforms" << std::endl;

  // Find Xilinx Plaftorm
  for (cl_uint plat_idx = 0; plat_idx < platform_count; plat_idx++) {
    err = clGetPlatformInfo(platforms[plat_idx], CL_PLATFORM_VENDOR, 1000,
                            (void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS) {
      std::cout << "[ERROR] clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!"
                << std::endl;
      std::cout << "Test failed" << std::endl;
      return ;
    }
    if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
      std::cout << "[INFO] Selected platform " << plat_idx << " from "
                << cl_platform_vendor << std::endl;
      platform_id = platforms[plat_idx];
      platform_found = 1;
      break;
    }
  }
  if (!platform_found) {
    std::cout << "[ERROR] Platform Xilinx not found. Exit." << std::endl;
    return ;
  }

  // Get Accelerator compute device
  cl_uint num_devices;
  cl_uint device_found = 0;
  cl_device_id devices[16];  // compute device id
  char cl_device_name[1001];
  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices,
                       &num_devices);
  std::cout << "[INFO] Found " << num_devices << " devices" << std::endl;
  if (err != CL_SUCCESS) {
    std::cout << "[ERROR] Failed to create a device group!" << std::endl;
    std::cout << "[ERROR] Test failed" << std::endl;
    return ;
  }

  //iterate all devices to select the target device.
  cl_uint device_idx = 0;
  if (std::getenv("XLNX_ENABLE_DEVICES") != NULL) {
    device_idx = std::stoi(std::getenv("XLNX_ENABLE_DEVICES"));
    std::cout << "[INFO] Taking the device_idx from XLNX_ENABLE_DEVICES: "
              << device_idx << std::endl;
    num_devices = device_idx + 1;
  }
  for (; device_idx < num_devices; device_idx++) {
    err = clGetDeviceInfo(devices[device_idx], CL_DEVICE_NAME, 1024,
                          cl_device_name, 0);
    if (err != CL_SUCCESS) {
      std::cout << "[Error] Failed to get device name for device "
                << device_idx << std::endl;
      std::cout << "Test failed" << std::endl;
      return ;
    }
    std::cout << "CL_DEVICE_NAME " << cl_device_name << std::endl;
    if(strcmp(cl_device_name, target_device_name) == 0) {
      device_id = devices[device_idx];
      device_found = 1;
      std::cout << "Selected " << cl_device_name << " as the target device"
                << std::endl;
      break;
    }
  }

  std::cout << "cl device name is " << cl_device_name << std::endl;
  if (!device_found) {
    std::cout << "Target device " << target_device_name << " not found. Exit."
              << std::endl;
    return ;
  }

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    std::cout << "[Error] Failed to create a compute context!" << std::endl;
    std::cout << "Test failed" << std::endl;
    return ;
  }
  std::cout << "[INFO] Context created on device_id: " << device_id
            << std::endl;
  //------------------------------------------------------------------------------
  // xclbin
  //------------------------------------------------------------------------------
  std::cout << "[INFO] loading xclbin " << xclBinary << std::endl;
  cl_uint n_i0 = load_file_to_memory(xclBinary.c_str(), (char**) &kernelbinary);
  if (n_i0 < 0) {
    std::cout << "failed to load kernel from xclbin: " << xclBinary.c_str()
              << std::endl;
    std::cout << "Test failed" << std::endl;
    return ;
  }

  size_t n0 = n_i0;
  // Create the compute program from offline
  program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
      (const unsigned char **) &kernelbinary, &status, &err);
  free(kernelbinary);
  if ((!program) || (err!=CL_SUCCESS)) {
    std::cout << "[Error] Failed to create compute program from binary " << err
              << std::endl;
    std::cout << "Test failed" << std::endl;
    return ;
  }

  // Build the program executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    std::cout << "[Error] Failed to build program executable!" << std::endl;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);
    std::cout << buffer << std::endl;
    std::cout << "Test failed" << std::endl;
    return ;
  }

  // Create the compute kernel in the program we wish to run
  of_kernel = clCreateKernel(program, lk_kName.c_str(), &err);
  if (!of_kernel || err != CL_SUCCESS) {
    std::cout << "[Error] Failed to create compute kernel!" << std::endl;
    std::cout << "Test failed" << std::endl;
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
    std::cout << "[ERROR] Failed to create command queue" << std::endl;
    std::cout << "[ERROR] code " << err << std::endl;
    return;
  }

  // allocate OF input and output opencl buffers
  err = CL_SUCCESS;

  ocl_of_in_prev_buf = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      (size_t)(of_width*of_height*sizeof(unsigned char)), NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "[ERROR] Failed to create ocl_in_prev_buf " << err
              << std::endl;
    return;
  }

  ocl_of_in_cur_buf = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      (size_t)(of_width*of_height *sizeof(unsigned char)), NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "[ERROR] Failed to create ocl_in_cur_buf " << err
              << std::endl;
    return;
  }

  ocl_flowx = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      (size_t)(of_width*of_height *sizeof(float)), NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "[ERROR] Failed to create ocl_outx0 " << err << std::endl;
    return;
  }

  ocl_flowy = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      (size_t)(of_width*of_height *sizeof(float)), NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "[ERROR] Failed to create ocl_outy0 " << err << std::endl;
    return;
  }
  flowx.reserve(of_width*of_height);
  flowy.reserve(of_width*of_height);
  // flowx.create(of_width, of_height, CV_32FC1);
  // flowy.create(of_width, of_height, CV_32FC1);
}


int OpticalFlowDenseNonPyrLK::getNumCUs(void)
{
  return 1;
}


int OpticalFlowDenseNonPyrLK::exec_async (
    vector<vart::TensorBuffer *>& in, vector<vart::TensorBuffer *>& out,
    AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{

  // std::cout << "[DBG] Starting OpticalFlowDenseNonPyrLK... " << std::endl;

  std::vector<int> inShape = in[0]->get_tensor()->get_shape();
  uint8_t* currData = reinterpret_cast<uint8_t*>(in[0]->data().first);

  std::vector<int> prevShape = in[1]->get_tensor()->get_shape();
  uint8_t* prevData = reinterpret_cast<uint8_t*>(in[1]->data().first);

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
    std::cout << "[Error] Failed to clSetKernelArg! = " << err << std::endl;
  }
  // Write prev & curr frame into DDR memory through OCL APIs
  err = clEnqueueWriteBuffer(
      command_queue, ocl_of_in_prev_buf, CL_TRUE, 0,
      (of_width*of_height)*sizeof(unsigned char), prevData, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to write to source array error = " << err << std::endl;
  }
  err = clEnqueueWriteBuffer(
      command_queue, ocl_of_in_cur_buf, CL_TRUE, 0,
      (of_width*of_height)*sizeof(unsigned char), currData, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to write to source array! = " << err << std::endl;
  }
  // Launch OF kernel
  cl_event event;

  err = clEnqueueTask(command_queue, of_kernel, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to clEnqueueTask error " << err << std::endl;
  }
  clFinish(command_queue);
  err = clWaitForEvents(1, (const cl_event*) &event);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to clEnqueueTask clWaitForEvents error "
              << err << std::endl;
  }

  cl_event readevent, readevent1;
  // Read back the results x and y flow vectors from the device to host memory
  err = clEnqueueReadBuffer(
      command_queue, ocl_flowx, CL_TRUE, 0,
      (of_height*of_width)*sizeof(float), flowx.data(), 0, NULL, &readevent);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to read output array! flowx0 " << err << std::endl;
  }
  err = clWaitForEvents(1, &readevent);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to clEnqueueReadBuffer flowx clWaitForEvents error "
              << err << std::endl;
  }

  err = clEnqueueReadBuffer(
      command_queue, ocl_flowy, CL_TRUE, 0,
      (of_height*of_width)*sizeof(float), flowy.data(), 0, NULL, &readevent1);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to read output array! flowy0 " << err << std::endl;
  }
  err = clWaitForEvents(1, &readevent1);
  if (err != CL_SUCCESS)
  {
    std::cout << "[Error] Failed to clEnqueueReadBuffer flowy clWaitForEvents error "
              << err << std::endl;
  }

  auto xFlowDD = new AKS::AksTensorBuffer(xir::Tensor::create(
    "xFlowTensor", { of_height, of_width }, xir::create_data_type<float>()));
  float* xFlowData = reinterpret_cast<float*>(xFlowDD->data().first);
  std::memcpy(xFlowData, flowx.data(), of_height*of_width*sizeof(float));
  out.push_back(xFlowDD);

  auto yFlowDD = new AKS::AksTensorBuffer(xir::Tensor::create(
    "yFlowTensor", { of_height, of_width }, xir::create_data_type<float>()));
  float* yFlowData = reinterpret_cast<float*>(yFlowDD->data().first);
  std::memcpy(yFlowData, flowy.data(), of_height*of_width*sizeof(float));
  out.push_back(yFlowDD);

  // std::cout << "[DBG] Finished OpticalFlowDenseNonPyrLK... " << std::endl;
  return 0;
}

