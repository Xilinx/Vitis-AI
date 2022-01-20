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

#include <iostream>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

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
    int width;
    int height;

    xrt::run runner;

    xrt::bo prev_img_input;
    xrt::bo curr_img_input;
    xrt::bo flow_x_output;
    xrt::bo flow_y_output;

    void *prev_img_input_m;
    void *curr_img_input_m;
    void *flow_x_output_m;
    void *flow_y_output_m;
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
  auto lk_kName = params->_stringParams.find("lk_kName") == params->_stringParams.end() ?
    "dense_non_pyr_of_accel" : params->getValue<std::string>("lk_kName");
  width = params->getValue<int>("of_w");
  height = params->getValue<int>("of_h");

  std::string xclBinary;
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

  cl_int err;                         // error code returned from api calls
  cl_platform_id platform_id;         // platform id

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

  // iterate all devices to select the target device.
  unsigned device_idx = 0;
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
      std::cout << "[ERROR] Failed to get device name for device "
                << device_idx << std::endl;
      return ;
    }
    std::cout << "[INFO] CL_DEVICE_NAME " << cl_device_name << std::endl;
    if(strcmp(cl_device_name, target_device_name) == 0) {
      device_found = 1;
      std::cout << "[INFO] Selected " << cl_device_name
                << " (idx: " << device_idx << ") as the target device"
                << std::endl;
      break;
    }
  }

  if (!device_found) {
    throw std::runtime_error("[ERR] Target device not found.\n");
  }

  auto device = xrt::device(device_idx);
  // Load XCLBIN
  std::cout << "[INFO] Loading xclbin " << xclBinary << std::endl;
  auto uuid = device.load_xclbin(xclBinary);

  // Get Kernel/Pre-Proc CU
  auto krnl = xrt::kernel(device, uuid, lk_kName);

  // // Get runner instance from xrt
  runner = xrt::run(krnl);
  runner.set_arg(4, height);
  runner.set_arg(5, width);

  prev_img_input = xrt::bo(device, width*height*sizeof(uint8_t), krnl.group_id(0));
  curr_img_input = xrt::bo(device, width*height*sizeof(uint8_t), krnl.group_id(1));
  flow_x_output = xrt::bo(device, width*height*sizeof(float), krnl.group_id(2));
  flow_y_output = xrt::bo(device, width*height*sizeof(float), krnl.group_id(3));

  // // Create input memory maps
  prev_img_input_m = prev_img_input.map<uint8_t*>();
  if (prev_img_input_m == nullptr)
    throw std::runtime_error("[ERRR] Input prev_img pointer is invalid\n");

  curr_img_input_m = curr_img_input.map<uint8_t*>();
  if (curr_img_input_m == nullptr)
    throw std::runtime_error("[ERRR] Input curr_img pointer is invalid\n");

  // // Create memory for flow_x_output image
  flow_x_output_m = flow_x_output.map<float*>();
  if (flow_x_output_m == nullptr)
    throw std::runtime_error("[ERRR] Output flow_x pointer is invalid\n");

  // // Create memory for flow_x_output image
  flow_y_output_m = flow_y_output.map<float*>();
  if (flow_y_output_m == nullptr)
    throw std::runtime_error("[ERRR] Output flow_y pointer is invalid\n");
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

  // Copy to input BO
  std::memcpy(prev_img_input_m, prevData, width*height*sizeof(uint8_t));
  std::memcpy(curr_img_input_m, currData, width*height*sizeof(uint8_t));

  // Send the input imageToDevice data to the device memory
  prev_img_input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);
  curr_img_input.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_TO_DEVICE);

  runner.set_arg(0, prev_img_input);
  runner.set_arg(1, curr_img_input);
  runner.set_arg(2, flow_x_output);
  runner.set_arg(3, flow_y_output);
  runner.start();
  runner.wait();  // Wait for execution to finish

  // Copy to output buffer
  flow_x_output.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);
  flow_y_output.sync(xclBOSyncDirection::XCL_BO_SYNC_BO_FROM_DEVICE);

  auto xFlowDD = new AKS::AksTensorBuffer(xir::Tensor::create(
    "xFlowTensor", { height, width }, xir::create_data_type<float>()));
  float* xFlowData = reinterpret_cast<float*>(xFlowDD->data().first);
  std::memcpy(xFlowData, flow_x_output_m, width*height*sizeof(float));
  out.push_back(xFlowDD);

  auto yFlowDD = new AKS::AksTensorBuffer(xir::Tensor::create(
    "yFlowTensor", { height, width }, xir::create_data_type<float>()));
  float* yFlowData = reinterpret_cast<float*>(yFlowDD->data().first);
  std::memcpy(yFlowData, flow_y_output_m, width*height*sizeof(float));
  out.push_back(yFlowDD);

  // std::cout << "[DBG] Finished OpticalFlowDenseNonPyrLK... " << std::endl;
  return 0;
}
