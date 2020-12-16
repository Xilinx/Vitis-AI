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
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include <CL/cl.h>
#include <xcl2.hpp>

#include <aks/AksKernelBase.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksNodeParams.h>

struct PreProcKernelHandle {
  cl::Context context;
  cl::Kernel kernel;
  cl::CommandQueue commandQueue;
  cl::Program program;
  cl_device_id deviceId;
  cl::Device device;
};

class PreProcKernel: public AKS::KernelBase
{
  public:
    struct PreProcKernelHandle _preProcHandle;
    int getNumCUs(void);
    void nodeInit(AKS::NodeParams* params);
    int exec_async (
        std::vector<AKS::DataDescriptor*> &in,
        std::vector<AKS::DataDescriptor*> &out, 
        AKS::NodeParams* nodeParams,
        AKS::DynamicParamValues* dynParams);
};

int PreProcKernel::getNumCUs(void)
{
  return 1;
}

extern "C" { /// Add this to make this available for python bindings and dlsym
  /**
   * @brief This function sets up the pre-processing kernel using OpenCL runtime APIs.
   * 
   * @param  params structure which contains all the parameters required
   * @returns Returns KernelBase class pointer
   */
  AKS::KernelBase* getKernel (AKS::NodeParams* params)
  {
    /// Update KernelBase and return
    PreProcKernel * handle = new PreProcKernel();
    return handle;

  } //init
}// extern "C"

/**
 * @brief This function execute the Pre-Processing Kernel on the device using OpenCL
 * runtime APIs.
 * 
 * @param  handle KernelBase class pointer 
 * @param  in Vector of input Data Descriptor
 * @param  out Vector of output Data Descriptor
 * @param  params NodeParams structure which contains all the parameter required
 * @returns Integer
 */
int PreProcKernel::exec_async (std::vector<AKS::DataDescriptor*> &in, 
    std::vector<AKS::DataDescriptor*> &out, 
    AKS::NodeParams* nodeParams,
    AKS::DynamicParamValues* dynParams)
{
  /// Get input and output data shapes
  /// Input could be batch array or batch of images
  const std::vector<int>& inShape = in[0]->getShape();
  int batchSize = inShape[0];

  /// Get output Dimensions (Network input dim)
  int outHeight    = nodeParams->_intParams["net_h"];
  int outWidth     = nodeParams->_intParams["net_w"];
  int outChannels  = 3;
  int nOutElemsPerImg = outChannels * outHeight * outWidth;

  /// Create output data buffer
  std::vector<int> shape      = { batchSize, outChannels, outHeight, outWidth };
  AKS::DataDescriptor * outDD = new AKS::DataDescriptor(shape, AKS::DataType::FLOAT32);
  float * outData = (float*) outDD->data();

  /// CV Mat to store output data from device memory
  cv::Mat result;
  result.create(cv::Size(outHeight, outWidth), CV_16SC3);

  /// Params for quantization kernel
  float kernelParams[9];
  kernelParams[3] = kernelParams[4] = kernelParams[5] = 0.0;
  /// Mean values
  auto meanIter = nodeParams->_floatVectorParams.find("mean");
  kernelParams[0] = meanIter->second[0];
  kernelParams[1] = meanIter->second[1];
  kernelParams[2] = meanIter->second[2];
  int th1 = nodeParams->_intParams.find("th1")->second;
  int th2 = nodeParams->_intParams.find("th2")->second;
  int act_img_h, act_img_w;

  /////////////////////////// CL ///////////////////////////////

  /// Get the kernel handle object
  PreProcKernel * preProcHandlePtr = this;

  int inHeight  = 0;
  int inWidth   = 0;
  int inChannel = 3;
  uint8_t* inData = nullptr;
  for (int b = 0; b < batchSize; ++b) {
    if(in[0]->dtype() == AKS::DataType::AKSDD) {
      auto& dd = in[0]->data<AKS::DataDescriptor>()[b];
      inData   = dd.data<uint8_t>();
      // Shape = (Batch=1, H, W, C=3)
      inHeight = dd.getShape()[1];
      inWidth  = dd.getShape()[2];
    }
    else {
      inHeight = inShape[1];
      inWidth  = inShape[2];
      int nInElemsPerImg  = inChannel * inHeight * inWidth;
      inData   = in[0]->data<uint8_t>() + nInElemsPerImg;
    }

    /// Buffer creation
    cl::Buffer imageToDevice (preProcHandlePtr->_preProcHandle.context,
        CL_MEM_READ_ONLY, inHeight * inWidth * 3);

    cl::Buffer imageFromDevice (preProcHandlePtr->_preProcHandle.context,
        CL_MEM_WRITE_ONLY, outHeight * outWidth * 3 * 2);

    cl::Buffer paramsBuf (preProcHandlePtr->_preProcHandle.context,
        CL_MEM_READ_ONLY, 9 * sizeof(float));

    /// Set kernel arguments
    preProcHandlePtr->_preProcHandle.kernel.setArg(0, imageToDevice);
    preProcHandlePtr->_preProcHandle.kernel.setArg(1, imageFromDevice);
    preProcHandlePtr->_preProcHandle.kernel.setArg(2, inHeight);
    preProcHandlePtr->_preProcHandle.kernel.setArg(3, inWidth);
    preProcHandlePtr->_preProcHandle.kernel.setArg(4, outHeight);
    preProcHandlePtr->_preProcHandle.kernel.setArg(5, outWidth);
    preProcHandlePtr->_preProcHandle.kernel.setArg(6, paramsBuf);
    preProcHandlePtr->_preProcHandle.kernel.setArg(7, th1);
    preProcHandlePtr->_preProcHandle.kernel.setArg(8, th2);

    /// Copy data from host to FPGA
    preProcHandlePtr->_preProcHandle.commandQueue.enqueueWriteBuffer(
        imageToDevice,
        CL_TRUE,
        0,
        inHeight * inWidth * 3,
        inData
        );

    preProcHandlePtr->_preProcHandle.commandQueue.enqueueWriteBuffer(
        paramsBuf,
        CL_TRUE,
        0,
        9 * 4,
        kernelParams
        );

    cl::Event execEvent;

    /// Launch the kernel
    preProcHandlePtr->_preProcHandle.commandQueue.enqueueTask(
        preProcHandlePtr->_preProcHandle.kernel, NULL, &execEvent);
    /// Wait for the tast to finish
    clWaitForEvents(1, (const cl_event*)&execEvent);

    /// Copy data from device to Host
    preProcHandlePtr->_preProcHandle.commandQueue.enqueueReadBuffer(
        imageFromDevice,
        CL_TRUE,
        0,
        outHeight * outWidth * 3 * 2,
        result.data
        );

    /// Profiling
    //cl_ulong start= 0;
    //cl_ulong end = 0;
    //double diff = 0.0;
    //execEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    //execEvent.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    //diff_prof = end - start;
    //std::cout << "Kernel latency = " << (diff / 1000000.0) << "ms" << std::endl;

    /// Stop Command Queue
    preProcHandlePtr->_preProcHandle.commandQueue.finish();

    /////////////////////////// end of CL /////////////////////////////

    /// Fixed to float conversion
    int idx = 0, frameCnt = 0;
    float *out  = outData + b * nOutElemsPerImg;
    float *dst1 = out;
    float *dst2 = out + outHeight * outWidth;
    float *dst3 = out + outHeight * outWidth * 2;

    short *img_data = (short *)result.data;
    for (int l_rows = 0; l_rows < 224; l_rows++) {
      for(int l_cols = 0; l_cols < 224; l_cols++) {
        dst1[idx] = (float)img_data[frameCnt++] / 128.0f;
        dst2[idx] = (float)img_data[frameCnt++] / 128.0f;
        dst3[idx] = (float)img_data[frameCnt++] / 128.0f;
        idx++;
      } //l_cols
    } //l_rows
  } //batch_size
  out.push_back(outDD);
  return 0;
}

void PreProcKernel::nodeInit(AKS::NodeParams* params)
{
  /// Enable Multi-process mode
  char mps_env[] = "XCL_MULTIPROCESS_MODE=1";
  if (putenv(mps_env) != 0) {
    std::cout << "[ERR] Multi-process environment setting Failed!" << std::endl;
    return ;
  }

  /// Find Xilinx device and create CL context
  std::vector<cl::Device> devices = xcl::get_xil_devices();
  cl::Device device = devices[0/*Fix with Butler?*/];
  cl::Context context (device);

  /// Load XCLBIN
  unsigned fileBufferSize = 0;
  auto fileBuffer = xcl::read_binary_file (params->getValue<string>("xclbin").c_str(), fileBufferSize);
  cl::Program::Binaries bins {{fileBuffer, fileBufferSize}};

  devices.resize(1);
  if (device != devices[0]) {
    std::cerr << "[ERR] Device Index must be 0 " << std::endl;
    return ;
  }

  /// Check for device name
  std::string dev_name ("xilinx_u200_xdma_201830_2");
  if(dev_name.compare(devices[0].getInfo<CL_DEVICE_NAME>()) != 0) {
    std::cout << "[ERR] Device Not Supported: " << dev_name << std::endl;
    return ;
  }

  /// Create clKernel and clProgram
  cl_device_info devInfo;
  cl_int err;

  cl::Program program(context, devices, bins, NULL, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "[ERR] CL Program Creation Failed !"  << std::endl;
    return ;
  }

  cl::Kernel kernel(program, "pp_pipeline_accel", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "[ERR] CL Kernel Creation Failed !"  << std::endl;
    return ;
  }

  /// Command Queue 
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

  _preProcHandle.kernel  = kernel;
  _preProcHandle.context = context;
  _preProcHandle.device  = device;
  _preProcHandle.program = program;
  _preProcHandle.commandQueue = q;
}
