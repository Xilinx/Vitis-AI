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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cassert>

/* header file for Vitis AI advanced APIs */
#include <dnndk/dnndk.h>

using namespace std;

/* DPU Kernel name for SSD */
#define KERNEL_SSD "ssd"
/* DPU node name for input */
#define INPUT_NODE "ssd300_conv1_conv1_1_Conv2D"

typedef struct {
  int8_t *addrVirt;  /* virtural address of DPU input/output memory buffer */
  int8_t *addrPhy;   /* pysical address of DPU input/output memory buffer */
  int    size;       /* byte size of DPU input/output memory buffer */
} dpuBufferInfo;

  /* Handle for DPU input/output buffer */
  void *inBufferHandle, *outBufferHandle;

int8_t strToInt8 (const string &str) {
  stringstream ss;
  ss << str;
  int32_t result;
  ss >> result;
  return (int8_t)result;
}

/**
 * @brief Print tensor info
 *
 * @param task - task pointer
 * @param tensorName - task pointer for model SSD
 *
 * @return none
 */
void PrintTensorInfo(DPUTask *task, char *tensorName) {
  /* Get the boundary input or output tensor of model SSD */
  DPUTensor *tensor = dpuGetBoundaryIOTensor(task, tensorName);

  /* Dump tensor's following info:
   * name, virtual address, size,
   * height/width/channel, quantized scale info
   */
  printf("Output Tensor [%s]:\n", tensorName);
  printf("--------------------------------------------------\n");

  printf("%15s : 0x%lx\n", "Address ",
                  (int64_t)dpuGetTensorAddress(tensor));
  printf("%15s : %d\n", "Size",
                  dpuGetTensorSize(tensor));
  printf("%15s : (%d*%d*%d)\n", "Shape(H*W*C)",
                  dpuGetTensorHeight(tensor),
		              dpuGetTensorWidth(tensor),
                  dpuGetTensorChannel(tensor));
  printf("%15s : %2f\n", "Scale: ",
                  dpuGetTensorScale(tensor));
}

/**
 * @brief Entry to run TensorFlow SSD model on DPU
 *
 * @param task - task pointer for model SSD
 *
 * @return none
 */
void RunSSD(DPUTask *task) {
  assert(task);
  string int8Str;

  /* Input comes from int8 type data within file input_int8.txt */
  ifstream inputStream("input_int8.txt");

  /* Get the input tensor address with name "image:0", which is the only
   * input tensor of SSD model. Tensor name info can be found from DNNC
   * log after model compilation.
   */
  int8_t* inputAddr =
          dpuGetTensorAddress(dpuGetBoundaryIOTensor(task, "image:0"));

  /* Fetch data of int8 type and feed into DPU input memory buffer */
  while(getline(inputStream, int8Str)) {
    *inputAddr = strToInt8(int8Str);
    inputAddr++;
  };

  inputStream.close();

  /* The input memory buffer allocated by dpuAllocMem() is cacheable.
   * dpuSyncMemToDev() is used to flush input data from CPU cache line
   * to DPU input memory buffer before DPU task running. For the input/
   * output memory management implemented by the users, it is up to the
   * users to flush/invalidate cache if memory buffer is cacheable.
   */
  dpuSyncMemToDev(inBufferHandle, 0, dpuGetInputTotalSize(task));

  /* Trigger the running of DPU Task */
  dpuRunTask(task);

  /* The output memory buffer allocated by dpuAllocMem() is cacheable.
   * dpuSyncMemToDev() is used to mark CPU cache line as invalid for
   * afterwards DPU output memory buffer accessing after DPU task
   * execution is completed.
   */
  dpuSyncDevToMem(outBufferHandle, 0, dpuGetOutputTotalSize(task));

  /* print two DPU output tensors info for model SSD
   * "ssd300_concat:0" and "ssd300_concat_1:0" are names for two output
   * tensors, which can be found from DNNC log after model compilation.
  */
  PrintTensorInfo(task, (char*)"ssd300_concat:0");
  PrintTensorInfo(task, (char*)"ssd300_concat_1:0");
}

/**
 * @brief Example code to illustrate Vitis AI advanced API feature split-IO memory
 * optimization over TensorFlow SSD model.
 *
 * @param argc - argument counter
 * @param argv - argument vector
 *
 * @return exit code
 */
int main(int argc, char **argv) {
  /* DPU Kernel and Task for running Tensorflow SSD model */
  DPUKernel *kernel;
  DPUTask *task;

  /* memory info for DPU input/output buffer */
  dpuBufferInfo inBufferInfo, outBufferInfo;

  /* Attach to DPU device and prepare for running */
  dpuOpen();

  /* Allocate two physical continuous memory buffers for input/output tensors
   * of model SSD. The size info for them is listed out after model compilation
   * by DNNC. If the model holds multiple input or output tensors, the
   * size of input/output memory buffer equals to the ending address of the
   * last tensor minus the starting address of the first tensor. Note that
   * multiple tensors' layout inside input/output memory buffer is determined
   * by DNNC compiler, and there maybe exist paddings between adjacent tensors.
   * For SSD, there is one input tensor with size 270000, and there are two
   * tensors with size total 218304.
   */
  inBufferInfo.size  = 270000;
  outBufferInfo.size = 218304;

  /* Using dpuAllocMem() to allocate physical continuous memory buffers for
   * input/output tensors. The virutal/physical addresses are returned and
   * written to dpuBufferInfo.
   * Note that for the production scenarios, the users should implement the
   * functionalities of pyhiscal memory buffer allocation and even cacheabley
   * memory management instead of directly using dpuAllocMem() and dpuFreeMem().
   */
  inBufferHandle = dpuAllocMem(inBufferInfo.size,
                     inBufferInfo.addrVirt, inBufferInfo.addrPhy);
  outBufferHandle = dpuAllocMem(outBufferInfo.size,
                     outBufferInfo.addrVirt, outBufferInfo.addrPhy);

  assert(inBufferHandle && outBufferHandle);

  /* Load DPU Kernel and create DPU Task for model SSD */
  kernel = dpuLoadKernel(KERNEL_SSD);
  task = dpuCreateTask(kernel, 0);

  /* Bind input/output memory buffer to Task */
  dpuBindInputTensorBaseAddress(task,
                     inBufferInfo.addrVirt, inBufferInfo.addrPhy);
  dpuBindOutputTensorBaseAddress(task,
                     outBufferInfo.addrVirt, outBufferInfo.addrPhy);

  RunSSD(task);

  /* Destroy DPU Task and Kernel and free resources */
  dpuDestroyTask(task);
  dpuDestroyKernel(kernel);

  /* Free input/output memory buffers */
  dpuFreeMem(inBufferHandle);
  dpuFreeMem(outBufferHandle);

  /* Detach from DPU device */
  dpuClose();

  return 0;
}
