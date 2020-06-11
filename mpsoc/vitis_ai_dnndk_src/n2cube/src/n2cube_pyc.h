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

#ifndef _N2CUBE_PYC__H_
#define _N2CUBE_PYC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

int pyc_dpuSetExceptionMode(int mode);
int pyc_dpuGetExceptionMode();
const char *pyc_dpuGetExceptionMessage(int error_code);

/* Open & initialize the usage of DPU device */
int pyc_dpuOpen();

/* Close & finalize the usage of DPU device */
int pyc_dpuClose();


/* Load a DPU Kernel and allocate DPU memory space for
   its Code/Weight/Bias segments */
DPUKernel *pyc_dpuLoadKernel(const char *netName);

/* Set mean values for DPU Kernel */
int pyc_dpuSetKernelMeanValue(DPUKernel *kernel, int mean1, int mean2, int mean3);

/* Destroy a DPU Kernel and release its associated resources */
int pyc_dpuDestroyKernel(DPUKernel *kernel);

/* Instantiate a DPU Task from one DPU Kernel, allocate its private
   working memory buffer and prepare for its execution context */
DPUTask *pyc_dpuCreateTask(DPUKernel *kernel, int mode);

/* Launch the running of DPU Task */
int pyc_dpuRunTask(DPUTask *task);

/* Remove a DPU Task, release its working memory buffer and destroy
   associated execution context */
int pyc_dpuDestroyTask(DPUTask *task);

/* Enable dump facility of DPU Task while running for debugging purpose */
int pyc_dpuEnableTaskDebug(DPUTask *task);

/* Enable profiling facility of DPU Task while running to get its performance metrics */
int pyc_dpuEnableTaskProfile(DPUTask *task);

/* Get the execution time of DPU Task */
long long pyc_dpuGetTaskProfile(DPUTask *task);

/* Get the execution time of DPU Node */
long long pyc_dpuGetNodeProfile(DPUTask *task, const char*nodeName);


/*
 * API for both single IO and multiple IO.
 * For multiply IO, should specify the input/output tensor idx.
 */

/* Get total number of input Tensor of DPU Task */
int pyc_dpuGetInputTensorCnt(DPUTask * task, const char * nodeName);

/* Get input Tensor of DPU Task */
DPUTensor* pyc_dpuGetInputTensor(DPUTask *task, const char*nodeName, int idx);

/* Get the start address of DPU Task's input Tensor */
int8_t* pyc_dpuGetInputTensorAddress(DPUTask *task, const char *nodeName, int idx);

/* Get the size (in byte) of one DPU Task's input Tensor */
int pyc_dpuGetInputTensorSize(DPUTask *task, const char *nodeName, int idx);

/* Get the scale value (DPU INT8 quantization) of one DPU Task's input Tensor */
float pyc_dpuGetInputTensorScale(DPUTask *task, const char *nodeName, int idx);

/* Get the height dimension of one DPU Task's input Tensor */
int pyc_dpuGetInputTensorHeight(DPUTask *task, const char *nodeName, int idx);

/* Get the width dimension of one DPU Task's input Tensor */
int pyc_dpuGetInputTensorWidth(DPUTask *task, const char *nodeName, int idx);

/* Get the channel dimension of one DPU Task's input Tensor */
int pyc_dpuGetInputTensorChannel(DPUTask *task, const char *nodeName, int idx);

/* Get total number of output Tensor of DPU Task */
int pyc_dpuGetOutputTensorCnt(DPUTask * task, const char * nodeName);

/* Get output Tensor of one DPU Task */
DPUTensor* pyc_dpuGetOutputTensor(DPUTask *task, const char *nodeName, int idx);

/* Get the start address of one DPU Task's output Tensor */
int8_t* pyc_dpuGetOutputTensorAddress(DPUTask *task, const char *nodeName, int idx);

/* Get the size (in byte) of one DPU Task's output Tensor */
int pyc_dpuGetOutputTensorSize(DPUTask *task, const char *nodeName, int idx);

/* Get the scale value (DPU INT8 quantization) of one DPU Task's output Tensor */
float pyc_dpuGetOutputTensorScale(DPUTask *task, const char *nodeName, int idx);

/* Get the height dimension of one DPU Task's output Tensor */
int pyc_dpuGetOutputTensorHeight(DPUTask *task, const char *nodeName, int idx);

/*  Get the channel dimension of one DPU Task's output Tensor */
int pyc_dpuGetOutputTensorWidth(DPUTask *task, const char *nodeName, int idx);

/* Get DPU Node's output tensor's channel */
int pyc_dpuGetOutputTensorChannel(DPUTask *task, const char *nodeName, int idx);

/* Set DPU Task's input Tensor with data stored under Caffe
   Blob's order (channel/height/width) in INT8 format */
int pyc_dpuSetInputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *data, int size, int idx);

/* Set DPU Task's input Tensor with data stored under Caffe
   Blob's order (channel/height/width) in FP32 format */
int pyc_dpuSetInputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *data, int size, int idx);

/* Set DPU Task's input Tensor with data stored under DPU
   Tensor's order (height/width/channel) in INT8 format */
int pyc_dpuSetInputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *data, int size, int idx);

/* Set DPU Task's input Tensor with data stored under DPU
   Tensor's order (height/width/channel) in FP32 format */
int pyc_dpuSetInputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *data, int size, int idx);

/* Get DPU Task's output Tensor and store them under Caffe
   Blob's order (channel/height/width) in INT8 format */
int pyc_dpuGetOutputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *data, int size, int idx);

/* Get DPU Task's output Tensor and store them under Caffe
   Blob's order (channel/height/width) in FP32 format */
int pyc_dpuGetOutputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *data, int size, int idx);

/* Get DPU Task's output Tensor and store them under DPU
   Tensor's order (height/width/channel) in INT8 format */
//int  pyc_dpuGetOutputTensorInHWCInt8(DPUTask *task, const char *nodeName, int idx);
int pyc_dpuGetOutputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *data, int size, int idx);

/* Get DPU Task's output Tensor and store them under DPU
   Tensor's order (height/width/channel) in FP32 format */
int pyc_dpuGetOutputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx);


/* Get the size of one DPU Tensor */
int pyc_dpuGetTensorSize(DPUTensor* tensor);

/* Get the start address of one DPU Tensor */
int8_t* pyc_dpuGetTensorAddress(DPUTensor* tensor);

/* Get the scale value of one DPU Tensor */
float pyc_dpuGetTensorScale(DPUTensor* tensor);

/* Get the height dimension of one DPU Tensor */
int pyc_dpuGetTensorHeight(DPUTensor* tensor);

/* Get the width dimension of one DPU Tensor */
int pyc_dpuGetTensorWidth(DPUTensor* tensor);

/* Get the channel dimension of one DPU Tensor */
int pyc_dpuGetTensorChannel(DPUTensor* tensor);

/* Compute softmax */
int pyc_dpuRunSoftmax(int8_t *input, float *output, int numClasses, int batchSize, float scale);

/* Set or get the priority of one DPU task. Priority range is 0 to 15, 0 has the highest priority.
   The priority of the task when it was created defaults to 15. */
int pyc_dpuSetTaskPriority(DPUTask *task, uint8_t priority);
uint8_t pyc_dpuGetTaskPriority(DPUTask *task);

/* Set or get the core mask of the task binding. Each bit represents a core.
   The default value of coreMask is 0x7 when the number of cores is 3. */
int pyc_dpuSetTaskAffinity(DPUTask *task, uint32_t coreMask);
uint32_t pyc_dpuGetTaskAffinity(DPUTask *task);

/* Get the total size in memory of all the inputs of the network,
   which involve the interleaves between two inputs. */
int pyc_dpuGetInputTotalSize(DPUTask *task);

/* Get the total size in memory of all the outputs of the network,
   which involve the interleaves between two outputs. */
int pyc_dpuGetOutputTotalSize(DPUTask *task);

DPUTensor *pyc_dpuGetBoundaryIOTensor(DPUTask *task, const char *tensorName);

#ifdef __cplusplus
}
#endif
#endif
