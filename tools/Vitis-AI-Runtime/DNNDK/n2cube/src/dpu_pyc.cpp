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
#include "dnndk/n2cube.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "dpu_def.h"
#include "n2cube_pyc.h"
/*
 * export function of DPU library
 *
 * initialization for running DPU, including
 * 1. open DPU device /dev/dpu
 * 2. allocate related resources for current process
 * 3. terminate process if fail
 */
EXPORT int pyc_dpuOpen()
{

    return dpuOpen();
}

/*
 * export function of DPU library
 *
 * finalization of DPU running, including
 * 1. notify DPU driver to release all resources hold by current process
 * 2. close DPU device /dev/dpu
 */
EXPORT int pyc_dpuClose()
{
    return dpuClose();
}


EXPORT DPUKernel *pyc_dpuLoadKernel(const char *netName)
{
    return dpuLoadKernel(netName);
}

EXPORT int pyc_dpuSetKernelMeanValue(DPUKernel *kernel, int mean1, int mean2, int mean3)
{
    return dpuSetKernelMeanValue(kernel, mean1, mean2, mean2);
}

EXPORT DPUTask *pyc_dpuCreateTask(DPUKernel *kernel, int mode)
{
    return dpuCreateTask(kernel,mode);
}

/*
 * export function of DPU library
 *
 * Launch the running of a sequential code on DPU, including
 * 1. produce a unique ID number (returned as kernel_handle_t) for the specified network
 * 2. allocate DPU memory space (consecutive in physical address)
 * 3. load DPU code and data (wights/bias) if network "net_id" is launched
 *    for the first time
 * 4. perform address relocation for DPU code if network "net_id" is launched
 *    for the first time
 *
 * For each DPU kernel, its code segment is loaded into DPU dedicated memory space
 * and will not be flushed out only when fini_dpu() is invoked by its process to
 * release DPU resources. However for DPU data segment (wights/bias/input/output),
 * the allocated memory space will be recycled when there is no enough memory space
 * to run new kernel. When the recycled DPU data is loaded into memory again, the
 * previous memory address space will be allocated to it again. The behind idea is
 * to avoid performing address relocation when DPU code is loaded into DPU
 * memory space. Such logic is implemented in the DPU driver.
 */
EXPORT int pyc_dpuRunTask(DPUTask *task)
{
    return dpuRunTask(task);
}

EXPORT int pyc_dpuDestroyTask(DPUTask *task)
{
    return dpuDestroyTask(task);
}

EXPORT int pyc_dpuEnableTaskDebug(DPUTask *task)
{
    return dpuEnableTaskDebug(task);
}

EXPORT int pyc_dpuEnableTaskProfile(DPUTask *task)
{
     return dpuEnableTaskProfile(task);
}

EXPORT int pyc_dpuDestroyKernel(dpu_kernel_t *kernel)
{
    return dpuDestroyKernel(kernel);
}

/*
 * Return DPU Task running time in us
 * supposed that:
 *  1. high resolutoin timer (64-bit-length) in Linux kernel never overflows
 *  2. task ending time should always be greater than starting time
 *
 */
EXPORT long long pyc_dpuGetTaskProfile(DPUTask *task)
{
    return dpuGetTaskProfile(task);
}
/*
 * Return DPU Task running time in us
 * supposed that:
 *  1. high resolutoin timer (64-bit-length) in Linux kernel never overflows
 *  2. task ending time should always be greater than starting time
 *
 */
EXPORT long long pyc_dpuGetNodeProfile(DPUTask *task, const char *nodeName)
{
    return dpuGetNodeProfile(task,nodeName);
}


/**
 * Get total number of input Tensor of DPU Task
 */
int pyc_dpuGetInputTensorCnt(DPUTask * task, const char * nodeName) {
    return dpuGetInputTensorCnt(task,nodeName);
}

/**
 * @brief Get kernel's input tensor (only for real Node)
 *
 * @note supposed that one kernel only have one input tensor
 *
 * @param kernel - the pointer to DPU kernel
 *
 * @return the tensor descriptor for this kernel's input
 */
EXPORT DPUTensor* pyc_dpuGetInputTensor(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensor(task,nodeName,idx);
}

/*
 * Get the start address of DPU Task's input Tensor, multiply IO supported.
 */
EXPORT int8_t* pyc_dpuGetInputTensorAddress(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorAddress(task,nodeName,idx);
}

/*
 * Get the size (in byte) of one DPU Task's input Tensor, multiply IO supported.
 */
EXPORT int pyc_dpuGetInputTensorSize(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorSize(task,nodeName,idx);
}

/*
 * Get the height dimension of one DPU Task's input Tensor, multiply IO supported.
 */
EXPORT int pyc_dpuGetInputTensorHeight(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorHeight(task,nodeName,idx);
}

/*
 * Get the width dimension of one DPU Task's input Tensor, multiple IO supported.
 */
EXPORT int pyc_dpuGetInputTensorWidth(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorWidth(task,nodeName,idx);
}

/*
 * Get the channel dimension of one DPU Task's input Tensor, multiple IO supported.
 */
EXPORT int pyc_dpuGetInputTensorChannel(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorChannel(task,nodeName,idx);
}

/*
 * Get the scale value (DPU INT8 quantization) of one DPU Task's input Tensor.
 * For multiple IO.
 */
EXPORT float pyc_dpuGetInputTensorScale(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetInputTensorScale(task,nodeName,idx);
}

/**
 * Get total number of output Tensor of DPU Task
 */
EXPORT int pyc_dpuGetOutputTensorCnt(DPUTask * task, const char * nodeName) {
    return dpuGetOutputTensorCnt(task,nodeName);
}

/**
 * @brief Get one layer's output tensor
 *
 * @note @ref
 *
 * @param kernel - the pointer to DPU kernel
 * @param layer_name - name of this layer
 *
 * @return the tensor descriptor for this layer's output
 */
EXPORT DPUTensor* pyc_dpuGetOutputTensor(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensor(task,nodeName,idx);
}

/*
 * Get the start address of one DPU Task's output Tensor, multiple IO supported.
 */
EXPORT int8_t* pyc_dpuGetOutputTensorAddress(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensorAddress(task,nodeName,idx);
}
/*
 * Get the size (in byte) of one DPU Task's output Tensor, multiple IO supported.
 */
EXPORT int pyc_dpuGetOutputTensorSize(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensorSize(task,nodeName,idx);
}

/*
 * Get the height dimension of one DPU Task's output Tensor, multiple IO supported.
 */
int pyc_dpuGetOutputTensorHeight(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensorHeight(task,nodeName,idx);
}

/*
 * Get the channel dimension of one DPU Task's output Tensor, multiple IO supported.
 */
int pyc_dpuGetOutputTensorWidth(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensorWidth(task,nodeName,idx);
}

/*
 * Get DPU Node's output tensor's channel, multiple IO supported.
 */
int pyc_dpuGetOutputTensorChannel(DPUTask *task, const char *nodeName, int idx)
{
    return dpuGetOutputTensorChannel(task,nodeName,idx);
}

/*
 * Get the scale value (DPU INT8 quantization) of one DPU Task's output Tensor.
 * For multiple IO.
 */
float pyc_dpuGetOutputTensorScale(DPUTask *task, const char *nodeName, int idx)
{
   return dpuGetOutputTensorScale(task,nodeName,idx);
}
/*
 * Get the size of tensor
 */
EXPORT int pyc_dpuGetTensorSize(DPUTensor* tensor)
{
    return dpuGetTensorSize(tensor);
}

/*
 * Get the start address (virtual) tensor
 */
EXPORT int8_t* pyc_dpuGetTensorAddress(DPUTensor* tensor)
{
    return dpuGetTensorAddress(tensor);
}

/*
 * Get the height dimension of tensor
 */
EXPORT int pyc_dpuGetTensorHeight(DPUTensor* tensor)
{
    return dpuGetTensorHeight(tensor);
}

/*
 * Get the width dimension of tensor
 */
EXPORT int pyc_dpuGetTensorWidth(DPUTensor* tensor)
{
    return dpuGetTensorWidth(tensor);
}

/*
 * Get the channel dimension of tensor
 */
EXPORT int pyc_dpuGetTensorChannel(DPUTensor* tensor)
{
    return dpuGetTensorChannel(tensor);
}

/*
 * Get the width dimension of tensor
 */
EXPORT float pyc_dpuGetTensorScale(DPUTensor* tensor)
{
    return dpuGetTensorScale(tensor);
}

/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be in in DPU Tensor order: height, width, channel;
 *       source data type must be int8_t;
 *       source data will be set without conversion
 *
 * @param task - pointer to DPU task
 * @param nodeName - Node name
 * @param buffer - pointer to source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT  int pyc_dpuSetInputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    return dpuSetInputTensorInHWCInt8(task,nodeName,buffer,size,idx);
}


/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be stored in Caffe blob order: channel, height, width;
 *       source data type must be int8_t;
 *       source data will be converted from Caffe order to DPU order
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to set input tensor
 * @param buffer - pointer to source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT int pyc_dpuSetInputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    return dpuSetInputTensorInCHWInt8(task,nodeName,buffer,size,idx);
}

/**
 * @brief Set DPU input tensor for a Node, multiple IO supported.
 *
 * @note source data must be stored in DPU Tensor order: height, width, channel
 *       source data type must be float
 *       source data will be converted from float to int_8
 *
 * @param task - pointer to DPU task
 * @param nodeName - DPU Node name to set input tensor
 * @param buffer - pointer to source data
 * @param size - size of source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT int pyc_dpuSetInputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    return dpuSetInputTensorInHWCFP32(task,nodeName,buffer,size,idx);
}

/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be stored in Caffe blob order: channel, height, width
 *       source data type must be float
 *       source data will be converted from float to int_8
 *
 * @param task - pointer to DPU task
 * @param nodeName - DPU Node name to set input tensor
 * @param buffer - pointer to source data
 * @param size - size of source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT int pyc_dpuSetInputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    return dpuSetInputTensorInCHWFP32(task,nodeName,buffer,size,idx);
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be in stored Caffe blob order: height, width， channel;
 *       target data type must be int8_t;
 *       target data will be got without conversion
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int pyc_dpuGetOutputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    return dpuGetOutputTensorInHWCInt8(task,nodeName,buffer,size,idx);
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be in stored Caffe blob order: channel, height, width;
 *       target data type must be int8_t;
 *       target data will be converted from DPU order to Caffe order
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int pyc_dpuGetOutputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    return dpuGetOutputTensorInCHWInt8(task,nodeName,buffer,size,idx);
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be stored in DPU Tensor order: height, width, channel;
 *       target data type must be float;
 *       target data will be converted from int8_t to float
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 * @param idx - tensor idx for multiple output, default as 0.
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT int pyc_dpuGetOutputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    return dpuGetOutputTensorInHWCFP32(task,nodeName,buffer,size,idx);
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be stored in Caffe bob order: channel, height, width;
 *       target data type must be float;
 *       target data will be converted from DPU order, int8_t to Caffe order, float
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int pyc_dpuGetOutputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    return dpuGetOutputTensorInCHWFP32(task,nodeName,buffer,size,idx);
}
/**
 * @brief softmax calculation
 *
 * @note length of input and output array should be num_classes x batch_size;
 *       the calculation will be performed on an acceletator if exists,
 *       otherwise the calculation will be done on CPU.
 *
 * @param input - pointer to source data(int8_t*)
 * @param output - pointer to target data(float*)
 * @param numClasses - the number of classes
 * @param batchSize - batch size of softmax calculation
 * @param scale - scale value in softmax
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int pyc_dpuRunSoftmax(int8_t *input, float *output, int numClasses, int batchSize, float scale)
{
    return dpuRunSoftmax(input,output,numClasses,batchSize,scale); 
}

/* Set or get the priority of one DPU task. Priority range is 0 to 15, 0 has the highest priority.
   The priority of the task when it was created defaults to 15. */
int pyc_dpuSetTaskPriority(DPUTask *task, uint8_t priority)
{
    return dpuSetTaskPriority(task, priority);
}
uint8_t pyc_dpuGetTaskPriority(DPUTask *task)
{
    return dpuGetTaskPriority(task);
}

/* Set or get the core mask of the task binding. Each bit represents a core.
   The default value of coreMask is 0x7 when the number of cores is 3. */
int pyc_dpuSetTaskAffinity(DPUTask *task, uint32_t coreMask)
{
    return dpuSetTaskAffinity(task, coreMask);
}
uint32_t pyc_dpuGetTaskAffinity(DPUTask *task)
{
    return dpuGetTaskAffinity(task);
}

/* Get the total size in memory of all the inputs of the network,
   which involve the interleaves between two inputs. */
int pyc_dpuGetInputTotalSize(DPUTask *task)
{
    return dpuGetInputTotalSize(task);
}

/* Get the total size in memory of all the outputs of the network,
   which involve the interleaves between two outputs. */
int pyc_dpuGetOutputTotalSize(DPUTask *task)
{
    return dpuGetOutputTotalSize(task);
}

DPUTensor *pyc_dpuGetBoundaryIOTensor(DPUTask *task, const char *tensorName)
{
    return dpuGetBoundaryIOTensor(task, tensorName);
}

#ifdef __cplusplus
}
#endif
