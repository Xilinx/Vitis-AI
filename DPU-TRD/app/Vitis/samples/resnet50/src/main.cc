/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and

-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
#include <assert.h>
#include <dirent.h>
#include <dnndk/dnndk.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include "main.h"
#include "dputils.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

/* DPU Kernel Name for ResNet50 CONV & FC layers */
#define KRENEL_CONV "resnet50"

#define CONV_INPUT_NODE "conv1"
#define CONV_OUTPUT_NODE "res5c_branch2c"
#define FC_INPUT_NODE "fc1000"
#define FC_OUTPUT_NODE "fc1000"

#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                          \
  {                                                                       \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "us" << endl;                 \
  }
#else
#define _T(func) func;
#endif

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(std::string const &path, std::vector<std::string> &images) {
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);

  if (!S_ISDIR(s.st_mode)) {
    images.push_back(path);
    return;
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "PNG") || (ext == "png")) {
        images.push_back(path + "/" + name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief softmax operation
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i]);
    sum += result[i];
  }

  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vwords - vector of words
 *
 * @return none
 */
void TopK(const float *d, int size, int k, const string vkind[]) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }
  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    printf("TimeTest[%d]  = %-8f  name = %s\n", i, d[ki.second], vkind[ki.second].c_str());
    q.pop();
  }
}

/**
 * @brief Compute average pooling on CPU
 *
 * @param conv - pointer to ResNet50 CONV Task
 * @param fc - pointer to ResNet50 FC Task
 *
 * @return none
 */
void CPUCalcAvgPool(DPUTask *conv, DPUTask *fc) {
  assert(conv && fc);

  /* Get output Tensor to the last Node of ResNet50 CONV Task */
  DPUTensor *outTensor = dpuGetOutputTensor(conv, CONV_OUTPUT_NODE);
  /* Get size, height, width and channel of the output Tensor */
  int tensorSize = dpuGetTensorSize(outTensor);
  int outHeight = dpuGetTensorHeight(outTensor);
  int outWidth = dpuGetTensorWidth(outTensor);
  int outChannel = dpuGetTensorChannel(outTensor);

  /* allocate memory buffer */
  float *outBuffer = new float[tensorSize];

  /* Get the input address to the first Node of FC Task */
  int8_t *fcInput = dpuGetInputTensorAddress(fc, FC_INPUT_NODE);

  /* Copy the last Node's output and convert them from IN8 to FP32 format */
  dpuGetOutputTensorInHWCFP32(conv, CONV_OUTPUT_NODE, outBuffer, tensorSize);

  /* Get scale value for the first input Node of FC task */
  float scaleFC = dpuGetInputTensorScale(fc, FC_INPUT_NODE);
  int length = outHeight * outWidth;
  float avg = (float)(length * 1.0f);

  float sum;
  for (int i = 0; i < outChannel; i++) {
    sum = 0.0f;
    for (int j = 0; j < length; j++) {
      sum += outBuffer[outChannel * j + i];
    }
    /* compute average and set into the first input Node of FC Task */
    fcInput[i] = (int8_t)(sum / avg * scaleFC);
  }

  delete[] outBuffer;
}

/**
 * @brief Run CONV Task and FC Task for ResNet50
 *
 * @param taskConv - pointer to ResNet50 CONV Task
 *
 * @return none
 */
void runResnet50(string imgpath, DPUTask *taskConv) {
  assert(taskConv);

  /* Get channel count of the output Tensor for FC Task  */
  int channel = dpuGetOutputTensorChannel(taskConv, FC_OUTPUT_NODE);
  float *softmax = new float[channel];
  float *FCResult = new float[channel];

  // cout << "Run resnet50" << endl;
  /* Load image and Set image into CONV Task with mean value */
  vector<string> images;
  ListImages(imgpath, images);
  for (auto &imgpath : images) {
    Mat img = imread(imgpath);
    cout << "Input Image size: " << img.cols << " x " << img.rows << " x " << img.channels()
         << endl;
    cout << "Input featuremap: " << dpuGetInputTensorWidth(taskConv, CONV_INPUT_NODE) << " x "
         << dpuGetInputTensorHeight(taskConv, CONV_INPUT_NODE) << endl;
    _T(dpuSetInputImage2(taskConv, CONV_INPUT_NODE, img));

    _T(dpuRunTask(taskConv));

    /* Get FC result and convert from INT8 to FP32 format */
    _T(dpuGetOutputTensorInHWCFP32(taskConv, FC_OUTPUT_NODE, FCResult, channel));
    /* Calculate softmax on CPU and show TOP5 classification result */
    _T(CPUCalcSoftmax(FCResult, channel, softmax));
    _T(TopK(softmax, channel, 1, words));
  }
  delete[] softmax;
  delete[] FCResult;
}

/**
 * @brief Entry for running ResNet50 neural network
 *
 * @note Neural network ResNet50 is divied into two seperate DPU
 *       Kernels: CONV and FC.
 *
 */
int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage of resnet50 demo: ./resnet50 file_name" << endl;
    cout << "\tfile_name: path to your image file for classfication" << endl;
    return -1;
  }

  DPUKernel *kernelConv;
  DPUTask *taskConv;

  dpuOpen();
  kernelConv = dpuLoadKernel(KRENEL_CONV);
  taskConv = dpuCreateTask(kernelConv, 0);

  runResnet50(argv[1], taskConv);

  dpuDestroyTask(taskConv);
  dpuDestroyKernel(kernelConv);
  dpuClose();

  return 0;
}
