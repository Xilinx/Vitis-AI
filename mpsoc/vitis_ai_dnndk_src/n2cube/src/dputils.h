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

#ifndef _DPUTILS_H_
#define _DPUTILS_H_

#include <opencv2/opencv.hpp>

struct  dpu_task;
typedef struct dpu_task   DPUTask;


/* Set image into DPU Task's input Tensor */
int dpuSetInputImage(DPUTask *task, const char *nodeName,
    const cv::Mat &image, float *mean, int idx = 0);

/* Set image into DPU Task's input Tensor with a specified scale parameter */
int dpuSetInputImageWithScale(DPUTask *task, const char *nodeName,
    const cv::Mat &image, float *mean, float scale, int idx = 0);

/* Set image into DPU Task's input Tensor (mean values automatically processed by N2Cube) */
int dpuSetInputImage2(DPUTask *task, const char *nodeName, const cv::Mat &image, int idx = 0);

#endif
