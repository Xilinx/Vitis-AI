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

#ifndef _N2CUBE_H_
#define _N2CUBE_H_

#ifdef __cplusplus
extern "C" {
#endif

int dpuLoadMean(DPUTask *task, float *mean, int size);

//int paraCheck(DPUTask * task, const char* nodeName, float *mean, int idx);

int inputMeanValueCheck(float* mean);

int pyc_dpuSetInputData(DPUTask *task, const char* nodeName, unsigned char* resized_data, int height, int width, int channel, float *mean, float scale, int idx);


#ifdef __cplusplus
}
#endif
#endif
