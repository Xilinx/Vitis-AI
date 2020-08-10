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
#include "dputils_pyc.h"
#ifdef USE_ARM_NEON
#include "neonopt.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#include "dpu_err.h"
#ifdef __cplusplus
}
#endif

using namespace std;
#ifdef __cplusplus
extern "C" {
#endif
EXPORT int loadMean(DPUTask *task, float *mean, int size) {
    return dpuGetKernelMean(task,mean,size);
}

//EXPORT int paraCheck(DPUTask * task, const char* nodeName, float *mean, int idx) {
//
//    N2CUBE_PARAM_CHECK(task);
//    N2CUBE_PARAM_CHECK(nodeName);
//    N2CUBE_PARAM_CHECK(mean);
//
//    if(idx > 0) {
//        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
//
//    }
//    return 0;
//}

EXPORT int pyc_dpuSetInputData(DPUTask *task, const char* nodeName, unsigned char* resized_data, int height, int width, int imageChannel,  float *mean, float scale, int idx) {
    float scaleFix;
    int8_t *inputAddr;
    int value, modelChannel;

    modelChannel = dpuGetInputTensorChannel(task, nodeName, idx);
    inputAddr = dpuGetInputTensorAddress(task, nodeName, idx);
    scaleFix = dpuGetInputTensorScale(task, nodeName, idx);

    N2CUBE_DPU_CHECK((imageChannel == 1) || (imageChannel == 3), N2CUBE_ERR_PARAM_VALUE,
        " for API %s. nodeName: %s", __func__, nodeName);
    N2CUBE_DPU_CHECK(imageChannel == modelChannel, N2CUBE_ERR_TENSOR_INPUT_CHANNEL,
        " for API %s. nodeName: %s", __func__, nodeName);
    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }
    scaleFix = scaleFix*scale;

    if (imageChannel == 1) {
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
                for (int idx_c=0; idx_c<modelChannel; idx_c++) {
                    value = *(resized_data+idx_h*width*modelChannel+idx_w*modelChannel+idx_c);
                    value = (int)((value - *(mean+idx_c)) * scaleFix);
                    inputAddr[idx_h*width+idx_w] = (char)value;
                }
            }
        }
    } else {
#ifdef USE_ARM_NEON
        dpuProcessNormalizion(inputAddr, resized_data, height, width, mean, scaleFix, width* imageChannel);
#else
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
                for (int idx_c=0; idx_c<3; idx_c++) {
		    value = (int)((resized_data[idx_h*width*3+idx_w*3+idx_c] - mean[idx_c])*scaleFix);
                    /* Invalid pixel values checking for input feature map */
                    if ((value>127) || (value<-128)) {
                        DPU_LOG_MSG("Invalid pixel value of input tensor: %d", value);
                        DPU_FAIL_ON_MSG("Please check if decent tool produces correct quantization info.");
                    };

                    inputAddr[idx_h*width*3+idx_w*3+idx_c] = (char)value;
                }
            }
        }
#endif
    }
    return N2CUBE_SUCCESS;
}
#ifdef __cplusplus
}
#endif
