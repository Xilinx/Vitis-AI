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
/**
 * dpu_elf.c
 *
 * Read various sections for DPU kernel from hybrid CPU-DPU ELF executable
 */
#ifdef __cplusplus
extern "C" {
#endif
#include "err.h"

const char* g_dpu_target_name[] = {
    "Unknown",
    "v1.1.3",
    "v1.3.0",
    "v1.3.1",
    "v1.3.2",
    "v1.3.3",
    "v1.3.4",
    "v1.3.5",
    "v1.4.0",
    "v1.4.1",
    "v1.4.2",
    "v1.3.6",
    "v1.3.7"
};

const char* g_dpu_arch_name[] = {
    "Unknown",
    "B1024F",
    "B1152F",
    "B4096F",
    "B256F",
    "B512F",
    "B800F",
    "B1600F",
    "B2048F",
    "B2304F",
    "B8192F",
    "B3136F",
    "B288F",
    "B144F",
    "B5184F"
};

const char N2CUBE_NO_MSG[] = "";
const char N2CUBE_MSG_OK[] = "Success";

const char COMMON_MSG_ERR_PARAM_NULL[] = "The parameter is null";
const char COMMON_MSG_ERR_PARAM_VALUE[] = "The value of parameter is wrong";

const char COMMON_MSG_ERR_TENSOR_SIZE[] = "The size of tensor is wrong";
const char COMMON_MSG_ERR_TENSOR_INPUT_INDEX[] = "Invalid DPU node input index";
const char COMMON_MSG_ERR_TENSOR_OUTPUT_INDEX[] = "Invalid DPU node output index";
const char COMMON_MSG_ERR_TENSOR_INPUT_SHAPE[] = "Invalid input image shape for DPU node";
const char COMMON_MSG_ERR_TENSOR_INPUT_CHANNEL[] = "The channel number of the imput image is incorrect";

const char COMMON_MSG_ERR_DPU_NONE[] = "No DPU core found in current configuration of DPU IP";
const char COMMON_MSG_ERR_DPU_TARGET[] = "Unknown target found in DPU IP";
const char COMMON_MSG_ERR_DPU_ARCH[] = "Unknown arch found in DPU IP";
const char COMMON_MSG_ERR_DPU_TARGET_MISMATCH[] = "DPU target version mismatch for kernel";
const char COMMON_MSG_ERR_DPU_ARCH_MISMATCH[] = "DPU arch version mismatch for dpu kernel";

const char COMMON_MSG_ERR_DPU_DRIVER_VERSION_NONE[] = "No version info in DPU Driver version file";
const char COMMON_MSG_ERR_DPU_DRIVER_MISMATCH[] = "DPU driver version mismatch with N2Cube";

const char COMMON_MSG_ERR_KERNEL_LOAD[] = "Fail to load DPU kernel";
const char COMMON_MSG_ERR_KERNEL_MEAN_VALUE[] = "Invalid mean value for DPU kernel";
const char COMMON_MSG_ERR_KERNEL_LOAD_SECTION[] = "Faild to load section from hybrid DPU file";
const char COMMON_MSG_ERR_KERNEL_ADDR_CODE[] = "Fail to verify DPU address of CODE symbol";
const char COMMON_MSG_ERR_KERNEL_ADDR_BIAS[] = "Fail to verify DPU address of BIAS";
const char COMMON_MSG_ERR_KERNEL_ADDR_WEIGHTS[] = "Fail to verify DPU address of WEIGHTS";

const char COMMON_MSG_ERR_ABI_VERSION[] = "Unsupport ABI version";
const char COMMON_MSG_ERR_ABI_SYMBOL_CODE[] = "Fail to locate DPU CODE symbol";
const char COMMON_MSG_ERR_ABI_SYMBOL_BIAS[] = "Fail to locate DPU BIAS symbol";
const char COMMON_MSG_ERR_ABI_SYMBOL_WEIGHTS[] = "Fail to locate DPU WEIGHTS symbol";
const char COMMON_MSG_ERR_ABI_CODE_SEGMENT_SIZE[] = "Invalid size of DPU CODE segment";
const char COMMON_MSG_ERR_ABI_BIAS_SEGMENT_SIZE[] = "Invalid size of DPU BIAS segment";
const char COMMON_MSG_ERR_API_WEIGHTS_SEGMENT_SIZE[] = "Invalid size of DPU WEIGHT segment";
const char COMMON_MSG_ERR_ABI_CODE_SEGMENT_COUNT[] = "Invalid count of DPU CODE segment";

const char COMMON_MSG_ERR_MALLOC_DPU_CAPABILITY[] = "Malloc fails when allocating memory space for data structure of DPU capability";
const char COMMON_MSG_ERR_LOCATE_HYBRID_ELF[] = "Fail to locate hybrid ELF";
const char COMMON_MSG_ERR_LOCATE_LIBRARY_DPU_KERNEL[] = "Fail to load DPU kernel";
const char COMMON_MSG_ERR_COMILATION_MODE_VALUE[] = "Compilation mode value %d NOT supported now";
const char COMMON_MSG_ERR_INVALID_HYBRID_ELFFILE[] = "Invalid Hybrid ELF file";
const char COMMON_MSG_ERR_MSB_FORMAT_NOT_SUPPORT[] = "MSB format ELF executable is not supported";
const char COMMON_MSG_ERR_ELF_READ_CONFIGURABLE[] = "ELF read configurable error";
const char COMMON_MSG_ERR_ELF_NO_FILE[] = "no elf file";
const char COMMON_MSG_N2CUBE_ERR_DPU_CONFIG_MISMATCH[] = "DPU configuration mismatch";

const struct error_message_t gN2cubeErrorMessage[] = {
    {N2CUBE_OK, N2CUBE_MSG_OK},
    {N2CUBE_ERR_PARAM_NULL, COMMON_MSG_ERR_PARAM_NULL},
    {N2CUBE_ERR_PARAM_VALUE, COMMON_MSG_ERR_PARAM_VALUE},
    {N2CUBE_ERR_TENSOR_SIZE, COMMON_MSG_ERR_TENSOR_SIZE},
    {N2CUBE_ERR_KERNEL_LOAD, COMMON_MSG_ERR_KERNEL_LOAD},
    {N2CUBE_ERR_DPU_DRIVER_VERSION_NONE, COMMON_MSG_ERR_DPU_DRIVER_VERSION_NONE},
    {N2CUBE_ERR_DPU_DRIVER_MISMATCH, COMMON_MSG_ERR_DPU_DRIVER_MISMATCH},
    {N2CUBE_ERR_TENSOR_INPUT_SHAPE, COMMON_MSG_ERR_TENSOR_INPUT_SHAPE},
    {N2CUBE_ERR_TENSOR_INPUT_CHANNEL, COMMON_MSG_ERR_TENSOR_INPUT_CHANNEL},
    {N2CUBE_ERR_KERNEL_MEAN_VALUE, COMMON_MSG_ERR_KERNEL_MEAN_VALUE},
    {N2CUBE_ERR_DPU_NONE, COMMON_MSG_ERR_DPU_NONE},
    {N2CUBE_ERR_DPU_TARGET, COMMON_MSG_ERR_DPU_TARGET},
    {N2CUBE_ERR_DPU_TARGET_MISMATCH, COMMON_MSG_ERR_DPU_TARGET_MISMATCH},
    {N2CUBE_ERR_DPU_ARCH, COMMON_MSG_ERR_DPU_ARCH},
    {N2CUBE_ERR_MALLOC_DPU_CAPABILITY, COMMON_MSG_ERR_MALLOC_DPU_CAPABILITY},
    {N2CUBE_ERR_TENSOR_INPUT_INDEX, COMMON_MSG_ERR_TENSOR_INPUT_INDEX},
    {N2CUBE_ERR_TENSOR_OUTPUT_INDEX, COMMON_MSG_ERR_TENSOR_OUTPUT_INDEX},
    {N2CUBE_ERR_DPU_ARCH_MISMATCH, COMMON_MSG_ERR_DPU_ARCH_MISMATCH},
    {N2CUBE_ERR_ABI_SYMBOL_CODE, COMMON_MSG_ERR_ABI_SYMBOL_CODE},
    {N2CUBE_ERR_ABI_CODE_SEGMENT_SIZE, COMMON_MSG_ERR_ABI_CODE_SEGMENT_SIZE},
    {N2CUBE_ERR_ABI_SYMBOL_BIAS, COMMON_MSG_ERR_ABI_SYMBOL_BIAS},
    {N2CUBE_ERR_ABI_BIAS_SEGMENT_SIZE, COMMON_MSG_ERR_ABI_BIAS_SEGMENT_SIZE},
    {N2CUBE_ERR_ABI_SYMBOL_WEIGHTS, COMMON_MSG_ERR_ABI_SYMBOL_WEIGHTS},
    {N2CUBE_ERR_API_WEIGHTS_SEGMENT_SIZE, COMMON_MSG_ERR_API_WEIGHTS_SEGMENT_SIZE},
    {N2CUBE_ERR_KERNEL_ADDR_CODE, COMMON_MSG_ERR_KERNEL_ADDR_CODE},
    {N2CUBE_ERR_KERNEL_ADDR_BIAS, COMMON_MSG_ERR_KERNEL_ADDR_BIAS},
    {N2CUBE_ERR_KERNEL_ADDR_WEIGHTS, COMMON_MSG_ERR_KERNEL_ADDR_WEIGHTS},
    {N2CUBE_ERR_ABI_VERSION, COMMON_MSG_ERR_ABI_VERSION},
    {N2CUBE_ERR_ABI_CODE_SEGMENT_COUNT, COMMON_MSG_ERR_ABI_CODE_SEGMENT_COUNT},
    {N2CUBE_ERR_KERNEL_LOAD_SECTION, COMMON_MSG_ERR_KERNEL_LOAD_SECTION},
    {ERR_LOCATE_HYBRID_ELF, COMMON_MSG_ERR_LOCATE_HYBRID_ELF},
    {ERR_LOCATE_LIBRARY_DPU_KERNEL, COMMON_MSG_ERR_LOCATE_LIBRARY_DPU_KERNEL},
    {ERR_COMILATION_MODE_VALUE, COMMON_MSG_ERR_COMILATION_MODE_VALUE},
    {ERR_INVALID_HYBRID_ELFFILE, COMMON_MSG_ERR_INVALID_HYBRID_ELFFILE},
    {ERR_MSB_FORMAT_NOT_SUPPORT, COMMON_MSG_ERR_MSB_FORMAT_NOT_SUPPORT},
    {ERR_ELF_READ_CONFIGURABLE, COMMON_MSG_ERR_ELF_READ_CONFIGURABLE},
    {ERR_ELF_NO_FILE, COMMON_MSG_ERR_ELF_NO_FILE},
    {N2CUBE_ERR_DPU_CONFIG_MISMATCH, COMMON_MSG_N2CUBE_ERR_DPU_CONFIG_MISMATCH}
};
const int error_message_count = ((sizeof(gN2cubeErrorMessage) / sizeof(struct error_message_t)));

const char *dpuGetExceptionMessage(int error_code)
{
    int i;

    for (i = 0; i < error_message_count; i++)
    {
        if (gN2cubeErrorMessage[i].error_code == error_code)
        {
            return gN2cubeErrorMessage[i].error_message;
        }
    }
    return N2CUBE_NO_MSG;
}

#ifdef __cplusplus
}
#endif
