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

#ifndef _ERR_H_
#define _ERR_H_
#ifdef __cplusplus
extern "C" {
#endif

/* error code */
#define N2CUBE_SUCCESS             (0)
#define N2CUBE_FAILURE             (-1)

#define ERR_TIMEOUT                (-2)

#define N2CUBE_OK                       0
#define N2CUBE_ERR_PARAM_NULL           -1
#define N2CUBE_ERR_PARAM_VALUE          -2

#define ERR                         -101
#define ERR_NAME                    -102
#define ERR_MEM_SIZE                -103
#define ERR_PTR_NULL                -104

#define ERR_ELF_NO_FILE             -105
#define ERR_ELF_INVALID             -106
#define ERR_ELF_INVALID_EXECUTABLE  -107
#define ERR_ELF_INVALID_SECTION     -108
#define ERR_ELF_NO_SHSTRTAB         -109
#define ERR_ELF_INVALID_SYMBOL      -110
#define ERR_ELF_UNMAPPED            -111
#define ERR_ELF_INVALID_SYM_NAME    -112

#define ERR_LD_INVALID_KERNEL       -113
#define ERR_LD_INVALID_SYSC_CALL    -114

#define ERR_OPNE_DPU_DEV            -115
#define ERR_INVALID_INST            -116
#define ERR_INVALID_LAYER_NAME      -117
#define ERR_INVALID_PROF            -118
#define ERR_INVALID_TASK            -119
#define ERR_LOCATE_HYBRID_ELF       -200
#define ERR_LOCATE_LIBRARY_DPU_KERNEL -201
#define ERR_COMILATION_MODE_VALUE    -202
#define ERR_INVALID_HYBRID_ELFFILE   -203
#define ERR_MSB_FORMAT_NOT_SUPPORT   -204
#define ERR_ELF_READ_CONFIGURABLE    -205

#define N2CUBE_ERR_DPU_NONE             -1000
#define N2CUBE_ERR_DPU_TARGET           -1001
#define N2CUBE_ERR_DPU_ARCH             -1002
#define N2CUBE_ERR_DPU_TARGET_MISMATCH  -1003
#define N2CUBE_ERR_DPU_ARCH_MISMATCH    -1004
#define N2CUBE_ERR_DPU_CONFIG_MISMATCH  -1005

#define N2CUBE_ERR_TENSOR_SIZE          -2000
#define N2CUBE_ERR_TENSOR_INPUT_INDEX   -2001
#define N2CUBE_ERR_TENSOR_OUTPUT_INDEX  -2002
#define N2CUBE_ERR_TENSOR_INPUT_SHAPE   -2003
#define N2CUBE_ERR_TENSOR_INPUT_CHANNEL -2004
#define N2CUBE_ERR_TENSOR_NAME          -2005

#define N2CUBE_ERR_KERNEL_LOAD             -3000
#define N2CUBE_ERR_KERNEL_MEAN_VALUE       -3001
#define N2CUBE_ERR_KERNEL_LOAD_SECTION     -3002
#define N2CUBE_ERR_KERNEL_ADDR_CODE        -3003
#define N2CUBE_ERR_KERNEL_ADDR_BIAS        -3004
#define N2CUBE_ERR_KERNEL_ADDR_WEIGHTS     -3005

#define N2CUBE_ERR_DPU_DRIVER_VERSION_NONE -4000
#define N2CUBE_ERR_DPU_DRIVER_MISMATCH     -4001

#define N2CUBE_ERR_ABI_VERSION              -5000
#define N2CUBE_ERR_ABI_SYMBOL_CODE          -5001
#define N2CUBE_ERR_ABI_SYMBOL_BIAS          -5002
#define N2CUBE_ERR_ABI_SYMBOL_WEIGHTS       -5003
#define N2CUBE_ERR_ABI_CODE_SEGMENT_SIZE    -5004
#define N2CUBE_ERR_ABI_BIAS_SEGMENT_SIZE    -5005
#define N2CUBE_ERR_API_WEIGHTS_SEGMENT_SIZE -5006
#define N2CUBE_ERR_ABI_CODE_SEGMENT_COUNT   -5007

#define N2CUBE_ERR_MALLOC_DPU_CAPABILITY   -6000

#define N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT 0
#define N2CUBE_EXCEPTION_MODE_RET_ERR_CODE 1

struct error_message_t {
    int error_code;
    const char *error_message;
};
extern const struct error_message_t gN2cubeErrorMessage[];
extern const char N2CUBE_NO_MSG[];
extern const char N2CUBE_MSG_OK[];
extern const int error_message_count;
const char *dpuGetExceptionMessage(int error_code);

#ifdef __cplusplus
}
#endif
#endif
