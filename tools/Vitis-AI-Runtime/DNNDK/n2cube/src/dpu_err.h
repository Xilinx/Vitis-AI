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

#ifndef _DPU_ERR_H
#define _DPU_ERR_H

#include <stdio.h>
#include <stdlib.h>
#include "dpu_def.h"
#include "../../common/dpu_types.h"
#include "../../common/err.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int dpuSetExceptionMode(int mode);
extern int dpuGetExceptionMode();
extern const char *dpuGetExceptionMessage(int error_code);

/**
 * Prefix used for tagging printed log output
 */
#define DPU_MSG_HEADER      "[DNNDK] "


static const char * DPU_MSG_ARRAY[MAX_NAME_LEN] = {
    /* ERR_OPEN_DPU_DEV */  "error to open DPU device file",
    /* ERR_OPEN_DPU_DEV */  "error to open DPU device file"
};

#define DNNRT_GET_ERR_MSG(error_id)    DPU_MSG_ARRAY[error_id]

/**
 * N2Cube assert function for internal bug/error verifying
 */
#define DPU_ASSERT(condition, error_id)                                                        \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            fprintf(stderr, "Xilinx DPU Runtime system internal error.\n");                    \
            fprintf(stderr, "Please contact Xilinx with the following info:\n");  \
            fprintf(stderr, "\tDebug info - Cond:\"%s\", File:%s, Function:%s, Line:%d.\n",\
                #condition, __FILE__, __func__, __LINE__);                                     \
            exit(error_id);                                                                    \
        }                                                                                      \
    } while (0)

/**
 * For failure condition check, log error message with varied arguments support
 */
#define N2CUBE_DPU_CHECK(condition, errorcode, format, ...)                              \
    do {                                                                      \
        if (!(condition)) {                                                   \
            if (N2CUBE_EXCEPTION_MODE_RET_ERR_CODE == dpuGetExceptionMode()) {            \
                return (errorcode);                                             \
            } else {                                                          \
                fprintf(stderr, DPU_MSG_HEADER "%s" format "\n",              \
                    dpuGetExceptionMessage(errorcode), ##__VA_ARGS__ );       \
                exit(-1);                                                     \
            }                                                                 \
        }                                                                     \
    } while (0)
#define N2CUBE_DPU_CHECK_AND_RET_NULL(condition, errorcode, format, ...)                              \
    do {                                                                      \
        if (!(condition)) {                                                   \
            if (N2CUBE_EXCEPTION_MODE_RET_ERR_CODE == dpuGetExceptionMode()) {            \
                return NULL;                                                  \
            } else {                                                          \
                fprintf(stderr, DPU_MSG_HEADER "%s" format "\n",              \
                    dpuGetExceptionMessage(errorcode), ##__VA_ARGS__ );      \
                exit(-1);                                                     \
            }                                                                 \
        }                                                                     \
    } while (0)

#define N2CUBE_PARAM_CHECK_AND_RET(param, ret)                                                                  \
    do {                                                                                           \
        if(!(param)) {                                                                             \
            if (N2CUBE_EXCEPTION_MODE_RET_ERR_CODE == dpuGetExceptionMode()) {                     \
                return ret;                                                                        \
            } else {                                                                               \
                fprintf(stderr, DPU_MSG_HEADER "Parameter %s is invalid for function %s.\n", #param, __func__);             \
                exit(-1);                                                                          \
            }                                                                                      \
        }                                                                                          \
    } while(0)

#define N2CUBE_PARAM_CHECK(param) N2CUBE_PARAM_CHECK_AND_RET(param,N2CUBE_ERR_PARAM_NULL)

/**
 * Log messages with varied arguments support
 */
#define DPU_LOG_MSG(format, ...)                                          \
    do {                                                                  \
        printf(DPU_MSG_HEADER format "\n", ##__VA_ARGS__ );               \
    } while (0)


#define DPU_FAIL_ON_ID(err_id)                                            \
    do {                                                                  \
        fprintf(stderr, DPU_MSG_HEADER DNNRT_GET_ERR_MSG(err_id) "\n");   \
        exit(error_id);                                                   \
    } while (0)


/**
 * Log failure messages with varied arguments support and exit
 */
#define DPU_FAIL_ON_MSG(format, ...)                                                      \
    do {                                                                                  \
        fprintf(stderr, DPU_MSG_HEADER format "\n", ##__VA_ARGS__ );                      \
        if (DPU_DEBUG_MSG()) {                                                            \
            fprintf(stderr, "\tDebug info - File:%s, Function:%s, Line:%d.\n",            \
                __FILE__, __func__, __LINE__);                                            \
        }                                                                                 \
        exit(-1);                                                                         \
    } while (0)


#define DPU_API_VER_CHECK(ver, ret)                                                                \
    do {                                                                                           \
        if(ver <= DPU_ABI_V1_0) {                                                                  \
            if (N2CUBE_EXCEPTION_MODE_RET_ERR_CODE == dpuGetExceptionMode()) {                     \
                return ret;                                                                        \
            } else {                                                                               \
                DPU_LOG_MSG("Multiply IO not supported for API %s for this ABI version.", __func__);   \
                DPU_LOG_MSG("Please update ABI to the version above v1.0.");                       \
                exit(-1);                                                                          \
            }                                                                                      \
        }                                                                                          \
    } while(0)


#define DPU_BUG_ON(expr, err_id)    DPU_ASSERT(!(expr), err_id)

#define CHECK_SEG_SIZE(size, log_name, kernel, elf)                                            \
    do {                                                                                       \
        if (!(size)) {                                                                         \
            elf_free(elf);                                                                     \
            N2CUBE_DPU_CHECK(0, N2CUBE_ERR_KERNEL_LOAD_SECTION,                                \
                    ". section:%s, hybrid ELF:%s\n"                                            \
                    "    1- Specified DPU Kernel name \"%s\" is right\n"                       \
                    "    2- DPU Kernel \"%s\" is compiled and linked into \"%s\" as expected", \
                    (log_name), kernel->base.elf_name, kernel->base.name,                      \
                    kernel->base.name, kernel->base.elf_name);                                 \
            }                                                                                  \
        } while(0)

#define DPU_CONFIG_CHECK_FIELD(elf_field, dpu_field, logstr)                                     \
    do{                                                                                          \
        N2CUBE_DPU_CHECK((elf_field) == (dpu_field),                                             \
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,                                             \
                     " for kernel %s - parameter: " logstr ", DPU kernel: %d, DPU IP: %d.",      \
                     kernel->base.name,                                                          \
                     (elf_field),                                                                \
                     (dpu_field));                                                               \
    } while(0)

#define DPU_CONFIG_CHECK_ARCH(elf_field, dpu_field, logstr)                                      \
    do{                                                                                          \
        N2CUBE_DPU_CHECK((elf_field) == (dpu_field),                                             \
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,                                             \
                     " for kernel %s - parameter: " logstr ", DPU kernel: B%d, DPU IP: B%d.",    \
                     kernel->base.name,                                                          \
                     (elf_field),                                                                \
                     (dpu_field));                                                               \
    } while(0)

#define DPU_CONFIG_CHECK_FIELD_ENABLE(elf_field, dpu_field, logstr)                              \
    do{                                                                                          \
        N2CUBE_DPU_CHECK((dpu_field) || ((elf_field) == (dpu_field)),                            \
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,                                             \
                     " for kernel %s - parameter: " logstr ", DPU kernel: %s, DPU IP: %s.",      \
                     kernel->base.name,                                                          \
                     ((elf_field) ? "Enabled" : "Disabled"),                                     \
                     ((dpu_field) ? "Enabled" : "Disabled"));                                    \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif // end of _DPU_ERR_H_
