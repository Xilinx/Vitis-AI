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

#ifndef _DDUMP_H_
#define _DDUMP_H_
#include "defs.h"
#include "../../common/dpu_node_v2.h"

/* error code */
#define DDUMP_SUCCESS             (0)
#define DDUMP_FAILURE             (-1)

#define DDUMP_MAX_STR_LEN (20)

#define DDUMP_SINGLE_LINE "----------------------------------------------------------------"
#define DDUMP_DUMP_HEADER "[DDump] "
#define DDUMP_ERROR_PRINT(s...) \
    printf(DDUMP_DUMP_HEADER); \
    printf("Error : "); \
    printf(s); \
    printf("\n");

#define BITS_PER_MB (1024 * 1024)
#define MIN_BITS_PER_MB (1024 * 1024 / 128)
#define BITS_PER_KB (1024)
#define MIN_BITS_PER_KB (1024 / 128)
#define NUMBER_PER_MOPS (1000 * 1000)
typedef enum {
    DDump_None = 0,
    DDump_All = 0xff,
    DDump_DpuV = 0x2,
    DDump_Dnnc = 0x4,
    DDump_Klist = 0x8,
    DDump_Version = 0x10000,
    DDump_File = 0x20000,
    DDump_Help = 0x40000,
    DDump_Senior = 0x80000,
    DDump_Model = 0x100000,
} DDump_Opt_t;

typedef struct DDump_kernel DDump_kernel_t;
struct DDump_kernel {
    kernel_t base;
    DDump_kernel_t *next;
};

typedef struct {
    DDump_kernel_t *list;
    int kernel_num;
} DDump_kernels_t;

/**
 * assert function for internal bug/error verifying
 */
#define DDUMP_ASSERT(condition)                                                        \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            fprintf(stderr, "Xilinx DDump tool internal error.\n");                    \
            fprintf(stderr, "Please contact Xilinx with the following info:\n");  \
            fprintf(stderr, "\tDebug info - Cond:\"%s\", File:%s, Function:%s, Line:%d.\n",\
                #condition, __FILE__, __func__, __LINE__);                                     \
            exit(-1);                                                                    \
        }                                                                                      \
    } while (0)

#endif
