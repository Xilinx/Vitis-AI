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

#ifndef _DPU_DEF_H_
#define _DPU_DEF_H_
#ifdef __cplusplus
extern "C" {
#endif
#include "dpu_task.h"

struct elf;
typedef struct elf elf_t;

#define EXPORT

#ifndef UNIT_TEST
#define INTERNAL           static
#else
#define INTERNAL
#endif

#define MEM_SIZE_PROF      (1024*1024)      /* DPU memory size for profiler - 1M */
#define MAX_MEM_SIZE       (500*1024*1024)  /* maximum DPU memory size - 500M */
#define UNIT_1M            ((1024*1024)*1.0f)           /* 1M */
#define UNIT_1G            ((1024*1024*1024)*1.0f)      /* 1G */
#define BIT_WIDTH_32       (32/8)
#define BIT_WIDTH_128      (128/8)

#ifdef MEM_WORK_AROUND
#define memset(start, value, size)       dpuMemset(start, value, size)
#define memcpy(dest, src, n)             dpuMemcpy(dest, src, n)
#define fread(ptr, size, nmemb, stream)  dpuFread(ptr, size, nmemb, stream)
#define fwrite(ptr, size, nmemb, stream) dpuWrite(ptr, size, nmemb, stream)
#endif

/* DPU Kernel Mode definitions */
#define K_MODE_NORMAL      (1<<0)
#define K_MODE_DEBUG       (1<<1)

#define FLAG_DEBUG_DRV      (1 << 0)  /* debug system call to DPU driver */
#define FLAG_DEBUG_LD       (1 << 1)  /* log message while loading kernel */
#define FLAG_DEBUG_LOG      (1 << 2)
#define FLAG_DEBUG_RELO     (1 << 3)
#define FLAG_DEBUG_RUN      (1 << 4)
#define FLAG_DEBUG_UNUSED1  (1 << 5)
#define FLAG_DEBUG_ELF      (1 << 6)
#define FLAG_DEBUG_MSG      (1 << 7)
#define FLAG_DEBUG_NO_CHECK (1 << 8)

#define MODE_RUNTIME_NORMAL  (1 << 8)
#define MODE_RUNTIME_DEBUG   (1 << 9)
#define MODE_RUNTIME_PROFILE    (1 << 10)
#define MODE_RUNTIME_DUMP    (1 << 11)

#define FLAG_DEBUG_ALL     (0xFFFFFFFF)

#define STR_RUNTIME_MODE_NORMAL     "normal"
#define STR_RUNTIME_MODE_DEBUG      "debug"
#define STR_RUNTIME_MODE_PROFILE    "profile"
#define STR_RUNTIME_MODE_DUMP       "dump"

#define DPU_DEV_NAME       "/dev/dpu"
#define DPU_ENV_DEBUG      "DPUDEBUG_"
#define DPU_TRACE_FILE     "dpu_trace"
#define DPU_LINE           "========================================================================\n"
#define DPU_LINE_LONG      "=====================================================================================================\n"
#define DPU_LINE_STAR      "*********************************************************************************************************\n"

#define STR_MB_S            "MB/S"
#define STR_GB_S            "GB/S"

int dpuDebug(unsigned long flag);
int dpuRuntimeMode(unsigned long flag);

int dpuTaskMode(dpu_task_t *task, int mode);
int dpuKernelMode(dpu_kernel_t *kernel, int mode);


int get_virtual_node_ID(dpu_task_t *task, const char *nodeName);
int get_virtual_node_ID_v2(dpu_task_t *task, const char *nodeName);
/* dump raw data of code/weights/bias/input/output for specified node into files for debugging */
int dpu_dump_node_by_ID(dpu_task_t *task, int nodeID);
void dpu_dump_node_when_timeout(dpu_task_t *task, char* nodeName);
int dump_get_dir_name(char *dirName);


#define DPU_DEBUG_LOG()      (dpuDebug(FLAG_DEBUG_LOG))
#define DPU_DEBUG_ELF()      (dpuDebug(FLAG_DEBUG_ELF))
#define DPU_DEBUG_LD()       (dpuDebug(FLAG_DEBUG_LD))
#define DPU_DEBUG_DRV()      (dpuDebug(FLAG_DEBUG_DRV))
#define DPU_DEBUG_RELO()     (dpuDebug(FLAG_DEBUG_RELO))
#define DPU_DEBUG_RUN()      (dpuDebug(FLAG_DEBUG_RUN))
#define DPU_DEBUG_MSG()      (dpuDebug(FLAG_DEBUG_MSG))
#define DPU_DEBUG_ALL()      (dpuDebug(FLAG_DEBUG_ALL))
#define DPU_DEBUG_NO_CHECK() (dpuDebug(FLAG_DEBUG_NO_CHECK))

#define DPU_RUNTIME_MODE_NORMAL()   (dpuRuntimeMode(MODE_RUNTIME_NORMAL))
#define DPU_RUNTIME_MODE_PROF()     (dpuRuntimeMode(MODE_RUNTIME_PROFILE))
#define DPU_RUNTIME_MODE_DEBUG()    (dpuRuntimeMode(MODE_RUNTIME_DEBUG))

#define KERNEL_IN_DEBUG(kernel) (dpuKernelMode(kernel, K_MODE_DEBUG))

#define TASK_IN_PROF(task)      (dpuTaskMode(task, T_MODE_PROFILE))
#define TASK_IN_DEBUG(task)     (dpuTaskMode(task, T_MODE_DEBUG))

#ifdef __cplusplus
}
#endif
#endif
