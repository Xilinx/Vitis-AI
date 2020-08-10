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

#ifndef _DPU_SYS_H_
#define _DPU_SYS_H_

#include <fcntl.h>
#include <sys/ioctl.h>

#include "../../common/dpu_types.h"

extern int dpu_dev_mem_alloc(mem_segment_t *seg, uint32_t size);
extern int dpu_dev_mem_free(mem_segment_t *seg);

//extern int dpu_dev_fd;

/* exported functions related to DPU driver */
EXPORT int dpu_attach();
EXPORT int dpu_dettach();

EXPORT int dpu_config();

EXPORT long dpu_launch_execution_session(dpu_kernel_t *kernel, dpu_task_t *task, char *nodeName, dpu_aol_run_t *session);

EXPORT int dpu_alloc_kernel_resource(dpu_kernel_t *kernel);
EXPORT int dpu_alloc_task_resource(dpu_task_t *task);

EXPORT int dpu_release_kernel_resource(dpu_kernel_t *kernel);
EXPORT int dpu_release_task_resource(dpu_task_t *task);

EXPORT int dpuCacheFlush(mem_segment_t *seg, uint32_t offset, uint32_t size);
EXPORT int dpuCacheInvalid(mem_segment_t *seg, uint32_t offset, uint32_t size);

EXPORT void *dpuMemset(void *dest, int value, size_t size);
EXPORT void *dpuMemcpy(void *dest, const void *src, size_t size);
EXPORT size_t dpuFread(void *ptr, size_t size, size_t nmemb, FILE *stream);
EXPORT size_t dpuWrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

int display_dpu_debug_info(void);
#endif
