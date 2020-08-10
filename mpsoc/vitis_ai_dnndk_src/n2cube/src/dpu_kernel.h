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

#ifndef _DPU_KERNEL_H_
#define _DPU_KERNEL_H_

#include "dpu_node_v1_virt.h"
#include "dpu_node_v1.h"

#include "../../common/dpu_types.h"

typedef struct dpu_kernel dpu_kernel_t;

/** @brief Descriptor for one DPU kernel which holds all the contexts
 *  for the runtime behaviours of one kernel
 *
 * - One kernel could be instantiated to many tasks which hold their own private
 *   Input/Output memory space individially.
 *
 */
struct dpu_kernel {
    kernel_t base;
    dpu_node_v1_virt_t     *virt_node_list;  /* virutal Node info for v1, will not be used in v2 */
    mem_segment_t   mem_code;       /* DPU code segment info */
    mem_segment_t   mem_weight;     /* DPU weight segment info */
    mem_segment_t   mem_bias;       /* DPU bias segment info */
    mem_segment_t   mem_param;      /* DPU param segment info, for ABIv2.0 */
    mem_segment_t   mem_prof;       /* DPU profiler segment info */

    /*
     * for the same network kernel
     * 1 - different DPUs share the same region of weights/bias & code
     * 2 - different DPUs have their own region of input/output & profiler
     */
    mem_region_t    region_code;    /* memory region for DPU code */
    mem_region_t    region_wb;      /* memory region for combination of weight/bias */
    mem_region_t    region_param;   /* memory region for DPU param, for ABIv2.0 */
    mem_region_t    region_prof;    /* memory region for DPU profiler */

};


#endif /* _DPU_KERNEL_H_ */
