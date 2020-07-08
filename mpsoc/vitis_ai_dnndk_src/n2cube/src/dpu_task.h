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

#ifndef _DPU_TASK_H_
#define _DPU_TASK_H_

#include "dpu_kernel.h"
#include "task_node.h"

typedef struct dpu_task dpu_task_t;

#define MEM_IO_USED     (1 << 0)
#define MEM_INPUT_USED  (1 << 1)
#define MEM_OUTPUT_USED (1 << 2)
#define SET_MEM_USED_MASK(task, val)   (task->mem_used_mask |= val)
#define SET_MEM_IO_USED(task)          SET_MEM_USED_MASK(task, MEM_IO_USED)
#define SET_MEM_INPUT_USED(task)       SET_MEM_USED_MASK(task, MEM_INPUT_USED)
#define SET_MEM_OUTPUT_USED(task)      SET_MEM_USED_MASK(task, MEM_OUTPUT_USED)
#define CHECK_MEM_USED(task, val)      (task->mem_used_mask & val)
#define IS_MEM_IO_USED(task)        CHECK_MEM_USED(task, MEM_IO_USED)
#define IS_MEM_INPUT_USED(task)     CHECK_MEM_USED(task, MEM_INPUT_USED)
#define IS_MEM_OUTPUT_USED(task)    CHECK_MEM_USED(task, MEM_OUTPUT_USED)

struct shape_t {
    uint32_t  n;
    uint32_t  h;
    uint32_t  w;
    uint32_t  c;
};

struct fixinfo_t {
    int32_t  width;
    int32_t  pos;
};

typedef struct {
    int8_t   *addr_virt;   // virtual addr.
    uint64_t  size;         // size of each tensor.
    struct shape_t   shape;
    struct fixinfo_t fix;
    float     scale;
    char*     tensor_name;  // tensor name.
} DPUTensorAttr;
/** @brief Descriptor for one DPU task which holds all the contexts
 *  for the runtime behaviours of one task
 *
 *  - Task is instantiated from one kernel with its own allocated Input/Output
 *    memory space.
 */
struct dpu_task {
    char                   name[MAX_NAME_LEN];     /* DPU task name in format: kernel name + task ID */
    int                    mode;        /* running mode: T_MODE_NORMAL/T_MODE_PROF/T_MODE_DEBUG */
    task_handle_t          task_id;     /* unique ID number for this task */
    dpu_kernel_t           *kernel;     /* pointer to kernel where task is instantiated from */
    uint8_t                schedule_priority;
    uint32_t               binding_core_mask;

    int                    mem_used_mask;
    mem_segment_t          mem_IO;      /* memory space for taks's Input/Output */
    mem_segment_t          mem_input;   /* seprately input mem */
    mem_segment_t          mem_output;  /* seprately output mem */

  uint64_t      inputTensorNum;    // Total number of all the inputs.
  DPUTensorAttr *inputTensorAttrs; // Attr array of all the inputs, 'inputNumber' as its total number.
                                   // Inputs order is same as DNNC summary after compilation.
  uint64_t      outputTensorNum;   // Total number of all the outputs.
  DPUTensorAttr *outputTensorAttrs;// Attr array of all the outputs, 'outputNumber' as its total number.
                                   // Outputs order is same as DNNC summary after compilation.

    task_node_t**          node_list;   /* header pointing to Node's list of current task */
    struct task_virtual_node_t *virt_node_list; /* virtual Node list of current task */

    /*
     * timestamp to trace the begin & stop for current DPU task
     * they are abtained via DPU device driver while task running time
     */
    signed long long       time_start;  /* the start timestamp in nano-second */
    signed long long       time_end;    /* the end timestamp in nano-second */
    signed long long       time_delta;  /* Delta time (ns) for Task running */
    signed long long       time_wall;   /* Wall time (ns): wall-time of (last layer - first layer) */
    int                    coreID;      /* DPU core ID which task runs on */

    struct port_profile_t  port_profile_start; /* start profile info for ports */
    struct port_profile_t  port_profile_end;   /* ending profile info for ports */
    unsigned long long     counter;
};


#endif /*_DPU_TASK_H_*/
