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

#ifndef _TASK_NODE_H_
#define _TASK_NODE_H_

#include <stdio.h>
#include "../../common/dpu_types.h"
#include "../../common/dpu_node.h"
#include "task_tensor.h"
#include "dpu_kernel.h"


typedef struct task_node task_node_t;
typedef struct dpu_task dpu_task_t;

typedef struct task_ops {
    void (*release) (task_node_t*);

    /* flush input tensor to memory */
    void (*cache_flush) (task_node_t*, dpu_node_t*);
    /* invalid cache for output tensor, in order to read data from memory directly */
    void (*cache_invalid_out) (task_node_t*, dpu_node_t*);

    task_tensor_t* (*get_tensorIn) (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel);
    task_tensor_t* (*get_tensorOut) (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel);

    void (*setup_tensor) (task_node_t*, dpu_task_t *, dpu_node_t*);
    void (*dump_addr_phy) (task_node_t*, FILE*, const char*);
    void (*dump_addr_virt) (task_node_t*, FILE*, const char*);
    void (*dump_input) (task_node_t *_this, dpu_task_t *task, dpu_node_t *node);
    void (*dump_output) (task_node_t *_this, dpu_task_t *task, dpu_node_t *node);
}task_ops_t;

/** @brief Descriptor for one DPU task
 *
 * - One kernel could be instantiated to many tasks which hold their own
 *   Input/Output memory space individially.
 *
 */
struct task_node {
    /* timestamp to trace the begin & stop for current network layer
     * they are specified by DPU device driver while DPU running */
    uint64_t       time_start;         /* the start timestamp (ns) when layer begins to run */
    uint64_t       time_end;           /* the end timestamp (ns) when layer finishes running */
    int                    coreID;             /* DPU coreID which Task Node runs on */

    struct port_profile_t  port_profile_start; /* start profile info for ports */
    struct port_profile_t  port_profile_end;   /* ending profile info for ports */

    task_ops_t             ops;
};

task_node_t* task_node_init(task_node_t*);
void task_node_free(task_node_t*);

#endif
