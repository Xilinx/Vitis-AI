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
#include "task_node.h"
#include "dpu_err.h"

inline static void cache_flush (task_node_t *_this, dpu_node_t *node) {
    DPU_FAIL_ON_MSG("No input tensor to flush for this task_node.");
}

inline static void cache_invalid_out (task_node_t *_this, dpu_node_t *node) {
    DPU_FAIL_ON_MSG("No output tensor to invalid for this task_node.");
}

inline static task_tensor_t* get_tensorIn (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_FAIL_ON_MSG("No Input tensor for this task_node.");
}

inline static task_tensor_t* get_tensorOut (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_FAIL_ON_MSG("No output tensor for this task_node.");
}

inline static void setup_tensor (task_node_t *_this,
                                 dpu_task_t *task,
                                 dpu_node_t  *node) {

    DPU_FAIL_ON_MSG("No tensor info for this task_node.");
}

inline static void dump_addr_phy (task_node_t *_this, FILE *stream, const char *format) {
    DPU_FAIL_ON_MSG("No addr_phy for this task_node.");
}

inline static void dump_addr_virt (task_node_t *_this, FILE *stream, const char *format) {
    DPU_FAIL_ON_MSG("No addr_virt for this task_node.");
}

inline static void dump_input (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {

}

inline static void dump_output (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {

}


void task_node_free(task_node_t *_this) {
    DPU_ASSERT(_this, ERR);
}


inline static void release (task_node_t *_this) {
    DPU_ASSERT(_this, ERR);

    task_node_free(_this);
}


task_node_t* task_node_init(task_node_t *_this) {
    DPU_ASSERT(_this, ERR);

    task_ops_t *ops = &(_this->ops);
    ops->cache_flush    = cache_flush;
    ops->cache_invalid_out    = cache_invalid_out;

    ops->get_tensorIn   = get_tensorIn;
    ops->get_tensorOut  = get_tensorOut;

    ops->setup_tensor   = setup_tensor;
    ops->release        = release;
    ops->dump_addr_phy  = dump_addr_phy;
    ops->dump_addr_virt = dump_addr_virt;
    ops->dump_input     = dump_input;
    ops->dump_output    = dump_output;

    return _this;
}
