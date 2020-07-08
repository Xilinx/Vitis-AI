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

#ifndef _TASK_TENSOR_H_
#define _TASK_TENSOR_H_

#include "../../common/dpu_types.h"

typedef struct task_tensor task_tensor_t;

typedef struct task_tensor_ops {
    void  (*release) (task_tensor_t *_this);
    float (*get_scale) (task_tensor_t *_this);
    void  (*setup_data) (task_tensor_t * _this,
                         tensor_shape_t* shape,
                         mem_segment_t *mem_base,
                         float           scale);
} task_tensor_ops_t;

/*
 * Structure for describing DPU Node's tensor (input or output)
 */
struct task_tensor {
    uint32_t              addr_phy;       /* start of tensor's physical address */
    int8_t*               addr_virt;      /* start of tensor's virtual address */
    mem_segment_t     *dev_mem;
    tensor_shape_t        *shape;         /* pointer to the tensor's entry stored in DPU Kernel */

    task_tensor_ops_t     ops;
};


/*
 * DPU Virtual Node for concact operation layer
 */
struct task_virtual_node_t {
    task_tensor_t    tensorOut;          /* virtual Node's output tensor */
};

task_tensor_t * task_tensor_init (task_tensor_t*);
void task_tensor_free(task_tensor_t*);

#endif
