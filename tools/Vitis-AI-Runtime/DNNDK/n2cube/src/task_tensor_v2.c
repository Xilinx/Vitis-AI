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
#include "task_tensor_v2.h"
#include "dpu_err.h"

inline static void setup_data(task_tensor_t  * _this,
                              tensor_shape_t *shape,
                              mem_segment_t  *mem_base,
                              float          scale) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(shape, ERR);
    DPU_ASSERT(mem_base, ERR);

    /* Set Node Tensor entry for Task */
    _this->shape = shape;
    _this->dev_mem = mem_base;
    /* set scale */
    /* set physical address */
    _this->addr_phy = mem_base->addr_phy + shape->offset;
    /* set virtual address */
    _this->addr_virt = (int8_t*)((unsigned long)mem_base->addr_virt +
                        shape->offset);
    _this->shape->scale = scale;
}

inline static float get_scale(task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);
    return _this->shape->scale;
}

/*
 * Release resources for task_tensor_v2_t itself.
 */
void task_tensor_v2_free(task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);
}

/*
 * Destructor of task_tensor_v2_t structure.
 */
static void release(task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);
    task_tensor_v2_free(_this);
    /* Call parent struct's destructor. */
    task_tensor_free(_this);
}

/*
 * Constructor of task_tensor_v2_t structure, need to be called somewhere explicitly
 */
task_tensor_v2_t * task_tensor_v2_init(task_tensor_t* _this) {
    DPU_ASSERT(_this, ERR);
    /* Call parent struct's constructor. */
    task_tensor_init(_this);

    task_tensor_ops_t *ops = &(_this->ops);
    ops->release    = release;
    ops->get_scale  = get_scale;
    ops->setup_data = setup_data;

    return (task_tensor_v2_t*)_this;
}
