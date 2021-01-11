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
#include "task_tensor.h"
#include "dpu_err.h"

inline static void setup_data(task_tensor_t * _this,
                                  tensor_shape_t* shape,
                                  mem_segment_t *mem_base,
                                  float           scale) {
    DPU_FAIL_ON_MSG("Can not come here.");
}


inline static float get_scale(task_tensor_t *_this) {
    DPU_FAIL_ON_MSG("No scale in task_tensor");
}

void task_tensor_free(task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);
}

inline static void release(task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);
    task_tensor_free(_this);
}

task_tensor_t * task_tensor_init (task_tensor_t *_this) {
    DPU_ASSERT(_this, ERR);

    task_tensor_ops_t *ops = &(_this->ops);
    ops->release    = release;
    ops->get_scale  = get_scale;
    ops->setup_data = setup_data;

    return _this;
}
