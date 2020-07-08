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

#ifndef _TASK_TENSOR_v2_H_
#define _TASK_TENSOR_v2_H_

#include "task_tensor.h"

typedef struct task_tensor_v2 {
    task_tensor_t base;
    float         scale;                 /* scale for fix info */
} task_tensor_v2_t;

task_tensor_v2_t * task_tensor_v2_init(task_tensor_t*);
void task_tensor_v2_free(task_tensor_t*);


#endif /* _TASK_TENSOR_v2_H_ */
