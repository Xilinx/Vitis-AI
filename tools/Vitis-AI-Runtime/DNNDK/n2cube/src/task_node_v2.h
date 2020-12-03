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

#ifndef _TASK_NODE_V2_H_
#define _TASK_NODE_V2_H_

#include "task_node.h"

/* task structure for ABIv2.0 */
typedef struct task_node_v2 {
    task_node_t   base;        /* base structure */
    task_tensor_t *tensorsIn;  /* input tensor list, multiple input supported, with total-number of dpu_node_v1:input_cnt */
    task_tensor_t *tensorsOut; /* output tensor list, multiply output supported, with total-number of dpu_node_v1:output_cnt  */
}task_node_v2_t;

task_node_v2_t* task_node_v2_init(task_node_t *_this, uint32_t inCnt, uint32_t outCnt);
void task_node_v2_free(task_node_t *_this);

#endif
