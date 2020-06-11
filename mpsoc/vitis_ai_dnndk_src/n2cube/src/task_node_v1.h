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

#ifndef _TASK_NODE_V1_H_
#define _TASK_NODE_V1_H_

#include "task_node.h"

/* task structure for ABIv1.0 */
typedef struct task_node_v1 {
    task_node_t      base;              /* base structure */
    task_tensor_t    tensorIn;          /* single input. */
    task_tensor_t    tensorOut;         /* single output. */
}task_node_v1_t;

task_node_v1_t* task_node_v1_init(task_node_t *);
void task_node_v1_free(task_node_t *);


#endif
