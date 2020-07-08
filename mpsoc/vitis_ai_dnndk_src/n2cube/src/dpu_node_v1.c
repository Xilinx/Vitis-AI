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
#include <stdlib.h>

#include "dpu_err.h"
#include "dpu_node_v1.h"

/* like destructure in C++
 * release resource alloced for node type itself, and either release base structure */
inline static void release (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v1_free((dpu_node_v1_t*)node);
    /* Call parent struct's destructor. */
    dpu_node_free(node);
}

/*
 * Constructor, need to be called somewhere explicitly
 */
dpu_node_v1_t * dpu_node_v1_init(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    /* Call parent struct's constructor. */
    dpu_node_init(node);

    dpu_node_ops_t *ops = &(node->ops);
    ops->release = release;
    return (dpu_node_v1_t *)node;
}

/* free resource alloced for node type itself, without considering of base structure */
void dpu_node_v1_free(dpu_node_v1_t *node) {
    DPU_ASSERT(node, ERR);
}
