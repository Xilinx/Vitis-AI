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
#include "dpu_node_v1_virt.h"
#include "dpu_err.h"

inline static int size() {
    return sizeof(dpu_node_v1_virt_t);
}

/* free resource alloced for node type itself, and either release base structure */
inline void dpu_node_v1_virt_free(dpu_node_v1_virt_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v1_free(&(node->base_v1));
}

/* like destructure in C++, unified interface used for different derived structure.
 * release resource alloced for node type itself, and either release base structure */
inline static void release(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v1_virt_free((dpu_node_v1_virt_t*)node);
}

/*
 * Constructor, need to be called somewhere explicitly
 */
inline dpu_node_v1_virt_t * dpu_node_v1_virt_init(dpu_node_v1_virt_t *node) {
    DPU_ASSERT(node, ERR);
    /* Call parent struct's constructor. */
    dpu_node_v1_init((dpu_node_t*)node);

    dpu_node_ops_t * ops = &(node->base_v1.base.ops);
    ops->size = size;
    ops->release = release;
    return node;
}
