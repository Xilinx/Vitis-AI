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
#include <string.h>
#include "dpu_node.h"

inline static char* get_name (dpu_node_t *node) {
    if (node == NULL) {
        return NULL;
    }
    return node->name;
}

inline static void set_name (dpu_node_t *node, char *nm) {
    if (node == NULL || nm == NULL) {
        return;
    }

    if(node->name == NULL) {
        uint32_t len = strlen(nm);

        node->name = (char*)malloc(sizeof(char*) * (len+1));
        strcpy(node->name, nm);
    }
}

inline static int size() {
    return sizeof(dpu_node_t);
}

inline static mem_segment_t * get_node_code (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No node_code for this node.");
    return NULL;
}

inline static mem_segment_t* get_node_bias (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No node_bias for this node.");
    return NULL;
}

inline static mem_segment_t* get_node_weight (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No node_weight for this node.");
    return NULL;
}

inline static mem_segment_t* get_node_params (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No node_params for this node.");
    return NULL;
}

inline static void dump_params (dpu_node_t *node, kernel_t *in2) {
    DPU_NODE_FAIL_ON_MSG("No param section for this node.");
}


inline static void trace_tensors (dpu_node_t *node, kernel_t *in2) {
    DPU_NODE_FAIL_ON_MSG("No tensor info for this node.");
}

inline static void trace_param_infos(dpu_node_t *node, kernel_t *in2) {
    DPU_NODE_FAIL_ON_MSG("No param info for this node.");
}

inline static void trace_addr_phy (dpu_node_t *node, FILE *stream, int nodeId) {
    DPU_NODE_FAIL_ON_MSG("No addr phy info for this node.");
}

inline static void trace_addr_virt (dpu_node_t *node, FILE *stream, int nodeId) {
    DPU_NODE_FAIL_ON_MSG("No addr virt info for this node.");
}


inline static void release(dpu_node_t *node) {
    if (node == NULL) {
        return;
    }
    dpu_node_free(node);
}

inline static uint64_t get_workload (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No workload for this node.");
    return 0;
}
inline static void  set_workload (dpu_node_t *node, uint64_t val) {
    DPU_NODE_FAIL_ON_MSG("No workload for this node.");
}

inline static uint64_t get_memload (dpu_node_t *node) {
    DPU_NODE_FAIL_ON_MSG("No memload for this node.");
    return 0;
}

inline static void set_memload (dpu_node_t *node, uint64_t val) {
    DPU_NODE_FAIL_ON_MSG("No memload for this node.");
}

inline static void update_addr (dpu_node_t *node, kernel_t *kernel) {
    if (node == NULL || kernel == NULL) {
        return;
    }
    DPU_NODE_FAIL_ON_MSG("No address to be updated.");
}

inline static void  alloc_dpu_mem_for_node_code (dpu_node_t *node,
        kernel_t *kernel, int mm_fd) {
    DPU_NODE_FAIL_ON_MSG("No node_code for this node.");
}



/* constructor of base structure, should be called explicitly */
dpu_node_t* dpu_node_init(dpu_node_t* node) {
    if (node == NULL) {
        return NULL;
    }
    node->name = 0;
    dpu_node_ops_t *ops = &(node->ops);

    /* setup member functions */
    ops->release  = release;
    ops->get_name = get_name;
    ops->set_name = set_name;
    ops->size     = size;
    ops->get_node_code   = get_node_code;
    ops->get_node_bias   = get_node_bias;
    ops->get_node_weight = get_node_weight;
    ops->get_node_params = get_node_params;
    ops->dump_params     = dump_params;
    ops->trace_tensors   = trace_tensors;
    ops->trace_param_infos = trace_param_infos;
    ops->trace_addr_phy  = trace_addr_phy;
    ops->trace_addr_virt = trace_addr_virt;
    ops->set_workload    = set_workload;
    ops->get_workload    = get_workload;
    ops->set_memload     = set_memload;
    ops->get_memload     = get_memload;
    ops->alloc_dpu_mem_for_node_code = alloc_dpu_mem_for_node_code;
    return node;
}

/* destructor of base structure, should be called explicitly */
void dpu_node_free(dpu_node_t *node) {
    if (node == NULL) {
        return;
    }
    if (node->name) {
        free(node->name);
    }
}
