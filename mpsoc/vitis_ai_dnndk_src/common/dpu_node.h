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

#ifndef  _DPU_NODE_H_
#define  _DPU_NODE_H_

#include <stdio.h>
#include <stdint.h>
#include "dpu_types.h"

#define DPU_NODE_FAIL_ON_MSG(a) printf(a)
typedef struct dpu_node_ops dpu_node_ops_t;
typedef struct dpu_node dpu_node_t;
/*
 * Node type
 */
typedef enum {
    T_NODE_DPU_REAL = 0x1,
    T_NODE_DPU_VIRT = 0x2,
    T_NODE_CPU = 0x80
} node_type_t;

struct dpu_node_ops{
    void (*release) (dpu_node_t *);

    void (*dump) (void *task);
    void (*dump_code) (void *task);
    /* dump bin info for param sections(symbols) in dpu ELF */
    void (*dump_params) (dpu_node_t *node, kernel_t *kernel);
    void (*trace_tensors) (dpu_node_t *node, kernel_t* kernel);
    /* trace related infos(fix) for param section(symbol) in Node-Pool */
    void (*trace_param_infos) (dpu_node_t *node, kernel_t* kernel);
    void (*trace_addr_phy) (dpu_node_t *node, FILE *stream, int nodeId);
    void (*trace_addr_virt) (dpu_node_t *node, FILE *stream, int nodeId);

    char* (*get_name) (dpu_node_t *node);
    void (*set_name) (dpu_node_t *, char *);
    int (*size) ();

    mem_segment_t* (*get_node_code) (dpu_node_t *);
    mem_segment_t* (*get_node_bias) (dpu_node_t *);
    mem_segment_t* (*get_node_weight) (dpu_node_t *);
    mem_segment_t* (*get_node_params) (dpu_node_t *);

    uint64_t (*get_workload) (dpu_node_t *);
    void     (*set_workload) (dpu_node_t *, uint64_t);
    uint64_t (*get_memload) (dpu_node_t *);
    void     (*set_memload) (dpu_node_t *, uint64_t);

    /* update each Node's code/bias/weight/input/output address info
     * after DPU kernel's resource is allocated  */
    void     (*update_addr) (dpu_node_t *, kernel_t *);
    void     (*alloc_dpu_mem_for_node_code) (dpu_node_t *, kernel_t*, int);
};

struct dpu_node {
    char            *name;                    /* Node's name */
    //node_type_t     type;                   /* Node's type, may be not needed */

    dpu_node_ops_t  ops;                      /* Node's operations, looks like member function in C++ */
};



/* constructor of base structure, should be called explicitly */
dpu_node_t* dpu_node_init(dpu_node_t *node);

/* destructor of base structure, should be called explicitly */
void dpu_node_free(dpu_node_t *);


#endif  /* _DPU_NODE_H_ */
