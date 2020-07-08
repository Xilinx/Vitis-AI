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

#ifndef _DPU_NODE_V1_REAL_H_
#define _DPU_NODE_V1_REAL_H_

#include "dpu_node_v1.h"

typedef struct dpu_node_v1_real dpu_node_v1_real_t;


/* Structure for dpu real node in ABIv1.0*/
struct dpu_node_v1_real {
    dpu_node_v1_t base_v1;

    uint64_t    workload;                   /* node's MAC computation workload */
    uint64_t    memload;                    /* node's memory load/store account in bytes */
    tensor_shape_t shapeIn;                 /* entry for input tensor */

    mem_segment_t   node_bias;              /* Node's bias info of DPU memory space*/
    mem_segment_t   node_weight;            /* Node's weights info of DPU memory space*/
    mem_segment_t   node_code;              /* Node's code segment info of DPU memory space*/
};

/* Constructor/destructor of dpu_node_v1_real_t, should be called explicitly */
dpu_node_v1_real_t * dpu_node_v1_real_init(dpu_node_t *);
void dpu_node_v1_real_free(dpu_node_v1_real_t *node);


#endif /* _DPU_NODE_V1_REAL_H_ */
