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

#ifndef _DPU_NODE_V1_H_
#define _DPU_NODE_V1_H_

#include "../../common/dpu_node.h"
#include "../../common/dpu_types.h"
#include "task_tensor.h"

typedef struct dpu_node_v1 dpu_node_v1_t;


/* Base structure for dpu node in ABIv1.0*/
struct dpu_node_v1{
    dpu_node_t base;                   /* base structure */

    uint8_t     weight_fix_width;           /* fixed info of weight: width, 8 by defaul at present */
    int8_t      weight_fix_pos;             /* fixed info of weight: position, -7~7 */

    uint8_t     bias_fix_width;             /* fixed info of bias: width, 8 by defaul at present */
    int8_t      bias_fix_pos;               /* fixed info of bias: position, -7~7 */

    tensor_shape_t shapeOut;                /* entry for output tensor */
};


dpu_node_v1_t * dpu_node_v1_init(dpu_node_t *);
void dpu_node_v1_free(dpu_node_v1_t *node);

#endif /* _DPU_NODE_V1_H_ */
