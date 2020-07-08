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

#ifndef _DPU_NODE_V1_VIRT_H_
#define _DPU_NODE_V1_VIRT_H_

#include "dpu_node_v1.h"

typedef struct dpu_node_v1_virt dpu_node_v1_virt_t;


/* Structure for dpu virtual node in ABIv1.0*/
struct dpu_node_v1_virt {
    dpu_node_v1_t base_v1;
};


/* Constructor/destructor of dpu_node_v1_real_t, should be called explicitly */
dpu_node_v1_virt_t * dpu_node_v1_virt_init(dpu_node_v1_virt_t *node);
void dpu_node_v1_virt_free(dpu_node_v1_virt_t *node);


#endif /* _DPU_NODE_V1_VIRT_H_ */
