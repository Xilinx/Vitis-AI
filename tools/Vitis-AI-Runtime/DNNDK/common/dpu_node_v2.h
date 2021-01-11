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

#ifndef _DPU_NODE_V2_H_
#define _DPU_NODE_V2_H_

#include <stdint.h>

#include "dpu_node.h"
#include "dpu_types.h"

typedef struct dpu_elf_sec dpu_elf_sec_t;
typedef struct dpu_elf_code_sec dpu_elf_code_sec_t;
typedef struct dpu_elf_param_sec dpu_elf_param_sec_t;
typedef struct dpu_node_v2  dpu_node_v2_t;

/* dpu section(symbol) info described in Node-Pool item, for ABIv2.0 */
struct dpu_elf_sec {
    uint32_t   name_idx;           /* name idx for this symbol in dpu elf .deephi.strtab section, name looks like _kernelName_nodeName_paramName */
    uint32_t   offset;             /* offset in byte relative to the begin of section */
    uint32_t   align;              /* alignment,offset in byte relative to addr in reg_id */
    uint32_t   size;               /* section(symbol) size in byte in dpu elf */
};

/* dpu code section(symbol) info  described in Node-Pool item, for ABIv2.0 */
struct dpu_elf_code_sec {
    dpu_elf_sec_t  base;
};

/* dpu param section(symbol) info  described in Node-Pool item, for ABIv2.0 */
struct dpu_elf_param_sec {
    dpu_elf_sec_t  base;
    uint32_t           fix_w;           /* fix width for this param */
    uint32_t           fix_p;           /* fix pos for this param */
    uint32_t           hight;           /* hight dimension for this param */
    uint32_t           width;           /* width dimension for this param */
    uint32_t           channel;         /* channel dimension for this param */
    uint32_t           out_channel;     /* output channel dimension for this param */
};

typedef enum data_type{
    T_DATA_IO  = 0,     // unique IO or seprate FM.
    T_DATA_OUTPUT = 1,  // seprate output
    T_DATA_PARAM  = 2,
    T_DATA_INPUT  = 3,  // seprate input
    T_DATA_CODE   = 4
} data_type_t;

typedef struct dpu_reg_assign{
    uint16_t        reg_id;      /* reg_id, 0~7 for DPU version after v1.3.x and tingtao */
    data_type_t     data_type;   /* data type assigned to register with reg_id */
} dpu_reg_assign_t;

/* structure for Node-Pool item for ABIv2.0 */
struct dpu_node_v2 {
    dpu_node_t base;

    node_type_t     type;              /* node type, real/virt/cpu */

    uint64_t        workload;          /* node's MAC computation workload */
    uint64_t        memload;           /* node's memory load/store account in bytes, for real node */

    uint32_t        reg_cnt;           /* count of registers can be assigned to this node */
    dpu_reg_assign_t *reg_type_list;    /* list of data types assigned to registers, index by reg_id */

    uint32_t        input_cnt;         /* count of input tensor */
    uint32_t        *input_list;       /* index list for input tensors */
    uint32_t        output_cnt;        /* count of output tensor */
    uint32_t        *output_list;      /* index list for output tensors */

    uint32_t            code_cnt;      /* count of code section(symbol) related to this node */
    dpu_elf_code_sec_t  elf_code;      /* code section info described in Node-Pool for ABIv2.0 */
    uint32_t            param_cnt;     /* count of param section(symbol) related to this node */
    dpu_elf_param_sec_t *elf_params;   /* list of param section info described in Node-Pool for ABIv2.0, weight/bias... */
    mem_segment_t       *node_params;  /* list of param segment infos of DPU memory space, total number: param_cnt */
    mem_segment_t       node_code;     /* Node's code segment info of DPU memory space, only exist in read node*/

    uint32_t        pre_cnt;           /* count of predecessor of this node */
    uint32_t*       pre_list;          /* predecessor list, each item is a index subscribe to node-pool */
    uint32_t        suc_cnt;           /* count of successor of this node */
    uint32_t*       suc_list;          /* successor list, each item is a index subscribe to node-pool */
};

dpu_node_v2_t * dpu_node_v2_init(dpu_node_t *);
void dpu_node_v2_free(dpu_node_v2_t *node);

#endif /* _DPU_NODE_V2_H_ */
