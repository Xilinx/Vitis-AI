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
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>


#include "../../common/dpu_node_v2.h"
#include "dpu_err.h"
#include "dpu_sys.h"

inline static int size() {
    return sizeof(dpu_node_v2_t);
}

inline static mem_segment_t * get_node_code (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v2_t *nd = (dpu_node_v2_t*)node;
    if(nd->type == T_NODE_DPU_REAL) {
        return &(((dpu_node_v2_t*)node)->node_code);
    } else {
        return NULL;
    }
}

inline static mem_segment_t* get_node_params (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return ((dpu_node_v2_t*)node)->node_params;
}


inline static uint64_t get_workload (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return ((dpu_node_v2_t*)node)->workload;
}
inline static void  set_workload (dpu_node_t *node, uint64_t val) {
    DPU_ASSERT(node, ERR);
    ((dpu_node_v2_t*)node)->workload = val;
}

inline static uint64_t get_memload (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return ((dpu_node_v2_t*)node)->memload;
}

inline static void set_memload (dpu_node_t *node, uint64_t val) {
    DPU_ASSERT(node, ERR);
    ((dpu_node_v2_t*)node)->memload = val;
}

/**
 * param_name is too long and has duplicate info:
 * it begins with "_dpu_" as prefix, and contines node_name and layer_name,
 * so we trim off it's prefix.
 */
inline static const char* trim_param_name(const char* param_name) {
    char *pos = NULL;
    return (pos = strstr(param_name, "_dpu_")) ? (pos + strlen("_dpu_")): param_name;
}

inline static void dump_params (dpu_node_t *node, kernel_t *in) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    int i;
    dpu_node_v2_t *nd;
    char out_file[MAX_NAME_LEN];

    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    nd = (dpu_node_v2_t *)node;
    for(i=0; i<nd->param_cnt; i++) {
        FILE *fp;
        sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s.bin",
                kernel->base.name, trim_param_name(nd->node_params[i].name));
        fp =fopen(out_file, "wb");
        fwrite(nd->node_params[i].addr_virt, sizeof(char), nd->node_params[i].length, fp);

        fflush(fp);
        fclose(fp);
    }
}

inline static void trace_tensors (dpu_node_t *node, kernel_t *kernel) {
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    int i;
    tensor_shape_t *in, *out, *tensors;
    dpu_node_v2_t * nd = (dpu_node_v2_t *)node;

    tensors = kernel->tensor_list;

    for(i=0; i<nd->input_cnt; i++) {
        in = &(tensors[nd->input_list[i]]);
        printf("%16s - %d\n", "Input Tensor", i);
        printf("%16s  0x%x\n", "I_attribute:", in->attr);
        printf("%16s  0x%x\n", "I_Height:",    in->height);
        printf("%16s  0x%x\n", "I_Width:",     in->width);
        printf("%16s  0x%x\n", "I_Channel:",   in->channel);
        printf("%16s  0x%x\n", "I_Address:",   in->offset);
        printf("%16s  0x%x\n", "I_Size:",      in->size);
        printf("%16s  0x%x\n", "I_Fix_Width:", in->fix_width);
        printf("%16s  %d\n",   "I_Fix_Pos:",   in->fix_pos);
        printf("%16s  %d\n",   "I_ch_stride:", in->channel_stride);
    }

    for(i=0; i<nd->output_cnt; i++) {
        out = &(tensors[nd->output_list[i]]);
        printf("%16s - %d\n", "Output Tensor", i);
        printf("%16s  0x%x\n", "O_attribute:", out->attr);
        printf("%16s  0x%x\n", "O_Height:",    out->height);
        printf("%16s  0x%x\n", "O_Width:",     out->width);
        printf("%16s  0x%x\n", "O_Channel:",   out->channel);
        printf("%16s  0x%x\n", "O_Address:",   out->offset);
        printf("%16s  0x%x\n", "O_Size:",      out->size);
        printf("%16s  0x%x\n", "O_Fix_Width:", out->fix_width);
        printf("%16s  %d\n",   "O_Fix_Pos:",   out->fix_pos);
        printf("%16s  %d\n",   "O_ch_stride:", out->channel_stride);
    }
}

inline static void trace_param_infos (dpu_node_t *node, kernel_t *kernel) {
    int i;

    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    /* there are no info for param in ABIv1.6 and before,
     * so dump when ABI after ABIv1.7. */
    if (kernel->abi_ver >= DPU_ABI_V1_7) {
        dpu_node_v2_t * nd = (dpu_node_v2_t *)node;

        for(i=0; i<nd->param_cnt; i++) {
            printf("%16s:\n", nd->node_params[i].name);
            printf("%16s  0x%x\n", " Fix_W:", nd->elf_params[i].fix_w);
            printf("%16s  %d\n",   " Fix_P:", nd->elf_params[i].fix_p);
        }
    }
}

/**
 * get pure name of param from oName according to Node name.
 * params :
 *   oName : param name string organized like _kernelName_nodeName_paramName.
 *   nName : copy paramName into nName
 * return : pointer to nName
 */
inline static char* get_param_pure_name(dpu_node_t *node, const char* oName, char* nName) {
    DPU_ASSERT(oName, ERR);
    DPU_ASSERT(nName, ERR);
    DPU_ASSERT(node, ERR);

    const char *ptr = oName;
    if(ptr = strstr(oName, node->name)) {
        ptr += strlen(node->name);
        DPU_ASSERT(*ptr != '\0', ERR);
        ptr++;
    }
    DPU_ASSERT(ptr, ERR);

    strcpy(nName, ptr);
    return nName;
}

inline static void trace_addr_phy (dpu_node_t *node,
                                         FILE *stream,
                                         int nodeId) {
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(stream, ERR);

    int i;
    bool newline = false;
    char paramName[MAX_NAME_LEN];
    dpu_node_v2_t *nd = (dpu_node_v2_t*)node;

    fprintf(stream, " Node-%d: %s\n", nodeId, node->name);

    if (nd->type == T_NODE_DPU_REAL) {
        fprintf(stream, "   %-25s : 0x%010lx 0x%lx\n", "Code",
                (unsigned long)(nd->node_code.addr_phy), (unsigned long)(nd->node_code.length));
    }

    for (i=0; i<nd->param_cnt; i++) {
        fprintf(stream,
                "   %-25s : 0x%010lx 0x%lx\n",
                get_param_pure_name(node, nd->node_params[i].name, paramName),
                (unsigned long)(nd->node_params[i].addr_phy), (unsigned long)(nd->node_params[i].length));
    }
}

inline static void trace_addr_virt (dpu_node_t *node,
                                    FILE       *stream,
                                    int        nodeId) {
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(stream, ERR);

    int i;
    bool newline = false;
    char paramName[MAX_NAME_LEN];
    dpu_node_v2_t *nd = (dpu_node_v2_t*)node;

    fprintf(stream, " Node-%d: %s\n", nodeId, node->name);

    if (nd->type == T_NODE_DPU_REAL) {
        fprintf(stream, "   %-25s : 0x%010lx 0x%lx\n", "Code",
                (unsigned long)(nd->node_code.addr_virt), (unsigned long)(nd->node_code.size));
    }

    for (i=0; i<nd->param_cnt; i++) {
        fprintf(stream,
                "   %-25s : 0x%010lx 0x%lx\n",
                get_param_pure_name(node, nd->node_params[i].name, paramName),
                (unsigned long)(nd->node_params[i].addr_virt), (unsigned long)(nd->node_params[i].size));
    }
}



/*
 * Update addr for symbols in Code/Param section.
 * After load data onto DPU, we will get start addr for section,
 * so need to update symbols addr.
 */
inline static void update_addr (dpu_node_t *node, kernel_t* in2) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    int i;
    dpu_node_v2_t *nd = (dpu_node_v2_t*)node;
    if(nd->type == T_NODE_DPU_REAL) {
        /* NOTE: phyical/virtual address for each Node's code alreay assigned */
        if (!KERNEL_IN_DEBUG(kernel)) {
            /* virtual/physicall address for each Node's code is alreday speicified in dump mode */
            nd->node_code.addr_virt =
                    (int8_t*)((unsigned long)nd->node_code.addr_phy +
                    (unsigned long)kernel->mem_code.addr_virt);
        }

        for(i=0; i<nd->param_cnt; i++) {
            /* update Node's virtual address for param */
            nd->node_params[i].addr_virt =
                (int8_t*)((unsigned long)nd->node_params[i].addr_phy +
                (unsigned long)kernel->mem_param.addr_virt);
        }

        /* update Node's physical address for code/param */
        if (!KERNEL_IN_DEBUG(kernel)) {
            nd->node_code.addr_phy =
               nd->node_code.addr_phy + kernel->mem_code.addr_phy;
        }

        for(i=0; i< nd->param_cnt; i++) {
           nd->node_params[i].addr_phy =
                nd->node_params[i].addr_phy + kernel->mem_param.addr_phy;
        }
    }
}

/*
 * Allocate dpu memory for code segment in this node.
 */
inline static void  alloc_dpu_mem_for_node_code (dpu_node_t *node,
                                                             kernel_t *in2,
                                                             int mm_fd) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;

    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    dpu_node_v2_t *nd = (dpu_node_v2_t *)node;
    if(nd->type == T_NODE_DPU_REAL) {
        mem_segment_t * seg = &(nd->node_code);

        /* setup for code */
        if (dpu_dev_mem_alloc(seg, seg->size) != 0) {
            DPU_FAIL_ON_MSG("Fail to map memory for DPU Kernel %s of Layer %s: Address: 0x%x Size: %d",
                kernel->base.name, node->name, seg->addr_phy, seg->size);
        }
        memset(seg->addr_virt, 0, seg->size);
        dpuCacheFlush(seg, 0, seg->size);

        kernel->region_code.region_size += seg->size;
    }
}


/*
 * Release resources for dpu_node_v2_t itself.
 */
void dpu_node_v2_free(dpu_node_v2_t *node) {
    DPU_ASSERT(node, ERR);

    if (node->reg_type_list) {
        free(node->reg_type_list);
    }
    if (node->output_list) {
        free(node->output_list);
    }
    if (node->input_list) {
        free(node->input_list);
    }
    if (node->pre_list) {
        free(node->pre_list);
    }
    if (node->suc_list) {
        free(node->suc_list);
    }
    if (node->elf_params) {
        free(node->elf_params);
    }
    if (node->node_params) {
        free(node->node_params);
    }
}


/* like destructure in C++, unified interface used for different derived structure.
 * release resource alloced for node type itself, and either release base structure */
inline static void release(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);

    dpu_node_v2_free((dpu_node_v2_t*)node);
    /* Call parent struct's destructor. */
    dpu_node_free(node);
}

/*
 * Constructor of dpu_node_v2_t structure, need to be called somewhere explicitly
 */
dpu_node_v2_t * dpu_node_v2_init(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    /* Call parent struct's constructor. */
    dpu_node_init(node);

    dpu_node_ops_t *ops = &(node->ops);
    ops->size = size;
    ops->dump_params       = dump_params;
    ops->trace_tensors     = trace_tensors;
    ops->trace_param_infos = trace_param_infos;
    ops->trace_addr_phy    = trace_addr_phy;
    ops->trace_addr_virt   = trace_addr_virt;
    ops->get_node_code   = get_node_code;
    ops->get_node_params = get_node_params;
    ops->get_memload     = get_memload;
    ops->set_memload     = set_memload;
    ops->get_workload    = get_workload;
    ops->set_workload    = set_workload;
    ops->update_addr     = update_addr;
    ops->release         = release;
    ops->alloc_dpu_mem_for_node_code = alloc_dpu_mem_for_node_code;

    return (dpu_node_v2_t *)node;
}
