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
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>

#include "dpu_node_v1_real.h"
#include "dpu_err.h"
#include "dpu_sys.h"



inline static int size() {
    return sizeof(dpu_node_v1_real_t);
}

inline static uint64_t get_workload (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return ((dpu_node_v1_real_t*)node)->workload;
}
inline static void  set_workload (dpu_node_t *node, uint64_t val) {
    DPU_ASSERT(node, ERR);
    ((dpu_node_v1_real_t*)node)->workload = val;
}

inline static uint64_t get_memload (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return ((dpu_node_v1_real_t*)node)->memload;
}

inline static void set_memload (dpu_node_t *node, uint64_t val) {
    DPU_ASSERT(node, ERR);
    ((dpu_node_v1_real_t*)node)->memload = val;
}

inline static mem_segment_t * get_node_code (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return  &(((dpu_node_v1_real_t*)node)->node_code);
}

inline static mem_segment_t* get_node_bias (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return &(((dpu_node_v1_real_t*)node)->node_bias);
}

inline static mem_segment_t* get_node_weight (dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    return &(((dpu_node_v1_real_t*)node)->node_weight);
}

INTERNAL void dump_node_bias(dpu_node_v1_real_t *node, dpu_kernel_t *kernel)
{
    FILE *fp;
    char out_file[MAX_NAME_LEN];

    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_b.bin",
        kernel->base.name, node->base_v1.base.name);

    fp = fopen(out_file, "wb");
    fwrite(node->node_bias.addr_virt, sizeof(char), node->node_bias.length, fp);

    fflush(fp);
    fclose(fp);
}

INTERNAL void dump_node_weights(dpu_node_v1_real_t *node, dpu_kernel_t *kernel)
{
    FILE *fp;
    char out_file[MAX_NAME_LEN];

    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_w.bin",
        kernel->base.name, node->base_v1.base.name);

    fp = fopen(out_file, "wb");
    fwrite(node->node_weight.addr_virt, sizeof(char), node->node_weight.length, fp);

    fflush(fp);
    fclose(fp);
}

inline static void dump_params (dpu_node_t *node, kernel_t *in2) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    dump_node_bias((dpu_node_v1_real_t *)node, kernel);
    dump_node_weights((dpu_node_v1_real_t *)node, kernel);
}

inline static void trace_tensors (dpu_node_t *node, kernel_t *in2) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    dpu_node_v1_real_t * nd = (dpu_node_v1_real_t *)node;
    tensor_shape_t *in = &(nd->shapeIn);
    tensor_shape_t *out = &(nd->base_v1.shapeOut);

    printf("%16s  0x%x\n", "I_Height:",    in->height);
    printf("%16s  0x%x\n", "I_Width:",     in->width);
    printf("%16s  0x%x\n", "I_Channel:",   in->channel);
    printf("%16s  0x%x\n", "I_Address:",   in->offset);
    printf("%16s  0x%x\n", "I_Size:",      in->size);
    printf("%16s  0x%x\n", "I_Fix_Width:", in->fix_width);
    printf("%16s  %d\n",   "I_Fix_Pos:",   in->fix_pos);

    printf("%16s  0x%x\n", "O_Height:",    out->height);
    printf("%16s  0x%x\n", "O_Width:",     out->width);
    printf("%16s  0x%x\n", "O_Channel:",   out->channel);
    printf("%16s  0x%x\n", "O_Address:",   out->offset);
    printf("%16s  0x%x\n", "O_Size:",      out->size);
    printf("%16s  0x%x\n", "O_Fix_Width:", out->fix_width);
    printf("%16s  %d\n",   "O_Fix_Pos:",   out->fix_pos);


}

inline static void trace_param_infos (dpu_node_t *node, kernel_t *in2) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    dpu_node_v1_real_t * nd = (dpu_node_v1_real_t *)node;

    printf("%16s  0x%x\n", "W_Fix_Width:", nd->base_v1.weight_fix_width);
    printf("%16s  %d\n",   "W_Fix_Pos:",   nd->base_v1.weight_fix_pos);

    printf("%16s  0x%x\n", "B_Fix_Width:", nd->base_v1.bias_fix_width);
    printf("%16s  %d\n",   "B_Fix_Pos:",   nd->base_v1.bias_fix_pos);
}

inline static void trace_addr_phy (dpu_node_t *node,
                                         FILE *stream,
                                         int nodeId) {
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(stream, ERR);

    dpu_node_v1_real_t *nd = (dpu_node_v1_real_t*)node;
    fprintf(stream, "%-4d", nodeId);
    fprintf(stream, "0x%08x  0x%08x  0x%08x",
           nd->node_code.addr_phy,
           nd->node_bias.addr_phy,
           nd->node_weight.addr_phy);
}

inline static void trace_addr_virt (dpu_node_t *node,
                                           FILE *stream,
                                           int nodeId) {
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(stream, ERR);

    dpu_node_v1_real_t *nd = (dpu_node_v1_real_t*)node;

    fprintf(stream, "%-4d0x%010lx  0x%010lx  0x%010lx\n",
        nodeId,
        (unsigned long)(nd->node_code.addr_virt),
        (unsigned long)(nd->node_bias.addr_virt),
        (unsigned long)(nd->node_weight.addr_virt));
}


inline static void update_addr (dpu_node_t *node, kernel_t* in2) {
    dpu_kernel_t *kernel = (dpu_kernel_t *)in2;
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(kernel, ERR);

    dpu_node_v1_real_t *nd = (dpu_node_v1_real_t*)node;
    /* NOTE: phyical/virtual address for each Node's code alreay assigned */
    if (!KERNEL_IN_DEBUG(kernel)) {
        /* virtual/physicall address for each Node's code is alreday speicified in dump mode */
        nd->node_code.addr_virt =
                (int8_t*)((unsigned long)nd->node_code.addr_phy +
                (unsigned long)kernel->mem_code.addr_virt);
        }

        /* update Node's virtual address for bias/weights */
        nd->node_bias.addr_virt =
            (int8_t*)((unsigned long)nd->node_bias.addr_phy +
            (unsigned long)kernel->mem_bias.addr_virt);

        nd->node_weight.addr_virt =
            (int8_t*)((unsigned long)nd->node_weight.addr_phy +
            (unsigned long)kernel->mem_weight.addr_virt);

        /* update Node's physical address for code/bias/weights */
        if (!KERNEL_IN_DEBUG(kernel)) {
            nd->node_code.addr_phy =
               nd->node_code.addr_phy + kernel->mem_code.addr_phy;
        }

        nd->node_bias.addr_phy =
            nd->node_bias.addr_phy + kernel->mem_bias.addr_phy;

        nd->node_weight.addr_phy =
            nd->node_weight.addr_phy + kernel->mem_weight.addr_phy;
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

    void *ret;
    dpu_node_v1_real_t *nd = (dpu_node_v1_real_t *)node;
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


/* free resource alloced for node type itself, and either release base structure */
inline void dpu_node_v1_real_free(dpu_node_v1_real_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v1_free(&(node->base_v1));
}

/* like destructure in C++, unified interface used for different derived structure.
 * release resource alloced for node type itself, and either release base structure */
inline static void release(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    dpu_node_v1_real_free((dpu_node_v1_real_t*)node);
}

/*
 * Constructor, need to be called somewhere explicitly
 */
inline dpu_node_v1_real_t * dpu_node_v1_real_init(dpu_node_t *node) {
    DPU_ASSERT(node, ERR);
    /* Call parent struct's constructor. */
    dpu_node_v1_init(node);

    dpu_node_ops_t * ops = &(node->ops);
    ops->size = size;
    ops->release = release;
    ops->dump_params       = dump_params;
    ops->trace_tensors     = trace_tensors;
    ops->trace_param_infos = trace_param_infos;
    ops->trace_addr_phy    = trace_addr_phy;
    ops->trace_addr_virt   = trace_addr_virt;
    ops->get_node_code     = get_node_code;
    ops->get_node_bias     = get_node_bias;
    ops->get_node_weight   = get_node_weight;
    ops->get_memload  = get_memload;
    ops->set_memload  = set_memload;
    ops->get_workload = get_workload;
    ops->set_workload = set_workload;
    ops->update_addr  = update_addr;
    ops->alloc_dpu_mem_for_node_code = alloc_dpu_mem_for_node_code;

    return (dpu_node_v1_real_t *)node;
}
