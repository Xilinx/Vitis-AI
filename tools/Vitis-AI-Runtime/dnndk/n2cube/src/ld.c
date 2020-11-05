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
#include "../../common/dpu_types.h"
#include "dpu_def.h"
#include "dpu_err.h"
#include "dpu_target.h"
#include "dpu_elf.h"
#include "dpu_sys.h"
#include "task_node_v1.h"
#include "task_node_v2.h"
#include "task_tensor_v1.h"
#include "../../common/dpu_node_v2.h"


#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <stdlib.h>
#include <stdint.h>

/* interanl functions used only inside DPU runtime loader */
INTERNAL int dpu_load_segment(dpu_kernel_t *kernel);
INTERNAL int dpu_do_relocation(dpu_kernel_t *kernel);
INTERNAL int dpu_update_kernel_node(dpu_kernel_t *kernel);


/*
 * Load DPU code and data (weights/bias/input/output) into DPU memory space
 * 1. open DPU device /dev/dpu
 * 2. allocate related resources for current process
 * 3. terminate process if fail
 *
 * Return value:
 */
EXPORT int load_kernel(dpu_kernel_t *kernel)
{
    int i;
    int ret;
    FILE *elf_fd;

    /* read kernel info from hybrid ELF binary */
    ret = dpu_elf_load_kernel(kernel);
    if (ret != N2CUBE_SUCCESS) {
        return ret;
    }

    /* debug the result after parsing hybird ELF */
    dpu_elf_load_debug(kernel);

    /* allocate resources for DPU kernel */
    dpu_alloc_kernel_resource(kernel);

    /* update Nodes info when kernel resource is allocated */
    dpu_update_kernel_node(kernel);

    /* load DPU code/weight/bias from ARM-DPU ELF binary into DPU memory space */
    dpu_load_segment(kernel);

    /* perform address rebasing-style relocation against DPU code segment */
    /* TODO */
    /* dpu_do_relocation(kernel); */

    return N2CUBE_SUCCESS;
}


EXPORT int unload_kernel(dpu_kernel_t *kernel)
{
    return N2CUBE_SUCCESS;
}


INTERNAL int hash_cnn_v1_inst(unsigned char inst_id)
{
    int i;

    for (i = 0; i < sizeof(cnn_v1_inst_tbl)/sizeof(struct dpu_inst_t); i++ ) {
        if (cnn_v1_inst_tbl[i].id == inst_id) {
            return i;
        }
    }

    return ERR_INVALID_INST;
}


INTERNAL int hash_cnn_v2_inst(unsigned char inst_id)
{
    int i;

    for (i = 0; i < sizeof(cnn_v2_inst_tbl)/sizeof(struct dpu_inst_t); i++ ) {
        if (cnn_v2_inst_tbl[i].id == inst_id) {
            return i;
        }
    }

    return ERR_INVALID_INST;
}

INTERNAL int dpu_check_endian(dpu_kernel_t *kernel)
{
    int i, j;
    unsigned long long source, result;
    unsigned char *pSrc, *pResult ;

    for (i = 0; i < kernel->mem_code.length; i += 8) {
        source = *(unsigned long long*)(&(kernel->mem_code.addr_virt[i]));
        pSrc = (unsigned char*)(&source);
        pResult = (unsigned char*)(&result);

        for (j=0;j<8;j++) {
            pResult[j] = pSrc[7-j];
        }

        *(unsigned long long *)(&(kernel->mem_code.addr_virt[i])) = result;
    }

#if 0
    if (DPU_DEBUG_RELO()) {
        printf("Dump DPU code after relocation from start:0x%08lx:\n", (unsigned long)(kernel->mem_code.addr_virt));
        for (i=0; i < kernel->mem_code.length; i++) {
            printf("%02x", kernel->mem_code.addr_virt[i] & 0xff);
            if ( (i+1)%8 == 0 ) printf("\n");
        }
    }
#endif

    return N2CUBE_SUCCESS;
}


/*
 * Perform relocation against DPU code after loading DPU code/data from
 * ARM-DPU-mixed ELF binary int DPU memory space.
 */
INTERNAL int dpu_do_relocation(dpu_kernel_t *kernel)
{
    int i, idx, pos;
    unsigned char inst_id;
    unsigned long relo_addr;

    dpu_check_endian(kernel);
    for (i = 0; i < kernel->mem_code.length; ) {
        /* get instruction's opcode and encodes */
        inst_id = (kernel->mem_code.addr_virt[i]) & 0x0f;
        idx = hash_cnn_v2_inst(inst_id);

        /* invalid instruction opcode */
        DPU_ASSERT(idx >= 0, ERR);

        if (cnn_v2_inst_tbl[idx].reloc) {
            pos = i + cnn_v2_inst_tbl[idx].reloc_pos;
            relo_addr = *(unsigned long *)(&(kernel->mem_code.addr_virt[pos]));

            /* for CNNv2 the start address for region of weight/bias is weight */
            *(unsigned long *)(&(kernel->mem_code.addr_virt[pos])) = (unsigned long)(relo_addr + kernel->mem_weight.addr_phy);
        } else {
            relo_addr = 0;
        }

#if 1
        if (DPU_DEBUG_LD()) {
            printf("mem offset:0x%x\tcode:0x%x\tindex:%d\tname:%s\tlength:%d \n",
               i+7, inst_id, idx, cnn_v2_inst_tbl[idx].name, cnn_v2_inst_tbl[idx].len);
            if (cnn_v2_inst_tbl[idx].reloc) {
                printf("mem base addr:0x%lx\trel addr:0x%lx\tafter relo:0x%lx\n",
                   (unsigned long)kernel->mem_weight.addr_phy, relo_addr, *(unsigned long *)(&(kernel->mem_code.addr_virt[pos])));
            }
        }
#endif

        /* update code offset */
        i += cnn_v2_inst_tbl[idx].len;
    }

    /* TODO */
    /* dpu_check_endian(kernel); */

    return N2CUBE_SUCCESS;
}


/*
 * Load DPU code or data from ARM-DPU-mixed ELF binary int DPU memory space
 */
INTERNAL int dpu_load_segment(dpu_kernel_t *kernel)
{
    int i, ret;
    mem_segment_t *node_code;
    dpu_node_t **nodes = kernel->base.node_list;

    FILE *fp = fopen(kernel->base.elf_name, "r");
    fseek(fp, kernel->base.elf_code.offset, SEEK_SET);

    if (KERNEL_IN_DEBUG(kernel)) {
        /* load DPU code in Node from hybird ELF binary into DPU memory space */
        for (i=0; i < kernel->base.node_cnt; i++) {
            if( node_code = nodes[i]->ops.get_node_code(nodes[i]) ) {
                ret = fread(node_code->addr_virt, 1,
                    node_code->length, fp);
                dpuCacheFlush(node_code, 0, node_code->length);
                if (ret < node_code->length) {
                    DPU_FAIL_ON_MSG("failure when reading code segment of Node:%s for DPU kenrel:%s.",
                        nodes[i]->ops.get_name(nodes[i]), kernel->base.name);
                }
            }
        }
    } else {
        /* load DPU code as a whole from hybrid ELF binary into DPU memory space */
        ret = fread(kernel->mem_code.addr_virt, 1, kernel->base.elf_code.size, fp);
        dpuCacheFlush(&kernel->mem_code, 0, kernel->base.elf_code.size);
        if (ret < kernel->base.elf_code.size) {
            DPU_FAIL_ON_MSG("failure when reading code segment for DPU kenrel:%s.", kernel->base.name);
        }

    }

    if (kernel->base.abi_ver <= DPU_ABI_V1_0) {
        /* for ABIv1.0 and Origin */
        /* load DPU weight from hybrid ELF binary into DPU memory space */
        fseek(fp, kernel->base.elf_weight.offset, SEEK_SET);
        ret = fread(kernel->mem_weight.addr_virt, 1, kernel->base.elf_weight.size, fp);
        dpuCacheFlush(&kernel->mem_weight, 0, kernel->base.elf_weight.size);
        if (ret < kernel->base.elf_weight.size) {
            DPU_FAIL_ON_MSG("failure when reading weight segment for DPU kenrel:%s.", kernel->base.name);
        }

        /* load DPU bias from hybrid ELF binary into DPU memory space */
        fseek(fp, kernel->base.elf_bias.offset, SEEK_SET);
        ret = fread(kernel->mem_bias.addr_virt, 1, kernel->base.elf_bias.size, fp);
        dpuCacheFlush(&kernel->mem_bias, 0, kernel->base.elf_bias.size);
        if (ret < kernel->base.elf_bias.size) {
            DPU_FAIL_ON_MSG("failure when reading weight segment for DPU kenrel:%s.", kernel->base.name);
        }
    } else {
        /* deal with params for ABIv2.0 */
        /* load DPU param from hybrid ELF binary into DPU memory space */
        fseek(fp, kernel->base.elf_param.offset, SEEK_SET);
        ret = fread(kernel->mem_param.addr_virt, 1, kernel->base.elf_param.size, fp);
        dpuCacheFlush(&kernel->mem_param, 0, kernel->base.elf_param.size);
        if (ret < kernel->base.elf_param.size) {
            DPU_FAIL_ON_MSG("failure when reading param segment for DPU kenrel:%s.", kernel->base.name);
        }

    }

    fclose(fp);

    return N2CUBE_SUCCESS;
}


/*
 * update each Node's code/bias/weight/input/output address info
 * after DPU kernel's resource is allocated
 */
INTERNAL int dpu_update_kernel_node(dpu_kernel_t *kernel)
{
    int i;
    dpu_node_t **nodes;

    DPU_ASSERT(kernel, ERR);
    nodes = kernel->base.node_list;

    /* update each Node's code/bias/weights/ physical address */
    for (i=0; i < kernel->base.node_cnt; i++) {
        nodes[i]->ops.update_addr(nodes[i], (kernel_t*)kernel);
    }

    /* for debug purpose */
    if (DPU_DEBUG_LD()) {
        DPU_LOG_MSG("After relocation of DPU kernel \"%s\":", kernel->base.name);
        printf(DPU_LINE);

        if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("%14s%12s%12s\n", "Physical Code", "Bias", "Weight");
        } else {
            printf("Physical address:\n");
            printf("%25s%12s%12s\n", "Section", "Addr", "Size");
        }
        for (i=0; i < kernel->base.node_cnt; i++) {
             nodes[i]->ops.trace_addr_phy(nodes[i], stdout, i);
        }
        printf(DPU_LINE);

        if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("\n%14s%12s%12s\n", "Virtual Code", "Bias", "Weight");
        } else {
            printf("Virtual address:\n");
        }
        for (i=0; i < kernel->base.node_cnt; i++) {
            nodes[i]->ops.trace_addr_virt(nodes[i], stdout, i);
        }

        printf(DPU_LINE);
    }
}

EXPORT int dpu_update_task_virtual_node(dpu_task_t *task)
{
    int i, size;
    uint32_t IOSpacePhy;
    int8_t *IOSpaceVirt;
    struct task_virtual_node_t *headTask;
    dpu_node_v1_virt_t *headKernel;

    /* return if no virtual Nodes */
    if (!task->kernel->base.virt_node_cnt)  return N2CUBE_SUCCESS;

    /* allocate memory space for Task's virtual Node */
    size = sizeof(struct task_virtual_node_t)*(task->kernel->base.virt_node_cnt);
    task->virt_node_list = (struct task_virtual_node_t *)malloc(size);
    memset(task->virt_node_list, 0, size);

    IOSpacePhy = task->mem_IO.addr_phy;
    IOSpaceVirt = task->mem_IO.addr_virt;

    headKernel = task->kernel->virt_node_list;
    headTask = task->virt_node_list;

    for (i=0; i < task->kernel->base.virt_node_cnt; i++) {
        headTask[i].tensorOut.shape =
            &(headKernel[i].base_v1.shapeOut);

        headTask[i].tensorOut.addr_phy =
            headKernel[i].base_v1.shapeOut.offset + IOSpacePhy;

        headTask[i].tensorOut.addr_virt =
            headKernel[i].base_v1.shapeOut.offset + IOSpaceVirt;

        task_tensor_v1_init(&(headTask[i].tensorOut));
    }

    return N2CUBE_SUCCESS;
}

/**
 * Create node_list in task
 * 1. Allocate memory space for node_list and its items.
 * 2. Init node_list items.
 */
INTERNAL void dpu_create_task_node(dpu_task_t *task) {
    int i, size;
    dpu_node_v2_t *node;
    DPU_ASSERT(task, ERR);

    size = sizeof(task_node_t*) * task->kernel->base.node_cnt;
    task->node_list = (task_node_t**)malloc(size);
    memset(task->node_list, 0, size);

    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        /* for ABIv1.0 and Origin*/
        for (i=0; i<task->kernel->base.node_cnt; i++) {
            size = sizeof(task_node_v1_t);
            task->node_list[i] = (task_node_t*)malloc(size);
            memset(task->node_list[i], 0, size);
            task_node_v1_init( task->node_list[i] );
        }
    } else {
        /* for ABI version after v1.6(contine)  */
        for (i=0; i<task->kernel->base.node_cnt; i++) {
            size =  sizeof(task_node_v2_t);
            task->node_list[i] = (task_node_t*)malloc(size);
            memset(task->node_list[i], 0, size);
            node = (dpu_node_v2_t*)(task->kernel->base.node_list[i]);
            task_node_v2_init( task->node_list[i], node->input_cnt, node->output_cnt );
        }
    }
}

/*
 * update each Node's code/bias/weight/input/output address info
 * after DPU kernel's resource is allocated. Virtual node is also
 * processed.
 */
EXPORT int dpu_update_task_node(dpu_task_t *task)
{
    int i;
    task_node_t **taskNode;
    dpu_node_t **nodes;
    mem_segment_t *node_code;

    DPU_ASSERT(task, ERR);

    /* allocate memory space for task core data structure */
    dpu_create_task_node(task);

    nodes = task->kernel->base.node_list;
    taskNode = task->node_list;

    /* update each Node's input/output physical address */
    for (i=0; i < task->kernel->base.node_cnt; i++) {
        taskNode[i]->ops.setup_tensor(taskNode[i],
                                      task,
                                      nodes[i]);
    }

    /* for debug purpose */
    if (DPU_DEBUG_LD()) {
        DPU_LOG_MSG("After memory allocation for DPU Task \"%s\":", task->name);
        printf(DPU_LINE);

        if(task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("%14s%12s%12s%12s%12s\n",
                "Physical-Code", "Bias", "Weight", "Input", "Output");
        } else {
            printf("Physical address:\n");
            printf("%25s%12s%12s\n", "Section", "Addr", "Size");
        }

        for (i=0; i < task->kernel->base.node_cnt; i++) {
            nodes[i]->ops.trace_addr_phy(nodes[i], stdout, i);
            taskNode[i]->ops.dump_addr_phy(taskNode[i], stdout, "  0x%08x  0x%08x\n");
        }
        printf(DPU_LINE);

        if(task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("\n%14s%12s%12s%12s%12s\n",
                "Virtual-Code", "Bias", "Weight", "Input", "Output");
        } else {
            printf("Virtual address:\n");
        }

        for (i=0; i < task->kernel->base.node_cnt; i++) {
            nodes[i]->ops.trace_addr_virt(nodes[i], stdout, i);
            taskNode[i]->ops.dump_addr_virt(taskNode[i], stdout, "  0x%08x  0x%08x\n");
        }

        printf(DPU_LINE);
    }

    return N2CUBE_SUCCESS;
}
