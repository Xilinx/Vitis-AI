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
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>

#include "../../common/dpu_types.h"
#include "dpu_def.h"
#include "dpu_err.h"
#include "dpu_sys.h"
#include "dpu_node_v1.h"
#include "../../common/dpu_node_v2.h"
#include "aol/dpu_aol.h"
#include "dpu_caps.h"
#include "dpu_scheduler.h"
#include "dpu_1to1.h"


/* descriptor for DPU devcie file */
//int dpu_dev_fd      = -1;
dpu_aol_dev_handle_t *gp_dpu_aol_handle = NULL;

#define PAGE_SIZE    sysconf(_SC_PAGE_SIZE)



INTERNAL int dpu_alloc_region_code(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_region_param(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_region_wb(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_region_io(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_region_prof(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_region_prof_fixed(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_code_in_whole(dpu_kernel_t *kernel, int fd);
INTERNAL int dpu_alloc_code_in_node(dpu_kernel_t *kernel, int fd);


/*
 * Attach current process to DPU device driver
 */
EXPORT int dpu_attach()
{
    int flags;

    /* open DPU device if it isn't opened yet */
    if (gp_dpu_aol_handle == NULL) {
        gp_dpu_aol_handle = dpu_aol_attach(DPU_SCHEDULE_MODE_SINGLE);

        /* error check */
        if (gp_dpu_aol_handle == NULL) {
            DPU_FAIL_ON_MSG("fail to open DPU device and exit ...\n");
        }

        if (gp_dpu_aol_handle->aol_version != 0x0100) {
            DPU_FAIL_ON_MSG("The AOL interface version should be v1.0.0 ...\n");
        }
    }

    //reset_dpus(gp_dpu_aol_handle);

    return N2CUBE_SUCCESS;
}


/*
 * Attach current process to DPU device driver
 */
EXPORT int dpu_config()
{
    int ret;
    int idx;
    int fd;
    char buf[16];
    memset(buf,0x0,sizeof(buf));
    unsigned char flag_1to1 = 0;

    //REQ_DPU_CAPS
    ret = get_dpu_caps(gp_dpu_aol_handle, &dpu_caps);
    if (ret == 1) {
        flag_1to1 = 1;
    } else if (ret != 0) {
        return N2CUBE_FAILURE;
    }

    dpu_caps.cache = 1; // fixed to 1

    //check signature
    if (dpu_caps.signature_valid) {
        N2CUBE_DPU_CHECK(dpu_caps.dpu_cnt>0, N2CUBE_ERR_DPU_NONE, "");
        if (dpu_caps.dpu_cnt>0) {
            // Configurable IP.
            if (dpu_caps.magic == DPU_CONF_MAGIC) {
                dpu_caps.p_dpu_info = malloc(sizeof(dpu_configurable_t)*dpu_caps.dpu_cnt);

                if (flag_1to1) {
                    ret = get_dpu_info_1to1(gp_dpu_aol_handle, (dpu_configurable_t *)dpu_caps.p_dpu_info, dpu_caps.dpu_cnt);
                } else {
                    ret = get_dpu_info_v1(gp_dpu_aol_handle, (dpu_configurable_t *)dpu_caps.p_dpu_info, dpu_caps.dpu_cnt);
                }
                if (ret != 0) {
                    return N2CUBE_FAILURE;
                }

                dpu_configurable_t *p_dpu_info = (dpu_configurable_t*)(dpu_caps.p_dpu_info);
                for (idx=0; idx<dpu_caps.dpu_cnt; idx++, p_dpu_info++) {
                    p_dpu_info->base.peak_perf = p_dpu_info->base.dpu_freq*p_dpu_info->base.dpu_arch/1000.0f;
                }
             } else {
                dpu_caps.p_dpu_info = malloc(sizeof(dpu_info_t)*dpu_caps.dpu_cnt);
                N2CUBE_DPU_CHECK(dpu_caps.p_dpu_info!=NULL, N2CUBE_ERR_MALLOC_DPU_CAPABILITY, "");

                ret = get_dpu_info_v0(gp_dpu_aol_handle, (dpu_info_t *)dpu_caps.p_dpu_info, dpu_caps.dpu_cnt);
                if (ret != 0) {
                    return N2CUBE_FAILURE;
                }
                ret = check_signature_default_v0(gp_dpu_aol_handle);
                if (ret != 0) {
                    return N2CUBE_FAILURE;
                }

                dpu_info_t *p_dpu_info = (dpu_info_t*)(dpu_caps.p_dpu_info);
                N2CUBE_DPU_CHECK(p_dpu_info->dpu_target < DPU_TARGET_RESERVE, N2CUBE_ERR_DPU_TARGET,
                  ". target number: %d", p_dpu_info->dpu_target );
                N2CUBE_DPU_CHECK(p_dpu_info->base.dpu_arch < DPU_ARCH_RESERVE, N2CUBE_ERR_DPU_ARCH,
                  ". arch number: %d", p_dpu_info->base.dpu_arch);

				for (idx=0; idx<dpu_caps.dpu_cnt; idx++) {
                    switch ((p_dpu_info+idx)->base.dpu_arch) {
                        case DPU_ARCH_B1024F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*1024/1000.0f;
                            break;
                        case DPU_ARCH_B1152F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*1152/1000.0f;
                            break;
                        case DPU_ARCH_B4096F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*4096/1000.0f;
                            break;
                        case DPU_ARCH_B256F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*256/1000.0f;
                            break;
                        case DPU_ARCH_B512F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*512/1000.0f;
                            break;
                        case DPU_ARCH_B800F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*800/1000.0f;
                            break;
                        case DPU_ARCH_B1600F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*1600/1000.0f;
                            break;
                        case DPU_ARCH_B2048F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*2048/1000.0f;
                            break;
                        case DPU_ARCH_B2304F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*2304/1000.0f;
                            break;
                        case DPU_ARCH_B8192F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*8192/1000.0f;
                            break;
                        case DPU_ARCH_B3136F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*3136/1000.0f;
                            break;
                        case DPU_ARCH_B288F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*288/1000.0f;
                            break;
                        case DPU_ARCH_B144F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*144/1000.0f;
                            break;
                        case DPU_ARCH_B5184F:
                            (p_dpu_info+idx)->base.peak_perf = (p_dpu_info+idx)->base.dpu_freq*5184/1000.0f;
                            break;
                        default:
                            DPU_FAIL_ON_MSG("Unknown arch number %d found in DPU IP.", (p_dpu_info+idx)->base.dpu_arch);
                    }
                }
            }
        }
    }

    return N2CUBE_SUCCESS;
}


/*
 * Dettach current process from DPU device driver
 */
EXPORT int dpu_dettach()
{
    int ret;

    if(gp_dpu_aol_handle != NULL) {
        ret = dpu_aol_detach(gp_dpu_aol_handle);
        gp_dpu_aol_handle = NULL;
        if (ret) {
            DPU_FAIL_ON_MSG("Fail to close DPU device.");
        }
    }

    if (dpu_caps.p_dpu_info) {
        free(dpu_caps.p_dpu_info);
        dpu_caps.p_dpu_info = NULL;
    }

    return N2CUBE_SUCCESS;
}

/*
 * Invoke DPU driver to run command via system call into ioctl
 */
EXPORT long dpu_launch_execution_session(dpu_kernel_t *kernel, dpu_task_t *task, char *nodeName, dpu_aol_run_t *session)
{
    int i = 0;
    long ret;
    uint32_t core_mask;

    DPU_ASSERT(session && task, ERR);

    if(kernel->base.dpu_target_ver == DPU_TARGET_V1_1_3) {
        session->regs[0].value >>= 12;
        session->regs[1].value >>= 12;
        session->regs[2].value >>= 12;
        session->regs[0].offset = 0x208; // addr_weight
        session->regs[1].offset = 0x204; // addr_io
        session->regs[2].offset = 0x20C; // addr_code
    } else {
        if (dpu_caps.signature_version >= 2) { // 1 to 1
            session->regs[session->reg_count - 1].offset = OFFSET_1t01_DPU_INSTR_ADDR_L;
        } else {
            session->regs[session->reg_count - 1].value >>= 12;
            session->regs[session->reg_count - 1].offset = 0x20C; // addr_code
        }
    }

    core_mask = dpu_scheduler_get_available_core_mask(task);
    session->core_mask = core_mask;
    session->ip_id = IP_ID_DPU;
    session->timeout = dpu_get_n2cube_timeout();
    ret = dpu_aol_run(gp_dpu_aol_handle, session);
    dpu_scheduler_release_dpu_core(core_mask, session->time_start, session->time_end);

    if (DPU_AOL_ERROR == ret) {
        if (!nodeName) {
            /* Run Task in normal mode */
            display_dpu_debug_info();
            reset_dpus(gp_dpu_aol_handle);
            printf("\n");
            DPU_FAIL_ON_MSG("DPU timeout while execute DPU Task:%s", task->name);
        } else {
            dpu_dump_node_when_timeout(task, nodeName);

            /* Run Task in dubug mode for a specified Node */
            display_dpu_debug_info();
            reset_dpus(gp_dpu_aol_handle);
            printf("\n");
            DPU_FAIL_ON_MSG("DPU timeout while execute DPU Task [%s] of Node [%s]", task->name, nodeName);
        }
    }

    return N2CUBE_SUCCESS;
}

INTERNAL void print_kernel_memory_map_v2(dpu_kernel_t *kernel)
{
    if (DPU_DEBUG_DRV()) {
        DPU_LOG_MSG("Allocate memory for DPU Kernel \"%s\" of Code/Param:",
            kernel->base.name);
        printf(DPU_LINE);

        printf("%24s%12s%14s\n","Size", "Phy-addr", "Virt-addr");
        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Code",
            kernel->mem_code.size,
            kernel->mem_code.addr_phy,
            (unsigned long)kernel->mem_code.addr_virt);

        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Param",
            kernel->mem_param.size,
            kernel->mem_param.addr_phy,
            (unsigned long)kernel->mem_param.addr_virt);

        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Profiler",
            kernel->mem_prof.size,
            kernel->mem_prof.addr_phy,
            (unsigned long)kernel->mem_prof.addr_virt);

        printf(DPU_LINE);

    }
}


INTERNAL void print_kernel_memory_map(dpu_kernel_t *kernel)
{
    if (DPU_DEBUG_DRV()) {
        DPU_LOG_MSG("Allocate memory for DPU Kernel \"%s\" of Code/Weight/Bias:",
            kernel->base.name);
        printf(DPU_LINE);

        printf("%24s%12s%14s\n","Size", "Phy-addr", "Virt-addr");
        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Code",
            kernel->mem_code.size,
            kernel->mem_code.addr_phy,
            (unsigned long)kernel->mem_code.addr_virt);

        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Weight",
            kernel->mem_weight.size,
            kernel->mem_weight.addr_phy,
            (unsigned long)kernel->mem_weight.addr_virt);

        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Bias",
            kernel->mem_bias.size,
            kernel->mem_bias.addr_phy,
            (unsigned long)kernel->mem_bias.addr_virt);

        printf("%12s  0x%08x  0x%08x  0x%010lx\n",
            "Profiler",
            kernel->mem_prof.size,
            kernel->mem_prof.addr_phy,
            (unsigned long)kernel->mem_prof.addr_virt);

        printf(DPU_LINE);

    }
}

/*
 * Perform DPU memory allocation via system call into driver module
 * @mem: the size of memory to be allocated. MUST be multiple of the system page
 *       size.
 * Return value: the physical address of DPU memory
 */
EXPORT int dpu_alloc_kernel_resource(dpu_kernel_t *kernel)
{
    int mm_fd = open(DPU_DEV_NAME, O_RDWR|O_SYNC, 0);

    /* allocate memory space for Code/Weight/Bias/Profiler */
    dpu_alloc_region_code(kernel, mm_fd);
    if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
        dpu_alloc_region_wb(kernel, mm_fd);
    } else {
        dpu_alloc_region_param(kernel, mm_fd);
    }

#if 0
    /* memory space for DPU profile is allocated to fixed position by DPU Driver */
	dpu_alloc_region_prof_fixed(kernel, mm_fd);
#endif

    /* Display Kernel's memory mapping info */
    if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
        print_kernel_memory_map(kernel);
    } else {
        print_kernel_memory_map_v2(kernel);
    }

    /* close memory device file */
    close(mm_fd);

    return N2CUBE_SUCCESS;

}

INTERNAL int dpu_alloc_region_code(dpu_kernel_t *kernel, int mm_fd)
{
    if (KERNEL_IN_DEBUG(kernel)) {
        dpu_alloc_code_in_node(kernel, mm_fd);
    } else {
        dpu_alloc_code_in_whole(kernel, mm_fd);
    }
}

INTERNAL int dpu_alloc_code_in_whole(dpu_kernel_t *kernel, int mm_fd)
{
    if (KERNEL_IN_DEBUG(kernel)) {
        DPU_FAIL_ON_MSG("Code of DPU kernel in debug mode %s should be allocated memory space by Node.\n",
            kernel->base.name);
    }

    /* note: due to the constrains from function mmap, memory allocation size for
     * DPU code must be a multiple of system page size
     */
    kernel->mem_code.length = kernel->base.elf_code.size;
	kernel->region_code.region_start  = &(kernel->mem_code);
    kernel->region_code.region_size = kernel->mem_code.length;

    /* update the size of code segment due to page size alignment */
	kernel->mem_code.size = kernel->region_code.region_size;

    /* allocate DPU memory for kernel code */
    if (dpu_dev_mem_alloc(kernel->region_code.region_start, kernel->region_code.region_size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for DPU Kernel %s: Size: %d",
            kernel->base.name, kernel->region_code.region_size);
    }

    memset(kernel->region_code.region_start->addr_virt, 0, kernel->region_code.region_size);
    dpuCacheFlush(kernel->region_code.region_start, 0, kernel->region_code.region_size);

    return N2CUBE_SUCCESS;
}

INTERNAL int dpu_alloc_code_in_node(dpu_kernel_t *kernel, int mm_fd)
{
    int i;
    dpu_node_t **nodes;
    mem_segment_t *seg_code, *code_begin;

    if (!KERNEL_IN_DEBUG(kernel)) {
        DPU_FAIL_ON_MSG("code of DPU kernel in non-debug mode %s should be allocated memory space as whole.\n",
            kernel->base.name);
    }

    nodes = kernel->base.node_list;
    code_begin = nodes[0]->ops.get_node_code(nodes[0]);
    kernel->region_code.region_size = 0;
    kernel->region_code.region_start = code_begin;

    for (i=0; i < kernel->base.node_cnt; i++) {
        if( seg_code = nodes[i]->ops.get_node_code(nodes[i]) ) {
            /* allocate DPU memory for code Node */
            if (dpu_dev_mem_alloc(seg_code, seg_code->size) != 0) {
                DPU_FAIL_ON_MSG("Fail to alloc memory for DPU Kernel %s of Layer %s: Size: %d",
                    kernel->base.name, nodes[i]->name, seg_code->size);
            }

            memset(seg_code->addr_virt, 0, seg_code->size);
            dpuCacheFlush(seg_code, 0, seg_code->size);

            kernel->region_code.region_size += seg_code->size;
        }
    }

    kernel->mem_code.addr_phy = code_begin->addr_phy;
    kernel->mem_code.addr_virt = code_begin->addr_virt;
    kernel->mem_code.length = kernel->region_code.region_size;
    kernel->mem_code.size = kernel->region_code.region_size;

    return N2CUBE_SUCCESS;

}

INTERNAL int dpu_alloc_region_param(dpu_kernel_t *kernel, int mm_fd)
{
    /* for kernel weights/bias section */
    kernel->mem_param.length = kernel->base.elf_param.size;
    kernel->mem_param.size = kernel->base.elf_param.size;

    kernel->region_param.region_size = kernel->mem_param.size;
    kernel->region_param.region_start = &(kernel->mem_param);
    kernel->region_param.region_size = kernel->region_param.region_size;

    /* update the size of bias segment */
    kernel->mem_param.size = kernel->region_param.region_size;

    /* allocate DPU memory for kernel data
       NOTE: weight/bias are designed as address contiguous.
       we allocate DPU memory address for them in one memory segment */
    if (dpu_dev_mem_alloc(kernel->region_param.region_start, kernel->region_param.region_size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for DPU Kernel %s: Size: %d",
            kernel->base.name, kernel->region_param.region_size);
    }

    memset(kernel->region_param.region_start->addr_virt, 0, kernel->region_param.region_size);
    dpuCacheFlush(kernel->region_param.region_start, 0, kernel->region_param.region_size);

    return N2CUBE_SUCCESS;
}


INTERNAL int dpu_alloc_region_wb(dpu_kernel_t *kernel, int mm_fd)
{
    int8_t *region_virt;
    uint32_t region_phy;

    /* for kernel weights/bias section */
    kernel->mem_bias.length = kernel->base.elf_bias.size;
    kernel->mem_bias.size = kernel->base.elf_bias.size;

    kernel->mem_weight.length = kernel->base.elf_weight.size;
    kernel->mem_weight.size = kernel->base.elf_weight.size;

    kernel->region_wb.region_size = kernel->mem_weight.size + kernel->mem_bias.size;
    kernel->region_wb.region_start = &(kernel->mem_weight);

    kernel->region_wb.region_size = kernel->region_wb.region_size;

    /* update the size of bias segment */
	kernel->mem_bias.size = kernel->region_wb.region_size - kernel->mem_weight.size;

    /* allocate DPU memory for kernel data
       NOTE: weight/bias are designed as address contiguous.
       we allocate DPU memory address for them in one memory segment */
    if (dpu_dev_mem_alloc(kernel->region_wb.region_start, kernel->region_wb.region_size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for DPU Kernel %s: Size: %d",
            kernel->base.name, kernel->region_wb.region_size);
    }

    /* set the address info for DPU bias layer */
    region_phy = kernel->region_wb.region_start->addr_phy;
    region_virt = kernel->region_wb.region_start->addr_virt;
    kernel->mem_bias.addr_phy =  region_phy + kernel->mem_weight.length;
    kernel->mem_bias.addr_virt = (int8_t*)(region_virt + kernel->mem_weight.length);

    memset(kernel->region_wb.region_start->addr_virt, 0, kernel->region_wb.region_size);
    dpuCacheFlush(kernel->region_wb.region_start, 0, kernel->region_wb.region_size);

    return N2CUBE_SUCCESS;
}

INTERNAL int dpu_alloc_region_prof(dpu_kernel_t *kernel, int mm_fd)
{
    /* set memory size for DPU profiler */
    kernel->mem_prof.type = MEM_PROF;
    kernel->mem_prof.size = MEM_SIZE_PROF;
    kernel->mem_prof.length = kernel->mem_prof.size;

    /* memory allocation size must be a multiple of system page size */
	kernel->region_prof.region_size = kernel->mem_prof.size;
	kernel->region_prof.region_start  = &(kernel->mem_prof);

    kernel->region_prof.region_size = kernel->region_prof.region_size;


    /* update the size of profiler segment */
	kernel->mem_prof.size = kernel->region_prof.region_size;

    /* allocate DPU memory for DPU profiler */
    if (dpu_dev_mem_alloc(kernel->region_prof.region_start, kernel->region_prof.region_size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for DPU Kernel %s: Size: %d",
            kernel->base.name, kernel->region_prof.region_size);
    }

    /* clear the allocated DPU memory segment */
    memset(kernel->region_prof.region_start->addr_virt, 0, kernel->region_prof.region_size);
    dpuCacheFlush(kernel->region_prof.region_start, 0, kernel->region_prof.region_size);

    return N2CUBE_SUCCESS;
}

INTERNAL int dpu_alloc_region_prof_fixed(dpu_kernel_t *kernel, int mm_fd)
{
    void *ret;
    int8_t *region_virt;
    uint32_t region_phy;


    /* set memory size for DPU profiler */
    kernel->mem_prof.type = MEM_PROF;
    kernel->mem_prof.size = MEM_SIZE_PROF;
    kernel->mem_prof.length = MEM_SIZE_PROF;

    /* memory allocation size must be a multiple of system page size */
	kernel->region_prof.region_size = kernel->mem_prof.size;
	kernel->region_prof.region_start  = &(kernel->mem_prof);
    kernel->region_prof.region_size = kernel->mem_prof.size;

    region_phy = 0x50000000; //suv

    ret = mmap(NULL, kernel->region_prof.region_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, mm_fd, (unsigned long)region_phy);

    if (ret == MAP_FAILED) {
        close(mm_fd);
        DPU_FAIL_ON_MSG("Fail to map memory for DPU Kernel %s: Address: 0x%x, Size: %d",
            kernel->base.name, region_phy, kernel->region_prof.region_size);
    }

	region_virt = (int8_t*)ret;
    kernel->region_prof.region_start->addr_phy = region_phy;
    kernel->region_prof.region_start->addr_virt = region_virt;

    /* clear the allocated DPU memory segment */
    memset(kernel->region_prof.region_start->addr_virt, 0, kernel->region_prof.region_size);
    dpuCacheFlush(kernel->region_prof.region_start, 0, kernel->region_prof.region_size);

    return N2CUBE_SUCCESS;
}

int dpu_setup_task_boundary_tensor(dpu_task_t *task,
                                    tensor_attr_t type) {
  int i, count = 0;
  tensor_shape_t *tensor, *tensors;
  N2CUBE_DPU_CHECK(task, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
  N2CUBE_DPU_CHECK(type & (TENSOR_ATTR_BOUNDRY_INPUT | TENSOR_ATTR_BOUNDRY_OUTPUT),
                                N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
  dpu_kernel_t *kernel = task->kernel;
  tensors = kernel->base.tensor_list;
  for (i = 0; i < kernel->base.tensor_cnt; i++) {
    count = (tensors[i].attr == type) ? (count + 1) : count;//count out all the tensors with "type" attr.
  }
  N2CUBE_DPU_CHECK(count < kernel->base.tensor_cnt, ERR, " invalid IO count for API %s", __func__);
  if (count == 0) return N2CUBE_SUCCESS;

  DPUTensorAttr *tAttr = (DPUTensorAttr*)malloc(count * sizeof(DPUTensorAttr));
  if (type == TENSOR_ATTR_BOUNDRY_INPUT) {
    task->inputTensorAttrs = tAttr;
    task->inputTensorNum = count;
  } else {
    task->outputTensorAttrs = tAttr;
    task->outputTensorNum = count;
  }

  count = 0;
  for (i = 0; i < kernel->base.tensor_cnt; i++) {
    if (tensors[i].attr == type) {
      tensor = &(tensors[i]);
      tAttr[count].addr_virt = task->mem_IO.addr_virt + tensor->offset;
      tAttr[count].size = tensor->size;
      tAttr[count].shape.n = 1;
      tAttr[count].shape.h = tensor->height;
      tAttr[count].shape.w = tensor->width;
      tAttr[count].shape.c = tensor->channel;
      tAttr[count].fix.width = tensor->fix_width;
      tAttr[count].fix.pos = tensor->fix_pos;
      tAttr[count].tensor_name = tensor->tensor_name;
      if (type == TENSOR_ATTR_BOUNDRY_INPUT) {
        tAttr[count].scale = (float)(pow(2, tensor->fix_pos));
      } else {
        tAttr[count].scale = (float)(pow(2, -(tensor->fix_pos)));
      }

      count++;
    }
  }
  return N2CUBE_SUCCESS;
}

/*
 * Allocate DPU memory for task's input/output sections
 * @param task - pointer to the task
 *
 * Return value: the physical address of DPU memory
 */
EXPORT int dpu_alloc_task_resource(dpu_task_t *task)
{
    dpu_kernel_t *kernel;

    kernel = task->kernel;

    /* for task input/output section
       note: not for the memory space size for input/output */
    task->mem_IO.length = kernel->base.IO_space_size;
    task->mem_IO.size = task->mem_IO.length;

    /* set the address info for DPU input/output layer
       NOTE1: the size/length info for them is not specified.
       NOTE2: virtual address should be updated before physical address.
    */
   if (dpu_dev_mem_alloc(&(task->mem_IO), task->mem_IO.size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for Task %s of DPU Kernel %s: Size: %d",
            task->name, kernel->base.name, task->mem_IO.size);
    }

    /* clear the allocated DPU memory segment */
    memset(task->mem_IO.addr_virt, 0xff, task->mem_IO.size);
    dpuCacheFlush(&task->mem_IO, 0, task->mem_IO.size);

    // setup boundary I/O tensor attributes
    dpu_setup_task_boundary_tensor(task, TENSOR_ATTR_BOUNDRY_INPUT);
    dpu_setup_task_boundary_tensor(task, TENSOR_ATTR_BOUNDRY_OUTPUT);
    return N2CUBE_SUCCESS;
}

INTERNAL void dpu_release_kernel_node(dpu_kernel_t *kernel) {
    int i;
    dpu_node_t **nodes = kernel->base.node_list;
    dpu_node_v1_virt_t * vnodes = kernel->virt_node_list;

    if (nodes) {
        for(i=0; i<kernel->base.node_cnt; i++) {
            nodes[i]->ops.release(nodes[i]);
            free(nodes[i]);
        }
        free(kernel->base.node_list);
        kernel->base.node_list = 0;
    }

    if (kernel->virt_node_list) {
        for(i=0; i<kernel->base.virt_node_cnt; i++) {
            vnodes[i].base_v1.base.ops.release((dpu_node_t*)&(vnodes[i]));
        }
        free(kernel->virt_node_list);
        kernel->virt_node_list = 0;
    }
}

INTERNAL void dpu_release_task_node (dpu_task_t *task) {
    int i;
    task_node_t **tn = task->node_list;

    if(tn) {
        for(i=0; i<task->kernel->base.node_cnt; i++) {
            tn[i]->ops.release(tn[i]);
            free(tn[i]);
        }
        free(tn);
        task->node_list = 0;
    }

    if(task->virt_node_list) {
        task_tensor_t *tensor = &(task->virt_node_list->tensorOut);
        tensor->ops.release(tensor);
        free(task->virt_node_list);
    }
    task->virt_node_list = 0;
}

/*
 * Release the kernel's resource from DPU driver
 */
EXPORT int dpu_release_kernel_resource(dpu_kernel_t *kernel)
{
    int i, ret;
    dpu_node_t **nodes = kernel->base.node_list;
    mem_segment_t * node_code;

    if (KERNEL_IN_DEBUG(kernel)) {
        /* remove each layer's code segment */
        for (i=0; i < kernel->base.node_cnt; i++) {
            if(node_code = nodes[i]->ops.get_node_code(nodes[i])) {
                ret = munmap(node_code->addr_virt, node_code->size);
            }
        }
    } else {
        ret = munmap(kernel->region_code.region_start->addr_virt,
            kernel->region_code.region_size);
    }

    /* remove the mapped DPU memory space from process's memory area */
    if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
        ret = munmap(kernel->region_wb.region_start->addr_virt, kernel->region_wb.region_size);
    } else {
        ret = munmap(kernel->region_param.region_start->addr_virt, kernel->region_param.region_size);
    }

#if 0
    ret = munmap(kernel->region_prof.region_start->addr_virt, kernel->region_prof.region_size);
#endif

    /* release DPU memory space */
    if (KERNEL_IN_DEBUG(kernel)) {
        /* free each layer's code segment memory */
        for (i=0; i < kernel->base.node_cnt; i++) {
            if(node_code = nodes[i]->ops.get_node_code(nodes[i])) {
                dpu_dev_mem_free(node_code);
            }
        }
    } else {
        dpu_dev_mem_free(kernel->region_code.region_start);
    }

    if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
        dpu_dev_mem_free(kernel->region_wb.region_start);
    } else {
        dpu_dev_mem_free(kernel->region_param.region_start);
    }

    dpu_release_kernel_node(kernel);

    for (i = 0; i < kernel->base.tensor_cnt; i++) {
        if(kernel->base.tensor_list[i].tensor_name) {
            free(kernel->base.tensor_list[i].tensor_name);
            kernel->base.tensor_list[i].tensor_name = NULL;
        }
    }
    free(kernel->base.tensor_list);
    kernel->base.tensor_list = 0;

    return N2CUBE_SUCCESS;
}

EXPORT int dpu_release_task_resource(dpu_task_t *task)
{
    dpu_dev_mem_free(&task->mem_IO);

    /* release memory space of task core data structure */
    if (task->inputTensorAttrs && (task->inputTensorNum > 0)) {
      free(task->inputTensorAttrs);
      task->inputTensorAttrs = NULL;
    }
    if (task->outputTensorAttrs && (task->outputTensorNum > 0)) {
      free(task->outputTensorAttrs);
      task->outputTensorAttrs = NULL;
    }
    dpu_release_task_node(task);

    return N2CUBE_SUCCESS;
}

EXPORT int dpuCacheFlush(mem_segment_t *seg, uint32_t offset, uint32_t size)
{
    if (seg->p_dev_mem) {
        dpu_aol_sync_to_dev(gp_dpu_aol_handle, seg->p_dev_mem, offset, size);
    }

    return N2CUBE_SUCCESS;
}


EXPORT int dpuCacheInvalid(mem_segment_t *seg, uint32_t offset, uint32_t size)
{
    if (seg->p_dev_mem) {
        dpu_aol_sync_from_dev(gp_dpu_aol_handle, seg->p_dev_mem, offset, size);
    }

    return N2CUBE_SUCCESS;
}

int dpu_dev_mem_alloc(mem_segment_t *seg, uint32_t size) {
    seg->p_dev_mem = dpu_aol_alloc_dev_mem(gp_dpu_aol_handle, size, DPU_AOL_MEM_PROT_READ | DPU_AOL_MEM_PROT_WRITE);
    if (seg->p_dev_mem == NULL)  {
        return -1;
    }

    seg->size = seg->p_dev_mem->size;
    seg->addr_phy = (uint32_t)seg->p_dev_mem->addr_phy;
    seg->addr_virt = (int8_t*)seg->p_dev_mem->addr_virt;

    return 0;
}

int dpu_dev_mem_free(mem_segment_t *seg) {
    return dpu_aol_free_dev_mem(gp_dpu_aol_handle, seg->p_dev_mem);
}

EXPORT void* dpuMemset(void *dest, int value, size_t size)
{
    int i;
    char *start;

    DPU_ASSERT(dest && size, ERR);

    if (DPU_DEBUG_LOG()) {
        DPU_LOG_MSG("DPU version memset used.");
    }

    start = (char *)dest;

    for (i=0; i<size; i++) {
        start[i] = (char)value;
    }

    return dest;
}


EXPORT void *dpuMemcpy(void *dest, const void *src, size_t size)
{
    int i;
    char *pdest, *psrc;

    DPU_ASSERT(dest && src && size, ERR);

    pdest = (char*)dest;
    psrc = (char*)src;

    if (DPU_DEBUG_LOG()) {
        DPU_LOG_MSG("DPU version memcpy used.");
    }

    for (i=0; i<size; i++) {
        pdest[i] = psrc[i];
    }

    return dest;
}

#ifdef fread
#undef fread
#endif
EXPORT size_t dpuFread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    int ret;
    char *buf, *pdest, *psrc;

    DPU_ASSERT(ptr && (size==1) && nmemb && stream, ERR);

    if (DPU_DEBUG_LOG()) {
        ;//DPU_LOG_MSG("DPU version fread used.");
    }

    buf = (char*)malloc(nmemb);

    ret = fread(buf, 1, nmemb, stream);
    dpuMemcpy((char*)ptr, buf, nmemb);

    free(buf);

    return nmemb;
}

#ifdef fwrite
#undef fwrite
#endif
EXPORT size_t dpuWrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    int i;
    char *buf, *pdest, *psrc;

    DPU_ASSERT(ptr && (size==1) && nmemb && stream, ERR);

    if (DPU_DEBUG_LOG()) {
        DPU_LOG_MSG("DPU version fwrite used.");
    }

    buf = (char*)malloc(nmemb);
    dpuMemset(buf, 0, nmemb);
    dpuMemcpy(buf, (char*)ptr, nmemb);
    fwrite(buf, 1, nmemb, stream);

    free(buf);

    return nmemb;
}

// Called by dexplorer
int display_dpu_debug_info(void) {
    int32_t ret;

    if (gp_dpu_aol_handle == NULL) {
        return -1;
    }

    /* display DPU core status */
    printf("[DPU mode]\n%s\n", dpu_get_n2cube_mode());

    /* display DPU core status */
    printf("\n[DPU timeout limitation (in seconds)]\n%lu\n", dpu_get_n2cube_timeout());

    /* display DPU core status */
    printf("\n");

    ret = get_dpu_info(gp_dpu_aol_handle, &dpu_caps);

    return ret;
}

uint32_t dpu_cache_status(void) {
    return dpu_caps.cache;
}
