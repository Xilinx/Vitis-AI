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
/**
 * dpu_elf.c
 *
 * Read various sections for DPU kernel from hybrid CPU-DPU ELF executable
 */
#ifdef __cplusplus
extern "C" {
#endif
#include "../../common/dpu_types.h"
#include "dpu_def.h"
#include "dpu_err.h"
#include "dpu_elf.h"
#include "dpu_node_v1_real.h"
#include "../../common/dpu_node_v2.h"

#include <ctype.h>
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/*
 * Entry offset for node section in unit of 32-bit length
 *
 * Format of Section ".deephi.node.netName"
 *
 * 0 - 32bit: Magic number ( 0 for real node; otherwise string index to deephi.strtab.kernel-name)
 * 1 - 64bit: Workload
 * 2 - 32bit: Input height
 * 3 - 32bit: Input width
 * 4 - 32bit: Input Channel
 * 5 - 32bit: Input address
 * 6 - 32bit: Input size
 * 7 - 32bit: Input-fixed-width (unsigned char)
 * 8 - 32bit: Input-fixed-pos (signed char)
 * 9 - 32bit: Output height
 * 10 - 32bit: Output width
 * 11 - 32bit: Output  channel
 * 12 - 32bit: Output address
 * 13 - 32bit: Output size
 * 14 - 32bit: Output-fixed-width  (unsigned char)
 * 15 - 32bit: Output-fixed-pos (signed char)
 * 16 - 32bit: Weights-fixed-width  (unsigned char)
 * 17 - 32bit: Weights-fixed-pos (signed char)
 * 18 - 32bit: Bias-fixed-width  (unsigned char)
 * 19 - 32bit: Bias-fixed-pos (signed char)
 *
 */
const int OFF_NODE_MAGIC              = 0;
const int OFF_NODE_WORKLOAD           = 1;
const int OFF_NODE_WORKLOAD_2         = 2;

/* for input tensor */
const int OFF_NODE_INPUT_HEIGHT       = 3;
const int OFF_NODE_INPUT_WIDTH        = 4;
const int OFF_NODE_INPUT_CHANNEL      = 5;
const int OFF_NODE_INPUT_ADDR         = 6;
const int OFF_NODE_INPUT_SIZE         = 7;
const int OFF_NODE_INPUT_FIX_WIDTH    = 8;
const int OFF_NODE_INPUT_FIX_POS      = 9;

/* for output tensor */
const int OFF_NODE_OUTPUT_HEIGHT      = 10;
const int OFF_NODE_OUTPUT_WIDTH       = 11;
const int OFF_NODE_OUTPUT_CHANNEL     = 12;
const int OFF_NODE_OUTPUT_ADDR        = 13;
const int OFF_NODE_OUTPUT_SIZE        = 14;
const int OFF_NODE_OUTPUT_FIX_WIDTH   = 15;
const int OFF_NODE_OUTPUT_FIX_POS     = 16;

/* fix info for weight/bias */
const int OFF_NODE_WEIGHT_FIX_WIDTH   = 17;
const int OFF_NODE_WEIGHT_FIX_POS     = 18;
const int OFF_NODE_BIAS_FIX_WIDTH     = 19;
const int OFF_NODE_BIAS_FIX_POS       = 20;

INTERNAL int dpu_setup_kernel_node(dpu_kernel_t *kernel, elf_t *elf);
INTERNAL int dpu_setup_node_metric(dpu_kernel_t * kernel, int nodeID);
INTERNAL int dpu_setup_kernel_virtual_node(dpu_kernel_t *kernel, elf_t *elf);
INTERNAL int dpu_setup_kernel_info(dpu_kernel_t *kernel, elf_t *elf);
INTERNAL int dpu_setup_kernel_node_v2_0(dpu_kernel_t *kernel, elf_t *elf);
INTERNAL int dpu_setup_configurable(dpu_kernel_t *kernel, elf_t *elf);

int (*setup_kernel_node) (dpu_kernel_t*, elf_t*);
void (*dump_kernel_elf) (dpu_kernel_t*);

/* setup function version to execute according to ABI version. */
void setup_func_ver(const uint32_t ver) {
    SETUP_FUNC_VER(ver, setup_kernel_node, dpu_setup_kernel_node, dpu_setup_kernel_node_v2_0);
}

/*
 * Check version between elf and dpu bit
 */
int version_check(dpu_kernel_t *kernel) {
    N2CUBE_DPU_CHECK(dpu_caps.magic != DPU_CONF_MAGIC,
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,
                     " for kernel %s.\n"\
                     "Please update DNNC to configurable version v3.0 and rebuild network to run on this DPU.",
                     kernel->base.name);
    dpu_info_t *p_dpu_info = (dpu_info_t*)(dpu_caps.p_dpu_info);

    N2CUBE_DPU_CHECK(kernel->base.dpu_arch_ver < DPU_ARCH_RESERVE, N2CUBE_ERR_DPU_ARCH,
              ". DPU arch version: %d, DPU kernel: %s",
              kernel->base.dpu_arch_ver, kernel->base.name);

    N2CUBE_DPU_CHECK(p_dpu_info->base.dpu_arch == kernel->base.dpu_arch_ver, N2CUBE_ERR_DPU_ARCH_MISMATCH,
              ". kernel: %s, DPU kernel: %s, DPU IP: %s",
              kernel->base.name,
              g_dpu_arch_name[kernel->base.dpu_arch_ver],
              g_dpu_arch_name[p_dpu_info->base.dpu_arch]);


    N2CUBE_DPU_CHECK(kernel->base.dpu_target_ver < DPU_TARGET_RESERVE, N2CUBE_ERR_DPU_TARGET,
              ". target version: %d, DPU kernel: %s",
              kernel->base.dpu_target_ver, kernel->base.name);

    N2CUBE_DPU_CHECK(p_dpu_info->dpu_target == kernel->base.dpu_target_ver, N2CUBE_ERR_DPU_TARGET_MISMATCH,
              ". kernel: %s, DPU kernel: %s, DPU IP: %s.",
              kernel->base.name,
              g_dpu_target_name[kernel->base.dpu_target_ver],
              g_dpu_target_name[p_dpu_info->dpu_target]);

    return N2CUBE_SUCCESS;
}

int configurable_check(dpu_kernel_t *kernel) {
    N2CUBE_DPU_CHECK(dpu_caps.magic == DPU_CONF_MAGIC,
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,
                     " for kernel %s.\n"\
                     "Please update the BOOT.BIN to version above v1.4.0, and update driver to version above v3.0.0",
                     kernel->base.name);

    dpu_configurable_t *elf_conf = &(kernel->base.dpu_conf);
    dpu_configurable_t *dpu_conf = (dpu_configurable_t *)(dpu_caps.p_dpu_info);
    DPU_CONFIG_CHECK_FIELD(elf_conf->sys.sys_ip_type, dpu_conf->sys.sys_ip_type, "DPU IP Type");
    DPU_CONFIG_CHECK_FIELD(elf_conf->sys.sys_regmap_ver, dpu_conf->sys.sys_regmap_ver, "DPU Regmap Version");

    //DPU_CONFIG_CHECK_FIELD(elf_conf->sub_version.ver_target, dpu_conf->sub_version.ver_target, "DPU Target Version");
    if (elf_conf->sub_version.ver_target != dpu_conf->sub_version.ver_target) {
        uint32_t elf_target_ver = elf_conf->sub_version.ver_target;
        uint32_t ip_target_ver = dpu_conf->sub_version.ver_target;
        N2CUBE_DPU_CHECK(elf_target_ver == ip_target_ver,
                         N2CUBE_ERR_DPU_CONFIG_MISMATCH,
                         " for kernel %s - parameter: DPU Target Version, DPU kernel: v%x.%x.%x, DPU IP: v%x.%x.%x.",
                         kernel->base.name,
                         (elf_target_ver & 0xf00) >> 8, (elf_target_ver & 0xf0) >> 4, elf_target_ver & 0xf,
                         (ip_target_ver & 0xf00) >> 8, (ip_target_ver & 0xf0) >> 4, ip_target_ver & 0xf);
    }

    /*DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_pp, dpu_conf->arch.arch_pp, "arch_pp");
    DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_ocp, dpu_conf->arch.arch_ocp, "arch_ocp");
    DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_icp, dpu_conf->arch.arch_icp, "arch_icp");*/
    DPU_CONFIG_CHECK_ARCH(2 * elf_conf->arch.arch_pp * elf_conf->arch.arch_icp * elf_conf->arch.arch_ocp,
                           dpu_conf->base.dpu_arch, "DPU Arch");

    //DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_img_bkgrp, dpu_conf->arch.arch_img_bkgrp, "RAM Usage");
    N2CUBE_DPU_CHECK(elf_conf->arch.arch_img_bkgrp == dpu_conf->arch.arch_img_bkgrp,
                     N2CUBE_ERR_DPU_CONFIG_MISMATCH,
                     " for kernel %s - parameter: RAM Usage, DPU kernel: %s, DPU IP: %s.",
                     kernel->base.name,
                     elf_conf->arch.arch_img_bkgrp == 2 ? "Low" : "High",
                     dpu_conf->arch.arch_img_bkgrp == 2 ? "Low" : "High");

    DPU_CONFIG_CHECK_FIELD(elf_conf->dwcv.dwcv_parallel, dpu_conf->dwcv.dwcv_parallel, "DepthwiseConv Parallel");
    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->dwcv.dwcv_alu_mode_enable, dpu_conf->dwcv.dwcv_alu_mode_enable, "DepthwiseConv ALU Mode");
    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->dwcv.dwcv_relu6_enable, dpu_conf->dwcv.dwcv_relu6_enable, "DepthwiseConv+Relu6");

    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->conv.conv_leakyrelu_enable, dpu_conf->conv.conv_leakyrelu_enable, "Conv+Leakyrelu");
    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->conv.conv_relu6_enable, dpu_conf->conv.conv_relu6_enable, "Conv+Relu6");
    DPU_CONFIG_CHECK_FIELD(elf_conf->conv.conv_wr_parallel, dpu_conf->conv.conv_wr_parallel, "Conv Write Parallel");

    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->pool.pool_average_enable, dpu_conf->pool.pool_average_enable, "Average Pool");

    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->load.load_augm_enable, dpu_conf->load.load_augm_enable, "Channel Augmentation");
    DPU_CONFIG_CHECK_FIELD(elf_conf->ram.ram_depth_mean, dpu_conf->ram.ram_depth_mean, "Mean Ram Depth");
    DPU_CONFIG_CHECK_FIELD(elf_conf->ram.ram_depth_img, dpu_conf->ram.ram_depth_img, "Image Ram Depth");
    DPU_CONFIG_CHECK_FIELD(elf_conf->ram.ram_depth_wgt, dpu_conf->ram.ram_depth_wgt, "Weight Ram Depth");
    DPU_CONFIG_CHECK_FIELD(elf_conf->ram.ram_depth_bias, dpu_conf->ram.ram_depth_bias, "Bias Ram Depth");
    DPU_CONFIG_CHECK_FIELD_ENABLE(elf_conf->load.load_img_mean_enable, dpu_conf->load.load_img_mean_enable, "Load Mean Opt");

    DPU_CONFIG_CHECK_FIELD(elf_conf->elew.elew_parallel, dpu_conf->elew.elew_parallel, "ElementWise Parallel");

    DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_hp_bw, dpu_conf->arch.arch_hp_bw, "M-AXI DPU HP Data Width");
    DPU_CONFIG_CHECK_FIELD(elf_conf->arch.arch_data_bw, dpu_conf->arch.arch_data_bw, "DPU Data Width");

    DPU_CONFIG_CHECK_FIELD(elf_conf->misc.misc_wr_parallel, dpu_conf->misc.misc_wr_parallel, "Load Parallel");
    return N2CUBE_SUCCESS;
}

/*
 * get dpu arch from dpu_version str in metadata, which looks like "DPU_1152A"
 */
INTERNAL dpu_arch_t get_dpu_arch_from_str(char *dpu_arch) {
    char* name[] = {
        "1024",
        "1152",
        "4096"
    };

    DPU_ASSERT(dpu_arch, ERR);

    for(int i=1; i<DPU_ARCH_RESERVE; i++) {
        if(strstr(dpu_arch, name[i-1])) {
            return (dpu_arch_t)(i);
        }
    }

    DPU_FAIL_ON_MSG("No valid arch info from metadata dpu_version string.");
    return DPU_ARCH_UNKNOWN;
}

/*
 * Load Metadata from dpu elf
 */
int dpu_elf_load_meta(dpu_kernel_t *kernel, elf_t *elf) {
    int ret;
    uint32_t* metaStart;
    char *strtabStart;
    int charIndex;
    uint32_t  nameIndex;

    metaStart = (uint32_t*)(elf->elf_data + kernel->base.elf_meta.offset);
    strtabStart = (char*)(elf->elf_data + kernel->base.elf_strtab.offset);

    /* setup DPU arch string */
    charIndex = 0;
    nameIndex = *(uint32_t*)(metaStart + OFF_META_DPU_ARCH);
    while (*(strtabStart+nameIndex) != '\0') {
        kernel->base.dpu_arch[charIndex] = *(strtabStart+nameIndex);
        nameIndex++;
        charIndex++;
    }

    /* setup dnnc version string */
    charIndex = 0;
    nameIndex = *(uint32_t*)(metaStart + OFF_META_VER_DNNC);
    while (*(strtabStart+nameIndex) != '\0') {
        kernel->base.dnnc_ver[charIndex] = *(strtabStart+nameIndex);
        nameIndex++;
        charIndex++;
    }

    /* setup for kernel mode */
    uint32_t mode = *(uint32_t*)(metaStart + OFF_META_MODE);

    if (mode & ELF_KERNEL_DEBUG) {
        kernel->base.mode = K_MODE_DEBUG;
    } else {
        kernel->base.mode = K_MODE_NORMAL;
    }

    /* kernel's node count: including virtual nodes */
    kernel->base.node_cnt = *(uint32_t*)(metaStart + OFF_META_NODE_CNT);
    /* kernel's node size */
    kernel->base.tensor_size = *(uint32_t*)(metaStart + OFF_META_TENSOR_SIZE);
    /* kernel's memory IO size */
    kernel->base.IO_space_size = *(uint32_t*)(metaStart + OFF_META_KERNEL_IO_SIZE);

    /* kernel's mean value for 3 channels */
    kernel->base.mean_c1= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C1));
    kernel->base.mean_c2= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C2));
    kernel->base.mean_c3= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C3));

    /* ABI version of dpu elf */
    kernel->base.abi_ver= (uint32_t)(*(uint32_t*)(metaStart + OFF_META_ABI_VER));
    N2CUBE_DPU_CHECK(kernel->base.abi_ver <= DPU_ABI_MAX, N2CUBE_ERR_ABI_VERSION,
              ".\n" \
              "    DPU ABI version v%d.%d found in DPU kernel %s.\n" \
              "    It's caused by the version mismatch between DNNC and N2Cube,\n" \
              "    please check the version info in your DNNDK release package.",
              ABI_MAJOR_VER(kernel->base.abi_ver), ABI_MINOR_VER(kernel->base.abi_ver), kernel->base.name);
    setup_func_ver(kernel->base.abi_ver);

    /* Version check */
    /* No configurable IP, check target info from metadata */
    if (kernel->base.abi_ver < DPU_ABI_V2_0) {
        if (kernel->base.abi_ver == DPU_ABI_ORIGIN) {
            /* set target v1.1.3 as default */
            kernel->base.dpu_target_ver = DPU_TARGET_V1_1_3;
            /* get arch from dpu_version string */
            kernel->base.dpu_arch_ver = get_dpu_arch_from_str(kernel->base.dpu_arch);
        } else {
            /* set arch/target version to the value readed from DPU elf metadata */
            uint32_t version   = *(uint32_t*)(metaStart + OFF_META_DPU_VER);
            kernel->base.dpu_arch_ver   = DPU_ARCH_VER(version);
            kernel->base.dpu_target_ver = DPU_TARGET_VER(version);
        }
    }
    if (kernel->base.abi_ver >= DPU_ABI_V2_1) {
        kernel->base.mem_mode = *(uint32_t*)(metaStart + OFF_META_MEM_MODE);
    }

    return N2CUBE_SUCCESS;
}

int dpu_version_check(dpu_kernel_t *kernel) {
    int ret = N2CUBE_SUCCESS;
    if (kernel->base.abi_ver >= DPU_ABI_V2_0) {
        ret = configurable_check(kernel);
    } else {
        /* when DPU bit has no signature, dpu_target will be set to DPU_TARGET_V1_1_3 as defualt.
        * we just let it run as usual without version check */
        if(dpu_caps.signature_valid) {
            ret = version_check(kernel);
        }
    }
    return ret;
}

int dpu_elf_check_seg_validity(dpu_kernel_t *kernel,
                               elf_t *elf,
                               dpu_segment_t *dpu_seg,
                               dpu_segment_t **v1_seg,
                               int v1_seg_cnt,
                               dpu_segment_t **v2_seg,
                               int v2_seg_cnt) {
    int i;
    DPU_ASSERT(kernel, ERR);

    /* check segment validity for v1 */
    if (kernel->base.abi_ver <= DPU_ABI_V1_0) {
        for(i = 1; i < v1_seg_cnt; i++) {
            CHECK_SEG_SIZE(v1_seg[i]->elf_seg.size,
                           v1_seg[i]->log_name,
                           kernel,
                           elf);
        }
    /* check segment validity for v2 */
    } else {
        for(i = 1; i < v2_seg_cnt; i++) {
            CHECK_SEG_SIZE(v2_seg[i]->elf_seg.size,
                           v2_seg[i]->log_name,
                           kernel,
                           elf);
        }
        // setup configurable segment, and then check in dpu_elf_load_meta.
        if (kernel->base.abi_ver >= DPU_ABI_V2_0) {
            const int conf_idx = 8;
            CHECK_SEG_SIZE(dpu_seg[conf_idx].elf_seg.size,
                        dpu_seg[conf_idx].log_name,
                        kernel,
                        elf);
        }
    }
    return N2CUBE_SUCCESS;
}

/*
 * Get DPU Kernel info from CPU-DPU mixed hybrid binary and fill
 * them into dpu_kernel_t.
 */
int dpu_elf_load_kernel(dpu_kernel_t *kernel)
{
    elf_t elf;
    int i, j, hybridmode = 1;
    int ret = N2CUBE_SUCCESS;
    dpu_segment_t dpu_seg[DPU_SEGMENT_NUM] = {0};
    char exe_name[MAX_NAME_LEN];
    char so_name[MAX_NAME_LEN];

    /* initial the dpu_segments information firstly. */
    elf_segment_init(dpu_seg);

    // for origin ABI & ABIv1.0
    dpu_segment_t* v1_seg[] = {&dpu_seg[0], &dpu_seg[1], &dpu_seg[2],
                               &dpu_seg[3], &dpu_seg[4], &dpu_seg[5]};
    dpu_segment_t* v2_seg[] = {&dpu_seg[0], &dpu_seg[1], &dpu_seg[4],
                               &dpu_seg[5], &dpu_seg[6], &dpu_seg[7]};

    DPU_ASSERT(kernel, ERR);

    /* construct segment names */
    for(i = 0; i < DPU_SEGMENT_NUM; i++) {
        strcat(dpu_seg[i].seg_name, kernel->base.name);
    }
    /* read data from ELF binary into buffer */
    hybridmode = 1 ; 
    /* get hybrid ELF binary name of myself */
    ret = elf_get_myself_name(&(kernel->base), hybridmode);
    sprintf(exe_name, "%s", kernel->base.elf_name);
    N2CUBE_DPU_CHECK(ret == N2CUBE_SUCCESS, ret,
            ". kernel:%s, hybrid ELF:%s", kernel->base.name, kernel->base.elf_name);
    ret = elf_read(&(kernel->base), &elf);

    if (DPU_DEBUG_ELF()) {
        DPU_LOG_MSG("Prepare to load DPU kernel \"%s\" from hybrid ELF \"%s\"",
            kernel->base.name, kernel->base.elf_name);
    }

    ret = elf_segment_read(&elf, dpu_seg);

    /* check metadate segment validity first. */
    if (!(dpu_seg[0].elf_seg.size)) {
        if (DPU_DEBUG_ELF()) {
            DPU_LOG_MSG("Don't find DPU kernel \"%s\" from ELF \"%s\", try to search libraries\n",
            kernel->base.name, kernel->base.elf_name);
        }
    } else {
        goto FOUND_KERNEL; // kernel found successfully, exit the loop
    }
    
    hybridmode = 0;
    ret = elf_get_myself_name(&(kernel->base), hybridmode);
    sprintf(so_name, "%s", kernel->base.elf_name);
    N2CUBE_DPU_CHECK(ret == N2CUBE_SUCCESS, ret,
            ": %s, check if: \n" \
            "    1- Specified DPU Kernel name \"%s\" is right.\n" \
            "    2- DPU Kernel \"%s\" is compiled and linked into \"%s\" as expected.\n" \
            "    3- DPU shared library \"%s\" is located in the system library paths.",
            kernel->base.name, kernel->base.name,
            kernel->base.name, exe_name,so_name);
    
    ret = elf_read(&(kernel->base), &elf);
    if (ret != N2CUBE_SUCCESS) {
        goto DONE;
    }

    ret = elf_segment_read(&elf, dpu_seg);
    if (!(dpu_seg[0].elf_seg.size)) {
        elf_free(&elf);

        N2CUBE_DPU_CHECK(dpu_seg[0].elf_seg.size, ERR_ELF_INVALID_SECTION,
            "Fail to load DPU kernel: %s, check if: \n" \
            "    1- Specified DPU Kernel name \"%s\" is right.\n" \
            "    2- DPU Kernel \"%s\" is compiled and linked into \"%s\" as expected.\n" \
            "    3- DPU shared library \"%s\" is located in the system library paths.",
            kernel->base.name, kernel->base.name,
            kernel->base.name, exe_name,so_name);
        if (DPU_DEBUG_ELF()) {
            DPU_LOG_MSG("Can't find metadata section for kernel \"%s\" from library \"%s\"\n",
                    kernel->base.name, kernel->base.elf_name);
        }
    }
    if (ret != N2CUBE_SUCCESS) {
        goto DONE;
    }
FOUND_KERNEL:
    /* copy the dpu_seg information to the kernel`s segment */
    elf_copy_segment_kernel(dpu_seg, &(kernel->base));

    /* load metadata of dpu elf, setup funcs according to ABI version */
    ret = dpu_elf_load_meta(kernel, &elf);
    if (ret != N2CUBE_SUCCESS) {
        goto DONE;
    }
    /* .strtab & .symtab will not be needed any more after ABIv1.7.
     * so a strip command can be made for hybrid DPU ELF */
    if(kernel->base.abi_ver <= DPU_ABI_V1_6) {
        ret = elf_read_symtab(&elf, kernel->base.elf_name);
        N2CUBE_DPU_CHECK(ret >= 0, ret,
                  ". hybrid DPU file: %s, section: .symtab\n"
                  "    Perhaps symbols are discarded out after you running command strip.",
                  kernel->base.elf_name);
        elf.sec_strtab = elf_get_section_by_name(&elf, ".strtab");
        if (!(elf.sec_strtab >= 0)) {
            elf_free(&elf);
            N2CUBE_DPU_CHECK(0, N2CUBE_ERR_KERNEL_LOAD_SECTION,
                ". section:.strtab, hybrid DPU file: %s\n"
                "    Perhaps symbols are discarded out after you running command strip.",
                kernel->base.elf_name,
                DPU_MSG_HEADER);
        }
    }
    /* check segment validity for v1 */
    dpu_elf_check_seg_validity(kernel, &elf, dpu_seg,
                               v1_seg, sizeof(v1_seg) / sizeof(struct dpu_segment_t*),
                               v2_seg, sizeof(v2_seg) / sizeof(struct dpu_segment_t*));

    if (kernel->base.abi_ver >= DPU_ABI_V2_0) {
            ret = elf_read_configurable(&(kernel->base), &elf);
            N2CUBE_DPU_CHECK(ret >= 0, ret,
                      ". hybrid DPU file: %s, read configurable failed\n",
                      kernel->base.elf_name);
    }
    /* config field version check */
    if (!DPU_DEBUG_NO_CHECK()) {
        dpu_version_check(kernel);
    }

    if(kernel->base.abi_ver <= DPU_ABI_V1_0) {
        /* setup virtual Node's info for kernel */
        dpu_setup_kernel_virtual_node(kernel, &elf);
    }
    /* setup Node's Code/Weights-bias info for kernel */
    ret = setup_kernel_node(kernel, &elf);
    if (N2CUBE_SUCCESS != ret) {
        goto DONE;
    }
DONE:
    elf_free(&elf);
    return ret;
}



/*
 * Create Node name from the specified fields
 */
INTERNAL int create_node_name(char *kernel_name, const char *type, char *node_name, char *retName)
{
    DPU_ASSERT(kernel_name && type &&  node_name && retName, ERR);
    sprintf(retName, "%s%s%s%s", "_dpu_", kernel_name, type, node_name);

    return 0;
}

/*
 * Setup each Node's (only real Node) name from hybrid ELF binary
 */
INTERNAL int dpu_setup_node_name(dpu_kernel_t *kernel, elf_t *elf)
{
    struct node_name_t {
        unsigned long value;
        char name[MAX_NAME_LEN];
    };

    int i, j, len, ret = 0;
    int code_node_cnt;
    struct node_name_t temp, *node_list;
    char sym_name[MAX_NAME_LEN];
    char code_node_name[MAX_NAME_LEN];

    code_node_cnt = 0;
    node_list = (struct node_name_t *)malloc(sizeof(struct node_name_t)*kernel->base.node_cnt);

    sprintf(code_node_name, "%s%s%s", "_dpu_", kernel->base.name, "_code_");
    len = strlen(code_node_name);

    /* iterate each one in symbol list of hybrid binary */
    for (i = 0; i < elf->num_symbols; i++) {
        elf_get_symbol_name(elf, i, sym_name); /* ignore errors */
        DPU_ASSERT(((strlen(sym_name)+1) <= MAX_NAME_LEN), ERR);

        /* for symbol with name like "_dpu_KernelName_code_" */
        if (strstr(sym_name, code_node_name)) {
            if (ELF_CLASS_32()) {
                if (elf->symtab.symtab_32[i].st_shndx == kernel->base.elf_code.link) {
                    /* only store the Node's name specified in prototxt */
                    strcpy(node_list[code_node_cnt].name, sym_name+len);
                    node_list[code_node_cnt++].value = elf->symtab.symtab_32[i].st_value;
                }
            } else {
                if (elf->symtab.symtab_64[i].st_shndx == kernel->base.elf_code.link) {
                    /* only store the Node's name specified in prototxt */
                    strcpy(node_list[code_node_cnt].name, sym_name+len);
                    node_list[code_node_cnt++].value = elf->symtab.symtab_64[i].st_value;
                }
            }
        }
    }

    /* NOTE: here kernel->base.node_cnt should be real Nodes */
    if (code_node_cnt != kernel->base.node_cnt) {
        if (!code_node_cnt) {
            DPU_LOG_MSG("Fail to read DPU symbols from hybrid ELF \"%s\"",
                kernel->base.elf_name);
            DPU_FAIL_ON_MSG("Please check if its symbols are stripped out.");
        } else {
            DPU_FAIL_ON_MSG("Invalid number of Nodes for Kernel \"%s\" in File \"%s\": %d %d",
                kernel->base.name, kernel->base.elf_name, code_node_cnt, kernel->base.node_cnt);
        }
    }

    /* sort Node's name according to Node's offset value relative to DPU input section */
    for (i = 0; i < (kernel->base.node_cnt-1); i++) {
        for (j = i+1; j < kernel->base.node_cnt; j++) {
            if (node_list[j].value < node_list[i].value) {
                strcpy(temp.name, node_list[i].name);
                temp.value = node_list[i].value;

                strcpy(node_list[i].name, node_list[j].name);
                node_list[i].value = node_list[j].value;

                strcpy(node_list[j].name, temp.name);
                node_list[j].value = temp.value;
            }
        }
    }

    /* specify Node's name specified in prototxt */
    for (i = 0; i < kernel->base.node_cnt; i++) {
        dpu_node_t* node = kernel->base.node_list[i];
        node->ops.set_name(node, node_list[i].name);
    }

    free(node_list);
    node_list = NULL;

    return N2CUBE_SUCCESS;
}

/*
 * Setup DPU kernel's node info
 * including address/size of input/output tensor and fixed info
 */
INTERNAL int dpu_setup_node_shape(dpu_kernel_t *kernel, dpu_node_v1_real_t *node, uint32_t *nodeOffset)
{
    DPU_ASSERT(node && nodeOffset, ERR);

    node->workload = *(uint64_t*)(nodeOffset + OFF_NODE_WORKLOAD);

    node->shapeIn.height =
        *(uint32_t*)(nodeOffset + OFF_NODE_INPUT_HEIGHT);
    node->shapeIn.width =
        *(uint32_t*)(nodeOffset + OFF_NODE_INPUT_WIDTH);
    node->shapeIn.channel=
        *(uint32_t*)(nodeOffset + OFF_NODE_INPUT_CHANNEL);
    node->shapeIn.offset =
        *(uint32_t*)(nodeOffset + OFF_NODE_INPUT_ADDR);
    node->shapeIn.size =
        *(uint32_t*)(nodeOffset + OFF_NODE_INPUT_SIZE);
    node->shapeIn.fix_width =
        (uint8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_INPUT_FIX_WIDTH));
    node->shapeIn.fix_pos =
        (int8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_INPUT_FIX_POS));

    DPU_ASSERT((node->shapeIn.size==
        (node->shapeIn.height*node->shapeIn.width*node->shapeIn.channel)), ERR);

    /* convert input FP32 to INT8: value*scale */
    node->shapeIn.scale = (float)pow(2, node->shapeIn.fix_pos);

    node->base_v1.shapeOut.height =
        *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_HEIGHT);
    node->base_v1.shapeOut.width =
        *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_WIDTH);
    node->base_v1.shapeOut.channel =
        *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_CHANNEL);
    node->base_v1.shapeOut.offset =
        *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_ADDR);
    node->base_v1.shapeOut.size =
        *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_SIZE);
    node->base_v1.shapeOut.fix_width =
        (uint8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_FIX_WIDTH));
    node->base_v1.shapeOut.fix_pos =
        (int8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_FIX_POS));

#if 0
    /* for concact node, output size may not be the real size of "height*width*channel" */
    DPU_ASSERT((node->shapeOut.size==
        (node->shapeOut.height*node->shapeOut.width*node->shapeOut.channel)), ERR);
#endif

    /* convert output INT8 to FP32: value*scale */
    node->base_v1.shapeOut.scale = (float)(pow(2, -node->base_v1.shapeOut.fix_pos));

    node->base_v1.weight_fix_width =
        (uint8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_WEIGHT_FIX_WIDTH));
    node->base_v1.weight_fix_pos =
        (int8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_WEIGHT_FIX_POS));

    node->base_v1.bias_fix_width =
        (uint8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_BIAS_FIX_WIDTH));
    node->base_v1.bias_fix_pos =
        (int8_t)(*(uint32_t*)(nodeOffset + OFF_NODE_BIAS_FIX_POS));

}

/*
 * find the count of DPU virtual node (for concat operation) from ELF Node section
 *
 */
INTERNAL int find_dpu_virtual_node(dpu_kernel_t *kernel, elf_t *elf)
{
    int i, cnt;
    uint32_t magic;
    char *nodeEntryStart;
    uint32_t *nodeEntryOffset;

    cnt = 0;
    nodeEntryStart = (char*)(elf->elf_data + kernel->base.elf_tensor.offset);

    for (i=0; i<kernel->base.node_cnt; i++) {
        nodeEntryOffset = (uint32_t*)(char*)(nodeEntryStart + kernel->base.tensor_size*i);
        magic = *(uint32_t*)(nodeEntryOffset + OFF_NODE_MAGIC);

        if (magic != ELF_NODE_REAL ) cnt++;

    }

    return cnt;
}

/**
 * @brief Setup DPU kernel's virtual Node info
 *
 * @note virtual Nodes info should be setup before real Nodes
 *
 * @param kernel - the pointer to DPU kernel
 *
 * @return
 */
INTERNAL int dpu_setup_kernel_virtual_node(dpu_kernel_t *kernel, elf_t *elf)
{
    char *nodeStart;
    char *strtabStart;
    uint32_t magic, *nodeOffset;
    tensor_shape_t *shapeOut;
    dpu_node_v1_t *virtNode;
    int i, size, nameIndex, nodeIndex, charIndex;

    /* process for DPU virtual nodes */
    kernel->base.virt_node_cnt = find_dpu_virtual_node(kernel, elf);

    /* return if no virtual Nodes */
    if (!kernel->base.virt_node_cnt) return N2CUBE_SUCCESS;

    /* allocate memory space for virtual Nodes of Kernel */
    size = sizeof(dpu_node_v1_virt_t) * kernel->base.virt_node_cnt;
    kernel->virt_node_list = (dpu_node_v1_virt_t*)malloc(size);
    memset(kernel->virt_node_list, 0, size);
    for(i=0; i<kernel->base.virt_node_cnt; i++) {
        dpu_node_v1_virt_init(&(kernel->virt_node_list[i]));
    }

    /* ELF start offset info */
    nodeStart = (char*)(elf->elf_data + kernel->base.elf_tensor.offset);
    strtabStart = (char*)(elf->elf_data + kernel->base.elf_strtab.offset);

    nodeIndex = 0;

    /* kernel->base.node_cnt holds both real and virtual Nodes */
    for (i=0; i<kernel->base.node_cnt; i++) {
        nodeOffset = (uint32_t*)(char*)(nodeStart + kernel->base.tensor_size*i);

        /* for DPU real Node, magic number is 0;
           for virtual Node, it is Node's name index to .deephi.strtab.kernelName */
        magic = *(uint32_t*)(nodeOffset + OFF_NODE_MAGIC);
        if (ELF_NODE_REAL == magic) continue;

        DPU_ASSERT((nodeIndex < kernel->base.virt_node_cnt), ERR);

        /* setup virtual Node shape info */
        shapeOut = &(kernel->virt_node_list[nodeIndex].base_v1.shapeOut);

        shapeOut->height =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_HEIGHT);
        shapeOut->width =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_WIDTH);
        shapeOut->channel =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_CHANNEL);
        shapeOut->offset =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_ADDR);
        shapeOut->size =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_SIZE);
        shapeOut->fix_width =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_FIX_WIDTH);
        shapeOut->fix_pos =
            *(uint32_t*)(nodeOffset + OFF_NODE_OUTPUT_FIX_POS);

        /* convert output from INT8 to FP32: value*scale */
        shapeOut->scale =
            (float)(pow(2, -shapeOut->fix_pos));

        virtNode = &(kernel->virt_node_list[nodeIndex].base_v1);
        virtNode->weight_fix_width =
            *(uint32_t*)(nodeOffset + OFF_NODE_WEIGHT_FIX_WIDTH);
        virtNode->weight_fix_pos =
            *(uint32_t*)(nodeOffset + OFF_NODE_WEIGHT_FIX_POS);
        virtNode->bias_fix_width =
            *(uint32_t*)(nodeOffset + OFF_NODE_BIAS_FIX_WIDTH);
        virtNode->bias_fix_pos =
            *(uint32_t*)(nodeOffset + OFF_NODE_BIAS_FIX_POS);

        virtNode->base.ops.set_name((dpu_node_t*)&(kernel->virt_node_list[nodeIndex]),
                                    strtabStart+magic);

        nodeIndex++;
    }

    /* update the number of real DPU Nodes */
    kernel->base.node_cnt -= kernel->base.virt_node_cnt;

    return N2CUBE_SUCCESS;
}

/**
 * Get string from .deephi.strtab section.
 * idx : index in byte which target string started from.
 * buf : target string will be copyed to buf.
 */
void dpu_get_str_from_strtab(dpu_kernel_t *kernel, elf_t *elf, uint32_t idx, char *buf) {
    char *strtabStart = (char*)(elf->elf_data + kernel->base.elf_strtab.offset);

    DPU_ASSERT(buf, ERR);
    strcpy(buf, strtabStart + idx);
}


/*
 * For ABI version after and ABIv1.7. Function:
 * 1. get symbol info according to offset in nodepool item and symtab
 * 2. set mem_segment struct from symbol info
 */
INTERNAL int dpu_elf_set_segment_info(dpu_kernel_t  *kernel,
                                   elf_t         *elf,
                                   dpu_node_v2_t *node,
                                   dpu_elf_sec_t *sec,
                                   mem_segment_t *seg) {
    uint32_t abi;
    DPU_ASSERT(elf, ERR);
    DPU_ASSERT(node, ERR);
    DPU_ASSERT(sec, ERR);
    DPU_ASSERT(seg, ERR);
    DPU_ASSERT(kernel, ERR);
    abi = kernel->base.abi_ver;
    N2CUBE_DPU_CHECK(abi >= DPU_ABI_V1_7, N2CUBE_ERR_ABI_VERSION,
              ".\n" \
              "    Linking DPU sections with node via offset available only for ABI above and v1.7,\n" \
              "    but version %d.%d found in DPU kernel %s.",
              ABI_MAJOR_VER(abi), ABI_MINOR_VER(abi), kernel->base.name);

    seg->size = sec->size;
    seg->length = sec->size;
    seg->addr_phy = sec->offset;
    dpu_get_str_from_strtab(kernel, elf, sec->name_idx, seg->name);

    return N2CUBE_SUCCESS;
}



/**
 * Looks up symbols relative to a Node by Node's name, temporary solution, just for ABIv1.6.
 */
int dpu_elf_get_symbols_by_node_name(dpu_kernel_t *kernel, elf_t* elf, dpu_node_v2_t *node)
{
    uint32_t abi;
    int i, size, paramIdx, paramCnt = 0;
    char sym_name[MAX_NAME_LEN];
    mem_segment_t *node_code, *node_params;

    DPU_ASSERT(kernel, ERR);
    DPU_ASSERT(elf, ERR);
    DPU_ASSERT(node, ERR);
    abi = kernel->base.abi_ver;
    N2CUBE_DPU_CHECK(kernel->base.abi_ver == DPU_ABI_V1_6, N2CUBE_ERR_ABI_VERSION,
              ".\n"
              "    Linking DPU sections with node via node name only available for ABIv1.6,\n"
              "    but ABIv%d.%d found in DPU kernel %s.",
              ABI_MAJOR_VER(abi), ABI_MINOR_VER(abi), kernel->base.name);

    node_code = &(node->node_code);
    for (i = 0; i < elf->num_symbols; i++) {
        elf_get_symbol_name(elf, i, sym_name); /* ignore errors */
        /* get symbol relative to Node */
        if (strstr(sym_name, node->base.name)) {
            if (ELF_CLASS_32()) {
                Elf32_Sym* symbol = &elf->symtab.symtab_32[i];
                /* set code symbol */
                if (symbol->st_shndx == kernel->base.elf_code.link) {
                    node_code->size = symbol->st_size;
                    node_code->length = symbol->st_size;
                    node_code->addr_phy = (uint32_t)(symbol->st_value);
                    strcpy(node_code->name, sym_name);
                    node->code_cnt++;
                    if(node->type == T_NODE_DPU_REAL) {
                        node->memload = node_code->length;
                    }
                /* count number of param symbols */
                } else if(symbol->st_shndx == kernel->base.elf_param.link){
                    paramCnt++;
                }
            } else {
                Elf64_Sym* symbol = &elf->symtab.symtab_64[i];
                /* set code symbol */
                if (symbol->st_shndx == kernel->base.elf_code.link) {
                    node_code->size = symbol->st_size;
                    node_code->length = symbol->st_size;
                    node_code->addr_phy = (uint32_t)(symbol->st_value);
                    strcpy(node_code->name, sym_name);
                    node->code_cnt++;
                    if(node->type == T_NODE_DPU_REAL) {
                        node->memload = node_code->length;
                    }
                /* count number of param symbols */
                } else if(symbol->st_shndx == kernel->base.elf_param.link){
                    paramCnt++;
                }
            }
        }
    }
    N2CUBE_DPU_CHECK(node->code_cnt <= 1, N2CUBE_ERR_ABI_CODE_SEGMENT_COUNT,
              ". Node: %s", node->base.name);


    /* alloc resource for node_params according to paramCnt. */
    size = sizeof(mem_segment_t) * paramCnt;
    node->node_params = (mem_segment_t*)malloc(size);
    memset(node->node_params, 0, size);
    paramIdx = 0;
    node_params = node->node_params;
    /* find and set node params. */
    for (i = 0; i < elf->num_symbols; i++) {
        elf_get_symbol_name(elf, i, sym_name); /* ignore errors */
        if (strstr(sym_name, node->base.name)) {
            if (ELF_CLASS_32()) {
                Elf32_Sym* symbol = &elf->symtab.symtab_32[i];
                /* set param symbols */
                if(symbol->st_shndx == kernel->base.elf_param.link){
                    node_params[paramIdx].size = symbol->st_size;
                    node_params[paramIdx].length = symbol->st_size;
                    node_params[paramIdx].addr_phy = (uint32_t)(symbol->st_value);
                    strcpy(node_params[paramIdx].name, sym_name);
                    if(node->type == T_NODE_DPU_REAL) {
                        node->memload = node_params[paramIdx].length;
                    }
                    paramIdx++;
                }
            } else {
                Elf64_Sym* symbol = &elf->symtab.symtab_64[i];
                /* set param symbols */
                if(symbol->st_shndx == kernel->base.elf_param.link) {
                    node_params[paramIdx].size = symbol->st_size;
                    node_params[paramIdx].length = symbol->st_size;
                    node_params[paramIdx].addr_phy = (uint32_t)(symbol->st_value);
                    strcpy(node_params[paramIdx].name, sym_name);
                    if(node->type == T_NODE_DPU_REAL) {
                        node->memload = node_params[paramIdx].length;
                    }
                    paramIdx++;
                }
            }
        }
    }

    DPU_ASSERT(paramCnt == paramIdx, ERR);
    node->param_cnt = paramCnt;

    return N2CUBE_SUCCESS;
}

/*
 * Read data from .deephi.tensor segment of DPU elf for ABI version after and ABIv1.6.
 */
INTERNAL void dpu_setup_tensor(dpu_kernel_t *kernel, elf_t *elf) {
    int i;
    uint32_t *readBase;
    char *strtabBase, *nameStr;
    tensor_shape_t *tensors;

    DPU_ASSERT(kernel, ERR);
    DPU_ASSERT(elf, ERR);

    readBase = (uint32_t*)(elf->elf_data + kernel->base.elf_tensor.offset);
    strtabBase = (char*)(elf->elf_data + kernel->base.elf_strtab.offset);

    kernel->base.tensor_cnt = kernel->base.elf_tensor.size/kernel->base.tensor_size;
    kernel->base.tensor_list = (tensor_shape_t*)malloc(sizeof(tensor_shape_t) * kernel->base.tensor_cnt);
    tensors = kernel->base.tensor_list;
    for(i=0; i<kernel->base.tensor_cnt; i++) {
        tensors[i].attr           = *readBase++;
        tensors[i].height         = *readBase++;
        tensors[i].width          = *readBase++;
        tensors[i].channel        = *readBase++;
        tensors[i].offset         = *readBase++;
        tensors[i].size           = *readBase++;
        tensors[i].fix_width      = *readBase++;
        tensors[i].fix_pos        = *readBase++;
        tensors[i].channel_stride = *readBase++;
        nameStr                   = strtabBase + *readBase++;
        tensors[i].tensor_name = (char*)malloc(strlen(nameStr) + 1);
        strcpy(tensors[i].tensor_name, nameStr);
        readBase++;  //reserved
        readBase++;  //reserved
        readBase++;  //reserved
    }
}


/*
 * setup node info from Node-Pool section in dpu elf for ABI version after and ABIv1.6.
 */
INTERNAL int dpu_setup_kernel_node_v2_0(dpu_kernel_t *kernel, elf_t *elf) {
    int ret;
    int i, j, size;
    uint32_t *readBase, tmp;
    char *strtabBase, *name;
    dpu_node_v2_t  *node;
    mem_segment_t *code, *param;

    /* setup tensor segment */
    dpu_setup_tensor(kernel, elf);

    readBase = (uint32_t*)(elf->elf_data + kernel->base.elf_node_pool.offset);
    strtabBase = (char*)(elf->elf_data + kernel->base.elf_strtab.offset);

    /* malloc mem for node list */
    size = sizeof(dpu_node_t*) * (kernel->base.node_cnt);
    DPU_ASSERT(kernel->base.node_cnt > 0, ERR);
    kernel->base.node_list = (dpu_node_t **)malloc(size);
    memset(kernel->base.node_list, 0, size);
    for(i=0; i<kernel->base.node_cnt; i++) {
        size = sizeof(dpu_node_v2_t);
        kernel->base.node_list[i] = (dpu_node_t*)malloc(size);
        memset(kernel->base.node_list[i], 0, size);
        dpu_node_v2_init(kernel->base.node_list[i]);
    }
    /* setup node-pool items */
    for (i=0; i<kernel->base.node_cnt; i++) {
        node = (dpu_node_v2_t*)(kernel->base.node_list[i]);

        node->type = (node_type_t)*readBase++;

        /* setup node name from strtab */
        name = strtabBase + *readBase;
        readBase++;
        node->base.ops.set_name((dpu_node_t*)node, name);

        /* setup workload */
        node->workload = *(uint64_t*)readBase;
        readBase += sizeof(uint64_t)/sizeof(uint32_t);

        /* setup reg assegment */
        node->reg_cnt = *readBase++;
        if(node->reg_cnt > 0) {
            node->reg_type_list =
                 (dpu_reg_assign_t*)malloc(sizeof(dpu_reg_assign_t) * node->reg_cnt);
            for(j=0; j<node->reg_cnt; j++) {
                tmp = *readBase++;
                node->reg_type_list[j].data_type = LSB16(tmp);
                node->reg_type_list[j].reg_id    = MSB16(tmp);
            }
        }

        /* setup input tensor */
        node->input_cnt = *readBase++;
        DPU_ASSERT(node->input_cnt > 0, ERR);
        node->input_list =
            (uint32_t*)malloc(sizeof(uint32_t) * node->input_cnt);
        for(j=0; j<node->input_cnt; j++) {
            uint32_t idx = *readBase++;
            node->input_list[j] = idx;
            tensor_shape_t * input = &(kernel->base.tensor_list[idx]);

            /* only real node has input scale */
            //tensors[i].scale = (float)(pow(2, tensors[i].fix_pos));
            if(node->type == T_NODE_DPU_REAL) {
                /* for memory access load for this node */
                node->memload += input->height * input->width * input->channel;
            }
        }

        /* setup output tensor */
        node->output_cnt = *readBase++;
        DPU_ASSERT(node->output_cnt > 0, ERR);
        node->output_list =
            (uint32_t*)malloc(sizeof(uint32_t) * node->output_cnt);
        for(j=0; j<node->output_cnt; j++) {
            uint32_t idx = *readBase++;
            node->output_list[j] = idx;
            tensor_shape_t * output = &(kernel->base.tensor_list[idx]);

            //output->scale = (float)(pow(2, -output->fix_pos));
            if(node->type == T_NODE_DPU_REAL) {
                /* for memory access load for this node */
                node->memload += output->height * output->width * output->channel;
            }
        }

        /* setup code & param symbols, link them with node by offset&size */
        if (kernel->base.abi_ver >= DPU_ABI_V1_7) {
            /* setup code section(symbol) */
            node->code_cnt = *readBase++;
            DPU_ASSERT(node->code_cnt<2, ERR);
            if(node->code_cnt > 0) {
                node->elf_code.base.offset = *readBase++;
                node->elf_code.base.size = *readBase++;
                node->elf_code.base.name_idx = *readBase++;
                node->elf_code.base.align = *readBase++;
                ret = dpu_elf_set_segment_info(kernel, elf, node,
                                     (dpu_elf_sec_t*)&(node->elf_code),
                                     &(node->node_code));
                if (N2CUBE_SUCCESS != ret) {
                    return ret;
                }

                if(node->type == T_NODE_DPU_REAL) {
                    /* for memory access load for this node */
                    node->memload += node->node_code.length;
                }
            }

            /* setup param sections(symbol) */
            node->param_cnt = *readBase++;
            if(node->param_cnt > 0) {
                node->elf_params = (dpu_elf_param_sec_t*)
                               malloc(sizeof(dpu_elf_param_sec_t) * node->param_cnt);
                node->node_params = (mem_segment_t*)malloc(sizeof(mem_segment_t) * node->param_cnt);
                for(j=0; j<node->param_cnt; j++) {
                    node->elf_params[j].base.offset = *readBase++;
                    node->elf_params[j].base.size = *readBase++;
                    node->elf_params[j].fix_w = *readBase++;
                    node->elf_params[j].fix_p = *readBase++;
                    node->elf_params[j].base.name_idx = *readBase++;
                    node->elf_params[j].base.align = *readBase++;
                    node->elf_params[j].hight = *readBase++;
                    node->elf_params[j].width = *readBase++;
                    node->elf_params[j].channel = *readBase++;
                    node->elf_params[j].out_channel = *readBase++;
                    ret = dpu_elf_set_segment_info(kernel, elf, node,
                                         (dpu_elf_sec_t*)&(node->elf_params[j]),
                                         &(node->node_params[j]));
                    if (N2CUBE_SUCCESS != ret) {
                        return ret;
                    }

                    if(node->type == T_NODE_DPU_REAL) {
                        /* for memory access load for this node */
                        node->memload += node->node_params[j].length;
                    }
                }
            }
        } else if (kernel->base.abi_ver == DPU_ABI_V1_6) {
            /* setup code & param symbols by node name, exist temperoly */
            ret = dpu_elf_get_symbols_by_node_name(kernel, elf, node);
            if (N2CUBE_SUCCESS != ret) {
                return ret;
            }
        } else {
            N2CUBE_DPU_CHECK(0, N2CUBE_ERR_ABI_VERSION,
                      ". ABI version: %x, kernel: %s",
                      kernel->base.abi_ver, kernel->base.name);
        }



        /* setup predecessor info */
        node->pre_cnt = *readBase++;
        if(node->pre_cnt > 0) {
            node->pre_list = (uint32_t*)malloc(sizeof(uint32_t) * node->pre_cnt);
            for(j=0; j<node->pre_cnt; j++) {
                node->pre_list[j] = *readBase++;
            }
        }

        /* setup successor info */
        node->suc_cnt = *readBase++;
        if(node->suc_cnt > 0) {
            node->suc_list = (uint32_t*)malloc(sizeof(uint32_t) * node->suc_cnt);
            for(j=0; j<node->suc_cnt; j++) {
                node->suc_list[j] = *readBase++;
            }
        }


        /* Accumulate total workload for Kernel */
        kernel->base.workloadTotal += (float)(node->workload);
        /* Accumulate total memory access for Kernel */
        kernel->base.memloadTotal += (float)(node->memload);
    }

    return N2CUBE_SUCCESS;
}

/*
 * Setup DPU kernel's Node info (real node)
 *
 */
INTERNAL int dpu_setup_kernel_node(dpu_kernel_t *kernel, elf_t *elf)
{
    int i, index, size, nodeIndex;
    void* symbol;
    char *nodeEntryStart;
    uint32_t magic, *nodeEntryOffset, *nodeOffset;
    char dpu_node_name[MAX_NAME_LEN];
    dpu_node_v1_real_t **nodes;
    mem_segment_t *seg_code, *seg_b, *seg_w;

    /* malloc mem for real node array */
    size = sizeof(dpu_node_t*) * (kernel->base.node_cnt);
    kernel->base.node_list = (dpu_node_t **)malloc(size);
    memset(kernel->base.node_list, 0, size);
    for(i=0; i<kernel->base.node_cnt; i++) {
        size = sizeof(dpu_node_v1_real_t);
        kernel->base.node_list[i] = (dpu_node_t *)malloc(size);
        memset(kernel->base.node_list[i], 0, size);
        dpu_node_v1_real_init(kernel->base.node_list[i]);
    }

    nodes = (dpu_node_v1_real_t**)(kernel->base.node_list);

    /* get each real Node's name
     * NOTE1: they may be different from names in prototxt due to dnnc reconstruction
     * NOTE2: Node name should be setup after the disclassifcation between real Node & virtual Node
     */
    dpu_setup_node_name(kernel, elf);

    nodeIndex = 0;
    nodeEntryStart = (char*)(elf->elf_data + kernel->base.elf_tensor.offset);

    for (i=0; i < (kernel->base.node_cnt + kernel->base.virt_node_cnt); i++) {
        nodeOffset = (uint32_t*)(char*)(nodeEntryStart + kernel->base.tensor_size*i);

        /* for DPU real Node, magic number is 0;
           for virtual Node, it is Node's name index to .deephi.strtab.kernelName */
        magic = *(uint32_t*)(nodeOffset + OFF_NODE_MAGIC);
        if (ELF_NODE_REAL != magic) continue;
        DPU_ASSERT((nodeIndex < kernel->base.node_cnt), ERR);

        /* setup for code */
        seg_code = &(nodes[nodeIndex]->node_code);
        create_node_name(kernel->base.name, "_code_", nodes[nodeIndex]->base_v1.base.name, seg_code->name);
        index = elf_get_symbol_by_name(elf, seg_code->name);
        N2CUBE_DPU_CHECK(index >= 0, N2CUBE_ERR_ABI_SYMBOL_CODE, ". symbol: %s", seg_code->name);
        seg_code->type = MEM_CODE;
        if (ELF_CLASS_32()) {
            Elf32_Sym* symbol = &elf->symtab.symtab_32[index];
            seg_code->size = symbol->st_size;
            seg_code->length = symbol->st_size;
            seg_code->addr_phy = (uint32_t)(symbol->st_value);
        } else {
            Elf64_Sym* symbol = &elf->symtab.symtab_64[index];
            seg_code->size = symbol->st_size;
            seg_code->length = symbol->st_size;
            seg_code->addr_phy = (uint32_t)(uint64_t)(symbol->st_value);
        }
        N2CUBE_DPU_CHECK((seg_code->size>0), N2CUBE_ERR_ABI_CODE_SEGMENT_SIZE,
            ". symbol:%s, hybrid ELF: %s", seg_code->name, kernel->base.elf_name);

        /* setup for bias */
        seg_b = &(nodes[nodeIndex]->node_bias);
        create_node_name(kernel->base.name, "_b_", nodes[nodeIndex]->base_v1.base.name, seg_b->name);
        index = elf_get_symbol_by_name(elf, seg_b->name);

        N2CUBE_DPU_CHECK(index >= 0, N2CUBE_ERR_ABI_SYMBOL_BIAS, ". BIAS symbol: %s", seg_b->name);
        seg_b->type = MEM_BIAS;
        if (ELF_CLASS_32()) {
            Elf32_Sym* symbol = &elf->symtab.symtab_32[index];
            seg_b->size = symbol->st_size;
            seg_b->length = symbol->st_size;
            seg_b->addr_phy = (uint32_t)(symbol->st_value);
        } else {
            Elf64_Sym* symbol = &elf->symtab.symtab_64[index];
            seg_b->size = symbol->st_size;
            seg_b->length = symbol->st_size;
            seg_b->addr_phy = (uint32_t)(uint64_t)(symbol->st_value);
        }
        N2CUBE_DPU_CHECK((seg_b->size>0), N2CUBE_ERR_ABI_BIAS_SEGMENT_SIZE,
            ". BIAS segment: %s, hybrid ELF: %s", seg_b->name, kernel->base.elf_name);

        /* setup for weights */
        seg_w = &(nodes[nodeIndex]->node_weight);
        create_node_name(kernel->base.name, "_w_", nodes[nodeIndex]->base_v1.base.name, seg_w->name);
        index = elf_get_symbol_by_name(elf, seg_w->name);
        N2CUBE_DPU_CHECK(index >= 0, N2CUBE_ERR_ABI_SYMBOL_WEIGHTS, ". WEIGHTS symbol: %s", seg_w->name);
        seg_w->type = MEM_WEIGHT;
        if (ELF_CLASS_32()) {
            Elf32_Sym* symbol = &elf->symtab.symtab_32[index];
            seg_w->size = symbol->st_size;
            seg_w->length = symbol->st_size;
            seg_w->addr_phy = (uint32_t)(symbol->st_value);
        } else {
            Elf64_Sym* symbol = &elf->symtab.symtab_64[index];
            seg_w->size = symbol->st_size;
            seg_w->length = symbol->st_size;
            seg_w->addr_phy = (uint32_t)(uint64_t)(symbol->st_value);
        }
        N2CUBE_DPU_CHECK((seg_w->size>0), N2CUBE_ERR_API_WEIGHTS_SEGMENT_SIZE,
            ". WEIGHT segment: %s, hybrid ELF: %s", seg_w->name, kernel->base.elf_name);


        if (nodeIndex > 0) {
            N2CUBE_DPU_CHECK((nodes[nodeIndex-1]->node_code.length +
                nodes[nodeIndex-1]->node_code.addr_phy) == seg_code->addr_phy,
                N2CUBE_ERR_KERNEL_ADDR_CODE, ". CODE: %s, Kernel: %s",
                seg_code->name, kernel->base.name);

#if 0
printf("pre len:%d  pre addr:%d  now addr:%d\n",
    (headNodeList[nodeIndex-1].node_bias.length),
    headNodeList[nodeIndex-1].node_bias.addr_phy,
    seg_b->addr_phy);
#endif

            N2CUBE_DPU_CHECK((nodes[nodeIndex-1]->node_bias.length +
                nodes[nodeIndex-1]->node_bias.addr_phy) == seg_b->addr_phy,
                N2CUBE_ERR_KERNEL_ADDR_BIAS, ". BIAS: %s, Kernel: %s",
                seg_b->name, kernel->base.name);


            N2CUBE_DPU_CHECK((nodes[nodeIndex-1]->node_weight.length +
                nodes[nodeIndex-1]->node_weight.addr_phy) == seg_w->addr_phy,
                N2CUBE_ERR_KERNEL_ADDR_WEIGHTS, ". WEIGHTS: %s, Kernel: %s",
                seg_w->name, kernel->base.name);

        }

        /* NOTE: physical address of input/output Node needs to be updated when
           they are located to fixed DPU memory space */
        nodeEntryOffset = (uint32_t*)(char*)(nodeEntryStart + kernel->base.tensor_size*nodeIndex);

        /* setup node info */
        dpu_setup_node_shape(kernel, nodes[nodeIndex], nodeEntryOffset);

        /* setup node metric info: computation load and memory access load */
        dpu_setup_node_metric(kernel, nodeIndex);

        /* update real Node count */
        nodeIndex++;

    }

    return N2CUBE_SUCCESS;
}


/*
 * Setup DPU kernel's node info
 * including address/size of input/output tensor and fixed info
 */
INTERNAL int dpu_setup_node_metric(dpu_kernel_t * kernel, int nodeID)
{
    dpu_node_v1_real_t *node;

    DPU_ASSERT(kernel, ERR);
    DPU_ASSERT((nodeID>=0) && (nodeID<kernel->base.node_cnt), ERR);

    node = (dpu_node_v1_real_t*)(kernel->base.node_list[nodeID]);

    /* Accumulate total workload for Kernel */
    kernel->base.workloadTotal += (float)(node->workload);

    /* for memory access load for this node */
    node->memload = node->shapeIn.height * node->shapeIn.width * node->shapeIn.channel +
        node->base_v1.shapeOut.height * node->base_v1.shapeOut.width * node->base_v1.shapeOut.channel +
        node->node_bias.length + node->node_weight.length + node->node_code.length;

    kernel->base.memloadTotal += (float)(node->memload);
}

inline static void dpu_kernel_print_size(uint32_t size, char *str) {
    if (size < (1024 / 128)) {
        sprintf(str, "%dBytes", size);
        return;
    } else if (size < (1024 * 1024 / 128)) {
        sprintf(str, "%.2fKB", (float)(size) / 1024);
        return;
    } else {
        sprintf(str, "%0.2fMB", (float)(size) / (1024 * 1024));
        return;
    }
}

void dpu_print_dpu_target_version(uint32_t ver, char *str) {
    uint32_t v1 = ver & 0x0000000f;
    uint32_t v2 = (ver & 0x000000f0) >> 4;
    uint32_t v3 = (ver & 0x00000f00) >> 8;
    //uint32_t v0 = (ver & 0x000ff000) >> 12;
    sprintf(str, "%x.%x.%x", v3, v2, v1);
}

/*
 * Get the kernel info from ARM-DPU mixed ELF binary and fill
 * them into dpu_kernel_t.
 */
int dpu_elf_load_debug(dpu_kernel_t *kernel)
{
    int i;
    dpu_node_t **nodes;
    dpu_node_v1_virt_t *vnode;
    tensor_shape_t *prtShapeOut;
    char temp_str[MAX_NAME_LEN] = {0};
    DPU_ASSERT(kernel, ERR);

    if (DPU_DEBUG_LD()) {
        DPU_LOG_MSG("Load DPU Kernel \"%s\" from hybrid ELF \"%s\"s",
            kernel->base.name, kernel->base.elf_name);
        printf(DPU_LINE);

        if (kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("%10s%12s%12s%12s%12s%12s\n",
                "Code(C)", "C-Size", "Weight(W)", "W-Size", "Bias(B)", "B-Size");
            printf("0x%08x  0x%08x  0x%08x  0x%08x  0x%08x  0x%08x\n",
                   kernel->base.elf_code.offset,
                   kernel->base.elf_code.size,
                   kernel->base.elf_weight.offset,
                   kernel->base.elf_weight.size,
                   kernel->base.elf_bias.offset,
                   kernel->base.elf_bias.size);
        } else {
            printf("%10s%12s%12s%12s%12s%12s\n",
                   "Code(C)", "C-Size", "Param(P)", "P-Size", "Tensor(T)", "T-Size");
            printf("0x%08x  0x%08x  0x%08x  0x%08x  0x%08x  0x%08x\n",
                   kernel->base.elf_code.offset,
                   kernel->base.elf_code.size,
                   kernel->base.elf_param.offset,
                   kernel->base.elf_param.size,
                   kernel->base.elf_tensor.offset,
                   kernel->base.elf_tensor.size);
        }
        printf(DPU_LINE);
        printf("Metadata for DPU Kernel: %s\n", kernel->base.name);
        printf("%s\n%s\n\n", "DPU arch of Kernel:", kernel->base.dpu_arch);
        printf(DPU_LINE);
        printf("%s\n%s\n\n", "Kernel built by compiler:", kernel->base.dnnc_ver);
        printf(DPU_LINE);
        printf("%18s  %s\n", "Mode:", (kernel->base.mode & K_MODE_DEBUG)? "DEBUG": "NORMAL");
        printf("%18s  %x\n", "DPU ABI Ver:", kernel->base.abi_ver);
        if (kernel->base.abi_ver < DPU_ABI_V2_0) {
            printf("%18s  %s\n", "DPU Target Ver:", g_dpu_target_name[kernel->base.dpu_target_ver]);
            printf("%18s  %s\n", "DPU Arch Type:", g_dpu_arch_name[kernel->base.dpu_arch_ver]);
        } else {
            dpu_print_dpu_target_version(kernel->base.dpu_conf.sub_version.ver_target, temp_str);
            printf("%25s  %s\n", "DPU Target Ver:", temp_str);
            printf("%25s  B%d\n", "DPU Arch Type:", 2 * kernel->base.dpu_conf.arch.arch_icp *
                    kernel->base.dpu_conf.arch.arch_ocp * kernel->base.dpu_conf.arch.arch_pp);
        }
        printf("%18s  %0.3fMOP\n", "Workload MACs:", kernel->base.workloadTotal / (1000*1000));
        dpu_kernel_print_size(kernel->base.memloadTotal, temp_str);
        printf("%18s  %s\n", "Memory Load Total:", temp_str);
        dpu_kernel_print_size(kernel->base.IO_space_size, temp_str);
        printf("%18s  %s\n", "IO Memory Space:", temp_str);

        printf("%18s  %d\n", "Node Entry Size:", kernel->base.tensor_size);
        printf("%18s  %d, %d, %d\n", "Mean Value:",
                kernel->base.mean_c1, kernel->base.mean_c2, kernel->base.mean_c3);
        printf("%18s  %d\n", "Node Count:", kernel->base.node_cnt);
        printf("%18s  %d\n", "Tensor Count:", kernel->base.tensor_cnt);
        printf("\n");
        printf(DPU_LINE);

        printf("DPU Node name List for Kernel - %s\n", kernel->base.name);
        printf("%10s\t%-30s\n", "ID", "Name");
        nodes = kernel->base.node_list;

        for (i=0; i < kernel->base.node_cnt; i++) {
            printf("%10d\t%-30s\n", i, nodes[i]->name);
        }

        printf(DPU_LINE);

        if (kernel->base.abi_ver <= DPU_ABI_V1_0) {
            printf("Real Node detail list for DPU Kernel: %s\n", kernel->base.name);
        } else {
            printf("Node detail list for DPU Kernel: %s\n", kernel->base.name);
        }

        for (i=0; i < kernel->base.node_cnt; i++) {
            printf("%s%d %s%s\n", "NodeID-", i, "Name-", nodes[i]->name);
            printf("%16s  0x%012llx\n", "Workload:", (long long)nodes[i]->ops.get_workload(nodes[i]));

            nodes[i]->ops.trace_tensors(nodes[i], (kernel_t*)kernel);
            nodes[i]->ops.trace_param_infos(nodes[i], (kernel_t*)kernel);
            printf("Physical address info:\n");
            if (kernel->base.abi_ver <= DPU_ABI_V1_0) {
                printf("%19s : %11s : %s\n", "Code", "Bias", "Weight");
            } else {
                printf("%19s : %11s : %s\n", "Section", "addr", "size");
            }
            nodes[i]->ops.trace_addr_phy(nodes[i], stdout, i);
            if (i != (kernel->base.node_cnt-1)) printf("\n");
        }


        if (kernel->base.virt_node_cnt) {
            vnode = kernel->virt_node_list;

            printf("Virtual Node list for DPU Kernel: %s\n", kernel->base.name);
            for (i=0; i < kernel->base.virt_node_cnt; i++) {
                prtShapeOut = &(vnode[i].base_v1.shapeOut);

                printf("%s%d %s%s\n", "NodeID-", i, "Name-", kernel->virt_node_list[i].base_v1.base.name);

                printf("%16s  0x%x\n", "O_Height:",    prtShapeOut->height);
                printf("%16s  0x%x\n", "O_Width:",     prtShapeOut->width);
                printf("%16s  0x%x\n", "O_Channel:",   prtShapeOut->channel);
                printf("%16s  0x%x\n", "O_Address:",   prtShapeOut->offset);
                printf("%16s  0x%x\n", "O_Size:",      prtShapeOut->size);
                printf("%16s  0x%x\n", "O_Fix_Width:", prtShapeOut->fix_width);
                printf("%16s  %d\n",   "O_Fix_Pos:",   prtShapeOut->fix_pos);

                printf("%16s  0x%x\n", "W_Fix_Width:", vnode[i].base_v1.weight_fix_width);
                printf("%16s  %d\n",   "W_Fix_Pos:",   vnode[i].base_v1.weight_fix_pos);

                printf("%16s  0x%x\n", "B_Fix_Width:", vnode[i].base_v1.bias_fix_width);
                printf("%16s  %d\n",   "B_Fix_Pos:",   vnode[i].base_v1.bias_fix_pos);

                if (i != (kernel->base.virt_node_cnt-1)) printf("\n");
            }
        }

        printf(DPU_LINE);
    }
}
#ifdef __cplusplus
}
#endif
