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
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <getopt.h>
#include <sys/stat.h>
#include "ddump.h"
#include "../../common/elf.h"
#include "../../common/err.h"

extern const char* g_dpu_target_name[];
extern const char* g_dpu_arch_name[];
/* for ddump Core library version */
char verDDump[] = "DDump version " DDUMP_VER_STRING;
char file_name[MAX_NAME_LEN] = {0};

/*
 * Read data from .deephi.tensor segment of DPU elf for ABI version after and ABIv1.6.
 */
INTERNAL void dpu_setup_tensor(kernel_t *kernel, elf_t *elf) {
    int i;
    uint32_t *readBase;
    tensor_shape_t *tensors;

    DDUMP_ASSERT(kernel != NULL);
    DDUMP_ASSERT(elf != NULL);

    readBase = (uint32_t*)(elf->elf_data + kernel->elf_tensor.offset);

    kernel->tensor_cnt = kernel->elf_tensor.size/kernel->tensor_size;
    kernel->tensor_list = (tensor_shape_t*)malloc(sizeof(tensor_shape_t) * kernel->tensor_cnt);
    tensors = kernel->tensor_list;
    for(i=0; i<kernel->tensor_cnt; i++) {
        tensors[i].attr           = *readBase++;
        tensors[i].height         = *readBase++;
        tensors[i].width          = *readBase++;
        tensors[i].channel        = *readBase++;
        tensors[i].offset         = *readBase++;
        tensors[i].size           = *readBase++;
        tensors[i].fix_width      = *readBase++;
        tensors[i].fix_pos        = *readBase++;
        tensors[i].channel_stride = *readBase++;
        readBase++;  //reserved
        readBase++;  //reserved
        readBase++;  //reserved
        readBase++;  //reserved
    }
}

/*
 * Constructor of dpu_node_v2_t structure, need to be called somewhere explicitly
 */
dpu_node_v2_t * ddump_node_v2_init(dpu_node_t *node) {
    /* Call parent struct's constructor. */
    dpu_node_init(node);
    return (dpu_node_v2_t *)node;
}

/*
 * setup node info from Node-Pool section in dpu elf for ABI version after and ABIv1.6.
 */
INTERNAL int ddump_setup_kernel_node_v2_0(kernel_t *kernel, elf_t *elf) {
    int ret;
    int i, j, size;
    uint32_t *readBase, tmp;
    char *strtabBase, *name;
    dpu_node_v2_t  *node;
    mem_segment_t *code, *param;

    /* setup tensor segment */
    dpu_setup_tensor(kernel, elf);

    readBase = (uint32_t*)(elf->elf_data + kernel->elf_node_pool.offset);
    strtabBase = (char*)(elf->elf_data + kernel->elf_strtab.offset);

    /* malloc mem for node list */
    size = sizeof(dpu_node_t*) * (kernel->node_cnt);
    kernel->node_list = (dpu_node_t **)malloc(size);
    memset(kernel->node_list, 0, size);

    for(i=0; i<kernel->node_cnt; i++) {
        size = sizeof(dpu_node_v2_t);
        kernel->node_list[i] = (dpu_node_t*)malloc(size);
        memset(kernel->node_list[i], 0, size);
        ddump_node_v2_init(kernel->node_list[i]);
    }

    /* setup node-pool items */
    for (i=0; i<kernel->node_cnt; i++) {
        node = (dpu_node_v2_t*)(kernel->node_list[i]);
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
            for(j=0; j<node->reg_cnt; j++) {
                tmp = *readBase++;
            }
        }

        /* setup input tensor */
        node->input_cnt = *readBase++;
        DDUMP_ASSERT(node->input_cnt >= 0);
        node->input_list =
            (uint32_t*)malloc(sizeof(uint32_t) * node->input_cnt);
        for(j=0; j<node->input_cnt; j++) {
            uint32_t idx = *readBase++;
            node->input_list[j] = idx;
            tensor_shape_t * input = &(kernel->tensor_list[idx]);

            /* only real node has input scale */
            //tensors[i].scale = (float)(pow(2, tensors[i].fix_pos));
            if(node->type == T_NODE_DPU_REAL) {
                /* for memory access load for this node */
                node->memload += input->height * input->width * input->channel;
            }
        }

        /* setup output tensor */
        node->output_cnt = *readBase++;
        DDUMP_ASSERT(node->output_cnt >= 0);
        node->output_list =
            (uint32_t*)malloc(sizeof(uint32_t) * node->output_cnt);
        for(j=0; j<node->output_cnt; j++) {
            uint32_t idx = *readBase++;
            node->output_list[j] = idx;
            tensor_shape_t * output = &(kernel->tensor_list[idx]);

            //output->scale = (float)(pow(2, -output->fix_pos));
            if(node->type == T_NODE_DPU_REAL) {
                /* for memory access load for this node */
                node->memload += output->height * output->width * output->channel;
            }
        }

        /* setup code & param symbols, link them with node by offset&size */
        if (kernel->abi_ver >= DPU_ABI_V1_7) {
            /* setup code section(symbol) */
            node->code_cnt = *readBase++;
            DDUMP_ASSERT(node->code_cnt >= 0);
            if(node->code_cnt > 0) {
                node->elf_code.base.offset = *readBase++;
                node->elf_code.base.size = *readBase++;
                node->node_code.length = node->elf_code.base.size;
                node->elf_code.base.name_idx = *readBase++;
                node->elf_code.base.align = *readBase++;

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
                    node->node_params[j].length = node->elf_params[j].base.size;
                    node->elf_params[j].fix_w = *readBase++;
                    node->elf_params[j].fix_p = *readBase++;
                    node->elf_params[j].base.name_idx = *readBase++;
                    node->elf_params[j].base.align = *readBase++;
                    node->elf_params[j].hight = *readBase++;
                    node->elf_params[j].width = *readBase++;
                    node->elf_params[j].channel = *readBase++;
                    node->elf_params[j].out_channel = *readBase++;
                    char *strtabStart = (char*)(elf->elf_data + kernel->elf_strtab.offset);

                     strcpy(node->node_params[j].name, strtabStart + node->elf_params[j].base.name_idx);

                    if(node->type == T_NODE_DPU_REAL) {
                        /* for memory access load for this node */
                        node->memload += node->node_params[j].length;
                    }
                }
            }
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
        kernel->workloadTotal += (float)(node->workload);
        /* Accumulate total memory access for Kernel */
        kernel->memloadTotal += (float)(node->memload);
    }
    return DDUMP_SUCCESS;
}

int print_version(void)
{
    char buildLabel[256];

    /* print dsight version */
    printf("%s\n", verDDump);
    sprintf(buildLabel, "Build Label: %s %s", __DATE__,  __TIME__);
    printf("%s\n", buildLabel);
    return 0;
}

void print_usage()
{
    printf("DDump - Xilinx DNNDK utility to parse and dump DPU ELF file or\n"
           "        DPU hybrid executable file\n");
    printf("Usage: ddump <option>\n");
    printf(" At least one of the following switches must be given:\n");
    printf(
        "  -f  --file       Specify DPU hybird executable or DPU ELF object\n"
        "  -k  --klist      Display each kernel general info from DPU ELF file\n"
        "                   or DPU hybrid executable file\n"
        "  -d  --dpu        Display DPU architecture info for each kernel\n"
        "  -c  --compiler   Display the DNCC compiler version for each kernel\n"
//        "  -m  --model      Display network model info for each kernel\n"
        "  -a  --all        Display all above info\n"
        "  -v  --version    Display DDump version info\n"
        "  -h  --help       Display this help info\n");
}

static int parser_options(int argc, char *argv[]) {
    int next_option;
    char *in_option;
    /* string listing valid short options letters. */
    const char* const short_options = "vhasmdkcf:";
    int option = DDump_None;

    /* array for describing valid long options. */
    const struct option long_options[] = {
        {"version",   no_argument,          NULL, 'v'},
        {"compiler",   no_argument,          NULL, 'c'},
        {"help",      no_argument,          NULL, 'h'},
        {"all",      no_argument,          NULL, 'a'},
        {"senior",      no_argument,          NULL, 's'},
        {"dpu",      no_argument,          NULL, 'd'},
        {"file",      required_argument,          NULL, 'f'},
        {"klist",      no_argument,          NULL, 'k'},
        {"model",      no_argument,          NULL, 'm'},
        {NULL,        0,                    NULL, 0}
    };

    do {
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
        switch (next_option) {
        case 'f':
            in_option = optarg;
            if (!in_option) {
                DDUMP_ERROR_PRINT(" specify a DPU ELF file or hybird executable.");
                return DDump_None;
            }
            if (strlen(file_name) != 0) {
                DDUMP_ERROR_PRINT(" specify only one DPU ELF file or hybird executable.");
                return DDump_None;
            }
            strcpy(file_name, in_option);
            option |= DDump_File;
            break;
        case 'v':
            option |= DDump_Version;
            break;
        case 'h':
            /* -h or --help */
            option |= DDump_Help;
            break;
        case 'a':
            option |= DDump_All;
            break;
        case 'm':
            option |= DDump_Model;
            break;
        case 'd':
            option |= DDump_DpuV;
            break;
        case 'c':
            option |= DDump_Dnnc;
            break;
        case 's':
            option |= DDump_Senior;
            break;
        case 'k':
            option |= DDump_Klist;
            break;
        case -1:
            if (optind == argc) {
                break;
            }
            // /home/fliu/work/DNNDK/tools/build/ddump: invalid option -- 'p'
            DDUMP_ERROR_PRINT(" %s: invalid option -- \"%s\".\n", argv[0], argv[optind]);
            return DDump_None;
        case '?':
        default:
            return DDump_None;
        }
    } while (next_option != -1);

    if (option == DDump_Version) {
        return option;
    }

    // if have not input any option use -k defaultly.
    if ((option & DDump_All) == DDump_None) {
        option = option | DDump_Klist;
    }

    return option;
}

inline static void ddump_trace_tensors(dpu_node_t *node, kernel_t *kernel) {
    int i;
    tensor_shape_t *tensors;
    dpu_node_v2_t * nd = (dpu_node_v2_t *)node;

    tensors = kernel->tensor_list;
    printf("%25s  ", "Input Tensors:");
    for(i=0; i<nd->input_cnt; i++) {
        printf("ID-%d", nd->input_list[i]);
        if (i == nd->input_cnt - 1) {
            printf("\n");
        } else {
            printf(", ");
        }
    }

    printf("%25s  ", "Output Tensors:");
    for(i=0; i<nd->output_cnt; i++) {
        printf("ID-%d", nd->output_list[i]);
        if (i == nd->output_cnt - 1) {
            printf("\n");
        } else {
            printf(", ");
        }
    }
}

void ddump_print_one_tensor(tensor_shape_t *tensor, int opt) {
    printf("%25s  %d\n", "Heigth:", tensor->height);
    printf("%25s  %d\n", "Width:", tensor->width);
    printf("%25s  %d\n", "Channel:", tensor->channel);
    printf("%25s  %d\n", "Fix Width:", tensor->fix_width);
    printf("%25s  %d\n", "Fix Position:", tensor->fix_pos);
    printf("%25s  %.3f\n", "Scale:", tensor->scale);
    if ((opt & DDump_Senior) != DDump_None) {
        printf("%25s  0x%x\n", "Address:", tensor->offset);
        printf("%25s  %d\n", "Channel Stride:", tensor->channel_stride);
        printf("%25s  %d\n", "Size:", tensor->size);
        printf("%25s  ", "Attribute:");
        if (tensor->attr & TENSOR_ATTR_NORMAL) {
            printf("%s ", "normal");
        }
        if (tensor->attr & TENSOR_ATTR_BOUNDRY_INPUT) {
            printf("%s ", "boundry input");
        }
        if (tensor->attr & TENSOR_ATTR_BOUNDRY_OUTPUT) {
            printf("%s ", "boundry output");
        }
        printf("\n");
    }
}

void ddump_trace_param_infos (dpu_node_t *node, kernel_t *kernel) {
    int i;

    if (kernel->abi_ver >= DPU_ABI_V1_7) {
        dpu_node_v2_t * nd = (dpu_node_v2_t *)node;
        for(i=0; i<nd->param_cnt; i++) {
            if (strstr(nd->node_params[i].name, "bias") != NULL) {
                printf("%25s  ", "Bias In:");
            } else {
                printf("%25s  ", "Weight In:");
            }
            printf("%s %d, ", "Fix_Width", nd->elf_params[i].fix_w);
            printf("%s %d\n",   "Fix_Position", nd->elf_params[i].fix_p);
        }
    }
}

static inline void ddump_get_abi_ver_idx(uint32_t abi_ver, char *abi_str) {
    char v_str[DDUMP_MAX_STR_LEN] = {0};
    strcpy(abi_str, "v");
    uint32_t highv = abi_ver >> 16;
    sprintf(v_str, "%d", highv);
    strcat(abi_str, v_str);
    strcat(abi_str, ".");
    uint32_t lowv = abi_ver & 0x0000ffff;
    sprintf(v_str, "%d", lowv);
    strcat(abi_str, v_str);
    return;
}

inline static void ddump_print_size(uint32_t size, char *str) {
    if (size < MIN_BITS_PER_KB) {
        sprintf(str, "%dBytes", size);
        return;
    } else if (size < MIN_BITS_PER_MB) {
        sprintf(str, "%.2fKB", (float)(size) / BITS_PER_KB);
        return;
    } else {
        sprintf(str, "%0.2fMB", (float)(size) / BITS_PER_MB);
        return;
    }
}

inline static void ddump_print_meanvalue(uint32_t value, char *str) {
    if (value == -1) {
        sprintf(str, "nul");
    } else {
        sprintf(str, "%d", value);
    }
    return;
}

void ddump_get_iptype_str(uint32_t type, char *str) {
    switch (type) {
    case 1:
        strcpy(str, "DPU");
        break;
    case 2:
        strcpy(str, "Softmax");
        break;
    case 3:
        strcpy(str, "Sigmod");
        break;
    case 4:
        strcpy(str, "Resize");
        break;
    case 5:
        strcpy(str, "SMFC");
        break;
    case 6:
        strcpy(str, "YRR");
        break;
    default:
        sprintf(str, "Error 0x%x", type);
        break;
    };
    return;
}

void ddump_print_dpu_target_version(uint32_t ver, char *str) {
    uint32_t v1 = ver & 0x0000000f;
    uint32_t v2 = (ver & 0x000000f0) >> 4;
    uint32_t v3 = (ver & 0x00000f00) >> 8;
    //uint32_t v0 = (ver & 0x000ff000) >> 12;
    sprintf(str, "%x.%x.%x", v3, v2, v1);
}

void ddump_print_ram_usage(uint32_t ram, char *str) {
    if (ram == 2) {
        strcpy(str, "low");
    } else if (ram == 3) {
        strcpy(str, "high");
    } else {
        strcpy(str, "unknown");
    }
    return;
}

void ddump_print_configurable(dpu_configurable_t *config, int opt) {
    char temp_str[MAX_NAME_LEN] = {0};
    printf("%25s\n", "DPU Configuration Parameters");
    if ((opt & DDump_Model) != DDump_None) {
        printf("%25s  0x%x\n", "DPU Arch:", config->base.dpu_arch);
        printf("%25s  0x%x\n", "DPU Freg:", config->base.dpu_freq);
        printf("%25s  %.2f\n", "DPU Peak Perf:", config->base.peak_perf);
        ddump_get_iptype_str(config->sys.sys_ip_type, temp_str);
        printf("%25s  %s\n", "Sys Ip Type:", temp_str);
    }
    ddump_print_dpu_target_version(config->sub_version.ver_target, temp_str);
    printf("%25s  %s\n", "DPU Target Ver:", temp_str);
    printf("%25s  B%d\n", "DPU Arch Type:",
            2 * config->arch.arch_icp * config->arch.arch_ocp * config->arch.arch_pp);
    if ((opt & DDump_Model) != DDump_None) {
        printf("%25s  %d\n", "DPU Regmap Version:", config->sys.sys_regmap_ver);
        printf("%25s  %d Bits\n", "DPU Data Width:", config->arch.arch_data_bw);
        printf("%25s  %d\n", "M-AXI DPU HP Data Width:", config->arch.arch_hp_bw);
    }

    ddump_print_ram_usage(config->arch.arch_img_bkgrp, temp_str);
    printf("%25s  %s\n", "RAM Usage:", temp_str);

    printf("%25s  %s\n", "DepthwiseConv:", config->dwcv.dwcv_parallel ? "Enabled" : "Disabled");
    if ((opt & DDump_Model) != DDump_None) {
      printf("%25s  %d\n", "DepthwiseConv Parallel:", config->dwcv.dwcv_parallel);
      printf("%25s  %s\n", "DepthwiseConv ALU Mode:", config->dwcv.dwcv_alu_mode_enable ? "Enabled" : "Disabled");
    }
    printf("%25s  %s\n", "DepthwiseConv+Relu6:",
            config->dwcv.dwcv_parallel && config->dwcv.dwcv_relu6_enable ? "Enabled" : "Disabled");

    if ((opt & DDump_Model) != DDump_None) {
      printf("%25s  %d\n", "Conv+Parallel:", config->conv.conv_wr_parallel);
    }
    printf("%25s  %s\n", "Conv+Leakyrelu:", config->conv.conv_leakyrelu_enable ? "Enabled" : "Disabled");
    printf("%25s  %s\n", "Conv+Relu6:", config->conv.conv_relu6_enable ? "Enabled" : "Disabled");

    printf("%25s  %s\n", "Channel Augmentation:", config->load.load_augm_enable ? "Enabled" : "Disabled");
    if ((opt & DDump_Model) != DDump_None) {
        printf("%25s  %s\n", "Load Mean Opt:", config->load.load_img_mean_enable ? "Enabled" : "Disabled");
        printf("%25s  %d\n", "Misc Write Parallel:", config->misc.misc_wr_parallel);
        printf("%25s  %d\n", "ElementWise Parallel", config->elew.elew_parallel);
    }
    printf("%25s  %s\n", "Average Pool:", config->pool.pool_average_enable ? "Enabled" : "Disabled");
    if ((opt & DDump_Model) != DDump_None) {
        printf("%25s  %d\n", "Mean Ram Depth:", config->ram.ram_depth_mean);
        printf("%25s  %d\n", "Image Ram Depth:", config->ram.ram_depth_img);
        printf("%25s  %d\n", "Weight Ram Depth:", config->ram.ram_depth_wgt);
        printf("%25s  %d\n", "Bias Ram Depth:", config->ram.ram_depth_bias);
    }
}

int ddump_print_one_kernel(kernel_t *kernel, int opt) {
    int i;
    //dpu_node_v1_virt_t *vnode;
    tensor_shape_t *prtShapeOut;
    char abi_str[DDUMP_MAX_STR_LEN] = {0};
    char size_str[DDUMP_MAX_STR_LEN] = {0};
    ddump_get_abi_ver_idx(kernel->abi_ver, abi_str);

    if ((opt & DDump_All) != DDump_None) {
        // dump the file and kernel name.
        printf("\n");
        printf("%s%s \n", "DPU Kernel name: ", kernel->name);
        printf(DDUMP_SINGLE_LINE);
    }

    if (kernel->abi_ver <= DPU_ABI_V1_0) {
        DDUMP_ERROR_PRINT(" lower DPU ABI version not supported.");
        return DDUMP_FAILURE;
    }

    // dump the mode info.
    if ((opt & DDump_Klist) != DDump_None) {
        printf("\n");
        printf(" -> DPU Kernel general info\n");
        printf("%25s  %s\n", "Mode:", (kernel->mode & K_MODE_DEBUG)? "DEBUG": "NORMAL");
        ddump_print_size(kernel->elf_code.size, size_str);
        printf("%25s  %s\n", "Code Size:", size_str);
        ddump_print_size(kernel->elf_param.size, size_str);
        printf("%25s  %s\n", "Param Size:", size_str);
        printf("%25s  %.3fMOP\n", "Workload MACs:", (kernel->workloadTotal / NUMBER_PER_MOPS));
        ddump_print_size(kernel->IO_space_size, size_str);
        printf("%25s  %s\n", "IO Memory Space:", size_str);

        printf("%25s  ", "Mean Value:");
        ddump_print_meanvalue(kernel->mean_c1, size_str);
        printf("%s, ", size_str);
        ddump_print_meanvalue(kernel->mean_c2, size_str);
        printf("%s, ", size_str);
        ddump_print_meanvalue(kernel->mean_c3, size_str);
        printf("%s\n", size_str);

        printf("%25s  %d\n", "Node Count:", kernel->node_cnt);
        printf("%25s  %d\n", "Tensor Count:", kernel->tensor_cnt);

        tensor_shape_t *tensors = kernel->tensor_list;
        // dump boudle input info.
        printf("%25s\n", "Tensor In(H*W*C)");
        for (i=0; i < kernel->tensor_cnt; i++) {
            if ((tensors[i].attr) & TENSOR_ATTR_BOUNDRY_INPUT) {
                sprintf(size_str, "Tensor ID-%d:", i);
                printf("%25s  %d*%d*%d\n", size_str,
                        tensors[i].height, tensors[i].width, tensors[i].channel);
            }
        }

        // dump boudle output info.
        printf("%25s\n", "Tensor Out(H*W*C)");
        for (i=0; i < kernel->tensor_cnt; i++) {
            if ((tensors[i].attr) & TENSOR_ATTR_BOUNDRY_OUTPUT) {
                sprintf(size_str, "Tensor ID-%d:", i);
                printf("%25s  %d*%d*%d\n", size_str,
                        tensors[i].height, tensors[i].width, tensors[i].channel);
            }
        }
    }

    // dump the DPU arch version.
    if ((opt & DDump_DpuV) != DDump_None) {

        printf("\n");
        printf(" -> DPU architecture info\n");
        printf("%25s  %s\n", "DPU ABI Ver:", abi_str);
        if (kernel->abi_ver < DPU_ABI_V2_0) {
            printf("%25s  %s\n", "DPU Target Ver:", g_dpu_target_name[kernel->dpu_target_ver]);
            printf("%25s  %s\n", "DPU Arch Type:", g_dpu_arch_name[kernel->dpu_arch_ver]);
        } else {
            ddump_print_configurable(&(kernel->dpu_conf), opt);
        }
        //printf("%25s  %s\n", "DPU Arch Kernel:", kernel->dpu_arch);
    }

    // dump the dnnc compiler version
    if ((opt & DDump_Dnnc) != DDump_None) {
        printf("\n");
        printf(" -> DNNC compiler info\n");
        printf("%12s %s\n", "DNNC Ver:", kernel->dnnc_ver);
    }

    // only dump the nodes name.
    if ((opt & DDump_Model) != DDump_None) {
        dpu_node_t **nodes = kernel->node_list;
        tensor_shape_t *tensors = kernel->tensor_list;
        printf("\n");

        // dump tensors info.
        printf(" -> Tensor List\n\n");
        for (i=0; i < kernel->tensor_cnt; i++) {
            printf("%25s  %d\n", "Tensor ID:", i);
            ddump_print_one_tensor(&tensors[i], opt);
            printf("\n");
        }

        // dump the nodes info.
        printf("\n");
        printf(" -> Node list\n\n");

        for (i=0; i < kernel->node_cnt; i++) {
            printf("%10s%d Name:  %s\n", "Node-", i, nodes[i]->name);

            ddump_trace_tensors(nodes[i], kernel);
            printf("%25s  %0.3fMOP\n", "Workload MACs:",
                    ((float)((dpu_node_v2_t*)nodes[i])->workload) / NUMBER_PER_MOPS);

            ddump_trace_param_infos(nodes[i], kernel);

            if (i != (kernel->node_cnt-1)) printf("\n");
        }
    }

#if 0 // now dnnc have not add the predecessor and successors info
    // dump the graph
    if ((opt & DDump_Model) != DDump_None) {
        printf("\n");
        DDUMP_ERROR_PRINT("DPU Graph for Kernel - %s\n", kernel->name);
        nodes = kernel->node_list;
        for (i=0; i < kernel->node_cnt; i++) {
            dpu_node_v2_t *node = (dpu_node_v2_t*)(nodes[i]);
            printf("%10d\t%-30s\n", i, node->base.name);
            printf("%s", "Pred nodes:");
            int k = 0;
            for (k = 0; k < node->pre_cnt; k++) {
                uint32_t preidx = node->pre_list[k];
                printf("%d %s", preidx, nodes[preidx]->name);
                if (k != node->pre_cnt - 1) {
                    printf(", ");
                }
            }
            printf("\n");

            printf("%s", "Succ nodes:");
            for (k = 0; k < node->suc_cnt; k++) {
                uint32_t succidx = node->suc_list[k];
                printf("%d %s", succidx, nodes[succidx]->name);
                if (k != node->suc_cnt - 1) {
                    printf(", ");
                }
            }
            printf("\n");

        }
    }
#endif
    // end dump.
    printf("\n");
    return DDUMP_SUCCESS;
}

int ddump_get_kernels_name(DDump_kernels_t *kernels) {
    elf_t elf = {0};
    kernel_t kernel = {0};
    int ret = DDUMP_FAILURE;
    /* read data from ELF binary into buffer */
    strcpy(kernel.elf_name, file_name);
    ret = elf_read(&kernel, &elf);
    if (ret != DDUMP_SUCCESS) {
        elf_free(&elf);
        DDUMP_ERROR_PRINT(" invalid DPU ELF file or hybrid executable specified.\n");
        return DDUMP_FAILURE;
    }

    char sec_name[MAX_NAME_LEN] = {0};
    int i = 0;
    DDump_kernel_t *pred_kernel = NULL;
    for (i = 0; i < elf.num_sections; i++) {
        /* search kernel's specific section name from symbol table */
        /* Report error and exit in case failure while reading section symbols */
        if (!(elf_get_section_name(&elf, i, sec_name) == DDUMP_SUCCESS)) {
            DDUMP_ERROR_PRINT(" read section name failed.\n");
            return DDUMP_FAILURE;
        }
        char* substr = strstr(sec_name, ELF_METADATA_SEGMENT_STR);
        /* search the ".deephi..." string to find the kernel name */
        if (substr != NULL) {
            DDump_kernel_t *kernel;
            kernel = (DDump_kernel_t *)malloc(sizeof(DDump_kernel_t));
            memset(kernel, 0, sizeof(DDump_kernel_t));

            if (NULL == kernel) {
                return DDUMP_FAILURE;
            }

            if (kernels->kernel_num == 0) {
                kernels->list = kernel;
            } else {
                pred_kernel->next = kernel;
            }

            strcpy(kernel->base.name, &sec_name[strlen(ELF_METADATA_SEGMENT_STR)]);
            kernels->kernel_num++;
            pred_kernel = kernel;
        }
    }
    elf_free(&elf);
    if (kernels->kernel_num <= 0) {
        DDUMP_ERROR_PRINT(" not found any DPU kernel.");
        return DDUMP_FAILURE;
    }

    int idx = 0;
    DDump_kernel_t *curr = NULL;
    curr = kernels->list;
    while (idx++ < kernels->kernel_num && curr != NULL) {
        DDump_kernel_t * next = curr->next;
        curr = next;
    }

    DDUMP_ASSERT(idx - 1 == kernels->kernel_num && curr == NULL);

    return DDUMP_SUCCESS;
}

/*
 * get dpu arch from dpu_version str in metadata, which looks like "DPU_1152A"
 */
dpu_arch_t ddump_get_arch_from_str(char *dpu_arch) {
    char name1[] = "1024";
    char name2[] = "1152";
    char name3[] = "4096";
    char* name[] = {name1, name2, name3};
    int i = 0;

    DDUMP_ASSERT(dpu_arch != NULL);

    for(i=1; i<DPU_ARCH_RESERVE; i++) {
        if(strstr(dpu_arch, name[i-1])) {
            return (dpu_arch_t)(i);
        }
    }

    DDUMP_ERROR_PRINT(" invalid arch info from metadata dpu_version string.");
    return DPU_ARCH_UNKNOWN;
}

/*
 * Load Metadata from dpu elf
 */
int ddump_elf_load_meta(kernel_t *kernel, elf_t *elf) {
    int ret;
    uint32_t* metaStart;
    char *strtabStart;
    int charIndex;
    uint32_t  nameIndex;

    metaStart = (uint32_t*)(elf->elf_data + kernel->elf_meta.offset);
    strtabStart = (char*)(elf->elf_data + kernel->elf_strtab.offset);
    /* setup DPU arch string */
    charIndex = 0;
    nameIndex = *(uint32_t*)(metaStart + OFF_META_DPU_ARCH);
    while (*(strtabStart+nameIndex) != '\0') {
        kernel->dpu_arch[charIndex] = *(strtabStart+nameIndex);
        nameIndex++;
        charIndex++;
    }
    /* setup dnnc version string */
    charIndex = 0;
    nameIndex = *(uint32_t*)(metaStart + OFF_META_VER_DNNC);
    while (*(strtabStart+nameIndex) != '\0') {
        kernel->dnnc_ver[charIndex] = *(strtabStart+nameIndex);
        nameIndex++;
        charIndex++;
    }
    /* setup for kernel mode */
    uint32_t mode = *(uint32_t*)(metaStart + OFF_META_MODE);

    if (mode & ELF_KERNEL_DEBUG) {
        kernel->mode = K_MODE_DEBUG;
    } else {
        kernel->mode = K_MODE_NORMAL;
    }
    /* kernel's node count: including virtual nodes */
    kernel->node_cnt = *(uint32_t*)(metaStart + OFF_META_NODE_CNT);
    /* kernel's node size */
    kernel->tensor_size = *(uint32_t*)(metaStart + OFF_META_TENSOR_SIZE);
    /* kernel's memory IO size */
    kernel->IO_space_size = *(uint32_t*)(metaStart + OFF_META_KERNEL_IO_SIZE);

    /* kernel's mean value for 3 channels */
    kernel->mean_c1= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C1));
    kernel->mean_c2= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C2));
    kernel->mean_c3= (int32_t)(*(uint32_t*)(metaStart + OFF_META_KERNEL_MEAN_C3));

    /* ABI version of dpu elf */
    kernel->abi_ver= (uint32_t)(*(uint32_t*)(metaStart + OFF_META_ABI_VER));

    /* Version check */
    /* No abi_ver, original elf version */
    if(kernel->abi_ver == DPU_ABI_ORIGIN) {
        /* set target v1.1.3 as default */
        kernel->dpu_target_ver = DPU_TARGET_V1_1_3;
        /* get arch from dpu_version string */
        kernel->dpu_arch_ver = ddump_get_arch_from_str(kernel->dpu_arch);
    } else {
        /* set arch/target version to the value readed from DPU elf metadata */
        uint32_t version   = *(uint32_t*)(metaStart + OFF_META_DPU_VER);
        kernel->dpu_arch_ver   = DPU_ARCH_VER(version);
        kernel->dpu_target_ver = DPU_TARGET_VER(version);
    }

    return DDUMP_SUCCESS;
}

int ddump_elf_load_kernel(kernel_t *kernel) {
    elf_t elf = {0};
    int ret = DDUMP_FAILURE;
    /* read data from ELF binary into buffer */
    ret = elf_read(kernel, &elf);
    if (ret != DDUMP_SUCCESS) {
        elf_free(&elf);
        DDUMP_ASSERT(0);
    }
    if (elf.elf_data == 0) {
        elf_free(&elf);
        DDUMP_ASSERT(0);
    }

    int i, j;
    dpu_segment_t dpu_seg[DPU_SEGMENT_NUM] = {0};

    /* initial the dpu_segments info firstly. */
    elf_segment_init(dpu_seg);

    /* construct segment names */
    for(i = 0; i < DPU_SEGMENT_NUM; i++) {
        strcat(dpu_seg[i].seg_name, kernel->name);
    }

    ret = elf_segment_read(&elf, dpu_seg);
    if (ret != DDUMP_SUCCESS) {
        DDUMP_ERROR_PRINT("elf_segment_read failture!");
        elf_free(&elf);
        return ret;
    }

    /* copy the dpu_seg info to the kernel`s segment */
    elf_copy_segment_kernel(dpu_seg, kernel);

    /* load metadata of dpu elf, setup funcs according to ABI version */
    ret = ddump_elf_load_meta(kernel, &elf);
    if (ret != DDUMP_SUCCESS) {
        elf_free(&elf);
        return ret;
    }

    ret = ddump_setup_kernel_node_v2_0(kernel, &elf);
    if (ret != DDUMP_SUCCESS) {
        elf_free(&elf);
        return ret;
    }

    if (kernel->abi_ver >= DPU_ABI_V2_0) {
        ret = elf_read_configurable(kernel, &elf);
    }

    return ret;
}

void free_kernels(DDump_kernels_t *kernels) {
    int idx = 0;
    DDump_kernel_t *curr = NULL;
    curr = kernels->list;
    while (curr != NULL) {
        DDump_kernel_t * next = curr->next;
        free(curr);
        curr = next;
    }
    return;
}

/*
 *  load each kernel`s info from elf file.
 */
int ddump_load_elf_kernels(DDump_kernels_t *kernels) {
    int ret = DDUMP_FAILURE;
    int idx = 0;
    DDump_kernel_t *curr = NULL;
    curr = kernels->list;
    while (curr != NULL) {
        strcpy(curr->base.elf_name, file_name);
        // get the kernel info
        ret = ddump_elf_load_kernel(&(curr->base));
        if (ret != DDUMP_SUCCESS) {
            DDUMP_ERROR_PRINT(" load DPU kernel %s from file %s failed.\n",
                    curr->base.name, curr->base.elf_name);
            return ret;
        }
        curr = curr->next;
    }
    return ret;
}

/*
 * get the kernel info and dump with option OPT.
 */
int ddump_print_kernels_klist(DDump_kernels_t *kernels, int opt) {
    int ret = DDUMP_FAILURE;
    int idx = 0;
    DDump_kernel_t *curr = NULL;
    // print kernels name list.
    printf("DPU Kernel List from file %s\n", file_name);
    printf("%25s  %s\n", "ID:", "Name");
    curr = kernels->list;
    while (curr != NULL) {
        printf("%24d:  %s\n", idx, curr->base.name);
        if (idx == kernels->kernel_num) {
            printf("\n");
        }
        curr = curr->next;
        idx++;
    }

    // print kernels other info.
    int bits = 0xffffff00 | (~DDump_Model);
    int head_opt = opt & bits;
    curr = kernels->list;
    idx = 0;
    while (curr != NULL) {
        ret = ddump_print_one_kernel(&(curr->base), head_opt);
        curr = curr->next;
    }

    return ret;
}

/*
 * get the kernel info and dump with option OPT.
 */
int ddump_print_kernels_model(DDump_kernels_t *kernels, int opt) {
    if ((opt & DDump_Model) == DDump_None) {
        return DDUMP_SUCCESS;
    }

    int ret = DDUMP_FAILURE;
    int idx = 0;
    DDump_kernel_t *curr = NULL;
    // print kernels other info.
    int bits = 0xffffff00 | ~(DDump_Klist | DDump_DpuV | DDump_Dnnc);
    int head_opt = opt & bits;
    curr = kernels->list;
    idx = 0;
    while (curr != NULL) {
        ret = ddump_print_one_kernel(&(curr->base), head_opt);
        curr = curr->next;
    }

    return ret;
}

int main(int argc, char *argv[])
{
    int ret = DDUMP_SUCCESS;
    int opt = 0;

    // parser the options firstly.
    opt = parser_options(argc, argv);
    // display version info
    if ((opt & DDump_Version) != DDump_None) {
        print_version();
        if (opt == DDump_Version) {
            return DDUMP_SUCCESS;
        }
    }

    // print the help info
    if (opt == DDump_None || ((opt & DDump_Help) != DDump_None)) {
        print_usage();
        return DDUMP_SUCCESS;
    }

    // need both input file and at least one display option.
    if ((opt & DDump_File) == DDump_None || (opt & DDump_All) == DDump_None) {
        DDUMP_ERROR_PRINT(" please specify options as below:\n");
        print_usage();
        return DDUMP_FAILURE;
    }

    /* check if trace file exist */
    struct stat fstat;
    if (stat(file_name, &fstat)) {
        DDUMP_ERROR_PRINT(" fail to locate the specified file %s.\n",
                file_name);
        return DDUMP_FAILURE;
    }

    DDump_kernels_t kernels = {0};
    // get kernel number and names from elf or hybird file.
    ret = ddump_get_kernels_name(&kernels);
    if (ret == DDUMP_FAILURE) {
        //DDUMP_ERROR_PRINT("get kernels failed");
        free_kernels(&kernels);
        return DDUMP_FAILURE;
    }

    // get each kernel info.
    ret = ddump_load_elf_kernels(&kernels);
    if (ret == DDUMP_FAILURE) {
        DDUMP_ERROR_PRINT(" load DPU kernel from file %s failed.", file_name);
        free_kernels(&kernels);
        return DDUMP_FAILURE;
    }

    // dump each kernel head
    ret = ddump_print_kernels_klist(&kernels, opt);
    if (ret == DDUMP_FAILURE) {
        DDUMP_ERROR_PRINT(" display DPU kernel list info failed.");
        free_kernels(&kernels);
        return DDUMP_FAILURE;
    }
    // dump each kernel nodes info.
    ret = ddump_print_kernels_model(&kernels, opt);
    if (ret == DDUMP_FAILURE) {
        DDUMP_ERROR_PRINT(" display network model info failed.");
        free_kernels(&kernels);
        return DDUMP_FAILURE;
    }
    free_kernels(&kernels);
    return ret;
}
