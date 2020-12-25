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

#ifndef _DPU_TYPES_H_
#define _DPU_TYPES_H_

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "../n2cube/src/aol/dpu_aol.h"

/* DPU Kernel Mode definitions */
#define K_MODE_NORMAL      (1<<0)
#define K_MODE_DEBUG       (1<<1)
#define MAX_NAME_LEN       (1024*2)

#define  LSB16(ver)     (ver & 0xFFFF)
#define  MSB16(ver)     ((ver >> 16) & 0xFFFF)
extern const int OFF_META_DPU_ARCH;
extern const int OFF_META_VER_DNNC;
extern const int OFF_META_MODE;
extern const int OFF_META_NODE_CNT;
extern const int OFF_META_TENSOR_SIZE;
extern const int OFF_META_KERNEL_IO_SIZE;
extern const int OFF_META_KERNEL_MEAN_C1;
extern const int OFF_META_KERNEL_MEAN_C2;
extern const int OFF_META_KERNEL_MEAN_C3;
extern const int OFF_META_ABI_VER;   //LSB(0~15): minor verion   MSB(16~31): major version
extern const int OFF_META_DPU_VER;  //LSB(0~15): dpu target   MSB(16~31): dpu arch type
extern const int OFF_META_MEM_MODE;
// Get ABI major version, hold MSB(16~31) of ABI version word
#define  ABI_MAJOR_VER(ver)    MSB16(ver)
// Get ABI minor version, hold LSB(0~15) of ABI version word
#define  ABI_MINOR_VER(ver)    LSB16(ver)

#define  DPU_ARCH_VER(ver)     MSB16(ver)
#define  DPU_TARGET_VER(ver)   LSB16(ver)
#define  IS_UNIQUE_MEM_MODE(kernel)  ((kernel)->base.mem_mode == 0)
#define  IS_SPLIT_IO_MODE(kernel)    ((kernel)->base.mem_mode == 1)

/*
 * kernel_handle_t is used as a unique ID number for a specified neural network
 * It is generated while calling function dpu_load_kernel(). For each process, it could
 * call it many times to run different neural networks. Therefore each network
 * have a unique name for differentiation among them.
 *
 */
#define kernel_handle_t unsigned long
#define task_handle_t   unsigned long

typedef struct {
    uint32_t dpu_arch;
    uint32_t dpu_freq;
    float peak_perf;
} dpu_info_base_t;

/** @Descriptor for DPU core
 */
typedef struct {
    dpu_info_base_t base;
    uint32_t dpu_target;
    uint32_t irq;
}dpu_info_t;

typedef struct {
    uint32_t sys_ip_type;
    uint32_t sys_regmap_ver;
} dpu_conf_sys_t;

typedef struct {
    uint32_t ver_target;
} dpu_conf_sub_version_t;

typedef struct {
    uint32_t arch_hp_bw;
    uint32_t arch_data_bw;
    uint32_t arch_img_bkgrp;
    uint32_t arch_pp;
    uint32_t arch_icp;
    uint32_t arch_ocp;
} dpu_conf_arch_t;

typedef struct {
    uint32_t ram_depth_mean;
    uint32_t ram_depth_bias;
    uint32_t ram_depth_wgt;
    uint32_t ram_depth_img;
} dpu_conf_ram_t;

typedef struct {
    uint32_t load_augm_enable;
    uint32_t load_img_mean_enable;
}dpu_conf_load_t;

typedef struct {
    uint32_t conv_leakyrelu_enable;
    uint32_t conv_relu6_enable;
    uint32_t conv_wr_parallel;
}dpu_conf_conv_t;

typedef struct {
    uint32_t pool_average_enable;
} dpu_conf_pool_t;

typedef struct {
    uint32_t elew_parallel;
} dpu_conf_elew_t;

typedef struct {
    uint32_t dwcv_alu_mode_enable;
    uint32_t dwcv_relu6_enable;
    uint32_t dwcv_parallel;
} dpu_conf_dwcv_t;

typedef struct {
    uint32_t misc_wr_parallel;
} dpu_misc_t;

#define DPU_CONF_MAGIC  0x434F4E46

typedef struct {
    dpu_info_base_t base;
    dpu_conf_sys_t sys;
    dpu_conf_sub_version_t sub_version;
    dpu_conf_arch_t arch;
    dpu_conf_ram_t  ram;
    dpu_conf_load_t load;
    dpu_conf_conv_t conv;
    dpu_conf_pool_t pool;
    dpu_conf_elew_t elew;
    dpu_conf_dwcv_t dwcv;
    dpu_misc_t      misc;
} dpu_configurable_t;

typedef enum tensor_attr {
    TENSOR_ATTR_NORMAL         = (1<<0),   /* normal tensor */
    TENSOR_ATTR_BOUNDRY_INPUT  = (1<<1),   /* tensor as original input of kernel boundry */
    TENSOR_ATTR_BOUNDRY_OUTPUT = (1<<2)    /* tensor as final output of kernel boundry */
} tensor_attr_t;

typedef struct {
    uint32_t  offset;       /* the offset into hybrid ARM-DPU binary executable file */
    uint32_t  size;         /* the section size of DPU weight/bias/code/input/output in byte */
    uint16_t  link;         /* the section index of DPU code/data in byte */
} elf_segment_t;

typedef struct tensor_shape {
    tensor_attr_t attr;           /* attribute of the tensor */
    uint32_t      height;         /* height of feature map */
    uint32_t      width;          /* width of feature map */
    uint32_t      channel;        /* channel of feature map */
    uint32_t      offset;         /* offset based on mem base addr of feature map */
    uint32_t      size;           /* size of feature map (NOTE: size maybe larger than height*width*channel due to layer concact) */

    uint8_t       fix_width;      /* fixed info of feature map: width, 8 by defaul at present */
    int8_t        fix_pos;        /* fixed info of feature map: position, -7~7 */

    uint32_t      channel_stride; /* stride for channel */
    char          *tensor_name;

    float         scale;          /* scale value for tensor */  //mv it to input/output for ABI v2
} tensor_shape_t;

enum elf_kernel_mode_t {
    ELF_KERNEL_NORMAL = 0,       /* normal kernel generated by DNNC */
    ELF_KERNEL_DEBUG  = 1        /* debug kernel generated by DNNC */
};

enum elf_node_type_t {
    ELF_NODE_REAL     = 0,       /* real computation node existing in computation graph */
    ELF_NODE_VIRTUAL  = 1        /* virtual node designed for concact operation */
};

typedef struct dpu_node dpu_node_t;
typedef struct baisc_kernel kernel_t;

struct baisc_kernel {
    char                   name[MAX_NAME_LEN];     /* kernel name corresponding to a neural network */
    char                   elf_name[MAX_NAME_LEN]; /* file name of hybrid CPU-DPU ELF binary */
    char                   dpu_arch[MAX_NAME_LEN]; /* DPU arch for this kernel */
    char                   dnnc_ver[MAX_NAME_LEN]; /* dnnc version info for this kernel */

    int                    dpu_dev_fd;     /* device file of /dev/dpu */
    kernel_handle_t        kernel_id;      /* unique ID number for this kernel */

    int                    mode;           /* kernel mode: K_MODE_NORMAL/K_MODE_DEBUG, from metadata section */
    uint32_t               tensor_size;    /* byte size of tensor: from metadata section */
    uint32_t               IO_space_size;  /* Kernel's input/output memory space: from metadata section */
    uint32_t               mean_c1;        /* Kernel's mean value for channel 1: from metadata section */
    uint32_t               mean_c2;        /* Kernel's mean value for channel 2: from metadata section */
    uint32_t               mean_c3;        /* Kernel's mean value for channel 3: from metadata section */
    uint32_t               abi_ver;        /* DPU ELF ABI version */
    uint32_t               mem_mode;
    uint32_t               dpu_arch_ver;   /* DPU arch read from dpu elf, 1:B1024F 2:B1152F  3:B4096F, discarded since from ABIv2.0 */
    uint32_t               dpu_target_ver; /* DPU target version read from dpu elf, 1:1.1.3  2:1.3.0, discarded since from ABIv2.0 */
    dpu_configurable_t     dpu_conf;       /* DPU configurable IP supported since from ABIv2.0 */

    uint32_t               tensor_cnt;     /* tensor count for ABI version after V1.0 */
    tensor_shape_t         *tensor_list;   /* array of all tensors in .deephi.tensor section for ABI v2 */

    float               workloadTotal;  /* Total compuation workload for Kernel */
    float               memloadTotal;   /* Total memory access workload for Kernel */

    uint32_t               node_cnt;       /* real Node count for v1, real&virt count for v2 */
    //dpu_node_t**           node_list;      /* real Node info for v1, real&virt node together for v2 */

    uint32_t               virt_node_cnt;    /* Virtual Node count for v1, will not be used in v2 */
    //dpu_node_v1_virt_t     *virt_node_list;  /* virutal Node info for v1, will not be used in v2 */

    elf_segment_t   elf_meta;       /* hybrid ELF metadata segment info */
    elf_segment_t   elf_code;       /* hybrid ELF code segment info */
    elf_segment_t   elf_weight;     /* hybrid ELF weight segment info */
    elf_segment_t   elf_bias;       /* hybrid ELF bias segment info */
    elf_segment_t   elf_tensor;     /* hybrid ELF node segment info */
    elf_segment_t   elf_strtab;     /* hybrid ELF strtab segment info */

    elf_segment_t   elf_param;      /* hybrid ELF param segment info, for ABIv2.0 */
    elf_segment_t   elf_node_pool;  /* hybrid ELF node-pool segment info, for ABIv2.0 */
    elf_segment_t   elf_conf;       /* hybrid ELF configurable segment info */
    dpu_node_t **node_list;         /* real Node info for v1, real&virt node together for v2 */
};

extern const char* g_dpu_target_name[];
extern const char* g_dpu_arch_name[];

#define SHORT_NAME_LEN     (50)

#define DPU_FUNCTION     "/sys/module/dpu/parameters/function"
#define DPU_EXTENSION    "/sys/module/dpu/parameters/extension"
#define DPU_CACHE        "/sys/module/dpu/parameters/cache"
#define DPU_MODE         "/sys/module/dpu/parameters/mode"
#define DPU_TIMIEOUT     "/sys/module/dpu/parameters/timeout"
#define DPU_PROFILER     "/sys/module/dpu/parameters/profiler"
#define DPU_DRV_VERSION  "/sys/module/dpu/parameters/version"

#define DPU_INPUT_SCALE(fix_p)  (float)(pow(2, fix_p))
#define DPU_OUTPUT_SCALE(fix_p)  (float)(pow(2, -fix_p))

/** @Descriptor for DPU external IP such as softmax, full-connect and resize etc.
 */
typedef struct {
    int valid;     /*whether this extension/IP exists in BOOT.BIN, 0: not exist; 1: exist*/
    int enable;    /*whether use this extension: 0: not use; 1: use*/
    int version;   /*ip verson info*/
    int irq;       /*interupt info*/
} dpu_extension_info_t;

/** @Descriptor for DPU function such as depthwise conv and serdes etc.
 */
typedef struct {
    int version;   /*version info; it means the function do not exist when version is 0.*/
} dpu_function_info_t;

/** @Descriptor for DPU capability
 */
typedef struct {
    uint32_t  magic;  /* 0x434F4E46: "CONF" for configurable IP, for previous version, it's 0 */
    uint32_t  cache;
    char hw_timestamp[32];      /*the time BOOT.BIN was release by hardware team*/
    uint32_t signature_valid;   /*whether BOOT.BIN has a vaild signature, 0: invalid, 1: valid*/
    uint32_t signature_version;
    uint32_t dpu_cnt;           /*DPU core avaiablity: [0, 1, ... dpu_cnt-1]*/
    void     *p_dpu_info; /*point to an arrry, which type is dpu_configurable_t for configurable IP, and dpu_info_t for previous IP*/

    uint32_t hp_width;
    uint32_t data_width;
    uint32_t bank_group;
    uint32_t reg_base;          /*register base address for all DPUs*/
    uint32_t reg_size;          /*register size covering all DPUs*/
    uint32_t irq_base0;
    uint32_t irq_base1;         /*irq_base0 and irq_base1 are used to caculate irqs for IPs in BOOT.BIN*/

    dpu_extension_info_t hdmi;
    dpu_extension_info_t bt1120;
    dpu_extension_info_t fullconnect;
    dpu_extension_info_t softmax;
    dpu_extension_info_t resize;

    dpu_function_info_t relu_p;
    dpu_function_info_t relu_leaky;
    dpu_function_info_t conv_depthwise;
    dpu_function_info_t avgpool;
    dpu_function_info_t serdes_nonlinear;
} dpu_caps_t;

extern dpu_caps_t dpu_caps;

typedef struct mem_segment mem_segment_t;

enum dpu_abi_ver {
    DPU_ABI_ORIGIN  = 0,         /* original ABI version before refactor */
    DPU_ABI_V1_0    = 0x10000,   /* ABIv1.0, add version check */
    DPU_ABI_V1_6    = 0x10006,   /* ABIv1.6, add param section, Node-Pool, multiply IO */
    DPU_ABI_V1_7    = 0x10007,   /* ABIv1.7, link code/param section with node through offset&size in Node-Pool */
    DPU_ABI_V2_0    = 0x20000,   /* ABIv2.0, configurable DPU IP supported. */
    DPU_ABI_V2_1    = 0x20001,
    DPU_ABI_MAX     = DPU_ABI_V2_1/* The biggest ABI version supported, used by ABI backward compatibility checking. */
};

/**
 * Set function ptr.
 * For ABIv1.0 and Origin, it goes to Origin path,
 * otherwise, it goes to v2 function path.
 */
#define SETUP_FUNC_VER(ver, funcPtr, funcOrigin, funcV2_0)            \
    do {                                                              \
        funcPtr = (ver <= DPU_ABI_V1_0) ?  (funcOrigin) : (funcV2_0); \
    }while(0)

#define ALLOWED_KERNEL_MODE (K_MODE_NORMAL|K_MODE_DEBUG)
#define ALLOWED_TASK_MODE   (T_MODE_NORMAL|T_MODE_PROFILE|T_MODE_DEBUG)

enum layer_type_t {
    LAYER_INPUT =  (1<<1),       /* input layer */
    LAYER_OUTPUT = (1<<2)        /* output layer */
};

enum dpu_mem_type_t {
    MEM_CODE   = (1<<1),
    MEM_DATA   = (1<<2),
    MEM_BIAS   = (1<<3),
    MEM_WEIGHT = (1<<4),
    MEM_INPUT  = (1<<5),
    MEM_OUTPUT = (1<<6),
    MEM_PROF   = (1<<7)
};

struct mem_segment {
    enum dpu_mem_type_t type;
    uint32_t            size;               /* segment size which could be larger than length due to alignment factor */
    uint32_t            length;             /* real length of kernel code/data segment in byte */
    uint32_t            addr_phy;           /* physical address of kernel code/data in DPU memory space */
    int8_t*             addr_virt;          /* virtual address of kernel code/data in DPU memory space */
    char                name[MAX_NAME_LEN]; /* layer's name specified in DPU assembly file */
    dpu_aol_dev_mem_t*  p_dev_mem;
};

/*
 * struct for describing DPU data memory region including weight/bias/input/output
 */
typedef struct {
    uint32_t                 region_size;   /* size of data memory region */
    mem_segment_t*           region_start;  /* pointer to the start segment */
} mem_region_t;

struct port_profile_t {
    unsigned int       port_hp0_read_byte;     /* bytes read in HP0 port (128-bit) */
    unsigned int       port_hp0_write_byte;    /* bytes write in HP0 port (128-bit) */

    unsigned int       port_hp1_read_byte;     /* bytes read in HP1 port (128-bit) */
    unsigned int       port_hp1_write_byte;    /* bytes write in HP1 port (128-bit) */

    unsigned int       port_hp2_read_byte;     /* bytes read in HP2 port (128-bit) */
    unsigned int       port_hp2_write_byte;    /* bytes write in HP2 port (128-bit) */

    unsigned int       port_hp3_read_byte;     /* bytes read in HP3 port (128-bit) */
    unsigned int       port_hp3_write_byte;    /* bytes write in HP3 port (128-bit) */

    unsigned int       port_gp_read_byte;      /* bytes read in GP port (32-bit for DPU inst) */
    unsigned int       port_gp_write_byte;     /* bytes write in GP port (32-bit for write profiler) */

    unsigned long long dpu_cycle;              /* the start timestamp before running (RETURNED) */
};

#define REG_NUM  8

typedef enum {
	DPU_HP_WIDTH_0 = 0,
	DPU_HP_WIDTH_1 = 1,
	DPU_HP_WIDTH_2 = 2,
	DPU_HP_WIDTH_3 = 3,
	DPU_HP_WIDTH_4 = 4,
	DPU_HP_WIDTH_5 = 5,
	DPU_HP_WIDTH_6 = 6,
	DPU_HP_WIDTH_7 = 7,
	DPU_HP_WIDTH_RESERVE = 8
} dpu_hp_width_t;

typedef enum {
	DPU_DATA_WIDTH_0 = 0,
	DPU_DATA_WIDTH_1 = 1,
	DPU_DATA_WIDTH_2 = 2,
	DPU_DATA_WIDTH_RESERVE = 3
} dpu_data_width_t;

typedef enum {
	DPU_BANK_GROUP_0 = 0,
	DPU_BANK_GROUP_1 = 1, /* Invalid value for DPU bank group */
	DPU_BANK_GROUP_2 = 2,
	DPU_BANK_GROUP_3 = 3,
	DPU_BANK_GROUP_4 = 4,
	DPU_BANK_GROUP_RESERVE = 5
} dpu_bank_group_t;

typedef enum {
    DPU_ARCH_UNKNOWN  = 0,        /* without dpu arch info */
    DPU_ARCH_B1024F   = 1,        /* value for dpu arch B1024F in dpu elf and bit signature */
    DPU_ARCH_B1152F   = 2,        /* value for dpu arch B1152F in dpu elf and bit signature */
    DPU_ARCH_B4096F   = 3,        /* value for dpu arch B4096F in dpu elf and bit signature */
    DPU_ARCH_B256F    = 4,        /* value for dpu arch B256F in dpu elf and bit signature */
    DPU_ARCH_B512F    = 5,        /* value for dpu arch B512F in dpu elf and bit signature */
    DPU_ARCH_B800F    = 6,        /* value for dpu arch B800F in dpu elf and bit signature */
    DPU_ARCH_B1600F   = 7,        /* value for dpu arch B800F in dpu elf and bit signature */
    DPU_ARCH_B2048F   = 8,        /* value for dpu arch B2048F in dpu elf and bit signature */
    DPU_ARCH_B2304F   = 9,        /* value for dpu arch B2304F in dpu elf and bit signature */
    DPU_ARCH_B8192F   = 10,       /* value for dpu arch B8192F in dpu elf and bit signature */
    DPU_ARCH_B3136F   = 11,       /* value for dpu arch B3136F in dpu elf and bit signature */
    DPU_ARCH_B288F    = 12,       /* value for dpu arch B288F in dpu elf and bit signature */
    DPU_ARCH_B144F    = 13,       /* value for dpu arch B144F in dpu elf and bit signature */
    DPU_ARCH_B5184F   = 14,       /* value for dpu arch B5184F in dpu elf and bit signature */
    DPU_ARCH_RESERVE  = 15
} dpu_arch_t;

typedef enum {
    DPU_TARGET_UNKNOWN  =  0,     /* without dpu target info */
    DPU_TARGET_V1_1_3   =  1,     /* value for dpu target v1.1.3 in dpu elf and bit signature */
    DPU_TARGET_V1_3_0   =  2,     /* value for dpu target v1.3.0 in dpu elf and bit signature */
    DPU_TARGET_V1_3_1   =  3,     /* value for dpu target v1.3.1 in dpu elf and bit signature */
    DPU_TARGET_V1_3_2   =  4,     /* value for dpu target v1.3.2 in dpu elf and bit signature */
    DPU_TARGET_V1_3_3   =  5,     /* value for dpu target v1.3.3 in dpu elf and bit signature */
    DPU_TARGET_V1_3_4   =  6,     /* value for dpu target v1.3.4 in dpu elf and bit signature */
    DPU_TARGET_V1_3_5   =  7,     /* value for dpu target v1.3.5 in dpu elf and bit signature */
    DPU_TARGET_V1_4_0   =  8,     /* value for dpu target v1.4.0 in dpu elf and bit signature */
    DPU_TARGET_V1_4_1   =  9,     /* value for dpu target v1.4.1 in dpu elf and bit signature */
    DPU_TARGET_V1_4_2   =  10,    /* value for dpu target v1.4.2 in dpu elf and bit signature */
    DPU_TARGET_V1_3_6   =  11,    /* value for dpu target v1.3.6 in dpu elf and bit signature */
    DPU_TARGET_V1_3_7   =  12,    /* value for dpu target v1.3.7 in dpu elf and bit signature */
    DPU_TARGET_RESERVE  =  13
} dpu_target_t;

#endif
