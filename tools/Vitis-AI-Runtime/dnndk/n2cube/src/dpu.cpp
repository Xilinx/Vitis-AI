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
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <pthread.h>

#include "dnndk/n2cube.h"
#include "vai/xdpurunner.h"

#ifdef USE_ARM_NEON
#include "neonopt.h"
#endif

#ifndef PATH_MAX
	#define PATH_MAX 1024
#endif
extern char vitisKernelPath[PATH_MAX];

#ifdef __cplusplus
extern "C" {
#endif
#include "../../common/dpu_types.h"
#include "dpu_task.h"
#include "dpu_err.h"
#include "dpu_sys.h"
#include "dpu_def.h"
#include "dpu_ld.h"
#include "../../common/dpu_node_v2.h"
#include "task_node_v1.h"
#include "task_node_v2.h"
#include "task_tensor.h"
#include "dpu_shm.h"
#include "dpu_caps.h"

#include "aol/dpu_aol.h"
#include "dpu_scheduler.h"
#include "dpu_1to1.h"

using namespace std;

mutex g_mtx;

static char copyright[] = "Copyright 2019 Xilinx Inc.";

struct n2cube_dpu_kernel_list_t {
    uint32_t count;
    dpu_kernel_t *kernel;
    struct n2cube_dpu_kernel_list_t *next;
};
struct n2cube_dpu_kernel_list_t *gp_n2cube_dpu_kernel_list = NULL;
pthread_mutex_t g_n2cube_dpu_kernel_list_mtx = PTHREAD_MUTEX_INITIALIZER;

extern const char* g_dpu_target_name[];
extern const char* g_dpu_arch_name[];

extern void dpuInitExceptionMode(void);
extern int dpu_dev_mem_alloc(mem_segment_t *seg, uint32_t size);
extern int dpu_dev_mem_free(mem_segment_t *seg);
extern int (*get_virtual_node_id) (dpu_task_t *, const char *);

extern dpu_aol_dev_handle_t *gp_dpu_aol_handle;

dpu_caps_t dpu_caps;
char g_extension_displayed;

/* the pointer to dpu trace file */
static FILE *dpu_trace_fd;
static mutex dpu_trace_mutex;
static mutex dpu_trace_print_mutex;

/* Directory name for dump */
typedef struct st_dpu_dump_thread_chain{
    int tid;
    struct st_dpu_dump_thread_chain *next;
} dpu_dump_thread_chain_t;
static dpu_dump_thread_chain_t *dpu_dump_thread_chain;
static char dpu_dump_dir_name[SHORT_NAME_LEN];
static mutex dpu_dump_mutex;

#ifdef __cplusplus
}
#endif

/* softmax calculation on arm*/
void softmax_on_arm(float *dst, int8_t *src, int size, float scale, int batch);

/* interanl functions used only inside DPU runtime library */
INTERNAL void setup_regs_from_assignment(dpu_kernel_t  *kernel,
                                                    dpu_task_t    *task,
                                                    uint32_t      node_id,
                                                    dpu_aol_run_t *session);
INTERNAL int run_task_as_whole(dpu_task_t *task);
INTERNAL int run_task_in_node(dpu_task_t *task);

INTERNAL int print_task_profile(dpu_task_t *task, int count);
INTERNAL int print_task_trace_time(dpu_task_t *task, int count);

INTERNAL char *get_node_name(dpu_task_t *task, int nodeID);

/* dump raw data in binary format by default */
INTERNAL int dump_node_code(dpu_task_t *task, int ID);
INTERNAL int dump_node_bias(dpu_task_t *task, int ID);
INTERNAL int dump_node_weights(dpu_task_t *task, int ID);
INTERNAL int dump_node_input(dpu_task_t *task, int ID);
INTERNAL int dump_node_output(dpu_task_t *task, int ID);
INTERNAL int get_node_id (dpu_task_t *, const char *);

#ifdef __cplusplus
extern "C" {
#endif

static int mask2id(uint32_t mask) {
    int i;
    uint32_t test = 1;
    for (i = 0; i < 32; i++) {
        if ((mask & test) != 0) {
            break;
        }
        test <<= 1;
    }

    return i;
}

/*
 * get verson for DNNDK
 */
EXPORT void dnndk_version(char buf[], int size)
{
    memset(buf, 0x0, size);
    snprintf(buf, size, "%s", VITIS_AI_VERSION);
}

/*
 * get copyright
 */
EXPORT void dnndk_copyright(char buf[], int size)
{
    memset(buf, 0x0, size);
    snprintf(buf, size, "%s", copyright);
}

/*
 * get verson for N2Cube
 */
EXPORT void n2cube_version(char buf[], int size)
{
    memset(buf, 0x0, size);
    snprintf(buf, size, "%s", N2Cube_VERSION);
}

/*
 * get builddate for N2Cube
 */
EXPORT void n2cube_builddate(char buf[], int size)
{
    memset(buf, 0x0, size);
    snprintf(buf, size, "%s %s", __DATE__, __TIME__);
}

static void dpuDumpExtensionTitle()
{
    if (0 == g_extension_displayed) {
        printf("\n[DPU Extension List]");
        g_extension_displayed = 1;
    }
}

static const char * getEnableStr(int enable)
{
    const static char ENABLED_STR[3][8] = {
        "No",
        "Yes",
        "Invalid"
    };

    if ((enable < 0) || (enable > 1)) {
        enable = 2;
    }

    return ENABLED_STR[enable];
}

void dump_configurable_signature() {
    int idx = 0;
    printf("\n[DPU Core Configuration List]\n");
    for (idx=0; idx< dpu_caps.dpu_cnt ; idx++) {
        printf("%-25s: #%d\n", "DPU Core", idx);
        printf("%-25s: %s\n", "DPU Enabled", getEnableStr(1));

        dpu_configurable_t *p_dpu_info = (dpu_configurable_t *)dpu_caps.p_dpu_info;
        //printf("%-25s: %d\n", "DPU Regmap Version", (p_dpu_info+idx)->sys.sys_regmap_ver);
        printf("%-25s: B%d\n", "DPU Arch", (p_dpu_info+idx)->base.dpu_arch);
        uint32_t ver_num = (p_dpu_info+idx)->sub_version.ver_target;
        printf("%-25s: v%x.%x.%x\n", "DPU Target Version",
                (ver_num & 0xf00) >> 8,
                (ver_num & 0xf0) >> 4,
                ver_num & 0xf);
        printf("%-25s: %d MHz\n", "DPU Freqency", (p_dpu_info+idx)->base.dpu_freq);
        //printf("%-25s: %d\n", "DPU Data Width", (p_dpu_info+idx)->arch.arch_data_bw);
        //printf("%-25s: %dbit\n", "M-AXI DPU HP Data Width", (p_dpu_info+idx)->arch.arch_hp_bw);
        printf("%-25s: %s\n", "Ram Usage", (p_dpu_info+idx)->arch.arch_img_bkgrp == 2 ? "Low" : "High");
        printf("%-25s: %s\n", "DepthwiseConv", (p_dpu_info+idx)->dwcv.dwcv_parallel ? "Enabled" : "Disabled");
        if ((p_dpu_info+idx)->dwcv.dwcv_parallel) {
            //printf("%-25s: %d\n", "DepthwiseConv Parallel", (p_dpu_info+idx)->dwcv.dwcv_parallel);
            //printf("%-25s: %s\n", "DepthwiseConv ALU Mode", (p_dpu_info+idx)->dwcv.dwcv_alu_mode_enable ? "Enabled" : "Disabled");
            printf("%-25s: %s\n", "DepthwiseConv+Relu6", (p_dpu_info+idx)->dwcv.dwcv_relu6_enable ? "Enabled" : "Disabled");
        } else {
            printf("%-25s: %s\n", "DepthwiseConv+Relu6", "Disabled");
        }
        //printf("%-25s: %d\n", "Conv Write Parallel", (p_dpu_info+idx)->conv.conv_wr_parallel);
        printf("%-25s: %s\n", "Conv+Leakyrelu", (p_dpu_info+idx)->conv.conv_leakyrelu_enable ? "Enabled" : "Disabled");
        printf("%-25s: %s\n", "Conv+Relu6", (p_dpu_info+idx)->conv.conv_relu6_enable ? "Enabled" : "Disabled");
        printf("%-25s: %s\n", "Channel Augmentation", (p_dpu_info+idx)->load.load_augm_enable ? "Enabled" : "Disabled");
        //printf("%-25s: %s\n", "Load Mean Opt", (p_dpu_info+idx)->load.load_img_mean_enable ? "Enabled" : "Disabled");
        //printf("%-25s: %d\n", "ElementWise Parallel", (p_dpu_info+idx)->elew.elew_parallel);
        printf("%-25s: %s\n", "Average Pool", (p_dpu_info+idx)->pool.pool_average_enable ? "Enabled" : "Disabled");
        //printf("%-25s: %d\n", "Mean Ram Depth", (p_dpu_info+idx)->ram.ram_depth_mean);
        //printf("%-25s: %d\n", "Image Ram Depth", (p_dpu_info+idx)->ram.ram_depth_img);
        //printf("%-25s: %d\n", "Weight Ram Depth", (p_dpu_info+idx)->ram.ram_depth_wgt);
        //printf("%-25s: %d\n", "Bias Ram Depth", (p_dpu_info+idx)->ram.ram_depth_bias);

        printf("\n");
    }
}

void dump_signature(char disVer) {
    int idx;
    char dpu_features[256];
    /* Get DPU features list */
    memset(dpu_features, '\0', sizeof(dpu_features));
    if (0 != dpu_caps.avgpool.version) {
        strcpy(dpu_features, "Avg-Pooling");
        if (disVer) {
            sprintf(dpu_features, "%s(Ver.%d)", dpu_features, dpu_caps.avgpool.version);
        }
    }
    if (0 != dpu_caps.relu_p.version) {
        if (strlen(dpu_features) != 0) {
            strcat(dpu_features, ", ");
        }
        strcat(dpu_features, "PReLU");
        if (disVer) {
            sprintf(dpu_features, "%s(Ver.%d)", dpu_features, dpu_caps.relu_p.version);
        }
    }
    if (0 != dpu_caps.relu_leaky.version) {
        if (strlen(dpu_features) != 0) {
            strcat(dpu_features, ", ");
        }

        // Bit0 CONV_LEAKYRELU, Bit1 CONV_RELU6, Bit2, DWCV_RELU6
        int relu_flag = 0;
        if (dpu_caps.relu_leaky.version & (~0x7)) {
            strcat(dpu_features, "UnknownReLU");
            if (disVer) {
                sprintf(dpu_features, "%s(Ver.%d)", dpu_features, dpu_caps.relu_leaky.version);
            }
        } else {
            if (dpu_caps.relu_leaky.version & 0x01) {
                relu_flag = 1;
                strcat(dpu_features, "LeakyReLU");
            }
            if (dpu_caps.relu_leaky.version & 0x02) {
                if (relu_flag) strcat(dpu_features, "/");
                relu_flag = 1;
                strcat(dpu_features, "ReLU6");
            }
            if (dpu_caps.relu_leaky.version & 0x04) {
                if (relu_flag) strcat(dpu_features, "/");
                relu_flag = 1;
                strcat(dpu_features, "DWCV_RELU6");
            }
        }
    }
    if (0 != dpu_caps.conv_depthwise.version) {
        if (strlen(dpu_features) != 0) {
            strcat(dpu_features, ", ");
        }
        strcat(dpu_features, "Depthwise Conv");
        if (disVer) {
            sprintf(dpu_features, "%s(Ver.%d)", dpu_features, dpu_caps.conv_depthwise.version);
        }
    }

    printf("\n");
    printf("[DPU Core List]\n");
    for (idx=0; idx< dpu_caps.dpu_cnt; idx++) {
        printf("%-25s: #%d\n", "DPU Core", idx);
        printf("%-25s: %s\n", "DPU Enabled", getEnableStr(1));
        dpu_info_t *p_dpu_info = (dpu_info_t *)dpu_caps.p_dpu_info;
        printf("%-25s: %s\n", "DPU Arch", g_dpu_arch_name[(p_dpu_info+idx)->base.dpu_arch]);
        printf("%-25s: %s\n", "DPU Target", g_dpu_target_name[(p_dpu_info+idx)->dpu_target]);
        printf("%-25s: %d MHz\n", "DPU Freqency", (p_dpu_info+idx)->base.dpu_freq);

 /* no longer display IRQ according Hardware team require */
 //       printf("%-16s: %d\n", "DPU IRQ", (dpu_caps.p_dpu_info+idx)->irq);

        if (strlen(dpu_features) != 0) {
            printf("%-25s: %s\n", "DPU Features", dpu_features);
        }
    }
}

EXPORT void dpuDumpSignatureInfo(char disVer)
{
    if(!dpu_caps.signature_valid) return;

    printf("[DPU IP Spec]\n");
    printf("%-25s: %s\n", "IP  Timestamp", dpu_caps.hw_timestamp);
    printf("%-25s: %d\n", "DPU Core Count", dpu_caps.dpu_cnt);

/* no longer display IRQ according Hardware team require */
//    printf("%-16s: %d\n", "IRQ Base 0", dpu_caps.irq_base0);
//    printf("%-16s: %d\n", "IRQ Base 1", dpu_caps.irq_base1);

    if (dpu_caps.magic == DPU_CONF_MAGIC) {
        dump_configurable_signature();
    } else {
        dump_signature(disVer);
    }

    g_extension_displayed = 0;
    if (0 != dpu_caps.softmax.valid) {
        dpuDumpExtensionTitle();
        printf("\nExtension Softmax\n");
        printf("%-25s: %s\n","Enabled", getEnableStr(dpu_caps.softmax.enable));

/* no longer display IRQ according Hardware team require */
//        printf("%-16s: %d\n","IRQ", dpu_caps.softmax.irq);

        if (disVer) {
            printf("%-25s: %d\n","Version", dpu_caps.softmax.version);
        }
    }

    if (0 != dpu_caps.resize.valid) {
        dpuDumpExtensionTitle();
        printf("\nExtension Resize\n");
        printf("%-25s: %s\n","Enabled", getEnableStr(dpu_caps.resize.enable));

/* no longer display IRQ according Hardware team require */
//        printf("%-16s: %d\n","IRQ", dpu_caps.resize.irq);

        if (disVer) {
            printf("%-25s: %d\n","Version", dpu_caps.resize.version);
        }
    }

    if (0 != dpu_caps.fullconnect.valid) {
        dpuDumpExtensionTitle();
        printf("\nExtension FC\n");
        printf("%-25s: %s\n","Enabled", getEnableStr(dpu_caps.fullconnect.enable));

/* no longer display IRQ according Hardware team require */
//        printf("%-16s: %d\n","IRQ", dpu_caps.fullconnect.irq);

        if (disVer) {
            printf("%-25s: %d\n","Version", dpu_caps.fullconnect.version);
        }
    }

    if (disVer) {
        if (0 != dpu_caps.bt1120.valid) {
            dpuDumpExtensionTitle();
            printf("\nExtension BT1120\n");
            printf("%-25s: %s\n","Enabled", getEnableStr(dpu_caps.bt1120.enable));

/* no longer display IRQ according Hardware team require */
//            printf("%-16s: %d\n","IRQ", dpu_caps.bt1120.irq);

            printf("%-25s: %d\n","Version", dpu_caps.bt1120.version);
        }

        if (0 != dpu_caps.hdmi.valid) {
            dpuDumpExtensionTitle();
            printf("\nExtension HDMI\n");
            printf("%-25s: %s\n","Enabled", getEnableStr(dpu_caps.hdmi.enable));

            /* no longer display IRQ according Hardware team require */
            // When there is no interrupt at version 1, otherwise print the interrupt number.
//            if (1 == dpu_caps.hdmi.version) {
//                printf("%-16s: %s\n","IRQ", "No Interrupt");
//            } else {
//                printf("%-16s: %d\n","IRQ", dpu_caps.hdmi.irq);
//            }

            printf("%-25s: %d\n","Version", dpu_caps.hdmi.version);
        }
    }
}

#ifdef __cplusplus
}
#endif

/*
 * Control entry for debugging N2Cube
 * refer to FLAG_DEBUG_* in include/dpu_def.h
 */
EXPORT int dpuDebug(unsigned long flag)
{
    unsigned long value;
    const char* var = getenv(DPU_ENV_DEBUG);

    if (var) {
        value = strtoul(var, 0, 0);
        return (value & flag);
    }

    return N2CUBE_SUCCESS;
}

/*
 * Control entry for debugging N2Cube
 * refer to FLAG_DEBUG_* in include/dpu_def.h
 */
EXPORT int dpuRuntimeMode(unsigned long flag)
{
    char mode[256];

    /* read runtime mode */
    strcpy(mode, dpu_get_n2cube_mode());

    if (flag == MODE_RUNTIME_NORMAL) {
        if (!strcasecmp(mode, STR_RUNTIME_MODE_NORMAL)) {
            return 1;
        } else {
            return 0;
        }
    } else if (flag == MODE_RUNTIME_PROFILE) {
        if (!strcasecmp(mode, STR_RUNTIME_MODE_PROFILE)) {
            return 1;
        } else {
            return 0;
        }
    } else if (flag == MODE_RUNTIME_DEBUG) {
        if (!strcasecmp(mode, STR_RUNTIME_MODE_DEBUG)) {
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}

/*
 * Control entry for message logging while DPU kernel runs
 * refer to MODE_* in include/dpu_def.h
 */
EXPORT int dpuKernelMode(dpu_kernel_t *kernel, int mode)
{
    return (kernel->base.mode & mode);
}

EXPORT int dpuTaskMode(dpu_task_t *task, int mode)
{
    return (task->mode & mode);
}

#if 0
static int dpu_check_driver_ver() {
    int ret, i;
    FILE *fd;
    char ch, ver[256], *pos;
    const char *ver_str = "DPU Driver version";

    fd = fopen(DPU_DRV_VERSION, "r");
    DPU_ASSERT(fd, ERR_ELF_NO_FILE);

    i = 0;
    ret = fread(&ch, sizeof(char), 1, fd);
    while ( (ret == 1) && (ch != '\n') && (ch != '\0')) {
        ver[i] = ch;
        ret = fread(&ch, 1, sizeof(char), fd);
        i++;
    }
    ver[i] = '\0';

    N2CUBE_DPU_CHECK(pos = strstr(ver, ver_str), N2CUBE_ERR_DPU_DRIVER_VERSION_NONE, "");
    pos = pos + strlen(ver_str);
    while(*pos == ' ') pos++;
    N2CUBE_DPU_CHECK(!(((*pos == '1') || (*pos == '2')) && (*(pos+1) == '.')),
                     N2CUBE_ERR_DPU_DRIVER_MISMATCH,
                     ". DPU driver version: %s, N2Cube version: %s\n"\
                     "    Please update DPU driver to version above v3.0.0",
                     pos, N2Cube_VERSION);

    fclose(fd);

    return N2CUBE_SUCCESS;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * export function of DPU library
 *
 * initialization for running DPU, including
 * 1. open DPU device /dev/dpu
 * 2. allocate related resources for current process
 * 3. terminate process if fail
 */
EXPORT int dpuOpen()
{
    int ret;
    dpuInitExceptionMode();

    dpu_trace_fd = NULL;
    dpu_dump_thread_chain = NULL;
    sprintf(dpu_dump_dir_name, "%s_%d", "dump", getpid());

    // confirm if version of dpu library matches driver
    //dpu_watch_hand();

    dpu_attach();

//    ret = dpu_check_driver_ver();
//    if (ret != N2CUBE_SUCCESS) {
//        return ret;
//    }

    ret = dpu_config();
    if (ret != N2CUBE_SUCCESS) {
        return ret;
    }

    if (dpu_config_shm() == DPU_SHM_FIRST_TIME) {
        dpu_scheduler_init_process(dpu_caps.dpu_cnt, 0, 1);
    } else if (dpu_config_shm() == DPU_SHM_SECOND_TIME) {
        dpu_scheduler_init_process(dpu_caps.dpu_cnt, 0, 0);
    } else {
        return N2CUBE_FAILURE;
    }

    return N2CUBE_SUCCESS;
}

/*
 * export function of DPU library
 *
 * finalization of DPU running, including
 * 1. notify DPU driver to release all resources hold by current process
 * 2. close DPU device /dev/dpu
 */
EXPORT int dpuClose()
{
    dpu_dettach();

    /* Close the file of dpu trace */
    if (dpu_trace_fd != NULL) {
        fclose(dpu_trace_fd);
        dpu_trace_fd = NULL;
    }

    dpu_dump_thread_chain_t *p = dpu_dump_thread_chain;
    while (p != NULL) {
        p = p->next;
        free(dpu_dump_thread_chain);
        dpu_dump_thread_chain = p;
    }

    return N2CUBE_SUCCESS;
}

#ifdef __cplusplus
}
#endif

EXPORT DPUKernel *dpuLoadKernel(const char *netName)
{
    int ret;
    dpu_kernel_t *kernel;
    struct n2cube_dpu_kernel_list_t *p;

    N2CUBE_DPU_CHECK_AND_RET_NULL(netName, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);

    // See if it has been loaded
    pthread_mutex_lock(&g_n2cube_dpu_kernel_list_mtx);
    p = gp_n2cube_dpu_kernel_list;
    while (p != NULL) {
        if (strcmp(p->kernel->base.name, netName) == 0) {
            p->count++;
	    kernel = p->kernel;
            pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
            return kernel;
        }
        p = p->next;
    }

    p = (struct n2cube_dpu_kernel_list_t *)malloc(sizeof(struct n2cube_dpu_kernel_list_t));
    if (NULL == p) {
        pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
        return NULL;
    }

    /* create a DPU kernel structure */
    kernel = (dpu_kernel_t *)malloc(sizeof(dpu_kernel_t));
    if (NULL == kernel) {
        pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
        return NULL;
    }
    memset(kernel, 0, sizeof(dpu_kernel_t));
    p->kernel = kernel;
    p->count = 1;
    p->next = gp_n2cube_dpu_kernel_list;
    gp_n2cube_dpu_kernel_list = p;

    /* set kernel name */
    strcpy(kernel->base.name, netName);
 //   kernel->base.dpu_dev_fd = dpu_dev_fd;

    /* create a unique kernel ID number for this kernel */
    kernel->base.kernel_id = dpu_gen_kernel_id();

    /* load DPU kernel from ELF image into DPU memory space */
    ret = load_kernel(kernel);
    pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
    N2CUBE_DPU_CHECK_AND_RET_NULL(ret==N2CUBE_SUCCESS, N2CUBE_ERR_KERNEL_LOAD, ". netName: %s", netName);

    return kernel;
}

EXPORT int dpuSetKernelMeanValue(DPUKernel *kernel, int mean1, int mean2, int mean3)
{
    N2CUBE_PARAM_CHECK(kernel);

    kernel->base.mean_c1 = mean1;
    kernel->base.mean_c2 = mean2;
    kernel->base.mean_c3 = mean3;

    return N2CUBE_SUCCESS;
}

int dpu_get_boundary_tensor_size(dpu_kernel_t *kernel,
                                 tensor_attr_t type) {
  int i, count = 0;
  int lastTensorIdx = 0;
  int biggestOffset = 0;
  tensor_shape_t *tensor, *tensors;
  DPU_ASSERT(kernel, N2CUBE_ERR_PARAM_NULL);
  DPU_ASSERT(type & (TENSOR_ATTR_BOUNDRY_INPUT | TENSOR_ATTR_BOUNDRY_OUTPUT),
             N2CUBE_ERR_PARAM_NULL);

  tensors = kernel->base.tensor_list;
  for (i = 0; i < kernel->base.tensor_cnt; i++) {
    count = (tensors[i].attr == type) ? (count + 1) : count;//count out all the tensors with "type" attr.
  }
  DPU_ASSERT(count < kernel->base.tensor_cnt, ERR);
  if (count == 0) return 0;

  for (i = 0; i < kernel->base.tensor_cnt; i++) {
    if (tensors[i].attr == type) {
      tensor = &(tensors[i]);

      if(tensor->offset > biggestOffset) {
        biggestOffset = tensor->offset;
        lastTensorIdx = i;
      }
    }
  }
  return tensors[lastTensorIdx].offset + tensors[lastTensorIdx].size;
}

int dpuGetInputTotalSize(DPUTask *task) {
    N2CUBE_PARAM_CHECK(task);
    return dpu_get_boundary_tensor_size(task->kernel, TENSOR_ATTR_BOUNDRY_INPUT);
}

int dpuGetOutputTotalSize(DPUTask *task) {
    N2CUBE_PARAM_CHECK(task);
    return dpu_get_boundary_tensor_size(task->kernel, TENSOR_ATTR_BOUNDRY_OUTPUT);
}

EXPORT DPUTask *dpuCreateTask(DPUKernel *kernel, int mode)
{
    uint32_t  mask;
    DPUTask * newTask;

    N2CUBE_DPU_CHECK_AND_RET_NULL(kernel, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);

    /* check the value of mode: mode 0 is OK */
    if (mode) N2CUBE_DPU_CHECK_AND_RET_NULL ((mode & ALLOWED_TASK_MODE), N2CUBE_ERR_PARAM_VALUE,
        " for API %s. mode: %d", __func__, mode);

    newTask = (DPUTask *)malloc(sizeof(DPUTask));
    memset(newTask, 0, sizeof(DPUTask));
    newTask->kernel = kernel;
    newTask->counter = 0;

    /* set the running mode kernel */
    newTask->mode = mode;

    /* set default priority */
    newTask->schedule_priority = 15;
    /* set default core mask */
    mask = 0x01;
    newTask->binding_core_mask = 0;
    for (int i = 0; i < dpu_caps.dpu_cnt; i++) {
        newTask->binding_core_mask |= mask;
        mask <<= 1;
    }

    if (DPU_RUNTIME_MODE_PROF()) {
        newTask->mode |= T_MODE_PROFILE;
    } else if (DPU_RUNTIME_MODE_DEBUG()) {
        newTask->mode |= T_MODE_DEBUG;
    }

    /* create a unique task ID number for this task */
    newTask->task_id = dpu_gen_task_id();

    /* create task name */
    sprintf(newTask->name, "%s-%ld",  kernel->base.name, newTask->task_id);

    /* allocate private memory space for DPU Task */
    dpu_alloc_task_resource(newTask);

    /* Update each Node's input/output memory address info */
    dpu_update_task_node(newTask);

    /* update Task's virtual Node */
    dpu_update_task_virtual_node(newTask);

    return newTask;
}

int dpuSetTaskPriority(DPUTask *task, uint8_t priority) {
    N2CUBE_PARAM_CHECK_AND_RET(task, ERR_INVALID_TASK);
    if (priority > 15) {
        return N2CUBE_FAILURE;
    }
    task->schedule_priority = priority;
    return N2CUBE_SUCCESS;
}
uint8_t dpuGetTaskPriority(DPUTask *task) {
    N2CUBE_PARAM_CHECK_AND_RET(task, 0xFF);
    return task->schedule_priority;
}

int dpuSetTaskAffinity(DPUTask *task, uint32_t coreMask) {
    uint32_t mask, temp;

    mask = 0;
    temp = 0x01;
    for (int i = 0; i < dpu_caps.dpu_cnt; i++) {
        mask |= temp;
        temp <<= 1;
    }

    N2CUBE_PARAM_CHECK_AND_RET(task, ERR_INVALID_TASK);
    N2CUBE_DPU_CHECK((((coreMask & (~mask)) == 0) && ((coreMask & mask) != 0)), N2CUBE_ERR_PARAM_VALUE,
         " for API %s. The core count is %d, but got coreMask 0x%04x", __func__, dpu_caps.dpu_cnt, coreMask);
    task->binding_core_mask = coreMask;
    return N2CUBE_SUCCESS;
}
uint32_t dpuGetTaskAffinity(DPUTask *task) {
    N2CUBE_PARAM_CHECK_AND_RET(task, 0);
    return task->binding_core_mask;
}

/*
 * export function of DPU library
 *
 * Launch the running of a sequential code on DPU, including
 * 1. produce a unique ID number (returned as kernel_handle_t) for the specified network
 * 2. allocate DPU memory space (consecutive in physical address)
 * 3. load DPU code and data (wights/bias) if network "net_id" is launched
 *    for the first time
 * 4. perform address relocation for DPU code if network "net_id" is launched
 *    for the first time
 *
 * For each DPU kernel, its code segment is loaded into DPU dedicated memory space
 * and will not be flushed out only when fini_dpu() is invoked by its process to
 * release DPU resources. However for DPU data segment (wights/bias/input/output),
 * the allocated memory space will be recycled when there is no enough memory space
 * to run new kernel. When the recycled DPU data is loaded into memory again, the
 * previous memory address space will be allocated to it again. The behind idea is
 * to avoid performing address relocation when DPU code is loaded into DPU
 * memory space. Such logic is implemented in the DPU driver.
 */
EXPORT int dpuRunTask(DPUTask *task)
{
    int ret, i;
    dpu_kernel_t *kernel;
    task_node_t *node;

    DPU_ASSERT(task, ERR_INVALID_TASK);
    if (IS_SPLIT_IO_MODE(task->kernel)) {
        N2CUBE_DPU_CHECK(IS_MEM_INPUT_USED(task) && IS_MEM_OUTPUT_USED(task),
                         ERR_ELF_INVALID,
                         "Kernel [%s] is compiled under split_io mode in DNNC.\n"
                         "%sPlease bind I/O base address using API"
                         " dpuBindInputTensorBaseAddress/dpuBindOutputTensorBaseAddress before API dpuRunTask.",
                         task->kernel->base.name, DPU_MSG_HEADER);
    }

    for (i=0; i < task->kernel->base.node_cnt; i++) {
        node = task->node_list[i];
        node->ops.cache_flush(node, task->kernel->base.node_list[i]);
    }

    kernel = task->kernel;
    if (KERNEL_IN_DEBUG(task->kernel)) {
        /* run DPU kernel in debug mode */
        ret = run_task_in_node(task);
    } else {
        static bool reminded = false;

        if (!reminded) {
            g_mtx.lock();
            if (!reminded) {
                reminded = true;

                if (TASK_IN_DEBUG(task)) {
                    DPU_LOG_MSG(\
                        "Warning: debug facility available only if DPU Kernel [%s] compiled in debug mode by dnnc\n" \
                        "%sWarning: DPU Kernel [%s] will run without debug info produced",  \
                        kernel->base.name, DPU_MSG_HEADER, kernel->base.name);
                } else if (TASK_IN_PROF(task)) {
                    DPU_LOG_MSG(\
                        "Warning: profile facility available only if DPU Kernel [%s] compiled in debug mode by dnnc\n" \
                        "%sWarning: DPU kernel [%s] will run without profile info output",  \
                        kernel->base.name, DPU_MSG_HEADER, kernel->base.name);
                }
            }
            g_mtx.unlock();
        }

        /* run DPU kernel in normal mode */
        ret = run_task_as_whole(task);
    }

    if (IS_UNIQUE_MEM_MODE(task->kernel)) {
        for (i=0; i < task->kernel->base.node_cnt; i++) {
            node = task->node_list[i];
            node->ops.cache_invalid_out(node, task->kernel->base.node_list[i]);
        }

        if(task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
            struct task_virtual_node_t *vnode;
            for (i=0; i < task->kernel->base.virt_node_cnt; i++) {
                vnode = &(task->virt_node_list[i]);
                if (TENSOR_ATTR_BOUNDRY_OUTPUT & vnode->tensorOut.shape->attr) {
                    dpuCacheInvalid(vnode->tensorOut.dev_mem, vnode->tensorOut.shape->offset, vnode->tensorOut.shape->size);
                }
            }
        }
    }

    if (TASK_IN_PROF(task)) {
        /*
         * log Task's each Node perfomrance metric info only for debug mode Kernel
         */
        if (KERNEL_IN_DEBUG(task->kernel)) {
            print_task_profile(task, kernel->base.node_cnt);
        }

        /* Log each running session timestamp of DPU Task
           1. log each Node's timestamp for debug mode Kernel
           2. log whole Task's timestamp for normal mode Kernel
         */
        print_task_trace_time(task, kernel->base.node_cnt);
    }

    task->counter++;
    return ret;
}

EXPORT int dpuDestroyTask(DPUTask *task)
{
    N2CUBE_PARAM_CHECK(task);

    dpu_release_task_resource(task);
    free(task);
    task = NULL;

    return N2CUBE_SUCCESS;
}

EXPORT int dpuEnableTaskDebug(DPUTask *task)
{
    N2CUBE_PARAM_CHECK(task);

    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_LOG_MSG("DPU Kernel [%s] is in non-debug mode", task->kernel->base.name);
        DPU_FAIL_ON_MSG("Debug facility avaialbe only for Kernel built by dnnc compiler in debug mode");
    }

    task->mode |= T_MODE_DEBUG;
    return N2CUBE_SUCCESS;
}

EXPORT int dpuEnableTaskProfile(DPUTask *task)
{
    N2CUBE_PARAM_CHECK(task);

    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_LOG_MSG("DPU Kernel [%s] is in non-debug mode", task->kernel->base.name);
        DPU_FAIL_ON_MSG("Profiling facility avaialbe only for Kernel built by dnnc compiler in debug mode");
    }

    task->mode |= T_MODE_PROFILE;
    return N2CUBE_SUCCESS;
}

EXPORT int dpuDestroyKernel(dpu_kernel_t *kernel)
{
    struct n2cube_dpu_kernel_list_t **pre_link;
    struct n2cube_dpu_kernel_list_t *p;
    N2CUBE_PARAM_CHECK(kernel);

    /* release all the memory space of DPU kernel*/
    pthread_mutex_lock(&g_n2cube_dpu_kernel_list_mtx);
    pre_link = &gp_n2cube_dpu_kernel_list;
    p = gp_n2cube_dpu_kernel_list;
    while (p != NULL) {
        if (p->kernel == kernel) {
            if (p->count > 1) {
                p->count--;
                pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
                return N2CUBE_SUCCESS;
            } else {
                dpu_release_kernel_resource(kernel);
                free(kernel);
                kernel = NULL;
                *pre_link = p->next;
		free(p);
                pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);
                return N2CUBE_SUCCESS;
            }
        }

        pre_link = &p->next;
        p = p->next;
    }
    pthread_mutex_unlock(&g_n2cube_dpu_kernel_list_mtx);

    return N2CUBE_SUCCESS;
}

EXPORT void* dpuAllocMem(int size, int8_t* &addrVirtual, int8_t* &addrPhysical)
{
    N2CUBE_DPU_CHECK_AND_RET_NULL(gp_dpu_aol_handle != NULL,
                                  ERR_OPNE_DPU_DEV,
                                  " Please use API dpuOpen before %s.",
                                  __func__);
    mem_segment_t *seg = (mem_segment_t *)malloc(sizeof(mem_segment_t));
    seg->size = size;
    if (dpu_dev_mem_alloc(seg, size) != 0) {
        DPU_FAIL_ON_MSG("Fail to alloc memory for size: %d", size);
    }
    addrVirtual = (int8_t*)seg->addr_virt;
    addrPhysical = (int8_t*)(int64_t)seg->addr_phy;

    memset(seg->addr_virt, 0xff, size);
    dpuCacheFlush(seg, 0, size);

    return seg;
}

EXPORT void dpuFreeMem(void* handler)
{
    dpu_dev_mem_free((mem_segment_t*)handler);
    free(handler);
}

EXPORT void dpuSyncMemToDev(void *handler, uint32_t offset, uint32_t size)
{
    dpu_aol_sync_to_dev(gp_dpu_aol_handle, ((mem_segment*)handler)->p_dev_mem, offset, size);
}

EXPORT void dpuSyncDevToMem(void *handler, uint32_t offset, uint32_t size)
{
    dpu_aol_sync_from_dev(gp_dpu_aol_handle, ((mem_segment*)handler)->p_dev_mem, offset, size);
}

int dpuBindInputTensorBaseAddress(DPUTask *task, int8_t *addrVirtual, int8_t *addrPhysical) {
    N2CUBE_PARAM_CHECK(task);
    N2CUBE_DPU_CHECK(IS_SPLIT_IO_MODE(task->kernel), ERR_ELF_INVALID,
                     "API [%s] only can be used under split_io mode.\n"
                     "%sPlease compile kernel [%s] with DNNC option --split_io_mem.",
                     __func__, DPU_MSG_HEADER, task->kernel->base.name);
    N2CUBE_DPU_CHECK(task->kernel->base.abi_ver != DPU_ABI_ORIGIN,
                     ERR_ELF_INVALID,
                     " The old ABI of binary can't support split memory, please rebuild it using DNNC with option --abi=1");
    task->mem_input.addr_virt = addrVirtual;
    task->mem_input.addr_phy = (uint64_t)addrPhysical;
    SET_MEM_INPUT_USED(task);
    dpu_update_task_node(task);
    return N2CUBE_SUCCESS;
}

int dpuBindOutputTensorBaseAddress(DPUTask *task, int8_t *addrVirtual, int8_t *addrPhysical) {
    N2CUBE_PARAM_CHECK(task);
    N2CUBE_DPU_CHECK(IS_SPLIT_IO_MODE(task->kernel), ERR_ELF_INVALID,
                     " API [%s] only can be used under split io mode,"
                     " please compile kernel [%s] with DNNC option --split_io_mem.",
                     __FILE__, task->kernel->base.name);
    N2CUBE_DPU_CHECK(task->kernel->base.abi_ver != DPU_ABI_ORIGIN,
                     ERR_ELF_INVALID,
                     " The old ABI of binary can't support split memory, please rebuild it using DNNC with option --abi=1");
    task->mem_output.addr_virt = addrVirtual;
    task->mem_output.addr_phy = (uint64_t)addrPhysical;
    SET_MEM_OUTPUT_USED(task);
    dpu_update_task_node(task);
    return N2CUBE_SUCCESS;
}

DPUTensor *dpuGetBoundaryIOTensor(DPUTask *task, const char *tensorName) {
  N2CUBE_PARAM_CHECK_AND_RET(task && task->kernel, NULL);
  N2CUBE_PARAM_CHECK_AND_RET(tensorName, NULL);
  N2CUBE_DPU_CHECK_AND_RET_NULL(
                   task->kernel->base.abi_ver >= DPU_ABI_V2_1,
                   N2CUBE_ERR_ABI_VERSION,
                   " %s only can be used with the ABI version since from ABIv2.1,"
                   " please update model binary for kernel [%s].",
                   __func__, task->kernel->base.name);
  int i, j;
  task_tensor_t  *tensor;
  task_node_v2_t *node;
  dpu_kernel_t   *kernel = task->kernel;
  for (i = 0; i < kernel->base.node_cnt; i++) {
    node = (task_node_v2_t*)(task->node_list[i]);
    for (j = 0; j < ((dpu_node_v2_t*)kernel->base.node_list[i])->input_cnt; j++) {
      if(0 == strcmp(node->tensorsIn[j].shape->tensor_name, tensorName)) {
        N2CUBE_DPU_CHECK_AND_RET_NULL(
                         node->tensorsIn[j].shape->attr & TENSOR_ATTR_BOUNDRY_INPUT,
                         N2CUBE_ERR_PARAM_NULL,
                         " The tenorName [%s] isn't a boundary tensor of kernel [%s].",
                         tensorName, kernel->base.name);
        return &node->tensorsIn[j];
      }
    }
    for (j = 0; j <((dpu_node_v2_t*)kernel->base.node_list[i])->output_cnt; j++) {
      if(0 == strcmp(node->tensorsOut[j].shape->tensor_name, tensorName)) {
        N2CUBE_DPU_CHECK_AND_RET_NULL(
                         node->tensorsOut[j].shape->attr & TENSOR_ATTR_BOUNDRY_OUTPUT,
                         N2CUBE_ERR_PARAM_NULL,
                         " The tenorName [%s] isn't a boundary tensor of kernel [%s].",
                         tensorName, kernel->base.name);
        return &node->tensorsOut[j];
      }
    }
  }
  N2CUBE_DPU_CHECK_AND_RET_NULL(0, N2CUBE_ERR_TENSOR_NAME,
                                " No tensor exists with the name [%s] for kernel [%s].",
                                tensorName, kernel->base.name);
}


/*
 * Return DPU Task running time in us
 * supposed that:
 *  1. high resolutoin timer (64-bit-length) in Linux kernel never overflows
 *  2. task ending time should always be greater than starting time
 *
 */
EXPORT long long dpuGetTaskProfile(DPUTask *task)
{
    return (task->time_delta/1000);
}

EXPORT float dpuGetTaskProfileInSecond(DPUTask *task)
{
    return ((float)(task->time_delta/1000000.0f)/1000.0f);
}

EXPORT long long dpuGetTaskWallProfile(DPUTask *task)
{
    return (task->time_wall/1000);
}

EXPORT float dpuGetTaskWallProfileInSecond(DPUTask *task)
{
    return ((float)(task->time_wall/1000000.0f)/1000.0f);
}

/*
 * Return DPU Task running time in us
 * supposed that:
 *  1. high resolutoin timer (64-bit-length) in Linux kernel never overflows
 *  2. task ending time should always be greater than starting time
 *
 */
EXPORT long long dpuGetNodeProfile(DPUTask *task, const char *nodeName)
{
    int nodeID = get_node_id(task, nodeName);
    return ((task->node_list[nodeID]->time_end - task->node_list[nodeID]->time_start)/1000); //in us
}

EXPORT float dpuGetNodeProfileInSecond(DPUTask *task, const char *nodeName)
{
    int nodeID = get_node_id(task, nodeName);

    return ((float)((task->node_list[nodeID]->time_end -
        task->node_list[nodeID]->time_start)/1000000.0f)/1000.0f);
}

EXPORT int dpuRunTaskToLayer(DPUTask *task, const char *nodeName)
{
    int i, nodeID;
    int64_t time_delta;
    uint32_t *addr_code;
    dpu_aol_run_t aol_run;
    dpu_kernel_t *kernel;
    dpu_node_t **nodes;
    task_node_t * node;
    mem_segment_t *node_code = NULL;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    kernel = task->kernel;
    nodes = kernel->base.node_list;
    task->time_delta = 0;
    time_delta = 0;

    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_LOG_MSG("DPU Kernel \"%s\" NOT built in debug mode by dnnc compiler.", kernel->base.name);
        DPU_FAIL_ON_MSG("Can't run in Node by Node mode.");
    }

    for (i=0; i < task->kernel->base.node_cnt; i++) {
        node = task->node_list[i];
        node->ops.cache_flush(node, nodes[i]);
    }

    nodeID = get_node_id(task, nodeName);
    for (i=0; i <= nodeID; i++) {
        node_code = nodes[i]->ops.get_node_code(nodes[i]);
        if (node_code != NULL) {
            setup_regs_from_assignment(kernel, task, i, &aol_run);
            addr_code = &(aol_run.regs[aol_run.reg_count].value);
            *addr_code = (uint32_t)(node_code->addr_phy);
            aol_run.reg_count++;

            /* bloked system call to launch DPU running session */
            dpu_launch_execution_session(kernel, task, get_node_name(task, i), &aol_run);

            /* specify starting timestamp (ns) for current node */
            task->node_list[i]->time_start = aol_run.time_start;
            task->node_list[i]->time_end = aol_run.time_end;
            task->node_list[i]->coreID = mask2id(aol_run.core_mask);
        }

        if(node_code) {
            time_delta += task->node_list[i]->time_end - task->node_list[i]->time_start;
        }
        /* dump node */
        if (TASK_IN_DEBUG(task)) {
            dpu_dump_node_by_ID(task, i);
        }
    }

    for (i=0; i <= nodeID; i++) {
        node = task->node_list[i];
        node->ops.cache_invalid_out(node, nodes[i]);
    }

    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        struct task_virtual_node_t *vnode;
        for (i=0; i < task->kernel->base.virt_node_cnt; i++) {
            vnode = &(task->virt_node_list[i]);
            if (TENSOR_ATTR_BOUNDRY_OUTPUT & vnode->tensorOut.shape->attr) {
                dpuCacheInvalid(vnode->tensorOut.dev_mem, vnode->tensorOut.shape->offset, vnode->tensorOut.shape->size);
            }
        }
    }

    task->time_delta = time_delta;
    task->time_wall = task->node_list[nodeID]->time_end -
        task->node_list[0]->time_start;

    if (TASK_IN_PROF(task)) {
        print_task_profile(task, nodeID+1);

        /* Log each running session timestamp of DPU Task */
        print_task_trace_time(task, nodeID+1);
    }

    return N2CUBE_SUCCESS;
}

/**
 * Setup addr info for REGs, according to reg assignment by dnnc.
 */
INTERNAL void setup_regs_from_assignment(dpu_kernel_t  *kernel,
                                                    dpu_task_t    *task,
                                                    uint32_t      node_id,
                                                    dpu_aol_run_t *session) {
    DPU_ASSERT(session, ERR);
    DPU_ASSERT(kernel, ERR);
    DPU_ASSERT(task, ERR);
    DPU_ASSERT(node_id < kernel->base.node_cnt, ERR);

    /* version has reg assigment(after ABIv1.0) */
    if(kernel->base.abi_ver > DPU_ABI_V1_0) {
        int i, reg_count;
        uint32_t *reg;
        uint32_t reg_offset;
        dpu_node_v2_t *node = (dpu_node_v2_t *)(kernel->base.node_list[node_id]);
        dpu_reg_assign_t *assigns = node->reg_type_list;

        DPU_ASSERT(node->reg_cnt <= REG_NUM, ERR);

        // Set AP_REG
        reg_count = 0;
        reg_offset = 0x224;
        if (dpu_caps.signature_version >= 2) { // 1 to 1
            session->regs[reg_count].value = 0;
            session->regs[reg_count].offset = OFFSET_1t01_DPU_CTRL;
            reg_count++;
            session->regs[reg_count].value = 1;
            session->regs[reg_count].offset = OFFSET_1t01_DPU_GLBL_IRQ;
            reg_count++;
            session->regs[reg_count].value = 3;
            session->regs[reg_count].offset = OFFSET_1t01_DPU_IER;
            reg_count++;
            session->regs[reg_count].value = 1;
            session->regs[reg_count].offset = OFFSET_1t01_DPU_IRQ_CLR;
            reg_count++;
            session->regs[reg_count].value = 0x07070f0f;
            session->regs[reg_count].offset = OFFSET_1t01_DPU_HP_BUS;
            reg_count++;
            reg_offset = OFFSET_1to1_DPU_BASE_ADDR_0_L;
        }

        /* init regs */
        session->reg_count = reg_count + REG_NUM;
        for(i=0; i<REG_NUM; i++) {
            session->regs[reg_count + i].value = 0;
            session->regs[reg_count + i].offset = reg_offset + (i * 8);
        }

        /* setup regs according to reg assigment. */
        for(i=0; i<node->reg_cnt; i++) {
            reg = &( session->regs[ reg_count + assigns[i].reg_id ].value );

            switch( assigns[i].data_type ) {
                case T_DATA_IO :                             //for Input
                    *reg = task->mem_IO.addr_phy;
                    break;
                case T_DATA_OUTPUT:                          //for Output
                    *reg = task->mem_output.addr_phy;
                    break;
                case T_DATA_PARAM :                          //for param
                    *reg = kernel->region_param.region_start->addr_phy;
                    break;
                case T_DATA_INPUT :
                    *reg = task->mem_input.addr_phy;
                    break;
                case T_DATA_CODE  :                          //for code
                    *reg = kernel->region_code.region_start->addr_phy;
                    break;
                default:
                    DPU_FAIL_ON_MSG("Invalid data type %d for Register assignment", assigns[i].data_type);
            }
        }
    } else {
        session->reg_count = 2;
        session->regs[0].value = (unsigned long)(kernel->region_wb.region_start->addr_phy);
        session->regs[1].value = (unsigned long)(task->mem_IO.addr_phy);
    }
}

/*
 * Begin to run DPU kernel via calling to driver system call
 */
INTERNAL int run_task_as_whole(dpu_task_t *task)
{
    uint32_t *addr_code;
    dpu_aol_run_t aol_run;
    dpu_kernel_t *kernel;

    kernel = task->kernel;
    DPU_ASSERT(kernel, ERR_LD_INVALID_KERNEL);

    if (KERNEL_IN_DEBUG(task->kernel)) {
        DPU_LOG_MSG("DPU Kernel \"%s\" built in debug mode by dnnc compiler.", kernel->base.name);
        DPU_FAIL_ON_MSG("Can't run in wholely mode.");
    }

    task->time_delta = 0;

    setup_regs_from_assignment(kernel, task, 0, &aol_run);
    addr_code = &(aol_run.regs[aol_run.reg_count].value);
    *addr_code = (uint32_t)(kernel->region_code.region_start->addr_phy);
    aol_run.reg_count++;

    /* bloked system call to launch DPU running session */
    dpu_launch_execution_session(kernel, task, NULL, &aol_run);

    /* specify starting timestamp (ns) */
    task->time_start = aol_run.time_start;
    task->time_end = aol_run.time_end;
    task->coreID = mask2id(aol_run.core_mask);

    task->time_delta = task->time_end - task->time_start;
    task->time_wall = task->time_delta;

    return N2CUBE_SUCCESS;
}

/*
 * Begin to run DPU kernel in debug mode via calling to driver system call
 */
INTERNAL int run_task_in_node(dpu_task_t *task)
{
    int i;
    int64_t time_delta;
    dpu_kernel_t *kernel;
    struct port_profile_t port_profile;
    dpu_node_t **nodes;
    mem_segment_t *node_code = NULL;
    uint32_t *addr_code;
    dpu_aol_run_t aol_run;

    DPU_ASSERT(task, ERR_INVALID_TASK);
    DPU_ASSERT(task->kernel, ERR_LD_INVALID_KERNEL);

    kernel = task->kernel;
    task->time_delta = 0;
    time_delta = 0;

    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_LOG_MSG("DPU Kernel \"%s\" NOT built in debug mode by dnnc compiler.", kernel->base.name);
        DPU_FAIL_ON_MSG("Can't run in debug mode.");
    }

    nodes = kernel->base.node_list;
    for (i=0; i < kernel->base.node_cnt; i++) {
        /* Virt Node(concat) does't have code seg,
         * so we just launch exe session for real Node when node_code is NULL */
        node_code = nodes[i]->ops.get_node_code(nodes[i]);
        if (node_code != NULL) {
            setup_regs_from_assignment(kernel, task, i, &aol_run);
            addr_code = &(aol_run.regs[aol_run.reg_count].value);
            *addr_code = (uint32_t)(node_code->addr_phy);
            aol_run.reg_count++;

            /* bloked system call to launch DPU running session */
            dpu_launch_execution_session(kernel, task, get_node_name(task, i), &aol_run);

            /* specify starting timestamp (ns) for current node */
            task->node_list[i]->time_start = aol_run.time_start;
            task->node_list[i]->time_end = aol_run.time_end;
            task->node_list[i]->coreID = mask2id(aol_run.core_mask);
        }

        if(node_code) {
            /* accumulate time for runing all the nodes */
            time_delta += task->node_list[i]->time_end - task->node_list[i]->time_start;
        }
        /* dump node */
        if (TASK_IN_DEBUG(task)) {
            dpu_dump_node_by_ID(task, i);
        }
    }

    /* profile timing for running all the nodes */
    task->time_delta = time_delta;
    task->time_wall = task->node_list[kernel->base.node_cnt-1]->time_end -
        task->node_list[0]->time_start;

    task->time_start = task->time_end = 0;

    return N2CUBE_SUCCESS;
}

EXPORT int dpuSetTaskInputTensor(DPUTask *task, int8_t *data, int size, int idx)
{
    int i;
    int8_t *input;

    DPU_ASSERT((task && data && (size > 0)), ERR);

    if(task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        task_node_v1_t *tn = (task_node_v1_t *)(task->node_list[0]);
        DPU_ASSERT((size == tn->tensorIn.shape->size), ERR);
        input = tn->tensorIn.addr_virt;
    } else {
        task_node_v2_t *tn = (task_node_v2_t *)(task->node_list[0]);
        DPU_ASSERT((size == tn->tensorsIn[idx].shape->size), ERR);
        input = tn->tensorsIn[idx].addr_virt;
    }

    for (i=0; i < size; i++) {
        input[i] = data[i] ;
    }

    return N2CUBE_SUCCESS;
}

/**
 * Get total number of input Tensor of DPU Task
 */
int dpuGetInputTensorCnt(DPUTask * task, const char * nodeName) {
    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        return 1;
    } else {
        dpu_node_v2_t * node = (dpu_node_v2_t*)(task->kernel->base.node_list[get_node_id(task, nodeName)]);
        return node->input_cnt;
    }
}

/**
 * @brief Get kernel's input tensor (only for real Node)
 *
 * @note supposed that one kernel only have one input tensor
 *
 * @param kernel - the pointer to DPU kernel
 *
 * @return the tensor descriptor for this kernel's input
 */
EXPORT DPUTensor* dpuGetInputTensor(DPUTask *task, const char *nodeName, int idx)
{
    int id, idVirtNode;
    task_tensor_t *tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, NULL);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, NULL);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, NULL);
    }

    /* only for real Node: virtual node for concact doesn't have valid input Tensor */
    id = get_node_id(task, nodeName);

    if ( id < 0) {
        if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
            idVirtNode = get_virtual_node_ID(task, nodeName);

            /* for virtual node constructed for concact super-layer, it has many input Tensors.
               We just report error here. */
            if (idVirtNode > 0 ) {
                DPU_FAIL_ON_MSG("No valid input Tensor for Node %s of DPU kernel %s.\n",
                    nodeName, task->kernel->base.name);

                /* never come here */
                return NULL;
            }
        }

        DPU_FAIL_ON_MSG("Invalid Node name %s specified for DPU kernel %s.\n",
            nodeName, task->kernel->base.name);

        /* never come here */
        return NULL;
    }

    tensor = task->node_list[id]->ops.get_tensorIn(task->node_list[id], idx, task->kernel->base.node_list[id], task->kernel);
    if (NULL == tensor) {
        return NULL;
    }
    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        tensor->shape->attr = (tensor_attr_t)(tensor->shape->attr | TENSOR_ATTR_BOUNDRY_INPUT);
    }
    return tensor;
}

/*
 * Get the start address of DPU Task's input Tensor, multiply IO supported.
 */
EXPORT int8_t* dpuGetInputTensorAddress(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, NULL);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, NULL);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, NULL);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return (tensor->addr_virt);
}

/*
 * Get the size (in byte) of one DPU Task's input Tensor, multiply IO supported.
 */
EXPORT int dpuGetInputTensorSize(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return (tensor->shape->size);
}

/*
 * Get the height dimension of one DPU Task's input Tensor, multiply IO supported.
 */
int dpuGetInputTensorHeight(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return (tensor->shape->height);
}

/*
 * Get the width dimension of one DPU Task's input Tensor, multiple IO supported.
 */
int dpuGetInputTensorWidth(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return (tensor->shape->width);
}

/*
 * Get the channel dimension of one DPU Task's input Tensor, multiple IO supported.
 */
int dpuGetInputTensorChannel(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return (tensor->shape->channel);
}

/*
 * Get the scale value (DPU INT8 quantization) of one DPU Task's input Tensor.
 * For multiple IO.
 */
float dpuGetInputTensorScale(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, 0);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, 0);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, 0);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    return tensor->ops.get_scale(tensor);
}

/**
 * Get total number of output Tensor of DPU Task
 */
int dpuGetOutputTensorCnt(DPUTask * task, const char * nodeName) {
    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        return 1;
    } else {
        dpu_node_v2_t * node = (dpu_node_v2_t*)(task->kernel->base.node_list[get_node_id(task, nodeName)]);
        return node->output_cnt;
    }
}

/**
 * @brief Get one layer's output tensor
 *
 * @note @ref
 *
 * @param kernel - the pointer to DPU kernel
 * @param layer_name - name of this layer
 *
 * @return the tensor descriptor for this layer's output
 */
EXPORT DPUTensor* dpuGetOutputTensor(DPUTask *task, const char *nodeName, int idx)
{
    int id;
    task_tensor_t * tensor;
    struct task_virtual_node_t *vtn;

    N2CUBE_PARAM_CHECK_AND_RET(task, NULL);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, NULL);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, NULL);
    }

    /* search for real Node first */
    if ((id = get_node_id(task, nodeName)) >=0 ) {
        task_node_t *tn = task->node_list[id];
        return tn->ops.get_tensorOut(tn, idx, task->kernel->base.node_list[id], task->kernel);
    }

    if (task->kernel->base.abi_ver <= DPU_ABI_V1_0) {
        /* search for virtual Node first */
        if ((id = get_virtual_node_ID(task, nodeName)) >=0 ) {
            vtn = &(task->virt_node_list[id]);
            tensor = &(vtn->tensorOut);
            if (!(tensor->shape->attr & TENSOR_ATTR_BOUNDRY_OUTPUT)) {
                tensor->shape->attr = (tensor_attr_t)(tensor->shape->attr | TENSOR_ATTR_BOUNDRY_OUTPUT);
                dpuCacheInvalid(tensor->dev_mem, tensor->shape->offset, tensor->shape->size);
            }
            return tensor;
        }
    }

    DPU_FAIL_ON_MSG("Invalid Node name %s specified for DPU kernel %s.\n",
        nodeName, task->kernel->base.name);

    /* never run to here */
    return NULL;
}

/*
 * Get the start address of one DPU Task's output Tensor, multiple IO supported.
 */
EXPORT int8_t* dpuGetOutputTensorAddress(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, NULL);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, NULL);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, NULL);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->addr_virt);
}

INTERNAL uint32_t dpuGetOutputTensorPhyAddress(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, 0);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, 0);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, 0);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->addr_phy);
}

/*
 * Get the size (in byte) of one DPU Task's output Tensor, multiple IO supported.
 */
EXPORT int dpuGetOutputTensorSize(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->shape->size);
}

/*
 * Get the height dimension of one DPU Task's output Tensor, multiple IO supported.
 */
int dpuGetOutputTensorHeight(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->shape->height);
}

/*
 * Get the channel dimension of one DPU Task's output Tensor, multiple IO supported.
 */
int dpuGetOutputTensorWidth(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->shape->width);
}

/*
 * Get DPU Node's output tensor's channel, multiple IO supported.
 */
int dpuGetOutputTensorChannel(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->shape->channel);
}

/*
 * Get the scale value (DPU INT8 quantization) of one DPU Task's output Tensor.
 * For multiple IO.
 */
float dpuGetOutputTensorScale(DPUTask *task, const char *nodeName, int idx)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, 0);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, 0);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, 0);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return tensor->ops.get_scale(tensor);
}

/* Get DPU Node's input tensor's fix pos */
INTERNAL int8_t dpuGetOutputTensorFixPos(DPUTask *task, const char *nodeName, int idx = 0)
{
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK_AND_RET(task, 0);
    N2CUBE_PARAM_CHECK_AND_RET(nodeName, 0);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, 0);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    return (tensor->shape->fix_pos);
}

/*
 * Get the size of tensor
 */
EXPORT int dpuGetTensorSize(DPUTensor* tensor)
{
    N2CUBE_DPU_CHECK(tensor, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
    return (tensor->shape->size);
}

/*
 * Get the start address (virtual) tensor
 */
EXPORT int8_t* dpuGetTensorAddress(DPUTensor* tensor)
{
    N2CUBE_DPU_CHECK_AND_RET_NULL(tensor, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
    return (tensor->addr_virt);
}

/*
 * Get the height dimension of tensor
 */
EXPORT int dpuGetTensorHeight(DPUTensor* tensor)
{
    N2CUBE_DPU_CHECK(tensor, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
    return (tensor->shape->height);
}

/*
 * Get the width dimension of tensor
 */
EXPORT int dpuGetTensorWidth(DPUTensor* tensor)
{
    N2CUBE_DPU_CHECK(tensor, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
    return (tensor->shape->width);
}

/*
 * Get the channel dimension of tensor
 */
EXPORT int dpuGetTensorChannel(DPUTensor* tensor)
{
    N2CUBE_DPU_CHECK(tensor, N2CUBE_ERR_PARAM_NULL, " for API %s", __func__);
    return (tensor->shape->channel);
}

/*
 * Get the width dimension of tensor
 */
EXPORT float dpuGetTensorScale(DPUTensor* tensor)
{
    N2CUBE_PARAM_CHECK_AND_RET(tensor, 0);

    return tensor->ops.get_scale(tensor);
}

/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be in in DPU Tensor order: height, width, channel;
 *       source data type must be int8_t;
 *       source data will be set without conversion
 *
 * @param task - pointer to DPU task
 * @param nodeName - Node name
 * @param buffer - pointer to source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
 int dpuSetInputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    int i, tensorSize;
    task_tensor_t* tensor;
    int8_t *input;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    tensorSize = tensor->shape->size;
    input = tensor->addr_virt;

    N2CUBE_DPU_CHECK(tensorSize == size, N2CUBE_ERR_TENSOR_SIZE,
        " for API %s. node: %s, size: %d", __func__, nodeName, size);

    for (i=0; i < size; i++) {
        input[i] = buffer[i];
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be stored in Caffe blob order: channel, height, width;
 *       source data type must be int8_t;
 *       source data will be converted from Caffe order to DPU order
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to set input tensor
 * @param buffer - pointer to source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    int tensorSize, channel, height, width, idx_c, idx_h, idx_w, idx_hwc, idx_chw;
    task_tensor_t* tensor;
    int8_t *input;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    tensorSize = tensor->shape->size;
    N2CUBE_DPU_CHECK(tensorSize == size, N2CUBE_ERR_TENSOR_SIZE,
        " for API %s. node: %s, size: %d", __func__, nodeName, size);

    channel = tensor->shape->channel;
    height = tensor->shape->height;
    width = tensor->shape->width;

    input = tensor->addr_virt;
    for (idx_h = 0; idx_h < height; idx_h++) {
        for (idx_w = 0; idx_w < width; idx_w++) {
            for (idx_c = 0; idx_c < channel; idx_c++) {
                idx_hwc = idx_h * width * channel + idx_w * channel + idx_c;
                idx_chw = idx_c * height * width + idx_h * width + idx_w;
                input[idx_hwc] = buffer[idx_chw];
            }
        }
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Set DPU input tensor for a Node, multiple IO supported.
 *
 * @note source data must be stored in DPU Tensor order: height, width, channel
 *       source data type must be float
 *       source data will be converted from float to int_8
 *
 * @param task - pointer to DPU task
 * @param nodeName - DPU Node name to set input tensor
 * @param buffer - pointer to source data
 * @param size - size of source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    int i, tensorSize, value;
    float scale;
    int8_t *start;
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    tensorSize = tensor->shape->size;

    N2CUBE_DPU_CHECK(tensorSize == size, N2CUBE_ERR_TENSOR_SIZE,
        " for API %s. node: %s, size: %d", __func__, nodeName, size);

    scale = tensor->ops.get_scale(tensor);
    start = tensor->addr_virt;

    for (i = 0; i < size; i++) {
        value = (int)(buffer[i] * scale);

        /* Invalid pixel values checking for input feature map */
        if ((value>127) || (value<-128)) {
            DPU_LOG_MSG("Invalid pixel value of input tensor: %d", value);
            DPU_FAIL_ON_MSG("Please check if decent tool produces correct quantization info.");
        };

        start[i] = (int8_t)value;
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Set DPU input tensor of a layer, multiple IO supported.
 *
 * @note source data must be stored in Caffe blob order: channel, height, width
 *       source data type must be float
 *       source data will be converted from float to int_8
 *
 * @param task - pointer to DPU task
 * @param nodeName - DPU Node name to set input tensor
 * @param buffer - pointer to source data
 * @param size - size of source data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    int value;
    float scale;
    int tensorSize, channel, height, width, idx_c, idx_h, idx_w, idx_hwc, idx_chw;
    task_tensor_t* tensor;
    int8_t *input;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetInputTensor(task, nodeName, idx);
    tensorSize = tensor->shape->size;
    N2CUBE_DPU_CHECK(tensorSize == size, N2CUBE_ERR_TENSOR_SIZE,
        " for API %s. node: %s, size: %d", __func__, nodeName, size);

    channel = tensor->shape->channel;
    height = tensor->shape->height;
    width = tensor->shape->width;
    scale = tensor->ops.get_scale(tensor);

    input = tensor->addr_virt;
    for (idx_h = 0; idx_h < height; idx_h++) {
        for (idx_w = 0; idx_w < width; idx_w++) {
            for (idx_c = 0; idx_c < channel; idx_c++) {
                idx_hwc = idx_h * width * channel + idx_w * channel + idx_c;
                idx_chw = idx_c * height * width + idx_h * width + idx_w;

                value = (int)(buffer[idx_chw] * scale);
                /* Invalid pixel values checking for input feature map */
                if ((value>127) || (value<-128)) {
                    DPU_LOG_MSG("Invalid pixel value of input tensor: %d", value);
                    DPU_FAIL_ON_MSG("Please check if decent tool produces correct quantization info.");
                };

                input[idx_hwc] = (int8_t)value;
            }
        }
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be in stored Caffe blob order: height, width, channel;
 *       target data type must be int8_t;
 *       target data will be got without conversion
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuGetOutputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    int8_t *outputAddr;
    int tensorSize;
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    outputAddr = tensor->addr_virt;
    tensorSize = tensor->shape->size;

    N2CUBE_DPU_CHECK((size <= tensorSize), N2CUBE_ERR_TENSOR_SIZE,
        " for API %s. node: %s, size: %d", __func__, nodeName, size);

    memcpy(buffer, outputAddr, size);

    return N2CUBE_SUCCESS;
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be in stored Caffe blob order: channel, height, width;
 *       target data type must be int8_t;
 *       target data will be converted from DPU order to Caffe order
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuGetOutputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *buffer, int size, int idx)
{
    int8_t *outputAddr;
    int tensorSize, channel, height, width, idx_c, idx_h, idx_w, idx_hwc, idx_chw;
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    outputAddr = tensor->addr_virt;
    tensorSize = tensor->shape->size;

#if 0
    /* suv: temporaly closed */
    DPU_CHECK(tensorSize >= size,
        "Invalid size \"%d\" specified for DPU Node \"%s\" for DNNDK API \"%s\".",
        size, nodeName, __func__);
#endif

    channel = tensor->shape->channel;
    height = tensor->shape->height;
    width = tensor->shape->width;

    for (idx_c = 0; idx_c < channel; idx_c++) {
        for (idx_h = 0; idx_h < height; idx_h++) {
            for (idx_w = 0; idx_w < width; idx_w++) {
                idx_chw = idx_c * height * width + idx_h * width + idx_w;
                idx_hwc = idx_h * width * channel + idx_w * channel + idx_c;

                if (idx_chw == size) return N2CUBE_SUCCESS;

                buffer[idx_chw] = outputAddr[idx_hwc];
            }
        }
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be stored in DPU Tensor order: height, width, channel;
 *       target data type must be float;
 *       target data will be converted from int8_t to float
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 * @param idx - tensor idx for multiple output, default as 0.
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
EXPORT int dpuGetOutputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    int8_t *outputAddr, *temp;
    int tensorSize, i;
    float scale;
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    outputAddr = tensor->addr_virt;
    tensorSize = tensor->shape->size;
    scale = tensor->ops.get_scale(tensor);

    for (i = 0; i < size; i++) {
        buffer[i] = outputAddr[i] * scale;
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief Get DPU output tensor of a layer, multiple IO supported.
 *
 * @note target data must be stored in Caffe bob order: channel, height, width;
 *       target data type must be float;
 *       target data will be converted from DPU order, int8_t to Caffe order, float
 *
 * @param task - pointer to DPU task
 * @param layerName - layer name to get output tensor
 * @param buffer - pointer to target data
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuGetOutputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *buffer, int size, int idx)
{
    int8_t *outputAddr;
    float scale;
    int tensorSize, channel, height, width, idx_c, idx_h, idx_w, idx_hwc, idx_chw;
    task_tensor_t* tensor;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);
    N2CUBE_PARAM_CHECK(buffer);
    N2CUBE_PARAM_CHECK(size);

    if(idx > 0) {
        DPU_API_VER_CHECK(task->kernel->base.abi_ver, N2CUBE_ERR_ABI_VERSION);
    }

    tensor = dpuGetOutputTensor(task, nodeName, idx);
    outputAddr = tensor->addr_virt;
    tensorSize = tensor->shape->size;

    channel = tensor->shape->channel;
    height = tensor->shape->height;
    width = tensor->shape->width;
    scale = tensor->ops.get_scale(tensor);

    for (idx_c = 0; idx_c < channel; idx_c++) {
        for (idx_h = 0; idx_h < height; idx_h++) {
            for (idx_w = 0; idx_w < width; idx_w++) {
                idx_chw = idx_c * height * width + idx_h * width + idx_w;
                idx_hwc = idx_h * width * channel + idx_w * channel + idx_c;

                if (idx_chw == size) return N2CUBE_SUCCESS;

                buffer[idx_chw] = outputAddr[idx_hwc] * scale;
            }
        }
    }

    return N2CUBE_SUCCESS;
}

/*
 * Dump running result of DPU kernel
 */
EXPORT int dpuDumpTaskOutput(DPUTask *task)
{
    int ret;
    DPU_ASSERT(task, ERR);

    ret = dpu_dump_node_by_ID(task, task->kernel->base.node_cnt-1);
    return ret;
}

/*
 *  Dump Node's code/input/output/bias/weights
 */
EXPORT int dpuDumpLayerOfTask(DPUTask *task, const char *nodeName)
{
    int nodeID, ret;

    nodeID = get_node_id(task, nodeName);
    ret = dpu_dump_node_by_ID(task, nodeID);

    return ret;
}

int dump_get_dir_name(char *dirName)
{
    int index = 0;
    int tid = syscall(SYS_gettid);

    dpu_dump_mutex.lock();

    if (NULL == dpu_dump_thread_chain) {
        dpu_dump_thread_chain = (dpu_dump_thread_chain_t *)malloc(sizeof(dpu_dump_thread_chain_t));
        dpu_dump_thread_chain->tid = tid;
        dpu_dump_thread_chain->next = NULL;
    } else {
        dpu_dump_thread_chain_t *p = dpu_dump_thread_chain;
        while (p->next != NULL) {
            if (p->tid == tid) {
                break;
            }
            index++;
            p = p->next;
        }

        if ((p->next == NULL) && (p->tid != tid)) {
            dpu_dump_thread_chain_t *pnode = (dpu_dump_thread_chain_t *)malloc(sizeof(dpu_dump_thread_chain_t));
            pnode->next = NULL;
            pnode->tid = tid;
            p->next = pnode;
            index++;
        }
    }

    if (index == 0) {
        sprintf(dirName, "%s", dpu_dump_dir_name);
    } else {
        sprintf(dirName, "%s/thread%d", dpu_dump_dir_name, index);
    }

    if (access(dirName, 0) == -1) {
        if (mkdir(dirName, 0777)) {
            DPU_FAIL_ON_MSG("fail to create dump file directory");
        }
    }

    dpu_dump_mutex.unlock();

    return strlen(dirName);
}

/*
 * Dump Node's code/input/output/bias/weights
 * Dump principly:
 *   only dump real node for v1 version;
 *   dump real/virt node for v2 version;
 */
int dpu_dump_node_by_ID(dpu_task_t *task, int nodeID)
{
    DPU_ASSERT(task && (nodeID >= 0) && (nodeID < (task->kernel->base.node_cnt)), ERR);

    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_FAIL_ON_MSG("dump facility avaialbe only for DPU Kernel built by dnnc compiler in debug mode.");
    }

    if (!TASK_IN_DEBUG(task)) {
        DPU_FAIL_ON_MSG("dump facility avaialbe only for DPU Task in debug mode.");
    }

    dpu_node_t  *node = task->kernel->base.node_list[nodeID];
    task_node_t *tn   = task->node_list[nodeID];
    DPU_LOG_MSG("Dump Code/Param/Input/Output of DPU Kernel [%s] Node [%s]",
        task->kernel->base.name, node->name);

    dump_node_code(task, nodeID);
    node->ops.dump_params(node, (kernel_t*)(task->kernel));
    tn->ops.dump_input(tn, task, node);
    tn->ops.dump_output(tn, task, node);

    return N2CUBE_SUCCESS;
}

/*
 * Dump Node's code/input/output/bias/weights when dpu launch execution timeout.
 * Dump principly:
 *   only dump real node for v1 version;
 *   dump real/virt node for v2 version;
 */
void dpu_dump_node_when_timeout(dpu_task_t *task, char* nodeName)
{
    DPU_ASSERT(task && nodeName, ERR);
    int id = get_node_id(task, nodeName);
    DPU_ASSERT(id >= 0, ERR);

    if (TASK_IN_DEBUG(task)) {
        dpu_node_t  *node = task->kernel->base.node_list[id];
        task_node_t *tn   = task->node_list[id];
        DPU_LOG_MSG("Dump Code/Param/Input/Output of DPU Kernel [%s] Node [%s]",
            task->kernel->base.name, node->name);

        dump_node_code(task, id);
        node->ops.dump_params(node, (kernel_t*)(task->kernel));
        tn->ops.dump_input(tn, task, node);
        tn->ops.dump_output(tn, task, node);
    }
}

INTERNAL int dump_node_code(dpu_task_t *task, int id)
{
    FILE *fp;
    char out_file[MAX_NAME_LEN];
    dpu_node_t *node;
    mem_segment_t *node_code;

    DPU_ASSERT(task && (id >=0), ERR);

    node = task->kernel->base.node_list[id];
    if( node_code = node->ops.get_node_code(node) ) {
        sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_code.bin",
                task->kernel->base.name, node->name);

        fp = fopen(out_file, "wb");
        fwrite(node_code->addr_virt, sizeof(char), node_code->length, fp);

        fflush(fp);
        fclose(fp);

        // dump fix info for each node
        int i;
        task_node_v2_t *tn = (task_node_v2_t *)(task->node_list[id]);
        dpu_node_v2_t * nd = (dpu_node_v2_t*)(node);
        if ((task->kernel->base.abi_ver > DPU_ABI_V1_0)
            && ((nd->input_cnt != 0) || (nd->output_cnt != 0) || (nd->param_cnt != 0))) {
            sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_fixinfo.txt",
                    task->kernel->base.name, node->name);
            fp = fopen(out_file, "w");

            // dump input node fix info
            for(i=0; i<nd->input_cnt; i++) {
                tensor_shape_t *shape = tn->tensorsIn[i].shape;
                fprintf(fp, "%d %d ", shape->fix_width, shape->fix_pos);
            }

            // dump output node fix info
            for(i=0; i<nd->output_cnt; i++) {
                tensor_shape_t *shape = tn->tensorsOut[i].shape;
                fprintf(fp, "%d %d ", shape->fix_width, shape->fix_pos);
            }

            // dump parameter node fix info
            for(i=0; i<nd->param_cnt; i++) {
                fprintf(fp, "%d %d ", nd->elf_params[i].fix_w, nd->elf_params[i].fix_p);
            }
            fprintf(fp, "\n");

            fclose(fp);
        }
    }

    return N2CUBE_SUCCESS;
}

/**
 * Get node ID by nodename, only for ABIv1.6 whose nodeName contines "Nxx_" prefix.
 * In order to make minimal change to deployment code, so avoid to use full nodeName.
 */
int get_node_ID_by_sub_name(dpu_task_t *task, const char *nodeName)
{
    int id, nodeId = N2CUBE_FAILURE;
    int cnt = 0;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    dpu_node_t **nodes = task->kernel->base.node_list;
    for (id = 0; id < task->kernel->base.node_cnt; id ++) {
        if (strstr(nodes[id]->ops.get_name(nodes[id]), nodeName)) {
            nodeId = id;
            cnt++;
        }
    }

    /* Has one single Node which node contines "nodeName". */
    if (cnt == 1) {
        DPU_ASSERT(nodeId >= 0, ERR);
        return nodeId;
    } else if (cnt > 1) {
        DPU_FAIL_ON_MSG("Multiply Node exist with name contines %s.\n"
                    "%sPlease check the full name after dnnc compilation.",
                    nodeName,
                    DPU_MSG_HEADER);
    }

    DPU_FAIL_ON_MSG("Node name %s for kernel %s not exist, please check.", nodeName, task->kernel->base.name);
    return N2CUBE_FAILURE;
}

int get_node_ID_by_full_name(dpu_task_t *task, const char *nodeName)
{
    int id;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    dpu_node_t **nodes = task->kernel->base.node_list;
    for (id = 0; id < task->kernel->base.node_cnt; id ++) {
        /* compare the Node's name ignoring case */
        if (!strcasecmp(nodes[id]->ops.get_name(nodes[id]), nodeName)) {
            return id;
        }
    }

    return N2CUBE_FAILURE;
}

INTERNAL int get_node_id (dpu_task_t *task, const char *nodeName) {
    return (task->kernel->base.abi_ver == DPU_ABI_V1_6) ?
              get_node_ID_by_sub_name(task, nodeName) : get_node_ID_by_full_name(task, nodeName);
}


INTERNAL inline char *get_node_name(dpu_task_t *task, int nodeID)
{
    int id;
    DPU_ASSERT(nodeID >= 0, ERR);

    return task->kernel->base.node_list[nodeID]->name;
}

int get_virtual_node_ID(dpu_task_t *task, const char *nodeName)
{
    int id;

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(nodeName);

    for (id = 0; id < task->kernel->base.virt_node_cnt; id ++) {
        /* compare the Node's name ignoring case */
        if (!strcasecmp(task->kernel->virt_node_list[id].base_v1.base.name, nodeName)) {
            return id;
        }
    }

    return N2CUBE_FAILURE;
}

inline unsigned long long port_data_count(unsigned int start, unsigned int end, int unit)
{
    if (end < start) {
        return (unsigned long long)((0xFFFFFFFF - start + end)*unit);
    } else {
        return (unsigned long long)((end - start)*unit);
    }
}

#ifdef FPGA_ENABLE_PORT_PROFILE
INTERNAL int print_task_port_profile(dpu_task_t *task, int count)
{
    int id;
    long long timing;
    float time_f, time_dpu;

    float HP0_R, HP0_W, HP1_R, HP1_W, HP2_R, HP2_W, HP3_R, HP3_W, GP_R, GP_W, Sum;
    float HP0_R_TOTAL, HP0_W_TOTAL, HP1_R_TOTAL, HP1_W_TOTAL, HP2_R_TOTAL, HP2_W_TOTAL;
    float HP3_R_TOTAL, HP3_W_TOTAL, GP_R_TOTAL, GP_W_TOTAL;

    struct port_profile_t  port_start;
    struct port_profile_t  port_end;

    HP0_R_TOTAL = 0.0f;
    HP0_W_TOTAL = 0.0f;
    HP1_R_TOTAL = 0.0f;
    HP1_W_TOTAL = 0.0f;
    HP2_R_TOTAL = 0.0f;
    HP2_W_TOTAL = 0.0f;
    HP3_R_TOTAL = 0.0f;
    HP3_W_TOTAL = 0.0f;
    GP_R_TOTAL = 0.0f;
    GP_W_TOTAL = 0.0f;

    printf(DPU_LINE_STAR);

    printf("%4s %30s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
        "ID", "NodeName", "HP0_R", "HP0_W", "HP1_R", "HP1_W",
        "HP2_R", "HP2_W", "HP3_R", "HP3_W", "GP_R", "GP_W", "Sum.");
    printf("%42s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
        STR_MB_S, STR_MB_S, STR_MB_S, STR_MB_S, STR_MB_S,
        STR_MB_S, STR_MB_S, STR_MB_S, STR_MB_S, STR_MB_S, STR_MB_S);

    for (id = 0; id < count; id ++) {
        time_f = dpuGetNodeProfileInSecond(task, task->kernel->base.node_list[id]->name);
        time_dpu = (task->node_list[id]->port_profile_end.dpu_cycle-
            task->node_list[id]->port_profile_start.dpu_cycle)*(1/322000.0f/1000.0f);

        port_start = task->node_list[id]->port_profile_start;
        port_end = task->node_list[id]->port_profile_end;

        HP0_R = (float)(port_data_count(port_start.port_hp0_read_byte,
            port_end.port_hp0_read_byte, BIT_WIDTH_128))/UNIT_1M;
        HP0_W = (float)(port_data_count(port_start.port_hp0_write_byte,
            port_end.port_hp0_write_byte, BIT_WIDTH_128))/UNIT_1M;

        HP1_R = (float)(port_data_count(port_start.port_hp1_read_byte,
            port_end.port_hp1_read_byte, BIT_WIDTH_128))/UNIT_1M;
        HP1_W = (float)(port_data_count(port_start.port_hp1_write_byte,
            port_end.port_hp1_write_byte, BIT_WIDTH_128))/UNIT_1M;

        HP2_R = (float)(port_data_count(port_start.port_hp2_read_byte,
            port_end.port_hp2_read_byte, BIT_WIDTH_128))/UNIT_1M;
        HP2_W = (float)(port_data_count(port_start.port_hp2_write_byte,
            port_end.port_hp2_write_byte, BIT_WIDTH_128))/UNIT_1M;

        HP3_R = (float)(port_data_count(port_start.port_hp3_read_byte,
            port_end.port_hp3_read_byte, BIT_WIDTH_128))/UNIT_1M;
        HP3_W = (float)(port_data_count(port_start.port_hp3_write_byte,
            port_end.port_hp3_write_byte, BIT_WIDTH_128))/UNIT_1M;

        GP_R = (float)(port_data_count(port_start.port_gp_read_byte,
            port_end.port_gp_read_byte, BIT_WIDTH_32))/UNIT_1M;
        GP_W = (float)(port_data_count(port_start.port_gp_write_byte,
            port_end.port_gp_write_byte, BIT_WIDTH_32))/UNIT_1M;

        Sum = HP0_R + HP0_W + HP1_R + HP1_W +
            HP2_R + HP2_W + HP3_R + HP3_W + GP_R + GP_W;

        printf("%4d %30s %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n",
            id, task->kernel->base.node_list[id]->name,
            HP0_R/time_f, HP0_W/time_f, HP1_R/time_f, HP1_W/time_f, HP2_R/time_f,
            HP2_W/time_f, HP3_R/time_f, HP3_W/time_f, GP_R/time_f, GP_W/time_f, Sum/time_f);

        printf("cpu: %f  dpu:%f\n", time_f, time_dpu);

        HP0_R_TOTAL += HP0_R;
        HP0_W_TOTAL += HP0_W;
        HP1_R_TOTAL += HP1_R;
        HP1_W_TOTAL += HP1_W;
        HP2_R_TOTAL += HP2_R;
        HP2_W_TOTAL += HP2_W;
        HP3_R_TOTAL += HP3_R;
        HP3_W_TOTAL += HP3_W;
        GP_R_TOTAL += GP_R;
        GP_W_TOTAL += GP_W;

    }

#if 0
    port_start = task->node_list[0].port_profile_start;
    port_end = task->node_list[count-1].port_profile_end;

    HP0_R_TOTAL = (float)(port_data_count(port_start.port_hp0_read_byte,
        port_end.port_hp0_read_byte, BIT_WIDTH_128))/UNIT_1M;
    HP0_W_TOTAL = (float)(port_data_count(port_start.port_hp0_write_byte,
        port_end.port_hp0_write_byte, BIT_WIDTH_128))/UNIT_1M;

    HP1_R_TOTAL = (float)(port_data_count(port_start.port_hp1_read_byte,
        port_end.port_hp1_read_byte, BIT_WIDTH_128))/UNIT_1M;
    HP1_W_TOTAL = (float)(port_data_count(port_start.port_hp1_write_byte,
        port_end.port_hp1_write_byte, BIT_WIDTH_128))/UNIT_1M;

    HP2_R_TOTAL = (float)(port_data_count(port_start.port_hp2_read_byte,
        port_end.port_hp2_read_byte, BIT_WIDTH_128))/UNIT_1M;
    HP2_W_TOTAL = (float)(port_data_count(port_start.port_hp2_write_byte,
        port_end.port_hp2_write_byte, BIT_WIDTH_128))/UNIT_1M;

    HP3_R_TOTAL = (float)(port_data_count(port_start.port_hp3_read_byte,
        port_end.port_hp3_read_byte, BIT_WIDTH_128))/UNIT_1M;
    HP3_W_TOTAL = (float)(port_data_count(port_start.port_hp3_write_byte,
        port_end.port_hp3_write_byte, BIT_WIDTH_128))/UNIT_1M;

    GP_R_TOTAL = (float)(port_data_count(port_start.port_gp_read_byte,
        port_end.port_gp_read_byte, BIT_WIDTH_32))/UNIT_1M;
    GP_W_TOTAL = (float)(port_data_count(port_start.port_gp_write_byte,
        port_end.port_gp_write_byte, BIT_WIDTH_32))/UNIT_1M;
#endif

    time_f = dpuGetTaskProfileInSecond(task);

    Sum = HP0_R_TOTAL + HP0_W_TOTAL + HP1_R_TOTAL + HP1_W_TOTAL +
        HP2_R_TOTAL + HP2_W_TOTAL + HP3_R_TOTAL + HP3_W_TOTAL +
        GP_R_TOTAL + GP_W_TOTAL;

    printf("\n%35s\n", "Total Nodes In Avg:");
    printf("%35s %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n",
        "All:",
        HP0_R_TOTAL/time_f, HP0_W_TOTAL/time_f, HP1_R_TOTAL/time_f, HP1_W_TOTAL/time_f, HP2_R_TOTAL/time_f,
        HP2_W_TOTAL/time_f, HP3_R_TOTAL/time_f, HP3_W_TOTAL/time_f, GP_R_TOTAL/time_f, GP_W_TOTAL/time_f, Sum/time_f);

    printf(DPU_LINE_STAR);

    return N2CUBE_SUCCESS;
}
#endif

INTERNAL int print_task_profile(dpu_task_t *task, int count)
{
    int id;
    int real_node_count;
    int zu9_wordaround;
    long long timing;
    float workload, perf, memload;
    dpu_node_t **nodes;
    float peak_perf;

    /* Profiler facility avaialbe only for kernel in debug mode */
    if (!KERNEL_IN_DEBUG(task->kernel)) {
        DPU_FAIL_ON_MSG("Profiler facility avaialbe only for kernel built by dnnc compiler in debug mode.");
    }

    /* check signature */
    if (!dpu_caps.signature_valid) {
        DPU_FAIL_ON_MSG("No description info contained in DPU IP, profiling not supported.");
    }

    dpu_trace_print_mutex.lock();

    DPU_LOG_MSG("Performance profile - DPU Kernel \"%s\" DPU Task \"%s\"",
        task->kernel->base.name, task->name);

    zu9_wordaround = 0;
    nodes = task->kernel->base.node_list;

    printf(DPU_LINE_LONG);

    printf("%4s %30s %13s %7s %11s %10s %11s %7s\n",
        "ID", "NodeName", "Workload(MOP)", "Mem(MB)",
        "RunTime(ms)", "Perf(GOPS)", "Utilization", "MB/S");
    real_node_count = 0;
    if (dpu_caps.magic == DPU_CONF_MAGIC) {
        peak_perf = ((dpu_configurable_t*)(dpu_caps.p_dpu_info) + task->coreID)->base.peak_perf;
    } else {
        peak_perf = ((dpu_info_t*)(dpu_caps.p_dpu_info) + task->coreID)->base.peak_perf;
    }
    for (id = 0; id < count; id ++) {
        /* Skip virtual node, it does not need to be executed on DPU. */
        if (nodes[id]->ops.get_node_code(nodes[id]) == NULL) {
            continue;
        }
        real_node_count++;

        /* Output profiling time for each Node */
        timing = dpuGetNodeProfile(task, nodes[id]->name);
        workload = (float)(nodes[id]->ops.get_workload(nodes[id])/1000000.0f); /* MOP */
        perf = (float)(workload/1000.0f)/(timing/1000000.0f);
        memload = (float)(nodes[id]->ops.get_memload(nodes[id])/UNIT_1M); /* MB */;

    printf("%4d %30s %13.3f %7.2f %11.3f %10.1f %11.1f%% %7.1f\n",
        real_node_count,
        nodes[id]->name,
        workload,
        memload,
        timing/1000.0f,
        perf,
        (float)(perf/peak_perf*100.0f),
        memload/(timing/1000000.0f));
    }

    printf("\n%35s\n", "Total Nodes In Avg:");
    timing = dpuGetTaskProfile(task); /* us */
    workload = (float)(task->kernel->base.workloadTotal/1000000.0f); /* MOP */
    perf = (float)(workload/1000.0f)/(timing/1000000.0f);
    memload = (float)(task->kernel->base.memloadTotal/1000000.0f);

   if (zu9_wordaround) {
       printf("%35s %13.3f %7.2f %11.3f %10.1f %11.1f%% %7.1f\n",
            "All",
            workload,
            memload,
            (float)(timing/1000.0f),
            perf*0.85,
            (float)(perf*0.85/peak_perf*100.0f),
            memload/(timing/1000000.0f));
    } else {
       printf("%35s %13.3f %7.2f %11.3f %10.1f %11.1f%% %7.1f\n",
           "All",
           workload,
           memload,
           (float)(timing/1000.0f),
           perf,
           (float)(perf/peak_perf*100.0f),
           memload/(timing/1000000.0f));
    }

    printf(DPU_LINE_LONG);

#ifdef FPGA_ENABLE_PORT_PROFILE
    print_task_port_profile(task, count);
#endif

    dpu_trace_print_mutex.unlock();

    return N2CUBE_SUCCESS;
}

INTERNAL int print_task_trace_time(dpu_task_t *task, int count)
{
    int id;
    long long timing;
    char record[256], fileName[128];
    float workload, perf, percent;
    dpu_node_t **nodes;
    float peak_perf;

    if (!TASK_IN_PROF(task)) {
        DPU_FAIL_ON_MSG("tracing facility avaialbe only DPU Task in profile mode.");
    }

    /* Open the file of dpu trace */
    if (NULL == dpu_trace_fd) {
        dpu_trace_mutex.lock();
        if (NULL == dpu_trace_fd) {
            sprintf(fileName, "%s_%d.prof", DPU_TRACE_FILE, getpid());
            dpu_trace_fd = fopen(fileName, "wb");
        }
        dpu_trace_mutex.unlock();
    }

    if (dpu_caps.magic == DPU_CONF_MAGIC) {
        peak_perf = ((dpu_configurable_t*)(dpu_caps.p_dpu_info) + task->coreID)->base.peak_perf;
    } else {
        peak_perf = ((dpu_info_t*)(dpu_caps.p_dpu_info) + task->coreID)->base.peak_perf;
    }
    /* output DPU timestamp for Kernel in debug mode */
    if (KERNEL_IN_DEBUG(task->kernel)) {
        nodes = task->kernel->base.node_list;
        for (id=0; id<count; id++) {
            /* Skip virtual node, it does not need to be executed on DPU. */
            if (nodes[id]->ops.get_node_code(nodes[id]) == NULL) {
                continue;
            }

            timing = dpuGetNodeProfile(task, nodes[id]->name);
            workload = (float)(nodes[id]->ops.get_workload(nodes[id])/1000000.0f); /* MOP */
            perf = (float)(workload/1000.0f)/(timing/1000000.0f);
            percent = (float)(perf/peak_perf*1.0f);

            sprintf(record, "%d\t%20s\t%lld\t%lld\t%0.3f\n",
                task->node_list[id]->coreID,         /* DPU Core ID */
                task->kernel->base.name,                 /* Kernel name */
                task->node_list[id]->time_start/1000,/* DPU start timestamp */
                task->node_list[id]->time_end/1000,  /* DPU end timestamp */
                percent);                           /* DPU utilization */

            dpu_trace_mutex.lock();
            fwrite(record, 1, strlen(record), dpu_trace_fd);
            dpu_trace_mutex.unlock();
        }
    } else {
        /* output DPU timestamp for Kernel in normal mode */
        timing = dpuGetTaskProfile(task); /* us */
        workload = (float)(task->kernel->base.workloadTotal/1000000.0f); /* MOP */
        perf = (float)(workload/1000.0f)/(timing/1000000.0f);
        percent = (float)(perf/peak_perf*1.0f);

        sprintf(record, "%d\t%20s\t%lld\t%lld\t%0.3f\n",
            task->coreID,               /* DPU Core ID */
            task->kernel->base.name,         /* Kernel name */
            task->time_start/1000,      /* DPU start timestamp */
            task->time_end/1000,        /* DPU end timestamp */
            percent);                   /* DPU utilization */

        dpu_trace_mutex.lock();
        fwrite (record, 1, strlen(record), dpu_trace_fd);
        dpu_trace_mutex.unlock();
    }

    fflush(dpu_trace_fd);

    return N2CUBE_SUCCESS;
}

/*
 * do softmax caculation by invoking SMFC IP with all resouce ready.
 * return value:
 *   0: success, otherwise failure;
 */
INTERNAL inline int _dpuRunSoftmaxInternal(uint32_t width, uint32_t exWidth, uint32_t height, int fix_pos, uint32_t phyInput, uint32_t phyOutput)
{
    // below 3 line describe the hardware limitation
    int widthMax = (1024-1);
    int heightMax = (65536-1);
    int sizeMax = (1048576-1);

    int i;
    int step, cnt, size;
    float offset=0.0f;
    long ret;

    dpu_aol_run_t aol_run;

    if (dpu_caps.softmax.valid && dpu_caps.softmax.enable) {
        ;
    } else {
        //SMFC ip not ready
        return N2CUBE_FAILURE;
    }
    if (width > widthMax) {
        //width too long
        return N2CUBE_FAILURE;
    }
    // deal with max entryies once
    if (height>heightMax) {
        step = heightMax;
    } else {
        step = height;
    }
    size = exWidth*step;
    if (size > sizeMax) {
        step = sizeMax/exWidth;
    }
    if (height > step) {
        step = (step/4)*4;
	cnt = height%step ? (height/step +1) : (height/step);
    } else {
        cnt = 1;
    }
    for(i = 0; i < cnt; i++) {
        int index = 0;
        if (dpu_caps.signature_version == 2) { //for 1 to 1
            aol_run.regs[index].value = 0;
            aol_run.regs[index].offset = OFFSET_SMFC_CRTL;
            index++;
            aol_run.regs[index].value = 1;
            aol_run.regs[index].offset = OFFSET_SMFC_GLBL_IRQ;
            index++;
            aol_run.regs[index].value = 0x3;
            aol_run.regs[index].offset = OFFSET_SMFC_IER;
            index++;
            aol_run.regs[index].value = 1;
            aol_run.regs[index].offset = OFFSET_SMFC_INT_CLR;

            // height
            index++;
            if((i == (cnt-1))&&(i !=0)){
                aol_run.regs[index].value = height-(step*i);
            }else{
                aol_run.regs[index].value = step;
            }
            aol_run.regs[index].offset = OFFSET_SMFC_CMD_Y_LEN;
            // width
            index++;
            aol_run.regs[index].value = width;
            aol_run.regs[index].offset = OFFSET_SMFC_CMD_X_LEN;
            // input
            index++;
            aol_run.regs[index].value = phyInput +  i*exWidth*step*1;
            aol_run.regs[index].offset = OFFSET_SMFC_SM_SRC_ADDR_L;
            // output
            index++;
            aol_run.regs[index].value = phyOutput + i*exWidth*step*4;
            aol_run.regs[index].offset = OFFSET_SMFC_SM_DST_ADDR_L;
            // scale
            index++;
            aol_run.regs[index].value = fix_pos;
            aol_run.regs[index].offset = OFFSET_SMFC_SM_CMD_SCALE;
            // offset
            index++;
            aol_run.regs[index].value = offset;
            aol_run.regs[index].offset = OFFSET_SMFC_SM_CMD_OFFSET;
            // mode
            index++;
            aol_run.regs[index].value = 0;
            aol_run.regs[index].offset = OFFSET_SMFC_CALC_MOD;

            index++;
            aol_run.reg_count = index;
        } else { //for 1 to more
            // height
            if((i == (cnt-1))&&(i !=0)){
                aol_run.regs[0].value = height-(step*i);
            }else{
                aol_run.regs[0].value = step;
            }
            aol_run.regs[0].offset = (uint32_t)((unsigned long)(&g_smfc_reg.sm_len_y) - (unsigned long)(&g_smfc_reg));
            // width
            aol_run.regs[1].value = width;
            aol_run.regs[1].offset = (uint32_t)((unsigned long)(&g_smfc_reg.sm_len_x) - (unsigned long)(&g_smfc_reg));
            // input
            aol_run.regs[2].value = phyInput +  i*exWidth*step*1;
            aol_run.regs[2].offset = (uint32_t)((unsigned long)(&g_smfc_reg.src) - (unsigned long)(&g_smfc_reg));
            // output
            aol_run.regs[3].value = phyOutput + i*exWidth*step*4;
            aol_run.regs[3].offset = (uint32_t)((unsigned long)(&g_smfc_reg.dst) - (unsigned long)(&g_smfc_reg));
            // scale
            aol_run.regs[4].value = fix_pos;
            aol_run.regs[4].offset = (uint32_t)((unsigned long)(&g_smfc_reg.scale) - (unsigned long)(&g_smfc_reg));
            // offset
            aol_run.regs[5].value = offset;
            aol_run.regs[5].offset = (uint32_t)((unsigned long)(&g_smfc_reg.sm_offset) - (unsigned long)(&g_smfc_reg));
            // mode
            aol_run.regs[6].value = 0;
            aol_run.regs[6].offset = (uint32_t)((unsigned long)(&g_smfc_reg.calc_mod) - (unsigned long)(&g_smfc_reg));
            aol_run.reg_count = 7;
        }

        //REQ_RUN_SOFTMAX
        aol_run.core_mask = 0x01;
        aol_run.ip_id = IP_ID_SOFTMAX;
        aol_run.timeout = dpu_get_n2cube_timeout();
        ret = dpu_aol_run(gp_dpu_aol_handle, &aol_run);
        if (DPU_AOL_ERROR == ret) {
            display_dpu_debug_info();
            reset_dpus(gp_dpu_aol_handle);
            printf("\n");
            DPU_FAIL_ON_MSG("DPU timeout while execute Softmax");
        }
    }

    return N2CUBE_SUCCESS;
}

/**
 * @brief softmax calculation
 *
 * @note length of input and output array should be num_classes x batch_size;
 *       the calculation will be performed on an acceletator if exists,
 *       otherwise the calculation will be done on CPU.
 *
 * @param input - pointer to source data(int8_t*)
 * @param output - pointer to target data(float*)
 * @param numClasses - the number of classes
 * @param batchSize - batch size of softmax calculation
 * @param scale - scale value in softmax
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuRunSoftmax(int8_t *input, float *output, int numClasses, int batchSize, float scale)
{
    unsigned long softmax_in_phy_base = 0;
    unsigned long softmax_in_virt_base = 0;
    unsigned long softmax_in_len = 0;
    unsigned long softmax_out_phy_base = 0;
    unsigned long softmax_out_virt_base = 0;
    unsigned long softmax_out_len = 0;
    mem_segment_t seg_in, seg_out;

    int i, j;
    int ret = N2CUBE_FAILURE;
    int fix_pos = -((int8_t)log2f(scale));
    int hw_scale;
    N2CUBE_PARAM_CHECK(input);
    N2CUBE_PARAM_CHECK(output);

    N2CUBE_DPU_CHECK((numClasses > 0)&&(batchSize > 0)&&(scale > 0), N2CUBE_ERR_PARAM_VALUE,
        ". numClasses: %d, batchSize: %d, scale: %f", numClasses, batchSize, scale);
    if (dpu_caps.signature_version < 2 ) {  //1 to multi
        hw_scale = 3;
    }
    else {    // 1 to 1
        hw_scale = 2;
    }
    if (dpu_caps.softmax.valid && dpu_caps.softmax.enable && (fix_pos>=hw_scale) && (numClasses<1024)) {
        void *addr;
        uint32_t phyInput;
        int8_t  *virtInput;
        uint32_t phyOutput;
        float   *virtOutput;

        uint32_t exWidth = numClasses%4 ? (numClasses/4 +1)*4 : numClasses;
        uint32_t sizeIn = exWidth * batchSize * sizeof(int8_t);
        uint32_t sizeOut = exWidth * batchSize * sizeof(float);

        // input
        if (dpu_dev_mem_alloc(&seg_in, sizeIn) != 0) {
            DPU_FAIL_ON_MSG("Fail to malloc memory for DPU softmax unit");
        }
        phyInput = (uint32_t)seg_in.addr_phy;
        virtInput = (int8_t*)seg_in.addr_virt;

        softmax_in_phy_base = (unsigned long)phyInput;
        softmax_in_virt_base = (unsigned long)virtInput;
        softmax_in_len = (unsigned long)seg_in.size;

        // output
        if (dpu_dev_mem_alloc(&seg_out, sizeOut) != 0) {
            DPU_FAIL_ON_MSG("Fail to malloc memory for DPU softmax unit");
        }
        phyOutput = (uint32_t)seg_out.addr_phy;
        virtOutput = (float*)seg_out.addr_virt;

        softmax_out_phy_base = (unsigned long)phyOutput;
        softmax_out_virt_base = (unsigned long)virtOutput;
        softmax_out_len = (unsigned long)seg_out.size;

        // Compute
        if (exWidth == numClasses) {
            memcpy(virtInput, input, batchSize*numClasses*sizeof(int8_t));
        } else {
            for(j=0; j<batchSize; j++){
                memcpy(virtInput+j*exWidth*sizeof(int8_t), input+j*numClasses*sizeof(int8_t), numClasses*sizeof(int8_t));
            }
        }
        dpuCacheFlush(&seg_in, 0, sizeIn);

        ret = _dpuRunSoftmaxInternal(numClasses, exWidth, batchSize, fix_pos, phyInput, phyOutput);
        if ( N2CUBE_SUCCESS == ret){
            dpuCacheInvalid(&seg_out, 0, sizeOut);
            if (exWidth == numClasses) {
                memcpy(output, virtOutput, batchSize*numClasses*sizeof(float));
            } else {
                for(j=0; j<batchSize; j++){
                    memcpy(output+j*numClasses,virtOutput+j*exWidth, numClasses*sizeof(float));
                }
            }
        }

        // free
        dpu_dev_mem_free(&seg_in);
        dpu_dev_mem_free(&seg_out);

        if (isnan(*output)==1 || isinf(*output)==1) {
            ret = N2CUBE_FAILURE;
        }
    }
    else {
//    if ( N2CUBE_FAILURE == ret) {
#if 0
        double *result = (double*)malloc(numClasses*sizeof(double));
        for (int i = 0; i < batchSize; i++) {
            long double sum = 0.0f;
            for (int j = 0; j < numClasses; j++) {
                float item = (*(input+i*numClasses+j))*scale;
                *(result+j) = exp(item);
                sum += *(result+j);
                if (isinf(sum)==1) {
                   DPU_FAIL_ON_MSG("Overflow while do softmax");
                }
            }
            for (int j = 0; j < numClasses; j++) {
                *(output+i*numClasses+j) = (float)(*(result+j)/sum);
            }
        }
#else
        softmax_on_arm(output,input,numClasses,scale,batchSize);
        ret = N2CUBE_SUCCESS;
#endif
    }

    return ret;
}

/**
 * @brief Get mean value from model
 * This API is only available for Caffe model
 * @param task - pointer to DPU task
 * @param mean - pointer to mean value
 * @param count - channle number of input
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuGetKernelMean(DPUTask *task, float *mean, int count) {

    N2CUBE_PARAM_CHECK(task);
    N2CUBE_PARAM_CHECK(mean);
    if (count == 3) {
        N2CUBE_DPU_CHECK((task->kernel->base.mean_c1 != -1) || (task->kernel->base.mean_c2 != -1) || (task->kernel->base.mean_c3 != -1),
            N2CUBE_ERR_KERNEL_MEAN_VALUE, ". kernel name: %s", task->kernel->base.name);
        mean[0] = task->kernel->base.mean_c1*1.0f;
        mean[1] = task->kernel->base.mean_c2*1.0f;
        mean[2] = task->kernel->base.mean_c3*1.0f;
    }
    else if (count == 1) {
        N2CUBE_DPU_CHECK((task->kernel->base.mean_c1 != -1),
            N2CUBE_ERR_KERNEL_MEAN_VALUE, ". kernel name: %s", task->kernel->base.name);
        mean[0] = task->kernel->base.mean_c1*1.0f;
    }
    else {
        DPU_FAIL_ON_MSG("Only 1 or 3 channel is supported by API %s",__func__);
	return -1;
    }

    return 0;
}

/**
 * @brief softmax calculation on arm
 *
 * @note length of src and dst array should be size x batch;
 *       if the size is 2 or 4 and the macro USING_NEON id defined, neon accelerated code
 *       will be implemented, otherwise the calculation will be done on CPU.
 *
 * @param dst - pointer to target data(float*)
 * @param src - pointer to source data(int8_t*)
 * @param size - the length of each array
 * @param scale - scale value in softmax
 * @param batch - how many array to be calculated
 *
 * @return void
 */
extern void softmax_batch(float* output, const int8_t* input, unsigned int size, float scale,
                    unsigned int batch);
void softmax_on_arm(float *dst, int8_t *src, int size, float scale, int batch)
{
#ifdef USE_ARM_NEON
    if (size == 2)
        neon_softmax2(dst, src, scale, batch);
    else if (size == 4)
        neon_softmax4(dst, src, scale, batch);
    else {
        // operations on NEON not used, using CPU instead
        softmax_batch(dst,src,size,scale,batch);
    }
#else   // not USE_ARM_NEON
    softmax_batch(dst,src,size,scale,batch);
#endif  // end USE_ARM_NEON
}

////////////////////////////////////////////////////////////////////////
// below section deal with unified api
////////////////////////////////////////////////////////////////////////

#include "dnndk/n2cube.h"
#include <map>
#include <fstream>
#include <string>
#include <streambuf>
#include <arm_neon.h>
#include <unordered_map>

#include "vai/dpu_runner.hpp"

#include <future>
#include <map>

#include <list>
#include <iterator>

#include <pthread.h>
#include <semaphore.h>
#include "cJSON.h"

using namespace std;

pthread_mutex_t  mutIsOpened; //protecting following two variables.
static int isOpened = 0;
extern char vitisKernelPath[];

namespace vitis {
namespace ai {
std::vector<std::unique_ptr<vitis::ai::DpuRunner>> DpuRunner::create_dpu_runner(
    const std::string& model_directory) {

    std::vector<std::unique_ptr<DpuRunner>> runners;
    XdpuRunner *runner = new XdpuRunner(model_directory);
    runners.emplace_back(runner);

    return runners;
}

void cvt_f32_s8_neon(const float* src, float scale, int size, int8_t* dst) {
    float32x4_t _scale = vdupq_n_f32(scale);
    float32x4x4_t _src, _dst;
    int32x4x4_t _dst_s32;
    int16x4x4_t _tmp_s16;
    int16x8x2_t _tmp2_s16;
    int8x8x2_t _tmp3_s8;
    int8x16_t _dst_s8;

    for (int i = 0; i < size / 16; i++) {
        for (int j = 0; j < 4; j++) {
            _src.val[j] = vld1q_f32(src + i * 16 + 4 * j);
            _dst.val[j] = vmulq_f32(_src.val[j], _scale);
            _dst_s32.val[j] = vcvtq_s32_f32(_dst.val[j]);
            _tmp_s16.val[j] = vmovn_s32(_dst_s32.val[j]);
        }
        _tmp2_s16.val[0] = vcombine_s16(_tmp_s16.val[0], _tmp_s16.val[1]);
        _tmp2_s16.val[1] = vcombine_s16(_tmp_s16.val[2], _tmp_s16.val[3]);

        _tmp3_s8.val[0] = vmovn_s16(_tmp2_s16.val[0]);
        _tmp3_s8.val[1] = vmovn_s16(_tmp2_s16.val[1]);

        _dst_s8 = vcombine_s8(_tmp3_s8.val[0], _tmp3_s8.val[1]);

        vst1q_s8(dst + i * 16, _dst_s8);
    }
    if (size % 16) {
        for (int i = size / 16 * 16; i < size; i++) {
            dst[i] = (int8_t)(src[i] * scale);
        }
    }
}


void cvt_s8_f32_neon(int8_t* src, float scale, int size, float* dst) {
    float32x4_t _scale = vdupq_n_f32(scale);

    int8x16_t _src_s8;
    int8x8x2_t _src_s8_lh;
    int16x8x2_t _tmp_s16_2;
    int16x4x4_t _t_s16x4x4;
    int32x4x4_t _t_s32x4x4;
    float32x4x4_t _dst_f32;

    for (int i = 0; i < size / 16; i++) {
        _src_s8 = vld1q_s8(src + i * 16);

        _src_s8_lh.val[0] = vget_low_s8(_src_s8);
        _src_s8_lh.val[1] = vget_high_s8(_src_s8);

        _tmp_s16_2.val[0] = vmovl_s8(_src_s8_lh.val[0]);
        _tmp_s16_2.val[1] = vmovl_s8(_src_s8_lh.val[1]);

        _t_s16x4x4.val[0] = vget_low_s16(_tmp_s16_2.val[0]);
        _t_s16x4x4.val[1] = vget_high_s16(_tmp_s16_2.val[0]);
        _t_s16x4x4.val[2] = vget_low_s16(_tmp_s16_2.val[1]);
        _t_s16x4x4.val[3] = vget_high_s16(_tmp_s16_2.val[1]);

        for (int j = 0; j < 4; j++) {
            _t_s32x4x4.val[j] = vmovl_s16(_t_s16x4x4.val[j]);
            _dst_f32.val[j] = vmulq_f32(vcvtq_f32_s32(_t_s32x4x4.val[j]), _scale);
            vst1q_f32(dst + i * 16 + 4 * j, _dst_f32.val[j]);
        }
    }

    if (size % 16) {
        for (int i = size / 16 * 16; i < size; i++) {
            dst[i] = (float)src[i] * scale;
        }
    }
}

XdpuRunner::XdpuRunner(const std::string &path)
: path_(path)
{
    // TODO FIXME make dynamic by reading from path/meta.json
    pthread_t threadId;
    typedef void* (*FUNC)(void*);
    FUNC callback;
    int cpylen;

    pthread_mutex_lock(&mutIsOpened);
    if (0 == isOpened) {
        dpuOpen();
    }
    isOpened++;

    {
        char exepath[PATH_MAX], jsonpath[PATH_MAX];
        std::string exepath_string, jsonpath_string;

        ssize_t cnt = readlink("/proc/self/exe", exepath, PATH_MAX);
        exepath_string = std::string(exepath, ( cnt > 0 )? cnt : 0);

        if (0 == path.find('/')) {
            jsonpath_string = path+"/meta.json";
        } else if (0 == path.find("./")) {
                    jsonpath_string = exepath_string.substr(0, exepath_string.find_last_of('/'))
                    + path.substr(1, string::npos)
                    + "/meta.json";
        } else {
            jsonpath_string = exepath_string.substr(0, exepath_string.find_last_of('/')+1)
                    + path
                    + "/meta.json";
        }

        memset(jsonpath, 0x0, PATH_MAX);
        strcpy(jsonpath, jsonpath_string.c_str());
	cpylen = strlen(jsonpath) - strlen("meta.json");

        FILE *file = fopen(jsonpath, "r" );
        if ( file == NULL ) {
            throw std::invalid_argument(jsonpath_string +": NOT Found.");
        } else {
            char *data;
            char *model;
            cJSON *metaJson;
            ssize_t sz;

            fseek(file, 0, SEEK_END);
            sz = ftell(file);
            fseek(file, 0, SEEK_SET);
            data = (char*)malloc(sz);
            if (data == NULL) {
                throw std::invalid_argument("prepare memory for meta.json fail.");
            }

            fread((void*)data, sz, 1, file);
            metaJson = cJSON_Parse(data);
            model = cJSON_GetObjectItem(metaJson, "vitis_dpu_kernel")->valuestring;

            N_THREAD_READ  = 1;
            N_THREAD_RUN   = 1;
            N_THREAD_WRITE = 1;
            N_TASK_POOL    = 4;
            if (cJSON_HasObjectItem(metaJson, "pre_processing_pool")) {
                N_THREAD_READ  = cJSON_GetObjectItem(metaJson, "pre_processing_pool")->valueint;
            }
            if (cJSON_HasObjectItem(metaJson, "dpu_thread_pool")) {
                N_THREAD_RUN  = cJSON_GetObjectItem(metaJson, "dpu_thread_pool")->valueint;
            }
            if (cJSON_HasObjectItem(metaJson, "post_processing_pool")) {
                N_THREAD_WRITE  = cJSON_GetObjectItem(metaJson, "post_processing_pool")->valueint;
            }
            if (cJSON_HasObjectItem(metaJson, "dpu_task_pool")) {
                N_TASK_POOL  = cJSON_GetObjectItem(metaJson, "dpu_task_pool")->valueint;
            }

            memset(vitisKernelPath, 0x0, PATH_MAX);
            strncpy(vitisKernelPath, jsonpath, cpylen);

            g_kernel = dpuLoadKernel(model);

            cJSON_Delete(metaJson);
            free(data);
            fclose( file );
        }
    }
    pthread_mutex_unlock(&mutIsOpened);

    sem_init(&semTsk,    0, N_TASK_POOL);
    sem_init(&semJob ,   0, 0);
    sem_init(&semTskJob, 0, 0);
    sem_init(&semOUT,    0, 0);

    pthread_mutex_init(&mutWL ,    NULL);

    pthread_mutex_init(&mutTsk ,   NULL);
    pthread_mutex_init(&mutJob ,   NULL);
    pthread_mutex_init(&mutTskJob, NULL);
    pthread_mutex_init(&mutOUT,    NULL);

    DPUTask *tsk;
    for (int idx = 0 ; idx < N_TASK_POOL; idx++) {
        //no race, no protection.
        tsk = dpuCreateTask(g_kernel, 0);
        lstTsk.push_back(tsk);
    }

    callback = (FUNC)&XdpuRunner::_read;
    for (int idx = 0 ; idx < N_THREAD_READ; idx++) {
        if(pthread_create(&threadId, NULL, callback, this)) {
            DPU_FAIL_ON_MSG("create _read thread pool fail@ %d", idx);
        }
    }

    callback = (FUNC)&XdpuRunner::_run;
    for (int idx = 0 ; idx < N_THREAD_RUN; idx++) {
        if(pthread_create(&threadId, NULL, _run,  this)) {
            DPU_FAIL_ON_MSG("create _run thread pool fail@  %d", idx);
        }
    }
    callback = (FUNC)&XdpuRunner::_write;
    for (int idx = 0 ; idx < N_THREAD_WRITE; idx++) {
        if(pthread_create(&threadId, NULL, _write, this)) {
            DPU_FAIL_ON_MSG("create _write thread pool fail@ %d", idx);
        }
    }

    g_graphinfo = (struct vaiGraphInfo *)malloc(sizeof(struct vaiGraphInfo));

    g_graphinfo->inTensorCnt  = tsk->inputTensorNum;
    g_graphinfo->outTensorCnt = tsk->outputTensorNum;
    g_graphinfo->inTensorList  = (struct vaiTensorShape *)malloc(tsk->inputTensorNum*sizeof(struct vaiTensorShape));
    g_graphinfo->outTensorList = (struct vaiTensorShape *)malloc(tsk->outputTensorNum*sizeof(struct vaiTensorShape));

    for (int idx = 0 ; idx < g_graphinfo->inTensorCnt; idx++) {
        struct vaiTensorShape * vts = g_graphinfo->inTensorList + idx;

        vts->height  = (tsk->inputTensorAttrs  + idx)->shape.h;
        vts->width   = (tsk->inputTensorAttrs  + idx)->shape.w;
        vts->channel = (tsk->inputTensorAttrs  + idx)->shape.c;

        g_graphinfo_iLength += vts->height * vts->width * vts->channel;

        std::string tname = (tsk->inputTensorAttrs  + idx)->tensor_name;
        std::vector<std::int32_t> dims;

        dims.push_back(1);
        dims.push_back(vts->height);
        dims.push_back(vts->width);
        dims.push_back(vts->channel);

        Tensor *t = new Tensor(tname, dims, Tensor::DataType::FLOAT);
        inputs_.push_back(t);
    }
    for (int idx = 0 ; idx < g_graphinfo->outTensorCnt; idx++) {
        struct vaiTensorShape * vts = g_graphinfo->outTensorList + idx;

        vts->height  = (tsk->outputTensorAttrs  + idx)->shape.h;
        vts->width   = (tsk->outputTensorAttrs  + idx)->shape.w;
        vts->channel = (tsk->outputTensorAttrs  + idx)->shape.c;

        g_graphinfo_oLength += vts->height * vts->width * vts->channel;

        std::string tname = (tsk->outputTensorAttrs  + idx)->tensor_name;
        std::vector<std::int32_t> dims;

        dims.push_back(1);
        dims.push_back(vts->height);
        dims.push_back(vts->width);
        dims.push_back(vts->channel);

        Tensor *t = new Tensor(tname, dims, Tensor::DataType::FLOAT);
        outputs_.push_back(t);
    }

}

XdpuRunner::~XdpuRunner() {
    for (int i=0; i < inputs_.size(); i++)
        delete inputs_[i];
    for (int i=0; i < outputs_.size(); i++)
        delete outputs_[i];

    DPUTask *tsk;
    for (int idx = 0 ; idx < N_TASK_POOL; idx++) {
        pthread_mutex_lock(&mutTsk);
        tsk = lstTsk.front();
        lstTsk.pop_front();
        dpuDestroyTask(tsk);
        pthread_mutex_unlock(&mutTsk);
    }

    dpuDestroyKernel(g_kernel);

    pthread_mutex_lock(&mutIsOpened);
    isOpened--;
    if (0 == isOpened) {
        dpuClose();
    }
    pthread_mutex_unlock(&mutIsOpened);
}

std::vector<Tensor *> XdpuRunner::get_input_tensors() {
  return inputs_;
}
std::vector<Tensor *> XdpuRunner::get_output_tensors() {
  return outputs_;
}

std::pair<uint32_t, int>
XdpuRunner::execute_async(const std::vector<TensorBuffer *> &inputs,
                          const std::vector<TensorBuffer *> &outputs) {
    std::unordered_map<std::string, std::vector<const float*> > inputPtrs;
    std::unordered_map<std::string, std::vector<float*> > outPtrs;

    int batchSize;

    for (unsigned j = 0; j < inputs.size(); j++){
        TensorBuffer *tb = inputs[j];
        float *dptr = (float*)(tb->data().first);
        batchSize = tb->get_tensor()->get_dim_size(0);
        int elsize = tb->get_tensor()->get_element_num() / batchSize;

        for (int b=0; b < batchSize; b++)
          inputPtrs[tb->get_tensor()->get_name()].push_back(dptr + (b*elsize));
    }

    for (unsigned j = 0; j < outputs.size(); j++) {
        TensorBuffer *tb = outputs[j];
        float *dptr = (float*)(tb->data().first);
        batchSize = tb->get_tensor()->get_dim_size(0);
        int elsize = tb->get_tensor()->get_element_num() / batchSize;

        for (int b=0; b < batchSize; b++)
          outPtrs[tb->get_tensor()->get_name()].push_back(dptr + (b*elsize));
    }


    DPUJid *jid = (DPUJid *)malloc(sizeof(DPUJid));

    //create and put an DPUJid into waitlist
    pthread_mutex_lock(&mutWL);
    jid->idx = g_jobs_idx;
    jid->cnt = batchSize;
    jid->msk = (1<<batchSize) - 1;
    pthread_mutex_init(&(jid->wait), NULL);
    pthread_mutex_lock(&(jid->wait));
    waitList.push_back(jid);
    g_jobs_idx += batchSize;
    pthread_mutex_unlock(&mutWL);

    pthread_mutex_lock(&mutJob);
    for (int idx=0; idx<batchSize; idx++ ) {
        DPUJob *job = new DPUJob();

        job->idx = jid->idx + idx;
        job->batch_offset = idx;

        job->in = inputPtrs;
        job->out = outPtrs;

        lstJob.push_back(job);
        sem_post(&semJob);
    }
    pthread_mutex_unlock(&mutJob);

    return {jid->idx, 0};
}

int XdpuRunner::wait(int jobId, int timeout) {

    DPUJid *jids;

    pthread_mutex_lock(&mutWL);
    for (auto const& i : waitList) {
        if (i->idx == jobId) {
            jids = i;
            break;
        }
    }
    pthread_mutex_unlock(&mutWL);

    //block on finish event
    pthread_mutex_lock(&(jids->wait));

    //remove jid from waitlist
    DPUJid *Jidele=NULL;
    pthread_mutex_lock(&mutWL);
    for (auto const& i : waitList) {
        if (i->idx == jobId) {
            pthread_mutex_destroy(&(i->wait));
            //waitList.remove(i);
            Jidele = i;
            break;
        }
    }
    waitList.remove(Jidele);
    free(Jidele);
    pthread_mutex_unlock(&mutWL);

    return 0;
}


void *XdpuRunner::_read(void *t){
    class XdpuRunner *self = (class XdpuRunner *) t;

    while (true){
        DPUJob  *job;
        DPUTask *tsk;
        DPUTJ *tj = (DPUTJ*)malloc(sizeof(DPUTJ));

        sem_wait(&(self->semJob));
        pthread_mutex_lock(&(self->mutJob));
        job = self->lstJob.front();
        self->lstJob.pop_front();
        pthread_mutex_unlock(&(self->mutJob));

        sem_wait(&(self->semTsk));
        pthread_mutex_lock(&(self->mutTsk));
        tsk = self->lstTsk.front();
        self->lstTsk.pop_front();
        pthread_mutex_unlock(&(self->mutTsk));

        tj->job = job;
        tj->tsk = tsk;

        for (int i=0; i< tsk->inputTensorNum; i++){
            DPUTensorAttr * tensorAttr = tsk->inputTensorAttrs + i;
            uint64_t size = tensorAttr->size;
            int8_t *buf = (int8_t *)(tensorAttr->addr_virt);

            cvt_f32_s8_neon(job->in[tensorAttr->tensor_name][job->batch_offset], tensorAttr->scale, size, buf);
        }

        pthread_mutex_lock(&(self->mutTskJob));
        self->lstTskJob.push_back(tj);
        pthread_mutex_unlock(&(self->mutTskJob));
        sem_post(&(self->semTskJob));
    }
    return NULL;
}

void *XdpuRunner::_run(void *t){
    class XdpuRunner *self = (class XdpuRunner *) t;

    while (true){
        DPUTJ * tj;

        sem_wait(&(self->semTskJob));
        pthread_mutex_lock(&(self->mutTskJob));
        tj = self->lstTskJob.front();
        self->lstTskJob.pop_front();
        pthread_mutex_unlock(&(self->mutTskJob));

        dpuRunTask(tj->tsk);

        pthread_mutex_lock(&(self->mutOUT));
        self->lstOUT.push_back(tj);
        pthread_mutex_unlock(&(self->mutOUT));
        sem_post(&(self->semOUT));
    }
    return NULL;
}

void *XdpuRunner::_write(void *t){
    class XdpuRunner *self = (class XdpuRunner *) t;

    while (true){
        DPUTJ * tj;

        sem_wait(&(self->semOUT));
        pthread_mutex_lock(&(self->mutOUT));
        tj = self->lstOUT.front();
        self->lstOUT.pop_front();
        pthread_mutex_unlock(&(self->mutOUT));

        for (int i=0; i< tj->tsk->outputTensorNum; i++){
            DPUTensorAttr * tensorAttr = tj->tsk->outputTensorAttrs + i;
            uint64_t size = tensorAttr->size;
            int8_t *buf = (int8_t *)(tensorAttr->addr_virt);

            cvt_s8_f32_neon(buf, tensorAttr->scale, size, tj->job->out[tensorAttr->tensor_name][tj->job->batch_offset]);
        }

        //wakeup waiting thread
        pthread_mutex_lock(&(self->mutWL));
        for (auto const& i : self->waitList) {
            if (i->idx <= tj->job->idx && tj->job->idx <= (i->idx+i->cnt) ) {
                i->msk = i->msk & ~(0x1<<(tj->job->idx - i->idx));
                if (i->msk ==0) {
                      pthread_mutex_unlock(&(i->wait));
                }
            }
        }
        pthread_mutex_unlock(&(self->mutWL));

        //put dputask back into the lstTsk
        pthread_mutex_lock(&(self->mutTsk));
        self->lstTsk.push_back(tj->tsk);
        pthread_mutex_unlock(&(self->mutTsk));
        sem_post(&(self->semTsk));

        // free job
        delete tj->job;
        // fre taskjob
        free(tj);
    }
    return NULL;
}

}
}

/**
 * singleton class to manage runners and cache its pytensors
 */
class DpuPyRunnerMgr {
public:
  static DpuPyRunnerMgr &instance() {
    static DpuPyRunnerMgr inst;
    return inst;
  }

  void *createRunner(char *path)
  {
    vitis::ai::XdpuRunner *runner = new vitis::ai::XdpuRunner(path);
    pyInputTensors.erase(runner);
    pyOutputTensors.erase(runner);

    std::vector<DpuPyTensor> inTensors, outTensors;

    std::vector<vitis::ai::Tensor*> in = runner->get_input_tensors();
    for (int i=0; i < in.size(); i++){
      inTensors.push_back(DpuPyTensor(*in[i]));
    }
    std::vector<vitis::ai::Tensor*> out = runner->get_output_tensors();
    for (int i=0; i < out.size(); i++){
      outTensors.push_back(DpuPyTensor(*out[i]));
    }

    pyInputTensors[runner] = inTensors;
    pyOutputTensors[runner] = outTensors;

    return runner;
  }

  void destroyRunner(void *runner)
  {
    pyInputTensors.erase(runner);
    pyOutputTensors.erase(runner);
  }

  std::vector<DpuPyTensor> &getInputTensors(void *runner)
  {
    return pyInputTensors[runner];
  }
  std::vector<DpuPyTensor> &getOutputTensors(void *runner)
  {
    return pyOutputTensors[runner];
  }

private:
  DpuPyRunnerMgr() {};
  DpuPyRunnerMgr(DpuPyRunnerMgr const&) {};
  DpuPyRunnerMgr& operator=(DpuPyRunnerMgr const&) = default;

  std::unordered_map<void*, std::vector<DpuPyTensor>> pyInputTensors;
  std::unordered_map<void*, std::vector<DpuPyTensor>> pyOutputTensors;
};

/**
 * C API implementation for python binding
 */
#ifdef __cplusplus
extern "C" {
#endif

std::vector<std::unique_ptr<vitis::ai::DpuRunner>> create_dpu_runner(
const std::string &model_directory) {

  std::vector<std::unique_ptr<vitis::ai::DpuRunner>> runners;
  vitis::ai::XdpuRunner *runner = new vitis::ai::XdpuRunner(model_directory);
  runners.emplace_back(runner);

  return runners;
}

void *DpuPyRunnerCreate(char *path)
{
  return DpuPyRunnerMgr::instance().createRunner(path);
}

void DpuPyRunnerGetInputTensors(void *runner, DpuPyTensor **tensors, int *tensor_count) {
  std::vector<DpuPyTensor> &vals
    = DpuPyRunnerMgr::instance().getInputTensors(runner);
  *tensors = vals.data();
  *tensor_count = vals.size();
}

void DpuPyRunnerGetOutputTensors(void *runner, DpuPyTensor **tensors, int *tensor_count) {
  std::vector<DpuPyTensor> &vals
    = DpuPyRunnerMgr::instance().getOutputTensors(runner);
  *tensors = vals.data();
  *tensor_count = vals.size();
}

int DpuPyRunnerGetTensorFormat(void * runner) {
    vitis::ai::DpuRunner *_runner = (vitis::ai::DpuRunner *)runner;
    return int(_runner->get_tensor_format());
}

int DpuPyRunnerExecuteAsync(void *runner_, void **pyin, void **pyout, int batch_sz, int *status)
{
  vitis::ai::XdpuRunner *runner = (vitis::ai::XdpuRunner*) runner_;
  std::vector<vitis::ai::Tensor *> inputs = runner->get_input_tensors();
  std::vector<vitis::ai::Tensor *> outputs = runner->get_output_tensors();
  std::vector<std::shared_ptr<vitis::ai::Tensor> > batchmodTensors;

  std::vector<vitis::ai::CpuFlatTensorBuffer> inputBuffers, outputBuffers;
  for (int i=0; i < inputs.size(); i++)
  {
    auto dims = inputs[i]->get_dims();
    dims[0] = batch_sz; // override batch size
    batchmodTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
      new vitis::ai::Tensor(
        inputs[i]->get_name(), dims, inputs[i]->get_data_type())));
    inputBuffers.push_back(vitis::ai::CpuFlatTensorBuffer(pyin[i],
      batchmodTensors.back().get()));
  }
  for (int i=0; i < outputs.size(); i++)
  {
    /*
     * FIXME?
     * If input batch size is N,
     * will output batch size be N as well, or merged into 1?
     */
    auto dims = outputs[i]->get_dims();
    dims[0] = batch_sz; // override batch size
    batchmodTensors.push_back(std::shared_ptr<vitis::ai::Tensor>(
      new vitis::ai::Tensor(
        outputs[i]->get_name(), dims, outputs[i]->get_data_type())));
    outputBuffers.push_back(vitis::ai::CpuFlatTensorBuffer(pyout[i],
      batchmodTensors.back().get()));
  }

  std::vector<vitis::ai::TensorBuffer*> inputBufferPtrs, outputBufferPtrs;
  for (int i=0; i < inputBuffers.size(); i++)
    inputBufferPtrs.push_back(&inputBuffers[i]);
  for (int i=0; i < outputBuffers.size(); i++)
    outputBufferPtrs.push_back(&outputBuffers[i]);

  auto response = runner->execute_async(inputBufferPtrs, outputBufferPtrs);
  *status = response.second;
  return response.first;
}

int DpuPyRunnerWait(void *runner_, int jobId)
{
  vitis::ai::XdpuRunner *runner = (vitis::ai::XdpuRunner*) runner_;
  return runner->wait(jobId);
}

void DpuPyRunnerDestroy(void *runner_)
{
  DpuPyRunnerMgr::instance().destroyRunner(runner_);
  vitis::ai::XdpuRunner *runner = (vitis::ai::XdpuRunner*) runner_;
  delete runner;
}

#ifdef __cplusplus
}
#endif
