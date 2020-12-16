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

#include "task_node_v2.h"
#include "../../common/dpu_node_v2.h"
#include "dpu_err.h"
#include "task_tensor_v2.h"


extern int dpuCacheFlush(mem_segment_t *seg, uint32_t offset, uint32_t size);
extern int dpuCacheInvalid(mem_segment_t *seg, uint32_t offset, uint32_t size);


inline static void cache_flush (task_node_t *_this, dpu_node_t *node) {
    tensor_shape_t *shape;
    task_tensor_t *tensor;
    uint32_t  total_len, cache_size;

    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(node, ERR);

    task_node_v2_t *tn = (task_node_v2_t*)_this;
    for(int i=0; i<((dpu_node_v2_t*)node)->input_cnt; i++) {
        tensor = &(tn->tensorsIn[i]);
        shape  = tensor->shape;
        if(TENSOR_ATTR_BOUNDRY_INPUT & shape->attr) {
            /* No stride, invalid continious cache aera. */
            if (shape->channel == shape->channel_stride) {
                dpuCacheFlush(tensor->dev_mem, shape->offset, shape->size);
            /* If exist stride, calculate addr size and invalid cache. */
            } else {
                cache_size = 0;
                total_len = 0;
                while(total_len < shape->size) {
                    cache_size += shape->channel_stride;
                    total_len += shape->channel;
                }
                DPU_ASSERT(total_len == shape->size, ERR);

                dpuCacheFlush(tensor->dev_mem, shape->offset, cache_size);
            }
        }
    }
}

inline static void cache_invalid_for_one_tensor(task_tensor_t *tensor) {
    tensor_shape_t *shape;
    uint32_t  total_len, cache_size;

    DPU_ASSERT(tensor, ERR);

    shape = tensor->shape;
    if(TENSOR_ATTR_BOUNDRY_OUTPUT & shape->attr) {
        /* No stride, invalid continious cache aera. */
        if (shape->channel == shape->channel_stride) {
            dpuCacheInvalid(tensor->dev_mem, shape->offset, shape->size);
        /* If exist stride, calculate addr size and invalid cache. */
        } else {
            cache_size = 0;
            total_len = 0;
            while(total_len < shape->size) {
                cache_size += shape->channel_stride;
                total_len += shape->channel;
            }
            DPU_ASSERT(total_len == shape->size, ERR);

            dpuCacheInvalid(tensor->dev_mem, shape->offset, cache_size);
        }
    }
}

inline static void cache_invalid_out (task_node_t *_this, dpu_node_t *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(node, ERR);

    task_node_v2_t *tn = (task_node_v2_t*)_this;
    for(int i=0; i<((dpu_node_v2_t*)node)->output_cnt; i++) {
        cache_invalid_for_one_tensor(&(tn->tensorsOut[i]));
    }
}

inline static task_tensor_t* get_tensorIn (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(nd, ERR);
    N2CUBE_DPU_CHECK_AND_RET_NULL(idx < ((dpu_node_v2_t*)nd)->input_cnt, N2CUBE_ERR_TENSOR_INPUT_INDEX,
              ". index: %d, Node: %s",
              idx,
              nd->name);

    task_node_v2_t *node = (task_node_v2_t*)_this;
    N2CUBE_DPU_CHECK_AND_RET_NULL(node->tensorsIn[idx].shape->attr == TENSOR_ATTR_BOUNDRY_INPUT, N2CUBE_ERR_TENSOR_INPUT_SHAPE,
              ". Node: %s, Kernel: %s\n"
              "    Please refer to DNNDK user guide for more info about \"Boundary Input Node\".",
              nd->name, kernel->base.name);

    return &(node->tensorsIn[idx]);
}

inline static task_tensor_t* get_tensorOut (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(nd, ERR);
    N2CUBE_DPU_CHECK_AND_RET_NULL(idx < ((dpu_node_v2_t*)nd)->output_cnt, N2CUBE_ERR_TENSOR_OUTPUT_INDEX,
              ". index: %d, Node: %s", idx, nd->name);

    task_node_v2_t *node = (task_node_v2_t*)_this;
    N2CUBE_DPU_CHECK_AND_RET_NULL(node->tensorsOut[idx].shape->attr == TENSOR_ATTR_BOUNDRY_OUTPUT, N2CUBE_ERR_TENSOR_INPUT_SHAPE,
              ". Node: %s, Kernel: %s\n"
              "    Please refer to DNNDK user guide for more info about \"Boundary Output Node\".",
              nd->name, kernel->base.name);
    cache_invalid_for_one_tensor(&(node->tensorsOut[idx]));
    return &(node->tensorsOut[idx]);
}


inline static void setup_tensor (task_node_t   *_this,
                                 dpu_task_t    *task,
                                 dpu_node_t    *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task && task->kernel, ERR);
    DPU_ASSERT(node, ERR);

    int i;
    mem_segment_t *mem_base;
    task_node_v2_t *tn = (task_node_v2_t*)_this;
    dpu_node_v2_t *nd = (dpu_node_v2_t*)node;
    tensor_shape_t* tensors, *tensor;

    tensors = task->kernel->base.tensor_list;

    /* setup input tensor info */
    for(i=0; i<nd->input_cnt; i++) {
        tensor = &(tensors[ nd->input_list[i] ]);
        if (tensor->attr == TENSOR_ATTR_BOUNDRY_INPUT
            && IS_MEM_INPUT_USED(task)) {
            mem_base = &(task->mem_input);
        } else mem_base = &(task->mem_IO);

        tn->tensorsIn[i].ops.setup_data(&(tn->tensorsIn[i]),
                                        tensor,
                                        mem_base,
                                        DPU_INPUT_SCALE(tensor->fix_pos));
    }

    /* setup output tensor info */
    for(i=0; i<nd->output_cnt; i++) {
        tensor = &(tensors[nd->output_list[i]]);
        if (tensor->attr == TENSOR_ATTR_BOUNDRY_OUTPUT
            && IS_MEM_OUTPUT_USED(task)) {
            mem_base = &(task->mem_output);
        } else mem_base = &(task->mem_IO);

        tn->tensorsOut[i].ops.setup_data(&(tn->tensorsOut[i]),
                                         tensor,
                                         mem_base,
                                         DPU_OUTPUT_SCALE(tensor->fix_pos));
    }
}

inline static void dump_addr_phy (task_node_t *_this,
                                        FILE *stream,
                                        const char *format) {
}

inline static void dump_addr_virt (task_node_t *_this,
                                         FILE *stream,
                                         const char *format) {
}

/*
 * Dump Input/Output tensor list into file seprately.
 * Consider of stride exist or not.
 *
 * tensors: tensor list of Input or Output.
 * tCnt   : tensor count of tensors.
 * fName   : name specified to dump file,
 *          such as "dir/kernelName_nodeName_in", "dir/kernelName_nodeName_out".
 */
inline static void dump_tensors(dpu_task_t *task, task_tensor_t * tensors, uint32_t tCnt, char *fName) {
    int i;
    char out_file[MAX_NAME_LEN];
    tensor_shape_t *shape;
    int8_t *ptr;
    uint32_t  total_len, cache_size;
    char *data;
    FILE *fp;

    /* Invalide Cache. */
    for(i=0; i<tCnt; i++) {
        shape = tensors[i].shape;

        /* No stride, invalid continious cache aera. */
        if (shape->channel == shape->channel_stride) {
            dpuCacheInvalid(tensors[i].dev_mem, shape->offset, shape->size);
        /* If exist stride, calculate addr size and invalid cache. */
        } else {
            cache_size = 0;
            total_len = 0;
            while(total_len < shape->size) {
                cache_size += shape->channel_stride;
                total_len += shape->channel;
            }
            DPU_ASSERT(total_len == shape->size, ERR);

            dpuCacheInvalid(tensors[i].dev_mem, shape->offset, cache_size);
        }
    }

    /* One single input dump into one file */
    for(i=0; i<tCnt; i++) {
        sprintf(out_file, "%s%d.bin", fName, i);
        fp = fopen(out_file, "wb");
        shape = tensors[i].shape;

        if (shape->channel == shape->channel_stride) {
            fwrite(tensors[i].addr_virt, sizeof(char), shape->size, fp);
        } else {
            /* Resize to original data according to stride. */
            ptr = tensors[i].addr_virt;
            total_len = 0;
            data = malloc(sizeof(char) * shape->size);
            memset(data, 0, sizeof(char) * shape->size);
            while(total_len < shape->size) {
                memcpy(data+total_len, ptr, shape->channel);
                ptr += shape->channel_stride;
                total_len += shape->channel;
            }
            /* total_len(after caculation) must be equal to shape->size. */
            if(total_len > shape->size) {
                free(data);
                fclose(fp);
                DPU_ASSERT(total_len == shape->size, ERR);
            }

            /* Dump to file. */
            fwrite(data, sizeof(char), shape->size, fp);
            free(data);
        }

        fflush(fp);
        fclose(fp);
    }
}

inline static void dump_input (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {
    char out_file[MAX_NAME_LEN];
    task_node_v2_t *tn;
    dpu_node_v2_t *nd;

    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task, ERR);
    DPU_ASSERT(node, ERR);

    tn = (task_node_v2_t*)_this;
    nd = (dpu_node_v2_t*)node;

    sprintf(out_file  + dump_get_dir_name(out_file), "/%s_%s_in",
            task->kernel->base.name, node->name);
    dump_tensors(task, tn->tensorsIn, nd->input_cnt, out_file);
}

inline static void dump_output (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {
    char out_file[MAX_NAME_LEN];
    task_node_v2_t *tn;
    dpu_node_v2_t *nd;

    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task, ERR);
    DPU_ASSERT(node, ERR);

    tn = (task_node_v2_t*)_this;
    nd = (dpu_node_v2_t*)node;

    sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_out",
            task->kernel->base.name, node->name);
    dump_tensors(task, tn->tensorsOut, nd->output_cnt, out_file);
}


/*
 * Release resources for task_node_v2_t itself.
 */
void task_node_v2_free(task_node_t *_this) {
    task_node_v2_t *tn = (task_node_v2_t*)_this;
    if (tn->tensorsIn) {
        free(tn->tensorsIn);
    }
    if (tn->tensorsOut) {
        free(tn->tensorsOut);
    }
}

/*
 * Destructor of task_node_v2_t structure.
 */
inline static void release (task_node_t *_this) {
    task_node_v2_free(_this);
    /* call parent struct's destructor. */
    task_node_free(_this);
}


/*
 * Constructor of task_node_v2_t structure, need to be called somewhere explicitly
 *
 * _this   : this pointer to a object constructed.
 * _inCnt  : count of tensorsIn list, used to malloc tensorsIn dynamicly.
 * _outCnt : count of tensorsOut list, used to malloc tensorsOut dynamicly.
 *
 * return value: this object pointer constructed.
 */
task_node_v2_t* task_node_v2_init(task_node_t *_this, uint32_t inCnt, uint32_t outCnt) {
    int i, size;
    task_tensor_t *tensor;
    task_node_v2_t *tn = (task_node_v2_t*)_this;

    DPU_ASSERT(_this, ERR);

    /* Call parent struct's constructor. */
    task_node_init(_this);

    tn->tensorsIn = 0;
    tn->tensorsOut = 0;
    if(inCnt>0) {
        size = sizeof(task_tensor_t) * inCnt;
        tn->tensorsIn = (task_tensor_t*)malloc(size);
        memset(tn->tensorsIn, 0, size);
        for(i=0; i<inCnt; i++) {
            tensor = &(tn->tensorsIn[i]);
            task_tensor_v2_init(tensor);
        }
    }
    if(outCnt>0) {
        size = sizeof(task_tensor_t) * outCnt;
        tn->tensorsOut = (task_tensor_t*)malloc(size);
        memset(tn->tensorsOut, 0, size);
        for(i=0; i<outCnt; i++) {
            tensor = &(tn->tensorsOut[i]);
            task_tensor_v2_init(tensor);
        }
    }

    task_ops_t *ops = &(_this->ops);
    ops->cache_flush       = cache_flush;
    ops->cache_invalid_out = cache_invalid_out;

    ops->setup_tensor   = setup_tensor;
    ops->get_tensorIn   = get_tensorIn;
    ops->get_tensorOut  = get_tensorOut;

    ops->release        = release;
    ops->dump_addr_phy  = dump_addr_phy;
    ops->dump_addr_virt = dump_addr_virt;
    ops->dump_input     = dump_input;
    ops->dump_output    = dump_output;

    return (task_node_v2_t*)_this;
}
