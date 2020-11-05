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
#include "task_node_v1.h"
#include "dpu_node_v1_real.h"
#include "dpu_err.h"
#include "task_tensor_v1.h"

extern int dpuCacheFlush(mem_segment_t *seg, uint32_t offset, uint32_t size);
extern int dpuCacheInvalid(mem_segment_t *seg, uint32_t offset, uint32_t size);



inline static void cache_flush (task_node_t *_this, dpu_node_t *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(node, ERR);

    task_node_v1_t *tn = (task_node_v1_t*)_this;
    if(TENSOR_ATTR_BOUNDRY_INPUT & tn->tensorIn.shape->attr) {
        dpuCacheFlush(tn->tensorIn.dev_mem, tn->tensorIn.shape->offset, tn->tensorIn.shape->size);
    }
}

inline static void cache_invalid_out (task_node_t *_this, dpu_node_t *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(node, ERR);

    task_node_v1_t *tn = (task_node_v1_t*)_this;
    if(TENSOR_ATTR_BOUNDRY_OUTPUT & tn->tensorOut.shape->attr) {
        dpuCacheInvalid(tn->tensorOut.dev_mem, tn->tensorOut.shape->offset, tn->tensorOut.shape->size);
    }
}
inline static task_tensor_t* get_tensorIn (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_ASSERT(_this, ERR);

    task_node_v1_t *node = (task_node_v1_t*)_this;
    return &(node->tensorIn);
}

inline static task_tensor_t* get_tensorOut (task_node_t* _this, int idx, dpu_node_t *nd, dpu_kernel_t *kernel) {
    DPU_ASSERT(_this, ERR);

    task_node_v1_t *node = (task_node_v1_t*)_this;
    task_tensor_t *tensor = &(node->tensorOut);
    if (!(tensor->shape->attr & TENSOR_ATTR_BOUNDRY_OUTPUT)) {
        tensor->shape->attr = (tensor_attr_t)(tensor->shape->attr | TENSOR_ATTR_BOUNDRY_OUTPUT);
        dpuCacheInvalid(tensor->dev_mem, tensor->shape->offset, tensor->shape->size);
    }
    return tensor;
}



inline static void setup_tensor (task_node_t   *_this,
                                 dpu_task_t    *task,
                                 dpu_node_t    *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task && task->kernel, ERR);
    DPU_ASSERT(node, ERR);

    task_node_v1_t *tn = (task_node_v1_t*)_this;
    dpu_node_v1_real_t *nd = (dpu_node_v1_real_t*)node;

    tensor_shape_t *tensor = &(nd->shapeIn);
    tn->tensorIn.ops.setup_data(&(tn->tensorIn),
                                tensor,
                                &(task->mem_IO),
                                DPU_INPUT_SCALE(tensor->fix_pos));

    tensor = &(nd->base_v1.shapeOut);
    tn->tensorIn.ops.setup_data(&(tn->tensorOut),
                                tensor,
                                &(task->mem_IO),
                                DPU_OUTPUT_SCALE(tensor->fix_pos));
}

inline static void dump_addr_phy (task_node_t *_this,
                                        FILE *stream,
                                        const char *format) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(stream, ERR);
    DPU_ASSERT(format, ERR);

    task_node_v1_t *node = (task_node_v1_t*)_this;
    fprintf(stream, format, node->tensorIn.addr_phy, node->tensorOut.addr_phy);
}

inline static void dump_addr_virt (task_node_t *_this,
                                         FILE *stream,
                                         const char *format) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(stream, ERR);
    DPU_ASSERT(format, ERR);

    task_node_v1_t *node = (task_node_v1_t*)_this;
    fprintf(stream, format, node->tensorIn.addr_virt, node->tensorOut.addr_virt);
}

inline static void dump_input (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task, ERR);
    DPU_ASSERT(node, ERR);

    FILE *fp;
    char out_file[MAX_NAME_LEN];
    task_node_v1_t *tn;

    tn = (task_node_v1_t*)_this;
    sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_in.bin",
            task->kernel->base.name, node->name);

    fp = fopen(out_file, "wb");
    dpuCacheInvalid(tn->tensorIn.dev_mem, tn->tensorIn.shape->offset, tn->tensorIn.shape->size);
    fwrite(tn->tensorIn.addr_virt, sizeof(char),
           tn->tensorIn.shape->size, fp);

    fflush(fp);
    fclose(fp);
}

inline static void dump_output (task_node_t *_this, dpu_task_t *task, dpu_node_t *node) {
    DPU_ASSERT(_this, ERR);
    DPU_ASSERT(task, ERR);
    DPU_ASSERT(node, ERR);

    FILE *fp;
    char out_file[MAX_NAME_LEN];
    task_node_v1_t *tn;

    tn = (task_node_v1_t*)_this;
    sprintf(out_file + dump_get_dir_name(out_file), "/%s_%s_out.bin",
        task->kernel->base.name, node->name);

    fp = fopen(out_file, "wb");
    dpuCacheInvalid(tn->tensorOut.dev_mem, tn->tensorOut.shape->offset, tn->tensorOut.shape->size);
    fwrite(tn->tensorOut.addr_virt, sizeof(char),
           tn->tensorOut.shape->size, fp);

    fflush(fp);
    fclose(fp);
}


/*
 * Release resources for task_node_v1_t itself.
 */
void task_node_v1_free(task_node_t *_this) {
    DPU_ASSERT(_this, ERR);
}

/*
 * Destructor of task_node_v1_t structure.
 */
inline static void release (task_node_t *_this) {
    DPU_ASSERT(_this, ERR);

    task_node_v1_free(_this);
    /* Call parent struct's destructor. */
    task_node_free(_this);
}

/*
 * Constructor of task_node_v1_t structure, need to be called somewhere explicitly
 */
task_node_v1_t* task_node_v1_init(task_node_t *_this) {
    DPU_ASSERT(_this, ERR);

    /* Call parent struct's constructor. */
    task_node_init(_this);

    /* Init members */
    task_node_v1_t *tn = (task_node_v1_t*)_this;
    task_tensor_v1_init(&(tn->tensorIn));
    task_tensor_v1_init(&(tn->tensorOut));

    task_ops_t *ops = &(_this->ops);
    ops->cache_flush    = cache_flush;
    ops->cache_invalid_out  = cache_invalid_out;

    ops->get_tensorIn   = get_tensorIn;
    ops->get_tensorOut  = get_tensorOut;

    ops->setup_tensor   = setup_tensor;
    ops->release        = release;
    ops->dump_addr_phy  = dump_addr_phy;
    ops->dump_addr_virt = dump_addr_virt;
    ops->dump_input     = dump_input;
    ops->dump_output    = dump_output;
}
