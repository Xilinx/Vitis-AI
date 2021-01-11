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

#ifndef __DPU_SCHEDULER_H__
#define __DPU_SCHEDULER_H__

#ifdef __cplusplus
extern "C" {
#endif

#define DPU_IDLE 0
#define DPU_RUNNING 1
#define DPU_DISABLE 2

typedef struct {
	uint32_t status;
	uint32_t _run_counter;
	uint32_t _int_counter;
	uint64_t time_start;
	uint64_t time_end;
    uint64_t pid;
	uint64_t task_id;
}dpu_status_t;

void dpu_scheduler_init_process(uint32_t core_count, int schedule_mode, uint8_t shm_init);

uint32_t dpu_scheduler_get_available_core_mask(dpu_task_t *task);
void dpu_scheduler_release_dpu_core(uint32_t mask, uint64_t time_start, uint64_t time_end);

void dpu_scheduler_get_status(uint32_t core_id, dpu_status_t *status);

uint32_t dpu_gen_kernel_id(void);
uint32_t dpu_gen_task_id(void);

const char *dpu_get_n2cube_mode(void);
void dpu_set_n2cube_mode(char *mode);

void dpu_set_n2cube_timeout(uint32_t second);
uint32_t dpu_get_n2cube_timeout(void);

void reset_dpus(dpu_aol_dev_handle_t *dev);

#ifdef __cplusplus
}
#endif

#endif
