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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <pthread.h>

#include "dpu_task.h"
#include "aol/dpu_aol.h"
#include "dpu_scheduler.h"
#include "dpu_err.h"
#include "dpu_shm.h"
#include "dpu_caps.h"

extern dpu_aol_dev_handle_t *gp_dpu_aol_handle;

#define FALSE 0
#define TRUE 1

#define FIFO_LEN 64

//#define DPU_SCH_DEBUG
#ifdef DPU_SCH_DEBUG
#define SCH_LOG_MSG(format, ...)                                          \
    do {                                                                  \
        printf("[SCH INFO, tid %d]" format "\n", syscall(SYS_gettid), ##__VA_ARGS__ );               \
    } while (0)
#else
	#define SCH_LOG_MSG(format, ...)
#endif

const uint32_t DPU_CORE_MASK[MAX_CORE_NUM] = {0x01, 0x02, 0x04, 0x08};

typedef struct {
	uint32_t dpu_count;
	dpu_status_t dpu[MAX_CORE_NUM];
	uint32_t n2cube_mode;
	uint32_t n2cube_timeout;
}g_dpu_scheduler_t;

typedef struct {
	pthread_mutex_t n2cube_mode_mtx;
	pthread_mutex_t task_mtx;
	uint32_t task_id_generater; //< task id of current task
	uint32_t kenel_id_generater;
}local_mutex_t;

typedef struct {
	pthread_cond_t cond;
	uint32_t priority;
	uint32_t bind_core_mask;
	uint32_t assign_core_mask;
	uint32_t next;
}schedule_fifo_t;

typedef struct {
	uint8_t                 proc_run_first; // proc run on the first time
	uint32_t                dpu_status[MAX_CORE_NUM];

	pthread_mutex_t         mutex_dpu_init;
	pthread_mutex_t         mutex_dpu;

	schedule_fifo_t         fifo[FIFO_LEN];
	uint32_t                fifo_head;
	pthread_cond_t          extra_thread_cond;
	uint32_t                extra_wait_count;
}local_task_t;

// Saving in share memory
static g_dpu_scheduler_t *gp_dpu_scheduler;
static local_mutex_t local_mutex;
static local_task_t local_task = {0};

static inline uint32_t convert_core_id(uint32_t mask);

// 0xFFFF for null
static uint32_t find_top_task(uint32_t priority, uint32_t core_mask) {
	uint32_t i = 0;
	uint32_t curr = local_task.fifo_head;
	if (curr != 0xFFFF) {
		while (1) {
			if (i >= FIFO_LEN) break;
			if (curr = 0xFFFF) break;

			if ((local_task.fifo[curr].priority <= priority) && (local_task.fifo[curr].bind_core_mask & core_mask)) {
				return curr;
			} else {
				curr = local_task.fifo[curr].next;
			}
			i++;
		}
	}
	return 0xFFFF;
}

static inline uint32_t find_fifo_seat(void) {
	uint32_t find;
	for (find = 0; find < FIFO_LEN; find++) {
		if (local_task.fifo[find].bind_core_mask == 0) {
			break;
		}
	}
	return find;
}

static uint32_t insert_task(uint32_t priority, uint32_t core_mask) {
	uint32_t i;
	uint32_t find;
	uint32_t curr, next;

	curr = local_task.fifo_head;
	if (curr == 0xFFFF) {
		// insert here
		find = find_fifo_seat();
		if (find >= FIFO_LEN) return 0xFFFF;
		local_task.fifo[find].priority = priority;
		local_task.fifo[find].bind_core_mask = core_mask;
		local_task.fifo[find].next = 0xFFFF;
		local_task.fifo_head = find;
		return find;
	} else if (local_task.fifo[curr].priority > priority) {
		// insert here
		find = find_fifo_seat();
		if (find >= FIFO_LEN) return 0xFFFF;
		local_task.fifo[find].priority = priority;
		local_task.fifo[find].bind_core_mask = core_mask;
		local_task.fifo[find].next = curr;
		local_task.fifo_head = find;
		return find;
	} else {
		i = 0;
		while (1) {
			if (i >= FIFO_LEN) break;
			next = local_task.fifo[curr].next;

			if ((next == 0xFFFF) || (local_task.fifo[next].priority > priority)) {
				// insert here
				find = find_fifo_seat();
				if (find >= FIFO_LEN) return 0xFFFF;
				local_task.fifo[find].priority = priority;
				local_task.fifo[find].bind_core_mask = core_mask;
				local_task.fifo[find].next = local_task.fifo[curr].next;
				local_task.fifo[curr].next = find;
				return find;
			} else {
				curr = next;
			}

			i++;
		}
	}
	return 0xFFFF;
}

static int signal_top_task(uint32_t core_mask) {
	uint32_t curr = local_task.fifo_head;
	uint32_t *pnext = &local_task.fifo_head;
	while (1) {
		if (curr == 0xFFFF) {
			SCH_LOG_MSG("No schdule task while release DPU %d", convert_core_id(core_mask));
			if (local_task.extra_wait_count) {
				SCH_LOG_MSG("FIFO extra signal");
				pthread_cond_signal(&local_task.extra_thread_cond);
			}
			return 0;
		}

		if (local_task.fifo[curr].bind_core_mask & core_mask) {
			local_task.fifo[curr].assign_core_mask = core_mask;
			*pnext = local_task.fifo[curr].next;
			SCH_LOG_MSG("FIFO signal at %d", curr);
			pthread_cond_signal(&local_task.fifo[curr].cond);
			return 1;
		} else {
			pnext = &local_task.fifo[curr].next;
			curr = local_task.fifo[curr].next;
		}
	}
}

static inline uint32_t convert_core_id(uint32_t mask) {
	uint32_t dpu_id;
	uint32_t tester;

	tester = 1;
	for (dpu_id = 0; dpu_id < MAX_CORE_NUM; dpu_id++) {
		if ((tester & mask) != 0) {
			break;
		}
		tester <<= 1;
	}

	return dpu_id;
}

uint32_t dpu_scheduler_get_available_core_mask(dpu_task_t *task) {
	uint32_t mask = 0;
	uint32_t dpu_id;
	uint32_t priority = task->schedule_priority;
	uint32_t bind_core_mask = task->binding_core_mask;
	uint64_t pid = getpid();
	uint64_t task_id = syscall(SYS_gettid);
	int insert;

	pthread_mutex_lock(&local_task.mutex_dpu);
WAKEUP_EXTRA:
	for (int i = 0; i < gp_dpu_scheduler->dpu_count; i++) {
		if ((bind_core_mask & DPU_CORE_MASK[i]) && (local_task.dpu_status[i] == DPU_IDLE)) {
			SCH_LOG_MSG("Find Idle DPU: %d", i);
			if (find_top_task(priority, DPU_CORE_MASK[i]) == 0xFFFF) {
				gp_dpu_scheduler->dpu[i].pid = pid;
				gp_dpu_scheduler->dpu[i].task_id = task_id;
				gp_dpu_scheduler->dpu[i]._run_counter++;
				local_task.dpu_status[i] = DPU_RUNNING;
				gp_dpu_scheduler->dpu[i].status = DPU_RUNNING;
				mask = DPU_CORE_MASK[i];
				pthread_mutex_unlock(&local_task.mutex_dpu);
				SCH_LOG_MSG("Schedule DPU: %d", convert_core_id(mask));
				return mask;
			}
		}
	}

	insert = insert_task(priority, bind_core_mask);
	if (insert == 0xFFFF) {
		SCH_LOG_MSG("The fifo is full now");
		goto WAIT_EXTRA;
	}
	SCH_LOG_MSG("FIFO wait at %d", insert);
	pthread_cond_wait(&local_task.fifo[insert].cond, &local_task.mutex_dpu);
	SCH_LOG_MSG("FIFO wake up at %d", insert);
	mask = local_task.fifo[insert].assign_core_mask;
	local_task.fifo[insert].bind_core_mask = 0;
	dpu_id = convert_core_id(mask);
	gp_dpu_scheduler->dpu[dpu_id].pid = pid;
	gp_dpu_scheduler->dpu[dpu_id].task_id = task_id;
	SCH_LOG_MSG("Runing at DPU: %d", dpu_id);
	if (local_task.extra_wait_count) {
		SCH_LOG_MSG("FIFO extra signal");
		pthread_cond_signal(&local_task.extra_thread_cond);
	}
	pthread_mutex_unlock(&local_task.mutex_dpu);

	return mask;

WAIT_EXTRA:
	SCH_LOG_MSG("FIFO extro wait");
	local_task.extra_wait_count++;
	pthread_cond_wait(&local_task.extra_thread_cond, &local_task.mutex_dpu);
	local_task.extra_wait_count--;
	SCH_LOG_MSG("FIFO extra wake up");
	goto WAKEUP_EXTRA;
}

void dpu_scheduler_release_dpu_core(uint32_t mask, uint64_t time_start, uint64_t time_end) {
	uint32_t dpu_id;
	uint32_t tester;

	tester = 1;
	for (dpu_id = 0; dpu_id < gp_dpu_scheduler->dpu_count; dpu_id++) {
		if ((tester & mask) != 0) {
			break;
		}
		tester <<= 1;
	}

	pthread_mutex_lock(&local_task.mutex_dpu);
	SCH_LOG_MSG("Release DPU: %d", convert_core_id(mask));
	gp_dpu_scheduler->dpu[dpu_id]._int_counter++;
	local_task.dpu_status[dpu_id] = DPU_IDLE;
	gp_dpu_scheduler->dpu[dpu_id].status = DPU_IDLE;
	gp_dpu_scheduler->dpu[dpu_id].time_start = time_start;
	gp_dpu_scheduler->dpu[dpu_id].time_end = time_end;
	if (signal_top_task(mask)) {
		gp_dpu_scheduler->dpu[dpu_id]._run_counter++;
		local_task.dpu_status[dpu_id] = DPU_RUNNING;
		gp_dpu_scheduler->dpu[dpu_id].status = DPU_RUNNING;
	}
	pthread_mutex_unlock(&local_task.mutex_dpu);
}

void dpu_scheduler_init_process(uint32_t core_count, int schedule_mode, uint8_t shm_init) {
	uint32_t i;

	if (sizeof(g_dpu_scheduler_t) > SIZEOF_DPU_SHM) {
		printf("[DNNDK] Share memory if full.\n");
		exit(1);
	}

	gp_dpu_scheduler = (g_dpu_scheduler_t *)gp_dpu_shm;
	if (shm_init) {
		// Initial share memory
		gp_dpu_scheduler->n2cube_timeout = 10;
		gp_dpu_scheduler->dpu_count = core_count;
		gp_dpu_scheduler->n2cube_mode = 0;
		for (i = 0; i < core_count; i++) {
			gp_dpu_scheduler->dpu[i].status = DPU_IDLE;
		}

		reset_dpus(gp_dpu_aol_handle);
	}

	pthread_mutex_lock(&local_task.mutex_dpu_init);

	if (local_task.proc_run_first) {
		pthread_mutex_unlock(&local_task.mutex_dpu_init);
		return;
	}
	local_task.proc_run_first = 1;
	SCH_LOG_MSG("This is the first time run of proc, init common ram!");

	local_task.fifo_head = 0xFFFF;
	for (i = 0; i < core_count; i++) {
		local_task.dpu_status[i] = DPU_IDLE;
	}

	pthread_cond_init(&local_task.extra_thread_cond, NULL);
	for (i = 0; i < FIFO_LEN; i++) {
		pthread_cond_init(&local_task.fifo[i].cond, NULL);
	}

	pthread_mutex_unlock(&local_task.mutex_dpu_init);
}

void dpu_scheduler_get_status(uint32_t core_id, dpu_status_t *status) {
	pthread_mutex_lock(&local_task.mutex_dpu);
	memcpy(status, &gp_dpu_scheduler->dpu[core_id], sizeof(dpu_status_t));
	pthread_mutex_unlock(&local_task.mutex_dpu);
}

uint32_t dpu_gen_kernel_id(void) {
	uint32_t kernel_id;
	pthread_mutex_lock(&local_mutex.task_mtx);
	kernel_id = local_mutex.kenel_id_generater++;
	pthread_mutex_unlock(&local_mutex.task_mtx);
	return kernel_id;
}

uint32_t dpu_gen_task_id(void) {
	uint32_t task_id;
	pthread_mutex_lock(&local_mutex.task_mtx);
	task_id = local_mutex.task_id_generater++;
	pthread_mutex_unlock(&local_mutex.task_mtx);
	return task_id;
}

const char n2cube_mode_str[][8] = {
	"normal",
	"profile",
	"debug",
	"dump"
};

const char *dpu_get_n2cube_mode(void) {
	const char *pmode;
	pthread_mutex_lock(&local_mutex.n2cube_mode_mtx);
	if (gp_dpu_scheduler->n2cube_mode > 3) {
		gp_dpu_scheduler->n2cube_mode = 0;
	}
	pmode = n2cube_mode_str[gp_dpu_scheduler->n2cube_mode];
	pthread_mutex_unlock(&local_mutex.n2cube_mode_mtx);
	return pmode;
}

void dpu_set_n2cube_mode(char *mode) {
	pthread_mutex_lock(&local_mutex.n2cube_mode_mtx);
	if (strcmp(mode, "normal") == 0) {
		gp_dpu_scheduler->n2cube_mode = 0;
	} else if (strcmp(mode, "profile") == 0) {
		gp_dpu_scheduler->n2cube_mode = 1;
	} else if (strcmp(mode, "debug") == 0) {
		gp_dpu_scheduler->n2cube_mode = 2;
	} else if (strcmp(mode, "dump") == 0) {
		gp_dpu_scheduler->n2cube_mode = 3;
	} else {
	}
	pthread_mutex_unlock(&local_mutex.n2cube_mode_mtx);
}

void dpu_set_n2cube_timeout(uint32_t second) {
	gp_dpu_scheduler->n2cube_timeout = second;
}

uint32_t dpu_get_n2cube_timeout(void) {
	return gp_dpu_scheduler->n2cube_timeout;
}

void reset_dpus(dpu_aol_dev_handle_t *dev) {
	int i, ret;
	dpu_aol_init_t aol_init;

	// Reset DPUs
    for (i = 0; i < DPU_AOL_REG_NUM; i++) {
        aol_init.regs_delay_us[i] = 0;
    }
    aol_init.core_mask = 0x01;
    aol_init.ip_id = IP_ID_DPU;
    // pmu.reset
    aol_init.regs[0].value = 0;
    aol_init.regs[0].offset = (uint32_t)((unsigned long)(&g_dpu_reg.pmu.reset) - (unsigned long)(&g_dpu_reg));
    // intreg.icr
    aol_init.regs[1].value = 0xFF;
    aol_init.regs[1].offset = (uint32_t)((unsigned long)(&g_dpu_reg.intreg.icr) - (unsigned long)(&g_dpu_reg));
    aol_init.regs_delay_us[1] = 1;
    aol_init.regs[2].value = 0;
    aol_init.regs[2].offset = (uint32_t)((unsigned long)(&g_dpu_reg.intreg.icr) - (unsigned long)(&g_dpu_reg));
    // pmu.reset
    aol_init.regs[3].value = 0xFFFFFFFF;
    aol_init.regs[3].offset = (uint32_t)((unsigned long)(&g_dpu_reg.pmu.reset) - (unsigned long)(&g_dpu_reg));

    aol_init.reg_count = 4;
    ret = dpu_aol_init(dev, &aol_init);
    if (DPU_AOL_OK != ret) {
        DPU_FAIL_ON_MSG("fail to init DPU and exit ...\n");
    }
}

// using for dexplorer
void dpu_clear_counter(void) {
	pthread_mutex_lock(&local_task.mutex_dpu);
	for (int i = 0; i < gp_dpu_scheduler->dpu_count; i++) {
		gp_dpu_scheduler->dpu[i]._run_counter = 0;
		gp_dpu_scheduler->dpu[i]._int_counter = 0;
	}
	pthread_mutex_unlock(&local_task.mutex_dpu);
}
