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
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>
#include <string.h>

#include "../common/dpu_types.h"
#include "aol/dpu_aol.h"
#include "dpu_caps.h"
#include "dpu_err.h"
#include "dpu_shm.h"
#include "dpu_scheduler.h"

#define DPU_CONFIGS_SHM_KEY 0x58445055 // "XDPU"

void *gp_dpu_shm;

int dpu_config_shm(void) {
  int shmid = shmget(DPU_CONFIGS_SHM_KEY, SIZEOF_DPU_SHM, 0666 | IPC_CREAT | IPC_EXCL);
  if (shmid == -1) {
    if (errno == EEXIST) {
      // key exist , don't create
      shmid = shmget(DPU_CONFIGS_SHM_KEY, SIZEOF_DPU_SHM, 0666);
      gp_dpu_shm = shmat(shmid, NULL, 0);
	  return DPU_SHM_SECOND_TIME;
    } else {
      DPU_LOG_MSG("shmget() error, errno = %d\n", errno);
      return DPU_SHM_ERRPR;
    }
  } else {
    gp_dpu_shm = shmat(shmid, NULL, 0);
    memset(gp_dpu_shm, 0, SIZEOF_DPU_SHM);

	return DPU_SHM_FIRST_TIME;
  }

  return DPU_SHM_ERRPR;
}
