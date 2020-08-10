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

#ifndef __DPU_SHM_H__
#define __DPU_SHM_H__

#define SIZEOF_DPU_SHM (1024)

extern void *gp_dpu_shm;

#define DPU_SHM_FIRST_TIME 1
#define DPU_SHM_SECOND_TIME 2
#define DPU_SHM_ERRPR 0
int dpu_config_shm(void);

#endif
