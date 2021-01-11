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

#ifndef _DPU_ELF_H_
#define _DPU_ELF_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <linux/elf.h>
#include <stdio.h>

#include "dpu_def.h"
#include "../../common/elf.h"
EXPORT int dpu_elf_load_kernel(dpu_kernel_t *kernel);
EXPORT int dpu_elf_load_debug(dpu_kernel_t *kernel);
//

#ifdef __cplusplus
}
#endif
#endif /* _DPU_ELF_H_ */
