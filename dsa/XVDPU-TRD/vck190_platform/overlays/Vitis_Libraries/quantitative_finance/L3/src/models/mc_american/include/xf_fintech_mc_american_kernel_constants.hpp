/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _XF_FINTECH_MC_AMERICAN_KERNEL_CONSTANTS_H_
#define _XF_FINTECH_MC_AMERICAN_KERNEL_CONSTANTS_H_

// MC American
typedef double KDataType;
#define TIMESTEPS (100)
#define COEF (4)
#define UN_K1 (2)
#define Unroll_STEP (2)
#define UN_K2_PATH (2)
#define UN_K3 (4)
#define ITERATION (4)
#define DEPTH_P (1024 * TIMESTEPS * ITERATION)
#define DEPTH_M (9 * TIMESTEPS)
#define SZ (8 * sizeof(KDataType))
#define COEF_DEPTH (1024)

#endif //_XF_FINTECH_MC_AMERICAN_KERNEL_CONSTANTS_H_
