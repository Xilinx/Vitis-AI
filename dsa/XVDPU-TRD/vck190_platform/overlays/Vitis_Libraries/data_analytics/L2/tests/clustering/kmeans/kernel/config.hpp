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
#ifndef __KERNEL_CONFIG_HPP_
#define __KERNEL_CONFIG_HPP_

//#define AP_INT_MAX_W 2048
#define NUM 100000 // 999 // 100*100*100*100
#define NS 100
#define KC 16  // 40  // 128 // 20 //512 //100 //512 //40 //512 //40 //128 //1024
#define DIM 16 // 24 // 512 // 16 //128 //1024 //24 // 512 //80 //128   //1024
#define PCU 8  // 16
#define PDV 1
#define DBATCH ((DIM + PDV - 1) / PDV)
#define KBATCH ((KC + PCU - 1) / PCU)
#define UramDepth (DBATCH * KBATCH) // (((D + PDV - 1) / PDV) * ((K + PCU - 1) / PCU))
// double
#define UBATCH (DIM / 8)
#define NDBATCH (NUM * UBATCH)
#define XF_DATA_ANALYTICS_DEBUG 1
//#define _HW_URAM_

typedef double DType;
#endif
