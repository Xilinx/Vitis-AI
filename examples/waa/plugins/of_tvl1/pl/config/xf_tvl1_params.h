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

#ifndef _XF_TVL1_CONFIG_H_
#define _XF_TVL1_CONFIG_H_

//NPC parameter
#define NPC_TVL_INNERCORE 2
#define NPC_TVL_OTHER 1
#define NPC_IMGPYR 8

//Internal Parameter
#define ERROR_BW 32
#define INVSCALE_BITWIDTH 32
#define IMG_BW 8
#define IMG_TYPE XF_8UC1
#define IMG_F_BITS 0
#define MAX_FLOW_VALUE 6
#define MAX_NUM_LEVELS 16
#define MEDIANBLURFILTERING 5
#define MB_INST 1
#define FLOW_BITWIDTH 32
#define FLOW_TYPE XF_32SC1
#define FLOW_F_BITS 16

// port widths
#define IMAGE_PTR_WIDTH 128
#define INT_PTR_WIDTH FLOW_BITWIDTH*NPC_TVL_INNERCORE
#define OUT_PTR_WIDTH 64*NPC_TVL_INNERCORE

///////Debug micros
#define LOOP_DEBUG 0
#define LOOP1_CNT 2

// Resize Micros
#define MAXDOWNSCALE 2
#define INTERPOLATION 1
#define NEWWIDTH 512  
#define NEWHEIGHT 512 
#define WIDTH 512
#define HEIGHT 512
#endif //_XF_TVL1_CONFIG_H_
