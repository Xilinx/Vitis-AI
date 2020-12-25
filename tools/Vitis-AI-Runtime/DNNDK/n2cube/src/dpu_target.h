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

#ifndef _DPU_TARGET_H_
#define _DPU_TARGET_H_

#include "dpu_def.h"

struct dpu_inst_t {
    unsigned char id;
    unsigned char len;                /* length of instruction in byte */
    unsigned char name[MAX_NAME_LEN]; /* name of instruction */
    unsigned char reloc;              /* if needs to be relocated */
    unsigned char reloc_pos;          /* postion (relative to the start of inst) to be relocated in byte */
    unsigned char reloc_len;          /* length of relocation address in byte */
};

/*
 * instruction format description for CNNv1
 */
static const struct dpu_inst_t cnn_v1_inst_tbl[] = {
    {0b0000,    3*8,    "load_i",      1,    8,    4},  /* load image */
    {0b0001,    2*8,    "load_w",      1,    8,    4},  /* load weight */
    {0b0010,    2*8,    "load_bias",   1,    8,    4},  /* load bias */
    {0b0100,    3*8,    "calc",        0,    0,    0},  /* calculation */
    {0b1000,    3*8,    "save",        1,    8,    4}   /* save */
};

/*
 * instruction format description for CNNv2
 */
static const struct dpu_inst_t cnn_v2_inst_tbl[] = {
    {0b0000,    3*4,    "load",       1,    8,    4},   /* load */
    {0b0100,    4*4,    "save",       1,   12,    4},   /* save */
    {0b1000,    4*4,    "conv",       0,    0,    0},   /* conv */
    {0b1001,    2*4,    "convinit",   0,    0,    0},   /* convinit */
    {0b1100,    3*4,    "pool",       0,    0,    0},   /* pool */
    {0b1101,    2*4,    "elewinit",   0,    0,    0},   /* elewinit */
    {0b1110,    2*4,    "elew",       0,    0,    0},   /* elew */
    {0b1111,    1*4,    "end",        0,    0,    0},   /* end */
    {0b1100,    3*4,    "pool2",      0,    0,    0},   /* pool2 */
};

#endif
