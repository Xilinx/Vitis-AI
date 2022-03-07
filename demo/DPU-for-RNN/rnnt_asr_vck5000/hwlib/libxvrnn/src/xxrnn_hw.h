/*
 * Copyright 2019 Xilinx Inc.
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

#define MAX_SIZE(type)      type##_MAX_SIZE
#define ADDR(type)          DDR_##type

#define LSB(value)           (value & 0xFFFFFFFF)
#define HSB(value)           ((value>>32) & 0xFFFFFFFF)
 
#define REG_AP_CTRL            (0x00000000)
#define REG_SOFT_RESET         (0x00000010)
#define REG_INSTR_LOW_ADDR     (0x00000014)
#define REG_INSTR_HIGH_ADDR    (0x00000018)
#define REG_VECTOR_LOW_ADDR    (0x0000011C)
#define REG_VECTOR_HIGH_ADDR   (0x00000120)
#define REG_BIAS_LOW_ADDR      (0x00000124)
#define REG_BIAS_HIGH_ADDR     (0x00000128)
#define REG_RESULT_LOW_ADDR    (0x0000012C)
#define REG_RESULT_HIGH_ADDR   (0x00000130)
#define REG_PROF_ADDR          (0x00000134)
#define REG_PROF_EN            (0x00000138)
#define REG_CFG_DONE           (0x0000013C)
#define REG_STATUS_0           (0x00000148)
#define REG_STATUS_1           (0x0000014C)
#define REG_STATUS_2           (0x00000150)
#define REG_STATUS_3           (0x00000154)
#define REG_STATUS_4           (0x00000158)
#define REG_STATUS_5           (0x0000015C)
#define REG_STATUS_6           (0x00000160)
#define REG_STATUS_7           (0x00000164)

#define DDR_WEIGHT             (0x00000000)
#define WEIGHT_MAX_SIZE        (0x02000000)
#define DDR_TANH               (0x02000000)
#define TANH_MAX_SIZE          (0x00100000)
#define DDR_SGMD               (0x02100000)
#define SGMD_MAX_SIZE          (0x00100000)
#define DDR_VECTOR             (0x03000000)
#define VECTOR_MAX_SIZE        (0x01000000)
#define DDR_BIAS               (0x04000000)
#define BIAS_MAX_SIZE          (0x01000000)
#define DDR_CELL               (0x05000000)
#define CELL_MAX_SIZE          (0x01000000)
#define DDR_RESL               (0x06000000)
#define RESL_MAX_SIZE          (0x01000000)
#define DDR_INSTR              (0x07000000)
#define INSTR_MAX_SIZE         (0x01000000)
#define DDR_PROF               (0x08000000)
#define PROF_MAX_SIZE          (0x00001000)

#define XXRNN_CTRL_ADDR_LEN_DATA    (0x2C0)

typedef struct 
{
    uint32_t addr;
    uint32_t value;
}xrnn_reg_t;

static xrnn_reg_t sentiment_regs[]={
    {0x00000168, 0x00000001},
    {0x0000016c, 0x000001f4},
    {0x00000170, 0x07000000},
    {0x00000190, 0x0000013b},
    {0x000001b0, 0x07001400},
    {0x000001d0, 0x00000034},
    {0x000001f0, 0x07001800},
    {0x000001f4, 0x00000040},
    {0x00000214, 0x00000100},
    {0x00000234, 0x03000000},
    {0x00000254, 0x06000000},
    {0x00000274, 0x06000000},
    {0x00000294, 0x00000001},

};

static xrnn_reg_t satisfaction_regs[]={
    {0x00000168, 0x00000001},
    {0x0000016c, 0x00000019},
    {0x00000170, 0x07000000},
    {0x00000190, 0x0000013b},
    {0x000001b0, 0x07001400},
    {0x000001d0, 0x00000034},
    {0x000001f0, 0x07001800},
    {0x000001f4, 0x00000040},
    {0x00000214, 0x00000100},
    {0x00000234, 0x03000000},
    {0x00000254, 0x06000000},
    {0x00000274, 0x06000000},
    {0x00000294, 0x00000001},
};


static xrnn_reg_t openie_regs[]={
    {0x00000168, 0x00000008},
    {0x0000016c, 0x00000000},
    {0x00000170, 0x07000000},
    {0x00000174, 0x07002900},
    {0x00000178, 0x07005200},
    {0x0000017c, 0x07007b00},
    {0x00000180, 0x0700a400},
    {0x00000184, 0x0700cd00},
    {0x00000188, 0x0700f600},
    {0x0000018c, 0x07011f00},
    {0x00000190, 0x000001fd},
    {0x00000194, 0x000001f9},
    {0x00000198, 0x000001f9},
    {0x0000019c, 0x000001f9},
    {0x000001a0, 0x000001f9},
    {0x000001a4, 0x000001f9},
    {0x000001a8, 0x000001f9},
    {0x000001ac, 0x000001f8},
    {0x000001b0, 0x07002000},
    {0x000001b4, 0x07004900},
    {0x000001b8, 0x07007200},
    {0x000001bc, 0x07009b00},
    {0x000001c0, 0x0700c400},
    {0x000001c4, 0x0700ed00},
    {0x000001c8, 0x07011600},
    {0x000001cc, 0x07013f00},
    {0x000001d0, 0x0000008e},
    {0x000001d4, 0x0000008e},
    {0x000001d8, 0x0000008e},
    {0x000001dc, 0x0000008e},
    {0x000001e0, 0x0000008e},
    {0x000001e4, 0x0000008e},
    {0x000001e8, 0x0000008e},
    {0x000001ec, 0x0000008e},
    {0x000001f0, 0x07014800},
    {0x000001f4, 0x000001c0},
    {0x000001f8, 0x00000280},
    {0x000001fc, 0x00000280},
    {0x00000200, 0x00000280},
    {0x00000204, 0x00000280},
    {0x00000208, 0x00000280},
    {0x0000020c, 0x00000280},
    {0x00000210, 0x00000280},
    {0x00000214, 0x00000280},
    {0x00000218, 0x00000280},
    {0x0000021c, 0x00000280},
    {0x00000220, 0x00000280},
    {0x00000224, 0x00000280},
    {0x00000228, 0x00000280},
    {0x0000022c, 0x00000280},
    {0x00000230, 0x00000280},
    {0x00000234, 0x03000000},
    {0x00000238, 0x06000000},
    {0x0000023c, 0x03000000},
    {0x00000240, 0x06000000},
    {0x00000244, 0x03000000},
    {0x00000248, 0x06000000},
    {0x0000024c, 0x03000000},
    {0x00000250, 0x06000000},
    {0x00000254, 0x06000000},  // index 59:  cal + (frm_num-1)*value in 0x214
    {0x00000258, 0x03000000},
    {0x0000025c, 0x06000000},
    {0x00000260, 0x03000000},
    {0x00000264, 0x06000000},
    {0x00000268, 0x03000000},
    {0x0000026c, 0x06000000},
    {0x00000270, 0x03000000},
    {0x00000274, 0x06000000},
    {0x00000278, 0x03000000},
    {0x0000027c, 0x06000000},
    {0x00000280, 0x03000000},
    {0x00000284, 0x06000000},
    {0x00000288, 0x03000000},
    {0x0000028c, 0x06000000},
    {0x00000290, 0x03000000}, // index 74
    {0x00000294, 0x00000000},
    {0x00000298, 0x00000000},
    {0x0000029c, 0x00000000},
    {0x000002a0, 0x00000000},
    {0x000002a4, 0x00000000},
    {0x000002a8, 0x00000000},
    {0x000002ac, 0x00000000},
    {0x000002b0, 0x00000000},
};



