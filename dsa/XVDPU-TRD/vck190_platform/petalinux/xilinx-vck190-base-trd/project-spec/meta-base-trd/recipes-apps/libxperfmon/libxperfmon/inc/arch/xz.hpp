/*
 * Copyright (C) 2020 – 2021 Xilinx, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * XILINX BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Xilinx shall not be used
 * in advertising or otherwise to promote the sale, use or other dealings in
 * this Software without prior written authorization from Xilinx.
 */

#ifndef __XAPM_HPP__
#include "xperfmon.hpp"
#endif

#define ZYNQMP_APM_CCI_BASE 0xFD490000
#define ZYNQMP_APM_DDR_BASE 0xFD0B0000
#define ZYNQMP_APM_OCM_BASE 0xFFA00000
#define ZYNQMP_APM_LPD_FPD_BASE 0xFFA10000

#define ZYNQMP_SMCR 0x200
#define ZYNQMP_APM_MSR 0x44
#define ZYNQMP_APM_MSR_OFFSET (ZYNQMP_APM_MSR + 4)

#define ZYNQMP_APM_SLOT_0_READ_BYTE 0x03
#define ZYNQMP_APM_SLOT_1_READ_BYTE (0x23 << 8)
#define ZYNQMP_APM_SLOT_2_READ_BYTE (0x43 << 16)
#define ZYNQMP_APM_SLOT_3_READ_BYTE (0x63 << 24)
#define ZYNQMP_APM_SLOT_4_READ_BYTE (0x83 << 8)
#define ZYNQMP_APM_SLOT_5_READ_BYTE (0xA3 << 16)

#define ZYNQMP_APM_SLOT_0_SMCR 0x200
#define ZYNQMP_APM_SLOT_1_SMCR 0x210
#define ZYNQMP_APM_SLOT_2_SMCR 0x220
#define ZYNQMP_APM_SLOT_3_SMCR 0x230
#define ZYNQMP_APM_SLOT_4_SMCR 0x240
#define ZYNQMP_APM_SLOT_5_SMCR 0x250

enum {
  zynqmp_cci_apm,
  zynqmp_ddr_apm,
  zynqmp_ocm_apm,
  zynqmp_fpd_apm,
  zynqmp_apm_count
};

std::vector<apm_t> ddr {
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 0,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_0_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_0_SMCR},
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 1,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_1_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_1_SMCR},
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 2,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_2_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_2_SMCR},
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 3,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_3_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_3_SMCR},
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 4,
     .msr_read = ZYNQMP_APM_MSR_OFFSET,
     .msr_write = ZYNQMP_APM_SLOT_4_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_4_SMCR},
    {.addr = ZYNQMP_APM_DDR_BASE,
     .slot = 5,
     .msr_read = ZYNQMP_APM_MSR_OFFSET,
     .msr_write = ZYNQMP_APM_SLOT_5_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_5_SMCR},
};

std::vector<apm_t> cci {
    {.addr = ZYNQMP_APM_CCI_BASE,
     .slot = 0,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_0_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_0_SMCR},
};

std::vector<apm_t> fpd {
    {.addr = ZYNQMP_APM_LPD_FPD_BASE,
     .slot = 0,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_0_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_0_SMCR},
};

std::vector<apm_t> ocm {
    {.addr = ZYNQMP_APM_OCM_BASE,
     .slot = 0,
     .msr_read = ZYNQMP_APM_MSR,
     .msr_write = ZYNQMP_APM_SLOT_0_READ_BYTE,
     .smcr = ZYNQMP_APM_SLOT_0_SMCR},
};
