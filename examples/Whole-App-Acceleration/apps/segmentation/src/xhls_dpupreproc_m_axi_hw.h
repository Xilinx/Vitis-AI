// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 4  - ap_continue (Read/Write/SC)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - enable ap_done interrupt (Read/Write)
//        bit 1  - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - ap_done (COR/TOW)
//        bit 1  - ap_ready (COR/TOW)
//        others - reserved
// 0x10 : Data signal of img_inp
//        bit 31~0 - img_inp[31:0] (Read/Write)
// 0x14 : Data signal of img_inp
//        bit 31~0 - img_inp[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of img_out
//        bit 31~0 - img_out[31:0] (Read/Write)
// 0x20 : Data signal of img_out
//        bit 31~0 - img_out[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of norm_fact
//        bit 31~0 - norm_fact[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of shift_fact
//        bit 31~0 - shift_fact[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of scale_fact
//        bit 31~0 - scale_fact[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of height
//        bit 15~0 - height[15:0] (Read/Write)
//        others   - reserved
// 0x44 : reserved
// 0x48 : Data signal of width
//        bit 15~0 - width[15:0] (Read/Write)
//        others   - reserved
// 0x4c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on
// Handshake)

#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_AP_CTRL 0x00
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_GIE 0x04
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IER 0x08
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_ISR 0x0c

#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_INP_DATA 0x10
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_IMG_INP_DATA 64
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_IMG_OUT_DATA 0x1c
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_IMG_OUT_DATA 64
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_NORM_FACT_DATA 0x28
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_NORM_FACT_DATA 32
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_SHIFT_FACT_DATA 0x30
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_SHIFT_FACT_DATA 32
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_SCALE_FACT_DATA 0x38
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_SCALE_FACT_DATA 32
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_HEIGHT_DATA 0x40
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_HEIGHT_DATA 16
#define XHLS_DPUPREPROC_M_AXI_CONTROL_ADDR_WIDTH_DATA 0x48
#define XHLS_DPUPREPROC_M_AXI_CONTROL_BITS_WIDTH_DATA 16
