// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xisppipeline_accel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XIsppipeline_accel_CfgInitialize(XIsppipeline_accel *InstancePtr, XIsppipeline_accel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_BaseAddress = ConfigPtr->Ctrl_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XIsppipeline_accel_Start(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL) & 0x80;
    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XIsppipeline_accel_IsDone(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XIsppipeline_accel_IsIdle(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XIsppipeline_accel_IsReady(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XIsppipeline_accel_EnableAutoRestart(XIsppipeline_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL, 0x80);
}

void XIsppipeline_accel_DisableAutoRestart(XIsppipeline_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_AP_CTRL, 0);
}

void XIsppipeline_accel_Set_width(XIsppipeline_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_WIDTH_DATA, Data);
}

u32 XIsppipeline_accel_Get_width(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_WIDTH_DATA);
    return Data;
}

void XIsppipeline_accel_Set_height(XIsppipeline_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_HEIGHT_DATA, Data);
}

u32 XIsppipeline_accel_Get_height(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_HEIGHT_DATA);
    return Data;
}

void XIsppipeline_accel_Set_bayer_phase(XIsppipeline_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_BAYER_PHASE_DATA, Data);
}

u32 XIsppipeline_accel_Get_bayer_phase(XIsppipeline_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_BAYER_PHASE_DATA);
    return Data;
}

void XIsppipeline_accel_InterruptGlobalEnable(XIsppipeline_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_GIE, 1);
}

void XIsppipeline_accel_InterruptGlobalDisable(XIsppipeline_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_GIE, 0);
}

void XIsppipeline_accel_InterruptEnable(XIsppipeline_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_IER);
    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_IER, Register | Mask);
}

void XIsppipeline_accel_InterruptDisable(XIsppipeline_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_IER);
    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_IER, Register & (~Mask));
}

void XIsppipeline_accel_InterruptClear(XIsppipeline_accel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIsppipeline_accel_WriteReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_ISR, Mask);
}

u32 XIsppipeline_accel_InterruptGetEnabled(XIsppipeline_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_IER);
}

u32 XIsppipeline_accel_InterruptGetStatus(XIsppipeline_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XIsppipeline_accel_ReadReg(InstancePtr->Ctrl_BaseAddress, XISPPIPELINE_ACCEL_CTRL_ADDR_ISR);
}

