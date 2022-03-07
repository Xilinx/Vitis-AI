// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XISPPIPELINE_ACCEL_H
#define XISPPIPELINE_ACCEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xisppipeline_accel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
#else
typedef struct {
    u16 DeviceId;
    u32 Ctrl_BaseAddress;
} XIsppipeline_accel_Config;
#endif

typedef struct {
    u32 Ctrl_BaseAddress;
    u32 IsReady;
} XIsppipeline_accel;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XIsppipeline_accel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XIsppipeline_accel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XIsppipeline_accel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XIsppipeline_accel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XIsppipeline_accel_Initialize(XIsppipeline_accel *InstancePtr, u16 DeviceId);
XIsppipeline_accel_Config* XIsppipeline_accel_LookupConfig(u16 DeviceId);
int XIsppipeline_accel_CfgInitialize(XIsppipeline_accel *InstancePtr, XIsppipeline_accel_Config *ConfigPtr);
#else
int XIsppipeline_accel_Initialize(XIsppipeline_accel *InstancePtr, const char* InstanceName);
int XIsppipeline_accel_Release(XIsppipeline_accel *InstancePtr);
#endif

void XIsppipeline_accel_Start(XIsppipeline_accel *InstancePtr);
u32 XIsppipeline_accel_IsDone(XIsppipeline_accel *InstancePtr);
u32 XIsppipeline_accel_IsIdle(XIsppipeline_accel *InstancePtr);
u32 XIsppipeline_accel_IsReady(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_EnableAutoRestart(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_DisableAutoRestart(XIsppipeline_accel *InstancePtr);

void XIsppipeline_accel_Set_width(XIsppipeline_accel *InstancePtr, u32 Data);
u32 XIsppipeline_accel_Get_width(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_Set_height(XIsppipeline_accel *InstancePtr, u32 Data);
u32 XIsppipeline_accel_Get_height(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_Set_bayer_phase(XIsppipeline_accel *InstancePtr, u32 Data);
u32 XIsppipeline_accel_Get_bayer_phase(XIsppipeline_accel *InstancePtr);

void XIsppipeline_accel_InterruptGlobalEnable(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_InterruptGlobalDisable(XIsppipeline_accel *InstancePtr);
void XIsppipeline_accel_InterruptEnable(XIsppipeline_accel *InstancePtr, u32 Mask);
void XIsppipeline_accel_InterruptDisable(XIsppipeline_accel *InstancePtr, u32 Mask);
void XIsppipeline_accel_InterruptClear(XIsppipeline_accel *InstancePtr, u32 Mask);
u32 XIsppipeline_accel_InterruptGetEnabled(XIsppipeline_accel *InstancePtr);
u32 XIsppipeline_accel_InterruptGetStatus(XIsppipeline_accel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
