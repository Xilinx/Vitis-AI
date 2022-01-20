// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xisppipeline_accel.h"

extern XIsppipeline_accel_Config XIsppipeline_accel_ConfigTable[];

XIsppipeline_accel_Config *XIsppipeline_accel_LookupConfig(u16 DeviceId) {
	XIsppipeline_accel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XISPPIPELINE_ACCEL_NUM_INSTANCES; Index++) {
		if (XIsppipeline_accel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XIsppipeline_accel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XIsppipeline_accel_Initialize(XIsppipeline_accel *InstancePtr, u16 DeviceId) {
	XIsppipeline_accel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XIsppipeline_accel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XIsppipeline_accel_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

