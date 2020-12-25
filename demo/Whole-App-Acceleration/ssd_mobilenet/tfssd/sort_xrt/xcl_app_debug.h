/**
 * Copyright (C) 2016-2017 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/**
 * Xilinx SDAccel HAL userspace driver extension APIs
 * Performance Monitoring Exposed Parameters
 * Copyright (C) 2015-2017, Xilinx Inc - All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _XCL_DEBUG_H_
#define _XCL_DEBUG_H_

// For performance counter definitions
#include "xclperf.h"

#ifdef __cplusplus
extern "C" {
#endif

/************************ AIM Debug Counters ********************************/
#define XAIM_DEBUG_SAMPLE_COUNTERS_PER_SLOT     9

/************************ AM Debug Counters ********************************/
#define XAM_DEBUG_SAMPLE_COUNTERS_PER_SLOT     8

/************************ ASM Debug Counters ********************************/
#define XASM_DEBUG_SAMPLE_COUNTERS_PER_SLOT    5

/*
 * LAPC related defs here
 */
#define XLAPC_MAX_NUMBER_SLOTS           31
#define XLAPC_STATUS_PER_SLOT            9

/* Metric counters per slot */
#define XLAPC_OVERALL_STATUS                0
#define XLAPC_CUMULATIVE_STATUS_0           1
#define XLAPC_CUMULATIVE_STATUS_1           2
#define XLAPC_CUMULATIVE_STATUS_2           3
#define XLAPC_CUMULATIVE_STATUS_3           4
#define XLAPC_SNAPSHOT_STATUS_0             5
#define XLAPC_SNAPSHOT_STATUS_1             6
#define XLAPC_SNAPSHOT_STATUS_2             7
#define XLAPC_SNAPSHOT_STATUS_3             8

/*
 * AXI Streaming Protocol Checker related defs here
 */
#define XSPC_MAX_NUMBER_SLOTS 31

/********************** Definitions: Enums, Structs ***************************/
enum xclDebugReadType {
  XCL_DEBUG_READ_TYPE_APM  = 0,
  XCL_DEBUG_READ_TYPE_LAPC = 1,
  XCL_DEBUG_READ_TYPE_AIM  = 2,
  XCL_DEBUG_READ_TYPE_ASM  = 3,
  XCL_DEBUG_READ_TYPE_AM   = 4,
  XCL_DEBUG_READ_TYPE_SPC  = 5
};

/* Debug counter results */
typedef struct {
  unsigned long long int WriteBytes     [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int WriteTranx     [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int ReadBytes      [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int ReadTranx      [XAIM_MAX_NUMBER_SLOTS];

  unsigned long long int OutStandCnts   [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastWriteAddr  [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastWriteData  [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastReadAddr   [XAIM_MAX_NUMBER_SLOTS];
  unsigned long long int LastReadData   [XAIM_MAX_NUMBER_SLOTS];
  unsigned int           NumSlots;
  char                   DevUserName    [256];
} xclDebugCountersResults;

typedef struct {
  unsigned int           NumSlots ;
  char                   DevUserName    [256] ;

  unsigned long long int StrNumTranx    [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrDataBytes   [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrBusyCycles  [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrStallCycles [XASM_MAX_NUMBER_SLOTS] ;
  unsigned long long int StrStarveCycles[XASM_MAX_NUMBER_SLOTS] ;
} xclStreamingDebugCountersResults ;

typedef struct {
  unsigned int           NumSlots ;
  char                   DevUserName    [256] ;

  unsigned long long CuExecCount        [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuExecCycles       [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuBusyCycles       [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxParallelIter  [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallExtCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallIntCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStallStrCycles   [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMinExecCycles    [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuMaxExecCycles    [XAM_MAX_NUMBER_SLOTS];
  unsigned long long CuStartCount       [XAM_MAX_NUMBER_SLOTS];
} xclAccelMonitorCounterResults;

enum xclCheckerType {
XCL_CHECKER_MEMORY = 0,
XCL_CHECKER_STREAM = 1
};

/* Debug checker results */
typedef struct {
  unsigned int   OverallStatus[XLAPC_MAX_NUMBER_SLOTS];
  unsigned int   CumulativeStatus[XLAPC_MAX_NUMBER_SLOTS][4];
  unsigned int   SnapshotStatus[XLAPC_MAX_NUMBER_SLOTS][4];
  unsigned int   NumSlots;
  char DevUserName[256];
} xclDebugCheckersResults;

typedef struct {
  unsigned int PCAsserted[XSPC_MAX_NUMBER_SLOTS];
  unsigned int CurrentPC [XSPC_MAX_NUMBER_SLOTS];
  unsigned int SnapshotPC[XSPC_MAX_NUMBER_SLOTS];
  unsigned int NumSlots;
  char DevUserName[256];
} xclDebugStreamingCheckersResults;

#ifdef __cplusplus
}
#endif
#endif
