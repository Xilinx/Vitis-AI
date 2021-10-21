.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


*************************************************
Internal Design of CDS Engine
*************************************************

Implementation
==============

Host
****
The host code (*main.cpp*) contains the OpenCL calls to invoke the CDS kernel and test for accuracy compared to the CPU model (cpu.cpp).

The golden results for the CDS spread have been obtained from the DG300 functions excel spreadsheet accompanying the John Hull's "Options, Futures and other Derivatives" book.


Kernel
******

The kernel (*CDS_kernel.cpp*) is configured using the defines specified within the *cds_engine_kernel.hpp* file.

These defines are used to specify the length of the term structure of interest rates (*IRLEN*), term strcuture of hazard rates (*HAZARDLEN*) and the number of CDS spread values (*N*) calculated by the kernel during a single execution.

The kernel returns an array of CDS fair values spread and takes the following list of inputs:

- a term structure of interest rates
- a term structure of hazard rates
- an array of notional values
- an array of recovery rates
- an array of maturities or life of the CDS
- an array of pay off frequencies (either 1, 2, 4 or 12 months)



