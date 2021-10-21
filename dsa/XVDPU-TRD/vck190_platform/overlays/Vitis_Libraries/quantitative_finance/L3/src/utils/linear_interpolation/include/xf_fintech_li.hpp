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

#ifndef _XF_FINTECH_LI_HPP_
#define _XF_FINTECH_LI_HPP_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// pFxyGrid has X columns and Y rows in contiguous memory
bool Xilinx_Interpolate(double* pFxyGrid,
                        double* pX,
                        double* pY,
                        int Size_X,
                        int Size_Y,
                        double Target_X,
                        double Target_Y,
                        double* pAnswer);

#ifdef __cplusplus
}
#endif

#endif /* _Xilinx__ */
