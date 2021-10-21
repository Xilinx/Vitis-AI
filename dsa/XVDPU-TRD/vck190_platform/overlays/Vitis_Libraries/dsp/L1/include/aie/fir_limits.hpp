/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_FIR_LIMITS_HPP_
#define _DSPLIB_FIR_LIMITS_HPP_

/* This file exists to define parameter scope limits of the FIR designs
   and hold static_asserts which assist checking these limits.
*/

// The following maximums are the maximums tested. The function may work for larger values.
#define FIR_LEN_MAX 240
#define FIR_LEN_MIN 4
#define SHIFT_MAX 62
#define SHIFT_MIN 0
#define ROUND_MAX 7
#define ROUND_MIN 0

#endif // _DSPLIB_FIR_LIMITS_HPP_
