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

#define RO 0 // Resource Optimized (8-pixel implementation)
#define NO 1 // Normal Operation (1-pixel implementation)

/* Conversion Type*/

#define XF_CONVERT16UTO8U 0  // set to convert bit depth from unsigned 16-bit to unsigned 8-bit
#define XF_CONVERT16STO8U 1  // set to convert bit depth from signed   16-bit to unsigned 8-bit
#define XF_CONVERT32STO8U 0  // set to convert bit depth from signed   32-bit to unsigned 8-bit
#define XF_CONVERT32STO16U 0 // set to convert bit depth from signed   32-bit to unsigned 16-bit
#define XF_CONVERT32STO16S 0 // set to convert bit depth from signed   32-bit to signed   16-bit
#define XF_CONVERT8UTO16U 0  // set to convert bit depth from unsigned 8-bit  to 16-bit unsigned
#define XF_CONVERT8UTO16S 0  // set to convert bit depth from unsigned 8-bit  to 16-bit signed
#define XF_CONVERT8UTO32S 0  // set to convert bit depth from unsigned 8-bit  to 32-bit unsigned
#define XF_CONVERT16UTO32S 0 // set to convert bit depth from unsigned 16-bit to 32-bit signed
#define XF_CONVERT16STO32S 0 // set to convert bit depth from signed   16-bit to 32-bit signed
