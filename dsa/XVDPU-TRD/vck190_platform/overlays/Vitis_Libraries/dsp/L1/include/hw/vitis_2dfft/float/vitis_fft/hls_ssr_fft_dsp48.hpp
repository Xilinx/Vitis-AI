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

// Filename : hls_ssr_fft_dsp48.hpp
#ifndef HLS_SSR_FFT_DSP48_H_
#define HLS_SSR_FFT_DSP48_H_

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 Two constants are defined which define input bit widh for DSP48
 multiplier input bit widths.

 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */
#ifndef _SSR_FFT_USE_DSP58_
//////////////////////////////////////////
#ifndef DSP48_OP1_BIT_WIDTH
#define DSP48_OP1_BIT_WIDTH 18
#endif // DSP48_OP1_BIT_WIDTH

#ifndef DSP48_OP2_BIT_WIDTH
#define DSP48_OP2_BIT_WIDTH 27
#endif // DSP48_OP2_BIT_WIDTH
//////////////////////////////////////////////
#else // Use DSP58
#ifndef DSP48_OP1_BIT_WIDTH
#define DSP48_OP1_BIT_WIDTH 24
#endif // DSP48_OP1_BIT_WIDTH

#ifndef DSP48_OP2_BIT_WIDTH
#define DSP48_OP2_BIT_WIDTH 27
#endif // DSP48_OP2_BIT_WIDTH
#endif

#endif // HLS_SSR_FFT_DSP48_H_
