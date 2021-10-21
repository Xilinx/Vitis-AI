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

#ifndef __FFT_BUFS_H__
#define __FFT_BUFS_H__

#include "fft_com_inc.h"

// Inter-rank temporary storage buffers
cint32_t fft_16_tmp1[FFT16_SIZE];
cint32_t fft_16_tmp2[FFT16_SIZE];
cint32_t fft_32_tmp1[FFT32_SIZE];
cint32_t fft_32_tmp2[FFT32_SIZE];
cint32_t fft_64_tmp1[FFT64_SIZE];
cint32_t fft_64_tmp2[FFT64_SIZE];
cint32_t fft_128_tmp1[FFT128_SIZE];
cint32_t fft_128_tmp2[FFT128_SIZE];
cint32_t fft_256_tmp1[FFT256_SIZE];
cint32_t fft_256_tmp2[FFT256_SIZE];
cint32_t fft_512_tmp1[FFT512_SIZE];
cint32_t fft_512_tmp2[FFT512_SIZE];
cint32_t fft_1024_tmp1[FFT1024_SIZE];
cint32_t fft_1024_tmp2[FFT1024_SIZE];
cint32_t fft_2048_tmp1[FFT2048_SIZE];
cint32_t fft_2048_tmp2[FFT2048_SIZE];
cint32_t fft_4096_tmp1[FFT4096_SIZE];
cint32_t fft_4096_tmp2[FFT4096_SIZE];

// Twiddle tables
// Half-size integer tables
// This is an optimization possible because in a radix4 unit, the second rank butterflies use the same
// twiddle just 90 degrees (minus j) rotated. Minus J rotation is supported by hw, so only the first
// quadrant need be stores - the other quadrant can be extracted by minus j rotation.
cint16_t fft_lut_tw1_half[FFT_1];
cint16_t fft_lut_tw2_half[FFT_2];
cint16_t fft_lut_tw4_half[FFT_4];
cint16_t fft_lut_tw8_half[FFT_8];
cint16_t fft_lut_tw16_half[FFT_16 / 2];
cint16_t fft_lut_tw32_half[FFT_32 / 2];
cint16_t fft_lut_tw64_half[FFT_64 / 2];
cint16_t fft_lut_tw128_half[FFT_128 / 2];
cint16_t fft_lut_tw256_half[FFT_256 / 2];
cint16_t fft_lut_tw512_half[FFT_512 / 2];
cint16_t fft_lut_tw1024_half[FFT_1024 / 2];
cint16_t fft_lut_tw2048_half[FFT_2048 / 2];

// Full (2 quadrant) integer tables
cint16_t fft_lut_tw1[FFT_1];
cint16_t fft_lut_tw2[FFT_2];
cint16_t fft_lut_tw4[FFT_4];
cint16_t fft_lut_tw8[FFT_8];
cint16_t fft_lut_tw16[FFT_16];
cint16_t fft_lut_tw32[FFT_32];
cint16_t fft_lut_tw64[FFT_64];
cint16_t fft_lut_tw128[FFT_128];
cint16_t fft_lut_tw256[FFT_256];
cint16_t fft_lut_tw512[FFT_512];
cint16_t fft_lut_tw1024[FFT_1024];
cint16_t fft_lut_tw2048[FFT_2048];

// Full (2 quadrant) float tables.
// Float cannot use the one quadrant trick because float cannot use radix4 functions.
// Why? The result of a butterfly for ints is an acc register, but in float it is a float reg.
// This means that the acc registers are unavailable to store data in float and this means
// there is not the capacity in registers required for the storage of inter-rank values in a radix 4
// stage, hence float uses radix2.
cint16_t fft_lut_tw1_cfloat[FFT_1];
cint16_t fft_lut_tw2_cfloat[FFT_2];
cint16_t fft_lut_tw4_cfloat[FFT_4];
cint16_t fft_lut_tw8_cfloat[FFT_8];
cint16_t fft_lut_tw16_cfloat[FFT_16];
cint16_t fft_lut_tw32_cfloat[FFT_32];
cint16_t fft_lut_tw64_cfloat[FFT_64];
cint16_t fft_lut_tw128_cfloat[FFT_128];
cint16_t fft_lut_tw256_cfloat[FFT_256];
cint16_t fft_lut_tw512_cfloat[FFT_512];
cint16_t fft_lut_tw1024_cfloat[FFT_1024];
cint16_t fft_lut_tw2048_cfloat[FFT_2048];

#endif /* __FFT_BUFS_H__ */
