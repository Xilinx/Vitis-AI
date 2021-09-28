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

#ifndef __FFT_COM_INC_H__
#define __FFT_COM_INC_H__

// FFT Sizes
#define FFT4096_SIZE 4096
#define FFT3072_SIZE 3072
#define FFT2048_SIZE 2048
#define FFT1536_SIZE 1536
#define FFT1200_SIZE 1200
#define FFT1024_SIZE 1024
#define FFT896_SIZE 896
#define FFT768_SIZE 768
#define FFT640_SIZE 640
#define FFT600_SIZE 600
#define FFT512_SIZE 512
#define FFT384_SIZE 384
#define FFT300_SIZE 300
#define FFT256_SIZE 256
#define FFT128_SIZE 128
#define FFT64_SIZE 64
#define FFT32_SIZE 32
#define FFT16_SIZE 16
#define FFT12_SIZE 12
#define FFT12_X128_SIZE 12 * 128
#define FFT12_X128_16B_SIZE 12 * 128
#define FFT24_SIZE 24
#define FFT36_SIZE 36

// IFFT Sizes
#define IFFT4096_SIZE 4096
#define IFFT3072_SIZE 3072
#define IFFT2048_SIZE 2048
#define IFFT1536_SIZE 1536
#define IFFT1200_SIZE 1200
#define IFFT1024_SIZE 1024
#define IFFT896_SIZE 896
#define IFFT512_SIZE 512
#define IFFT768_SIZE 768
#define IFFT640_SIZE 640
#define IFFT600_SIZE 600
#define IFFT384_SIZE 384
#define IFFT300_SIZE 300
#define IFFT256_SIZE 256
#define IFFT128_SIZE 128
#define IFFT12_SIZE 12
#ifndef IFFT12_BATCH_SIZE
#define IFFT12_BATCH_SIZE 128
#endif
#define IFFT12_BATCH_INPUT_SIZE IFFT12_SIZE* IFFT12_BATCH_SIZE
#define IFFT12_X128_16B_SIZE 12 * IFFT12_BATCH_SIZE
#define IFFT12_32_SIZE 12 * 32
#define IFFT24_SIZE 24
#define IFFT36_SIZE 36

// FFT block sizes
#define FFT_4096 4096
#define FFT_3072 3072
#define FFT_2048 2048
#define FFT_1536 1536
#define FFT_1200 1200
#define FFT_1024 1024
#define FFT_768 768
#define FFT_640 640
#define FFT_600 600
#define FFT_512 512
#define FFT_448 448
#define FFT_384 384
#define FFT_320 320
#define FFT_300 300
#define FFT_256 256
#define FFT_224 224
#define FFT_200 200
#define FFT_192 192
#define FFT_160 160
#define FFT_150 150
#define FFT_128 128
#define FFT_112 112
#define FFT_100 100
#define FFT_96 96
#define FFT_80 80
#define FFT_75 75
#define FFT_64 64
#define FFT_60 60
#define FFT_56 56
#define FFT_48 48
#define FFT_40 40
#define FFT_32 32
#define FFT_56 56
#define FFT_28 28
#define FFT_24 24
#define FFT_20 20
#define FFT_18 18
#define FFT_16 16
#define FFT_15 15
#define FFT_14 14
#define FFT_12 12
#define FFT_10 10
#define FFT_9 9
#define FFT_8 8
#define FFT_6 6
#define FFT_5 5
#define FFT_4 4
#define FFT_3 3
#define FFT_2 2
#define FFT_1 1

// FFT shift values
#define FFT_SHIFT15 15
#define FFT_SHIFT17 17
#define FFT_SHIFT18 18
#define FFT_SHIFT19 19
#define FFT_SHIFT20 20
#define FFT_SHIFT21 21
#define FFT_SHIFT23 23
#define FFT_SHIFT1 15
#define FFT_UPSHIFT1 15

// FFT batch sizes
#define FFT_BATCH_128_SIZE 128

// IFFT block sizes
#define IFFT_4096 4096
#define IFFT_3072 3072
#define IFFT_2048 2048
#define IFFT_1536 1536
#define IFFT_1200 1200
#define IFFT_1024 1024
#define IFFT_768 768
#define IFFT_640 640
#define IFFT_600 600
#define IFFT_512 512
#define IFFT_448 448
#define IFFT_384 384
#define IFFT_320 320
#define IFFT_300 300
#define IFFT_256 256
#define IFFT_224 224
#define IFFT_200 200
#define IFFT_192 192
#define IFFT_160 160
#define IFFT_128 128
#define IFFT_112 112
#define IFFT_100 100
#define IFFT_96 96
#define IFFT_80 80
#define IFFT_64 64
#define IFFT_60 60
#define IFFT_56 56
#define IFFT_48 48
#define IFFT_40 40
#define IFFT_32 32
#define IFFT_28 28
#define IFFT_24 24
#define IFFT_20 20
#define IFFT_16 16
#define IFFT_14 14
#define IFFT_12 12
#define IFFT_10 10
#define IFFT_8 8
#define IFFT_6 6
#define IFFT_5 5
#define IFFT_4 4
#define IFFT_2 2
#define IFFT_1 1

// IFFT shift values
#define IFFT_SHIFT15 15
#define IFFT_SHIFT17 17
#define IFFT_SHIFT20 20
#define IFFT_SHIFT22 22
#define IFFT_SHIFT23 23
#define IFFT_SHIFT24 24
#define IFFT_SHIFT27 27
#define IFFT_SHIFT1 15
#define IFFT_UPSHIFT1 15

// IFFT batch sizes
#define IFFT_BATCH_128_SIZE 128

// FFT/IFFT Final scale factor for ME kernel output
// n can be +ve or -ve, 2^n: shift right by n for +ve n and left shift for -ve n
// Include the user defined parameter file of the user defined flag is set
// These are user tuneable parameters, which can be modified from user_params.h

#ifndef FFT12_FINAL_SHIFT
#define FFT12_FINAL_SHIFT 0
#endif
#ifndef FFT16_FINAL_SHIFT
#define FFT16_FINAL_SHIFT 0
#endif
#ifndef FFT24_FINAL_SHIFT
#define FFT24_FINAL_SHIFT 0
#endif
#ifndef FFT32_FINAL_SHIFT
#define FFT32_FINAL_SHIFT 0
#endif
#ifndef FFT36_FINAL_SHIFT
#define FFT36_FINAL_SHIFT 0
#endif
#ifndef FFT64_FINAL_SHIFT
#define FFT64_FINAL_SHIFT 0
#endif
#ifndef FFT128_FINAL_SHIFT
#define FFT128_FINAL_SHIFT 0
#endif
#ifndef FFT256_FINAL_SHIFT
#define FFT256_FINAL_SHIFT 0
#endif
#ifndef FFT512_FINAL_SHIFT
#define FFT512_FINAL_SHIFT 0
#endif
#ifndef FFT1024_FINAL_SHIFT
#define FFT1024_FINAL_SHIFT 0
#endif
#ifndef FFT2048_FINAL_SHIFT
#define FFT2048_FINAL_SHIFT 0
#endif
#ifndef FFT4096_FINAL_SHIFT
#define FFT4096_FINAL_SHIFT 0
#endif
#ifndef FFT300_FINAL_SHIFT
#define FFT300_FINAL_SHIFT 0
#endif
#ifndef FFT600_FINAL_SHIFT
#define FFT600_FINAL_SHIFT 0
#endif
#ifndef FFT1200_FINAL_SHIFT
#define FFT1200_FINAL_SHIFT 0
#endif
#ifndef FFT384_FINAL_SHIFT
#define FFT384_FINAL_SHIFT 0
#endif
#ifndef FFT640_FINAL_SHIFT
#define FFT640_FINAL_SHIFT 0
#endif
#ifndef FFT896_FINAL_SHIFT
#define FFT896_FINAL_SHIFT 0
#endif
#ifndef FFT1536_FINAL_SHIFT
#define FFT1536_FINAL_SHIFT 0
#endif
#ifndef FFT3072_FINAL_SHIFT
#define FFT3072_FINAL_SHIFT 0
#endif
#ifndef FFT12_FINAL_SHIFT
#define FFT12_FINAL_SHIFT 0
#endif

#ifndef IFFT12_FINAL_SHIFT
#define IFFT12_FINAL_SHIFT 0
#endif
#ifndef IFFT24_FINAL_SHIFT
#define IFFT24_FINAL_SHIFT 0
#endif
#ifndef IFFT36_FINAL_SHIFT
#define IFFT36_FINAL_SHIFT 0
#endif
#ifndef IFFT128_FINAL_SHIFT
#define IFFT128_FINAL_SHIFT 0
#endif
#ifndef IFFT256_FINAL_SHIFT
#define IFFT256_FINAL_SHIFT 0
#endif
#ifndef IFFT512_FINAL_SHIFT
#define IFFT512_FINAL_SHIFT 0
#endif
#ifndef IFFT1024_FINAL_SHIFT
#define IFFT1024_FINAL_SHIFT 0
#endif
#ifndef IFFT2048_FINAL_SHIFT
#define IFFT2048_FINAL_SHIFT 0
#endif
#ifndef IFFT4096_FINAL_SHIFT
#define IFFT4096_FINAL_SHIFT 0
#endif
#ifndef IFFT300_FINAL_SHIFT
#define IFFT300_FINAL_SHIFT 0
#endif
#ifndef IFFT600_FINAL_SHIFT
#define IFFT600_FINAL_SHIFT 0
#endif
#ifndef IFFT1200_FINAL_SHIFT
#define IFFT1200_FINAL_SHIFT 0
#endif
#ifndef IFFT384_FINAL_SHIFT
#define IFFT384_FINAL_SHIFT 0
#endif
#ifndef IFFT640_FINAL_SHIFT
#define IFFT640_FINAL_SHIFT 0
#endif
#ifndef IFFT896_FINAL_SHIFT
#define IFFT896_FINAL_SHIFT 0
#endif
#ifndef IFFT1536_FINAL_SHIFT
#define IFFT1536_FINAL_SHIFT 0
#endif
#ifndef IFFT3072_FINAL_SHIFT
#define IFFT3072_FINAL_SHIFT 0
#endif
#ifndef IFFT12_FINAL_SHIFT
#define IFFT12_FINAL_SHIFT 0
#endif

// Force LLVM frontend to inline the functions (CRVO-1430)
#ifdef __clang__
#define INLINE_DECL inline __attribute__((always_inline))
#else
#define INLINE_DECL inline
#endif

#endif /* __FFT_COM_INC_H__ */
