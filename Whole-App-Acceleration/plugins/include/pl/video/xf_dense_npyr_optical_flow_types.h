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

#ifndef __XF_DENSE_NONPYR_OPTICAL_FLOW_TYPES__
#define __XF_DENSE_NONPYR_OPTICAL_FLOW_TYPES__

typedef unsigned char pix_t;

template <int BYTES_PER_CYCLE>
struct mywide_t {
    pix_t data[BYTES_PER_CYCLE];
};

typedef struct __yuv { pix_t y, u, v; } yuv_t;

typedef struct __rgb { pix_t r, g, b; } rgb_t;

// kernel returns this type. Packed structs on axi need to be powers-of-2.
typedef struct __rgba {
    pix_t r, g, b;
    pix_t a; // can be unused
} rgba_t;

typedef struct __hsv { pix_t h, s, v; } hsv_t;

#endif
