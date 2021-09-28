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
#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <adf.h>
#include "aie_api/utils.hpp"

// Vector Print functions. These are overloaded according to vector type for ease of use.

//--------------------------------------------------------------------------------------
// int16

// No support for v2int16 or v4int16 as less than the shortest AIE vector size supported

namespace xf {
namespace dsp {
namespace aie {
void vPrintf(const char* preamble, const v8int16 myVec) {
    const int kVLen = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16int16 myVec) {
    const int kVLen = 16;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v32int16 myVec) {
    const int kVLen = 32;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v64int16 myVec) {
    const int kVLen = 64;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

//--------------------------------------------------------------------------------------
// cint16

// No support for v2cint16 as this is less than the shortest AIE vector size supported

void vPrintf(const char* preamble, const v4cint16 myVec) {
    const int kVLen = 4;
    cint16_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem = ext_elem(myVec, i);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v8cint16 myVec) {
    const int kVLen = 8;
    cint16_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem = ext_elem(myVec, i);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16cint16 myVec) {
    const int kVLen = 16;
    const int kLineLim = 8;
    cint16_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem = ext_elem(myVec, i);
        printf("(%d, %d), ", myElem.real, myElem.imag);
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v32cint16 myVec) {
    const int kVLen = 32;
    const int kLineLim = 8;
    cint16_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem = ext_elem(myVec, i);
        printf("(%d, %d), ", myElem.real, myElem.imag);
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

// No support for v64cint16 as this exceeds the largest AIE vector size supported

//--------------------------------------------------------------------------------------float

// No support for v2int32 as this is less than the shortest AIE vector size supported

void vPrintf(const char* preamble, const v4int32 myVec) {
    const int kVLen = 4;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v8int32 myVec) {
    const int kVLen = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16int32 myVec) {
    const int kVLen = 16;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v32int32 myVec) {
    const int kVLen = 32;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%d, ", ext_elem(myVec, i));
        // if ((i % kLineLim) == kLineLim-1) printf("\n");
    }
    printf(")\n");
}

// No support for v64int32 as this exceeds the largest AIE vector size supported

//--------------------------------------------------------------------------------------
// cint32
// Ext_elem is unavailable for cint32 at time of writing, so a cast to int32 is used.
void vPrintf(const char* preamble, const v2cint32 myVec) {
    const int kVLen = 2;
    v4int32 myVCast = as_v4int32(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v4cint32 myVec) {
    const int kVLen = 4;
    v8int32 myVCast = as_v8int32(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v8cint32 myVec) {
    const int kVLen = 8;
    v16int32 myVCast = as_v16int32(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16cint32 myVec) {
    const int kVLen = 16;
    const int kLineLim = 8;
    v32int32 myVCast = as_v32int32(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
        // if ((i % kLineLim) == kLineLim-1) printf("\n");
    }
    printf(")\n");
}

// No support for v32cint32 or v64cint32 as these exceed the largest AIE vector size supported

//--------------------------------------------------------------------------------------
// float

// No support for v2float as this is less than the shortest AIE vector size supported

void vPrintf(const char* preamble, const v4float myVec) {
    const int kVLen = 4;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%f, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v8float myVec) {
    const int kVLen = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%f, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16float myVec) {
    const int kVLen = 16;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%f, ", ext_elem(myVec, i));
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v32float myVec) {
    const int kVLen = 32;
    const int kLineLim = 8;
    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        printf("%f, ", ext_elem(myVec, i));
        // if ((i % kLineLim) == kLineLim-1) printf("\n");
    }
    printf(")\n");
}

// No support for v64float as this exceeds the largest AIE vector size supported

//--------------------------------------------------------------------------------------
// cfloat
// Ext_elem is unavailable for cfloat at time of writing, so a cast to float is used.
void vPrintf(const char* preamble, const v2cfloat myVec) {
    const int kVLen = 2;
    v4float myVCast = as_v4float(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v4cfloat myVec) {
    const int kVLen = 4;
    v8float myVCast = as_v8float(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v8cfloat myVec) {
    const int kVLen = 8;
    v16float myVCast = as_v16float(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
    }
    printf(")\n");
}

void vPrintf(const char* preamble, const v16cfloat myVec) {
    const int kVLen = 16;
    const int kLineLim = 8;
    v32float myVCast = as_v32float(myVec);
    cint32_t myElem;

    printf("%s = (", preamble);
    for (int i = 0; i < kVLen; ++i) {
        myElem.real = ext_elem(myVCast, i * 2);
        myElem.imag = ext_elem(myVCast, i * 2 + 1);
        printf("(%d, %d), ", myElem.real, myElem.imag);
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

template <typename T, unsigned Elems>
void vPrintf(const char* preamble, const ::aie::vector<T, Elems>& myVec) {
    ::aie::print(myVec, true, preamble);
}

// No support for v32cfloat or v64cfloat as these exceed the largest AIE vector size supported

//--------------------------------------------------------------------------------------
// Arrays

void vPrintf(const char* preamble, int16_t* array, size_t len) {
    const int kLineLim = 16;
    printf("%s = (", preamble);
    for (int i = 0; i < len; ++i) {
        printf("%d, ", array[i]);
        if ((i % kLineLim) == kLineLim - 1) printf("\n");
    }
    printf(")\n");
}

void vPrintf(const char* preamble, cint16_t* array, size_t len) {
    const int kLineLim = 16;
    printf("%s = (", preamble);
    for (int i = 0; i < len; ++i) {
        printf("(%d, %d), ", array[i].real, array[i].imag);
    }
    printf(")\n");
}
}
}
}
#endif // DEBUG_UTILS_H
