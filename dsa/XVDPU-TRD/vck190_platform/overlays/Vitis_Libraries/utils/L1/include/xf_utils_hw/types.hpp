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

/**
 * @file types.hpp
 * @brief This file contains type definitions.
 *
 * This file is part of Vitis Utility Library.
 */

#ifndef XF_UTILS_HW_TYPES_H
#define XF_UTILS_HW_TYPES_H

// Fixed width integers
#if __cplusplus >= 201103L
#include <cstdint>
#endif

namespace xf {
namespace common {
namespace utils_hw {
#if __cplusplus >= 201103L

using ::int8_t;
using ::int16_t;
using ::int32_t;
using ::int64_t;

using ::uint8_t;
using ::uint16_t;
using ::uint32_t;
using ::uint64_t;

#else // __cplusplus

typedef signed char int8_t;
typedef short int int16_t;
typedef int int32_t;
// MSVC does not have this macro, but Windows is LLP64.
#if __LP64__ == 1
typedef long int int64_t;
#else
typedef long long int int64_t;
#endif // __LP64__

typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
// MSVC does not have this macro, but Windows is LLP64.
#if __LP64__ == 1
typedef unsigned long int uint64_t;
#else
typedef unsigned long long int uint64_t;
#endif // __LP64__

#endif // __cplusplus

} // namespace utils_hw
} // namespace common
} // namespace xf

#if defined(AP_INT_MAX_W) and AP_INT_MAX_W < 4096
#warning "AP_INT_MAX_W has been defined to be less than 4096"
#endif
#undef AP_INT_MAX_W
#define AP_INT_MAX_W 4096

#include "ap_int.h"
#include "hls_stream.h"

#endif // ifndef XF_UTILS_HW_TYPES_H
