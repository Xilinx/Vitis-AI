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
 * @brief This files includes finite differences swaption engine base G2 model class.
 *
 */

#ifndef XF_FINTECH_TYPES_HPP
#define XF_FINTECH_TYPES_HPP

namespace xf {
namespace fintech {

typedef unsigned int Size;

#define XF_MIN_DOUBLE (2.22507e-308)
#define XF_MAX_DOUBLE (1.79769e+308)
#define XF_MAX_FLOAT (3.40282e+38)
#define XF_MAX_INT (2147483647)
#define XF_EPSILON (2.22045e-16)

} // fintech
} // xf

#endif // XF_FINTECH_TYPES_H
