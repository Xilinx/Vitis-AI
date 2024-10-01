/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once
#include <any>
#include <cassert>
#include <cstring>

#include "./attrs2_primitive_values.hpp"
#include "xir/attrs/attrs_imp.hpp"
#include "xir/cxir.h"

namespace {
template <>
struct convert<xir_attr_value_t, any> {
  static any conv(xir_attr_value_t value);
};

template <>
struct convert<any, xir_attr_value_t> {
  static xir_attr_value_t conv(const any& value);
};

}  // namespace
