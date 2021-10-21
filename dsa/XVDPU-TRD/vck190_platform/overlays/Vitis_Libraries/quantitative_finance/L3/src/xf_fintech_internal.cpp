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

#include <stdio.h>

#include <cstddef>

#include "xf_fintech_internal.hpp"

using namespace xf::fintech;

void xf::fintech::ContextErrorCallback(const char* errInfo, const void* privateInfo, size_t cb, void* userData) {
    printf("[XLNX] ********************************\n");
    printf("[XLNX] CONTEXT ERROR - %s\n", errInfo);
    printf("[XLNX] ********************************\n");
}