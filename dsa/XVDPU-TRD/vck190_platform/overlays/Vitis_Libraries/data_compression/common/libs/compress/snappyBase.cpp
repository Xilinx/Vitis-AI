/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#include "snappyBase.hpp"
#include "xxhash.h"

uint8_t snappyBase::writeHeader(uint8_t* out) {
    int fileIdx = 0;

    // Snappy Stream Identifier
    out[fileIdx++] = 0xff;
    out[fileIdx++] = 0x06;
    out[fileIdx++] = 0x00;
    out[fileIdx++] = 0x00;
    out[fileIdx++] = 0x73;
    out[fileIdx++] = 0x4e;
    out[fileIdx++] = 0x61;
    out[fileIdx++] = 0x50;
    out[fileIdx++] = 0x70;
    out[fileIdx++] = 0x59;

    return fileIdx;
}

uint8_t snappyBase::readHeader(uint8_t* in) {
    uint8_t fileIdx = 0;
    uint8_t header_byte = 10;
    fileIdx = header_byte;
    return fileIdx;
}
