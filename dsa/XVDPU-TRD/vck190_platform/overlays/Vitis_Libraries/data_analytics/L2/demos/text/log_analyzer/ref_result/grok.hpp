/*
 * Copyright 2020 Xilinx, Inc.
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
#include <string.h>
#include "oniguruma.h"
#include <iostream>

#define N16 65536

int regexTop(int subNum, regex_t* reg, OnigRegion* region, UChar* str, uint32_t* offsets) {
    int r;
    int index = 0;
    unsigned char *start, *range, *end;
    end = str + strlen((char*)str);
    start = str;
    range = end;
    r = onig_match(reg, str, end, start, region, ONIG_OPTION_NONE);
    int i;

    if (r > 0 && (region->num_regs + 1 == subNum)) {
        offsets[index++] = subNum;
        for (i = 0; i < region->num_regs; i++) {
            offsets[index++] = region->beg[i] + region->end[i] * N16;
        }
    }

    return r;
}
