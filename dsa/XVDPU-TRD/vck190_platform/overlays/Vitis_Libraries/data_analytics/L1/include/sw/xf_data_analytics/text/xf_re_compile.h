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

#ifndef XF_TEXT_RE_COMPILE_H
#define XF_TEXT_RE_COMPILE_H

#include <stdint.h>

#define XF_UNSUPPORTED_OPCODE (-1000)

/**
 * @brief Data sturcture for each insturction.
 *
 * @param oprand_2 The 3rd operand.
 * @param oprand_1 The 2nd operand.
 * @param oprand_0 The 1st operand.
 * @param mode_len Concatenated mode and length.
 * @param opcode Op code.
 *
 */
typedef struct {
    union {
        struct {
            uint16_t oprand_2;
            uint16_t oprand_1;
            uint16_t oprand_0;
            uint8_t mode_len;
            uint8_t opcode;
        } inst_format;
        uint64_t d;
    };
} xf_instruction;

/**
 * @brief Software compiler for pre-compiling input regular expression.
 *
 * @param pattern Input regular expression.
 * @param bitset Bit set map for each character class.
 * @param instructions Compiled instruction list derived from input pattern.
 * @param instr_num Number of instructions.
 * @param cclass_num Number of character classes.
 * @param cpgp_nm Number of capturing groups.
 * @param cpgp_name_val Buffer of every name of each capturing group.
 * @param cpgp_name_oft Starting offset addresses for the name of each capturing group.
 *
 */
extern int xf_re_compile(const char* pattern,
                         unsigned int* bitset,
                         uint64_t* instructions,
                         unsigned int* instr_num,
                         unsigned int* cclass_num,
                         unsigned int* cpgp_nm,
                         uint8_t* cpgp_name_val,
                         uint32_t* cpgp_name_oft);

#endif // XF_TEXT_RE_COMPILE_H
