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
 * @file enums.hpp
 * @brief Header of common enums.
 *
 * This file is part of Vitis Search Library, and should contain only contain
 * definitions of enums and constants, so that it can be included in both host
 * and kenrel code without introducing complex dependency.
 *
 */

#ifndef XF_TEXT_ENUMS_H
#define XF_TEXT_ENUMS_H

namespace xf {
namespace data_analytics {
namespace text {

// This sub-namespace allows enums to be imported to other
// namespace all together.
namespace enums {

/// @brief OP list for regex VM.
enum {
    OP_FINISH = 0,
    OP_END = 1,
    OP_STR_1 = 2,
    OP_STR_2 = 3,
    OP_STR_3 = 4,
    OP_STR_4 = 5,
    OP_STR_5 = 6,
    OP_STR_N = 7,
    OP_STR_MB2N1 = 8,
    OP_STR_MB2N2 = 9,
    OP_STR_MB2N3 = 10,
    OP_STR_MB2N = 11,
    OP_STR_MB3N = 12,
    OP_STR_MBN = 13,
    OP_CCLASS = 14,
    OP_CCLASS_MB = 15,
    OP_CCLASS_MIX = 16,
    OP_CCLASS_NOT = 17,
    OP_CCLASS_MB_NOT = 18,
    OP_CCLASS_MIX_NOT = 19,
    OP_ANYCHAR = 20,
    OP_ANYCHAR_ML = 21,
    OP_ANYCHAR_STAR = 22,
    OP_ANYCHAR_ML_STAR = 23,
    OP_ANYCHAR_STAR_PEEK_NEXT = 24,
    OP_ANYCHAR_ML_STAR_PEEK_NEXT = 25,
    OP_WORD = 26,
    OP_WORD_ASCII = 27,
    OP_NO_WORD = 28,
    OP_NO_WORD_ASCII = 29,
    OP_WORD_BOUNDARY = 30,
    OP_NO_WORD_BOUNDARY = 31,
    OP_WORD_BEGIN = 32,
    OP_WORD_END = 33,
    OP_TEXT_SEGMENT_BOUNDARY = 34,
    OP_BEGIN_BUF = 35,
    OP_END_BUF = 36,
    OP_BEGIN_LINE = 37,
    OP_END_LINE = 38,
    OP_SEMI_END_BUF = 39,
    OP_CHECK_POSITION = 40,
    OP_BACKREF1 = 41,
    OP_BACKREF2 = 42,
    OP_BACKREF_N = 43,
    OP_BACKREF_N_IC = 44,
    OP_BACKREF_MULTI = 45,
    OP_BACKREF_NULTI_IC = 46,
    OP_BACKREF_WITH_LEVEL = 47,
    OP_BACKREF_WITH_LEVEL_IC = 48,
    OP_BACKREF_CHECK = 49,
    OP_BACKREF_CHECK_WITH_LEVEL = 50,
    OP_MEM_START = 51,
    OP_MEM_START_PUSH = 52,
    OP_MEM_END_PUSH = 53,
    OP_MEM_END_PUSH_REC = 54,
    OP_MEM_END = 55,
    OP_MEM_END_REC = 56,
    OP_FAIL = 57,
    OP_JUMP = 58,
    OP_PUSH = 59,
    OP_PUSH_SUPER = 60,
    OP_POP = 61,
    OP_POP_TO_MARK = 62,
    OP_PUSH_OR_JUMP_EXACT1 = 63,
    OP_PUSH_IF_PEEK_NEXT = 64,
    OP_REPEAT = 65,
    OP_REPEAT_NG = 66,
    OP_REPEAT_INC = 67,
    OP_REPEAT_INC_NG = 68,
    OP_EMPTY_CHECK_START = 69,
    OP_EMPTY_CHECK_END = 70,
    OP_EMPTY_CHECK_END_MEMST = 71,
    OP_EMPTY_CHECK_END_MEMST_PUSH = 72,
    OP_MOVE = 73,
    OP_STEP_BACK_START = 74,
    OP_STEP_BACK_NEXT = 75,
    OP_CUT_TO_MARK = 76,
    OP_MARK = 77,
    OP_SAVE_VAL = 78,
    OP_UPDATE_VAR = 79,
    OP_CALL = 80,
    OP_RETURN = 81,
    OP_CALLOUT_CONTENTS = 82,
    OP_CALLOUT_NAME = 83
};

} // namespace enums

using namespace enums;

} // namespace text
} // namespace data_analytics
} // namespace xf

#endif // XF_SEARCH_ENUMS_H
