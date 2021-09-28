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
 * @file regexVM.hpp
 * @brief Implementation for regular expression processor based on VM.
 *
 * This regexVM return a boolean indicating whether the input string is matched or not,
 * and a buffer to provide the offset position of the captured groups in the string.
 *
 */

#ifndef XF_TEXT_REGEX_VM_H
#define XF_TEXT_REGEX_VM_H

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_data_analytics/text/enums.hpp"

namespace xf {
namespace data_analytics {
namespace text {
namespace internal {

/**
 * @brief Pop one element out of stack.
 *
 * @param stkPtr Pointer for last item in stack.
 *
 */
static inline void popStack(unsigned int& stkPtr) {
#pragma HLS inline
    stkPtr--;
} // end popStack

/**
 * @brief Peeking stack data for optimization.
 *
 * @param stackBuff Internal stack buffer.
 * @param stkPtr Pointer for last item in stack.
 * @param dt Collection of stack type, memory id, memory start offset, memory end offset, op address, and string
 * pointer.
 *
 */
static inline void peekStack(ap_uint<54>* stackBuff, unsigned int stkPtr, ap_uint<54>& dt) {
#pragma HLS inline
    dt = stackBuff[stkPtr - 1];
} // end peekStack

/**
 * @brief Stack data splitter.
 *
 * @param dt Collection of stack type, memory id, memory start offset, memory end offset, op address, and string
 * pointer.
 * @param stkType Type of current item to be pushed into stack.
 * @param memID Current group index.
 * @param memStart Start offset address of current capturing group.
 * @param memEnd End offset address of current capturing group.
 * @param opAddr Instruction address.
 * @param strPtr Pointer for current character in input string.
 *
 */
static inline void split(ap_uint<54> dt,
                         ap_uint<6>& stkType,
                         ap_uint<16>& memID,
                         ap_uint<16>& memStart,
                         ap_uint<16>& memEnd,
                         ap_uint<16>& opAddr,
                         ap_uint<16>& strPtr) {
    if (dt.range(53, 48) == 0) {
        stkType = 0;
        opAddr = dt.range(31, 16);
        strPtr = dt.range(15, 0);
    } else if (dt.range(53, 48) == 1) {
        stkType = 1;
        memID = dt.range(47, 32);
        memEnd = dt.range(31, 16);
        memStart = dt.range(15, 0);
    } else if (dt.range(53, 48) == 2) {
        stkType = 2;
        memID = dt.range(47, 32);
        strPtr = dt.range(15, 0);
    } else if (dt.range(53, 48) == 3) {
        stkType = 3;
        memID = dt.range(47, 32);
        opAddr = dt.range(31, 16);
        strPtr = dt.range(15, 0);
    }
} // end split

/**
 * @brief Push one element into stack.
 *
 * @param stackBuff Internal stack buffer.
 * @param stkPtr Pointer for last item in stack.
 * @param dt Collection of data to be pushed into stack.
 *
 */
static inline void pushStack(ap_uint<54>* stackBuff, unsigned int& stkPtr, ap_uint<54> dt) {
#pragma HLS inline
    stackBuff[stkPtr++] = dt;
} // end pushStack

/**
 * @brief Push one element into stack without increasing the stack pointer (for optimized regexVM only).
 *
 * @param stackBuff Internal stack buffer.
 * @param stkPtr Pointer for last item in stack.
 * @param dt Collection of data to be pushed into stack.
 *
 */
static inline void pushStack_opt(ap_uint<54>* stackBuff, unsigned int stkPtr, ap_uint<54> dt) {
#pragma HLS inline
    stackBuff[stkPtr] = dt;
} // end pushStack_opt

/**
 * @brief Stack data combiner.
 *
 * @param dt Collection of stack type, memory id, memory start offset, memory end offset, op address, and string
 * pointer.
 * @param stkType Type of current item to be pushed into stack.
 * @param memID Current group index.
 * @param memStart Start offset address of current capturing group.
 * @param memEnd End offset address of current capturing group.
 * @param opAddr Instruction address.
 * @param strPtr Pointer for current character in input string.
 *
 */
static inline void combine(ap_uint<54>& dt,
                           ap_uint<6> stkType,
                           ap_uint<16> memID,
                           ap_uint<16> memStart,
                           ap_uint<16> memEnd,
                           ap_uint<16> opAddr,
                           ap_uint<16> strPtr) {
    // push 0, opaddr, str_ptr
    if (stkType == 0) {
        dt = (ap_uint<54>)(((ap_uint<32>)opAddr << 16) | strPtr) & 0xffffffff;
        // push mem-id, mem-start, mem-end
    } else if (stkType == 1) {
        dt = ((ap_uint<54>)1 << 48) | ((ap_uint<48>)memID << 32) | ((ap_uint<32>)memEnd << 16) | memStart;
        // push mark-id, 0, str_ptr
    } else if (stkType == 2) {
        dt = ((ap_uint<54>)2 << 48) | ((ap_uint<48>)memID << 32) | ((ap_uint<32>)0 << 16) | strPtr;
        // push remaining step, opaddr, str_ptr
    } else if (stkType == 3) {
        dt = ((ap_uint<54>)3 << 48) | ((ap_uint<48>)memID << 32) | ((ap_uint<32>)opAddr << 16) | strPtr;
    }
} // end combine

} // namespace internal

/**
 * @brief Implementation for regular expression VM (1 instruction per iteration).
 *
 * @tparam STACK_SIZE Size of internal stack.
 *
 * @param bitSetBuff Bit set map for character class.
 * @param instrBuff Instruction buffer.
 * @param msgBuff Message buffer as input string.
 * @param lenMsg Length of input string.
 * @param match Flag to indicate whether the input string is matched or not, 0 for mismatch, 1 for match, 2 for stack
 * overflow.
 * @param offsetBuff Offset address for each capturing group.
 *
 */
template <int STACK_SIZE>
void regexVM(ap_uint<32>* bitSetBuff,
             ap_uint<64>* instrBuff,
             ap_uint<32>* msgBuff,
             unsigned int lenMsg,
             ap_uint<2>& match,
             ap_uint<16>* offsetBuff) {
    // op address
    ap_uint<16> opaddr = 0;
    // internal stack
    ap_uint<54> stack_buff[STACK_SIZE];
#pragma HLS bind_storage variable = stack_buff type = ram_2p impl = uram
    // stack pointer
    unsigned int stk = 0;
    // stack type
    ap_uint<6> stk_type = 0;
    // input string pointer
    ap_uint<16> str_ptr = 0;
    // current sub-string (32-bit / 4 characters)
    ap_uint<32> cur_str = 0;
    // string start position
    ap_uint<16> str_start = 0;
    // initialize end flag
    bool end_flag = false;
    // initialize the match flag
    bool match_s = false;
    // initialize stack overflow flag
    bool stk_ovf = false;

    // counter for OP_REPEAT & OP_REPEAT_INC
    ap_uint<16> repeat_cnt = 0;

    // push finish code to stack

    // intermediate registers to overcome RAW dependency
    ap_uint<16> mem_adr_reg_1 = 0;
    ap_uint<16> mem_adr_reg_2 = 0;
    ap_uint<16> mem_dt_reg_1 = 0;
    ap_uint<16> mem_dt_reg_2 = 0;

    ap_uint<32> stk_adr_reg = 0;
    ap_uint<54> stk_dt_reg = 0;

    ap_uint<54> stk_dt_n = 0;
    internal::combine(stk_dt_n, stk_type, 0, 0, 0, 0,
                      0); // stk_type, push_mem_id, mem_start, mem_end, push_addr, push_str_ptr);
    stk_adr_reg = stk + 1;
    stk_dt_reg = stk_dt_n;
    internal::pushStack(stack_buff, stk, stk_dt_n);
// VM for processing particular Regex
#ifndef __SYNTHESIS__
    int exec_tm = 0;
#endif
LOOP_VM:
    while (!end_flag) {
#pragma HLS pipeline II = 2
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS dependence variable = offsetBuff inter false
#pragma HLS dependence variable = stack_buff inter false
#ifndef __SYNTHESIS__
        exec_tm++;
#endif
        // decode instruction
        ap_uint<64> instruction = instrBuff[opaddr];
        ap_uint<8> opcode = instruction.range(63, 56);
        ap_uint<4> mode = instruction.range(55, 52);
        ap_uint<4> len = instruction.range(51, 48);
        ap_uint<16> operand1 = instruction.range(47, 32);
        ap_uint<16> operand2 = instruction.range(31, 16);
        ap_uint<16> operand3 = instruction.range(15, 0);
        // predict the next address of instruction
        ap_uint<16> predict_op_addr = opaddr + 1;
        ap_uint<16> predict_jump_addr = operand1;
        // position of current character in current sub-string
        ap_uint<2> cur_pos = str_ptr.range(1, 0);
        // previous sub-string (32-bit / 4 characters)
        ap_uint<32> prev_str = 0;
        // next sub-string (32-bit / 4 characters)
        ap_uint<32> next_str = msgBuff[(str_ptr >> 2) + 1];

        if ((str_ptr % 4) == 0 && str_ptr > 0) {
            prev_str = cur_str;
        }
        // fetch current sub-string
        cur_str = msgBuff[str_ptr >> 2];
        // read comparing string
        ap_uint<32> cmp_str_1 = instruction.range(47, 16);
        // prepare string to be compared from message
        ap_uint<32> cmp_str_2 = 0;
        ap_uint<3> len_0 = 4 - cur_pos;
        ap_uint<3> len_1 = 4 - len_0;
        // concatenate the actual sub-string (32-bit) for matching
        cmp_str_2.range(8 * len_0 - 1, 0) = cur_str.range((cur_pos + len_0) * 8 - 1, cur_pos * 8);
        if (len_0 != 4) cmp_str_2.range(31, 8 * len_0) = next_str.range(len_1 * 8 - 1, 0);

        ap_uint<4> cmp_flag = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            // characters in instruction is aligned with MSB, where the one in message buffer is aligned with LSB
            // natually
            cmp_flag[k] = (cmp_str_1.range((4 - k) * 8 - 1, (3 - k) * 8) == cmp_str_2.range((k + 1) * 8 - 1, k * 8));
        }
        ap_uint<3> predict_match_len = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            if (k >= len || !cmp_flag[k])
                break;
            else
                predict_match_len++;
        }

        // read current character
        ap_uint<8> cur_char = cur_str.range((cur_pos + 1) * 8 - 1, cur_pos * 8);
        ap_uint<8> prev_char = 0;
        // read previous character
        if (str_ptr > 0) {
            if ((str_ptr % 4) == 0) {
                prev_char = prev_str.range(31, 24);
            } else {
                prev_char = cur_str.range(cur_pos * 8 - 1, (cur_pos - 1) * 8);
            }
        }
        // pre-fetch bitset buffer
        ap_uint<10> bit_addr = ((unsigned int)cur_char >> 5) + (unsigned int)operand1;
        bool predict_cc_cmp = ((bitSetBuff[bit_addr] & (1u << ((unsigned int)cur_char & 0x1f))) != 0);

        // pre-fetch start/end offset address
        ap_uint<16> mem_start;
        ap_uint<16> mem_end;
        // to avoid RAW dependency issue
        if (operand1 == mem_adr_reg_1)
            mem_start = mem_dt_reg_1;
        else if (operand1 == mem_adr_reg_2)
            mem_start = mem_dt_reg_2;
        else
            mem_start = offsetBuff[operand1];

        if (operand1 + 1 == mem_adr_reg_1)
            mem_end = mem_dt_reg_1;
        else if (operand1 + 1 == mem_adr_reg_2)
            mem_end = mem_dt_reg_2;
        else
            mem_end = offsetBuff[operand1 + 1];

        // peek the stack in advance
        ap_uint<6> peek_stk_type = 0;
        ap_uint<16> peek_mem_id = 0;
        ap_uint<16> peek_mem_start = 0;
        ap_uint<16> peek_mem_end = 0;
        ap_uint<16> peek_opaddr = 0;
        ap_uint<16> peek_str_ptr = 0;

        // printf("stk = %d\n", (unsigned int)stk);
        // assert(stk >= 0);
        // assert(stk < STACK_SIZE);
        // if(stk == 0) {
        //    printf("debug point\n");
        //}
        ap_uint<54> stk_dt = 0;
        if (stk == stk_adr_reg)
            stk_dt = stk_dt_reg;
        else
            internal::peekStack(stack_buff, stk, stk_dt);

        internal::split(stk_dt, peek_stk_type, peek_mem_id, peek_mem_start, peek_mem_end, peek_opaddr, peek_str_ptr);

        // flag for indicating where next op goes
        ap_uint<2> nt_flag = 0;
        // stack control signal
        bool pop_stk = false;
        bool push_stk = false;
        // flag to indicate if we are in OP_POP_TO_MARK
        bool in_pop_to_mark = false;
        // input string pointer for stack push
        ap_uint<16> push_str_ptr = 0;
        // mark id or mem id for stack push
        ap_uint<16> push_mem_id = 0;
        // address register for stack push
        ap_uint<16> push_addr = 0;

        // printf("opcode = %d\n", (unsigned int)opcode);
        switch (opcode) {
            case (OP_END): {
                match_s = true;
                end_flag = true;
                break;
            }
            case (OP_STR_N): {
                if (str_ptr + len > lenMsg) {
                    pop_stk = true;
                } else if (predict_match_len == len) {
                    str_ptr += predict_match_len;
                    nt_flag = 1;
                } else {
                    pop_stk = true;
                }
                break;
            }
            // OP_CCLASS_NOT is merged into OP_CCLASS in advance in compiler
            case (OP_CCLASS): {
                if (str_ptr + 1 > lenMsg) {
                    pop_stk = true;
                } else {
                    if (!predict_cc_cmp) {
                        pop_stk = true;
                    } else {
                        str_ptr++;
                        nt_flag = 1;
                    }
                }
                break;
            }
            // begin-buf is contained in begin-line
            case (OP_BEGIN_BUF):
            case (OP_BEGIN_LINE): {
                if (str_ptr == 0) {
                    nt_flag = 1;
                } else if (str_ptr != lenMsg && prev_char == 0x0a && opcode == OP_BEGIN_LINE) {
                    nt_flag = 1;
                } else {
                    pop_stk = true;
                }
                break;
            }
            // end-buf is contained in end-line
            case (OP_END_BUF):
            case (OP_END_LINE): {
                if (str_ptr == lenMsg) {
                    nt_flag = 1;
                } else if (str_ptr < lenMsg && cur_char == 0x0a && opcode == OP_END_LINE) {
                    nt_flag = 1;
                } else {
                    pop_stk = true;
                }
                break;
            }
            // mem-start is contained in mem-start-push
            case (OP_MEM_START):
            case (OP_MEM_START_PUSH): {
                mem_adr_reg_1 = operand1;
                mem_dt_reg_1 = str_ptr;
                offsetBuff[operand1] = str_ptr;
                nt_flag = 1;
                if (opcode == OP_MEM_START_PUSH) {
                    push_stk = true;
                    stk_type = 1;
                    push_mem_id = operand1;
                }
                break;
            }
            case (OP_MEM_END): {
                mem_adr_reg_1 = operand1;
                mem_dt_reg_1 = str_ptr;
                offsetBuff[operand1] = str_ptr;
                nt_flag = 1;
                break;
            }
            case (OP_JUMP): {
                nt_flag = 2;
                break;
            }
            case (OP_PUSH): {
                push_stk = true;
                stk_type = 0;
                push_str_ptr = str_ptr;
                push_addr = predict_jump_addr;
                nt_flag = 1;
                break;
            }
            case (OP_POP): {
                internal::popStack(stk);
                nt_flag = 1;
                break;
            }
            case (OP_PUSH_OR_JUMP_EXACT1): {
                if (str_ptr < lenMsg && cur_char == operand2.range(15, 8)) {
                    push_stk = true;
                    stk_type = 0;
                    push_str_ptr = str_ptr;
                    push_addr = predict_jump_addr;
                    nt_flag = 1;
                } else {
                    nt_flag = 2;
                }
                break;
            }
            // repeat is the starting point of a repetition operation
            case (OP_REPEAT):
            case (OP_REPEAT_INC): {
                if (opcode == OP_REPEAT)
                    repeat_cnt = 0;
                else
                    repeat_cnt++;
                // push stack
                push_str_ptr = str_ptr;
                push_addr = predict_op_addr;
                stk_type = 0;
                // when reach the lower bound, push to stack
                if ((opcode == OP_REPEAT && operand2 == 0) || (repeat_cnt >= operand2 && repeat_cnt < operand3)) {
                    push_stk = true;
                    nt_flag = 2;
                    // if not reach the lower bound, jump
                } else if (opcode == OP_REPEAT_INC && repeat_cnt < operand2) {
                    nt_flag = 2;
                } else {
                    nt_flag = 1;
                }
                break;
            }
            case (OP_ANYCHAR): {
                if ((str_ptr < lenMsg && cur_char == 0x0a) || str_ptr + 1 > lenMsg) {
                    pop_stk = true;
                } else {
                    str_ptr++;
                    nt_flag = 1;
                }
                break;
            }
            case (OP_ANYCHAR_STAR): {
                push_addr = predict_op_addr;
                stk_type = 0;
                push_str_ptr = str_ptr;
                // two conditions to goto fail
                if ((str_ptr < lenMsg && cur_char == 0x0a) || str_ptr + 1 > lenMsg) {
                    // if mismatch, stop and jump to next instruciton
                    nt_flag = 1;
                } else {
                    // keep execute the current instruction and push to stack until mismatch.
                    push_stk = true;
                    // increase the string pointer if not failed
                    str_ptr++;
                }
                break;
            }
            // push mark-id to internal stack
            case (OP_MARK): {
                push_stk = true;
                stk_type = 2;
                push_mem_id = operand1;
                nt_flag = 1;
                break;
            }
            // step back for back-tracking
            case (OP_STEP_BACK_START): {
                ap_int<16> signed_str_ptr = (ap_int<16>)str_ptr;
                signed_str_ptr -= operand2;
                if (signed_str_ptr < 0) {
                    pop_stk = true;
                } else {
                    str_ptr -= operand2;
                    if (operand3 != 0) {
                        push_stk = true;
                        stk_type = 0;
                        push_addr = predict_op_addr;
                        push_str_ptr = str_ptr;
                        nt_flag = 2;
                    } else {
                        nt_flag = 1;
                    }
                }
                break;
            }
            // fail section just pop the stack
            case (OP_FAIL): {
                pop_stk = true;
                break;
            }
            // pop the internal stack until mark-id found and matched
            case (OP_POP_TO_MARK): {
                // when reach the mark_id, stop execute this instruction and go to next instruction
                pop_stk = true;
                if (peek_stk_type == 2 && peek_mem_id == operand1) {
                    nt_flag = 1;
                    in_pop_to_mark = false;
                    // if not, keep execute this instruction and pop the stack
                } else {
                    in_pop_to_mark = true;
                }
                break;
            }
            default: {
                // the rest OPs are not supported currently
                break;
            }
        } // end switch

        // do real push or pop operation on stack
        if (pop_stk) {
            // finish code popped
            if (stk == 1) {
                end_flag = true;
            }
            if (peek_stk_type == 0 && !in_pop_to_mark) {
                nt_flag = 3;
                str_ptr = peek_str_ptr;
            } else if (peek_stk_type == 1) {
                mem_dt_reg_1 = peek_mem_start;
                mem_dt_reg_2 = peek_mem_end;
                mem_adr_reg_1 = peek_mem_id;
                mem_adr_reg_2 = peek_mem_id + 1;
                offsetBuff[peek_mem_id] = peek_mem_start;
                offsetBuff[peek_mem_id + 1] = peek_mem_end;
            }
            internal::popStack(stk);
        } else if (push_stk) {
            ap_uint<54> stk_dt_n = 0;
            internal::combine(stk_dt_n, stk_type, push_mem_id, mem_start, mem_end, push_addr, push_str_ptr);
            // printf("stk_type = %d, push_mem_id = %d, mem_start = %d, mem_end = %d, push_addr = %d, push_str_ptr =
            // %d\n",
            //(unsigned int)stk_type, (unsigned int)push_mem_id, (unsigned int)mem_start, (unsigned int)mem_end,
            //(unsigned int)push_addr, (unsigned int)push_str_ptr);
            if (stk == STACK_SIZE - 1) {
                end_flag = true;
                stk_ovf = true;
            }
            stk_adr_reg = stk + 1;
            stk_dt_reg = stk_dt_n;
            internal::pushStack(stack_buff, stk, stk_dt_n);
        }

        // switch to the op pointed by next flag
        if (nt_flag == 1) {
            opaddr = predict_op_addr;
        } else if (nt_flag == 2) {
            opaddr = predict_jump_addr;
        } else if (nt_flag == 3) {
            opaddr = peek_opaddr;
        }
        // printf("nt_flag = %d, opaddr = %d, str_ptr = %d\n", (unsigned int)nt_flag, (unsigned int)opaddr, (unsigned
        // int)str_ptr);

    } // end while
#ifndef __SYNTHESIS__
    printf("total exec time = %d\n", exec_tm);
#endif

    // if string matched, mark the whole sequence
    if (match_s) {
        match = 1;
        offsetBuff[0] = str_start;
        offsetBuff[1] = str_ptr;
    } else {
        // stack overflow
        if (stk_ovf) match = 2;
        // mismatch
        else
            match = 0;
    }
} // end regexVM

/**
 * @brief Implementation for regular expression VM (2 instructions per iteration).
 *
 * @tparam STACK_SIZE Size of internal stack.
 *
 * @param bitSetBuff Bit set map for cclass.
 * @param instrBuff Instruction buffer.
 * @param msgBuff Message buffer as input string.
 * @param lenMsg Length of input string.
 * @param match Flag to indicate whether the input string is matched or not, 0 for mismatch, 1 for match, 2 for stack
 * overflow.
 * @param offsetBuff Offset address for each capturing group.
 *
 */
template <int STACK_SIZE>
void regexVM_opt(ap_uint<32>* bitSetBuff,
                 ap_uint<64>* instrBuff,
                 ap_uint<32>* msgBuff,
                 unsigned int lenMsg,
                 ap_uint<2>& match,
                 ap_uint<16>* offsetBuff) {
    // op address
    ap_uint<16> opaddr = 0;
    ap_uint<16> n_opaddr = 1;
    // internal stack
    ap_uint<54> stack_buff[STACK_SIZE];
#pragma HLS bind_storage variable = stack_buff type = ram_2p impl = uram
    // stack pointer
    unsigned int stk = 0;
    // stack type
    ap_uint<6> stk_type = 0;
    // input string pointer
    ap_uint<16> str_ptr = 0;
    // current sub-string (64-bit / 8 characters)
    ap_uint<32> prev_str = 0;
    // string start position
    ap_uint<16> str_start = 0;
    // initialize end flag
    bool end_flag = false;
    // initialize the match flag
    bool match_s = false;
    // initialize the stack overflow flag
    bool stk_ovf = false;

    // counter for OP_REPEAT & OP_REPEAT_INC
    ap_uint<16> repeat_cnt = 0;

    // register to fix RAW dependency
    ap_uint<16> mem_adr_reg_1 = 0;
    ap_uint<16> mem_adr_reg_2 = 0;
    ap_uint<16> mem_dt_reg_1 = 0;
    ap_uint<16> mem_dt_reg_2 = 0;

    ap_uint<32> stk_adr_reg = 0;
    ap_uint<54> stk_dt_reg = 0;

    // internal::pushStack(stack_buff, stk, stk_type, 0, 0, 0, 0, 0);
    ap_uint<54> stk_dt_n = 0;
    internal::combine(stk_dt_n, stk_type, 0, 0, 0, 0,
                      0); // stk_type, push_mem_id, mem_start, mem_end, push_addr, push_str_ptr);
    stk_adr_reg = stk + 1;
    stk_dt_reg = stk_dt_n;
    internal::pushStack(stack_buff, stk, stk_dt_n);
// VM for processing particular Regex
#ifndef __SYNTHESIS__
    int exec_tm = 0;
#endif
LOOP_VM:
    while (!end_flag) {
#pragma HLS pipeline II = 3
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS dependence variable = offsetBuff inter false
#pragma HLS dependence variable = stack_buff inter false
#ifndef __SYNTHESIS__
        exec_tm++;
        assert(opaddr + 1 == n_opaddr);
#endif
        // decode instruction
        ap_uint<64> instruction_1 = instrBuff[opaddr];
        ap_uint<8> opcode_1 = instruction_1.range(63, 56);
        ap_uint<4> mode_1 = instruction_1.range(55, 52);
        ap_uint<4> len_1 = instruction_1.range(51, 48);
        ap_uint<16> operand1_1 = instruction_1.range(47, 32);
        ap_uint<16> operand2_1 = instruction_1.range(31, 16);
        ap_uint<16> operand3_1 = instruction_1.range(15, 0);
        // decode the next instruction
        ap_uint<64> instruction_2 = instrBuff[n_opaddr];
        ap_uint<8> opcode_2 = instruction_2.range(63, 56);
        ap_uint<4> mode_2 = instruction_2.range(55, 52);
        ap_uint<4> len_2 = instruction_2.range(51, 48);
        ap_uint<16> operand1_2 = instruction_2.range(47, 32);
        ap_uint<16> operand2_2 = instruction_2.range(31, 16);
        ap_uint<16> operand3_2 = instruction_2.range(15, 0);
        // predict the next address of instruction
        ap_uint<16> predict_op_addr_1 = n_opaddr;
        ap_uint<16> predict_jump_addr_1 = operand1_1;
        ap_uint<16> n_predict_op_addr_1 = n_opaddr + 1;
        ap_uint<16> n_predict_jump_addr_1 = operand1_1 + 1;

        ap_uint<16> predict_op_addr_2 = n_opaddr + 1;
        ap_uint<16> predict_jump_addr_2 = operand1_2;

        ap_uint<16> n_predict_op_addr_2 = n_opaddr + 2;
        ap_uint<16> n_predict_jump_addr_2 = operand1_2 + 1;

        // position of current character in current sub-string
        ap_uint<2> cur_pos = str_ptr.range(1, 0);
        // next sub-string (32-bit / 4 characters)
        ap_uint<32> next_str = msgBuff[(str_ptr >> 2) + 1];

        // fetch previous character before updating previous string
        ap_uint<8> prev_char = 0x0a;
        if (str_ptr > 0) {
            ap_uint<2> prev_pos = (str_ptr - 1) % 4;
            prev_char = prev_str.range((prev_pos + 1) * 8 - 1, prev_pos * 8);
        }
        // fetch current sub-string
        ap_uint<32> cur_str = msgBuff[str_ptr >> 2];
        // previous sub-string (32-bit / 4 characters)
        prev_str = cur_str;

        // pre-exec the comparator of string
        ap_uint<32> cmp_str_1_1 = instruction_1.range(47, 16);
        ap_uint<32> cmp_str_2_1 = instruction_2.range(47, 16);
        ap_uint<32> cmp_str_2 = 0;

        ap_uint<3> len_0_t = 4 - cur_pos;
        ap_uint<3> len_1_t = 4 - len_0_t;
        // concatenate the actual sub-string (32-bit) for matching
        cmp_str_2.range(8 * len_0_t - 1, 0) = cur_str.range((cur_pos + len_0_t) * 8 - 1, cur_pos * 8);
        if (len_0_t != 4) cmp_str_2.range(31, 8 * len_0_t) = next_str.range(len_1_t * 8 - 1, 0);

        ap_uint<4> cmp_flag_1 = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            // characters in instruction is aligned with MSB, where the one in message buffer is aligned with LSB
            // natually
            cmp_flag_1[k] =
                (cmp_str_1_1.range((4 - k) * 8 - 1, (3 - k) * 8) == cmp_str_2.range((k + 1) * 8 - 1, k * 8));
        }
        ap_uint<3> predict_match_len_1 = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            if (!cmp_flag_1[k])
                break;
            else
                predict_match_len_1++;
        }

        ap_uint<4> cmp_flag_2 = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            // characters in instruction is aligned with MSB, where the one in message buffer is aligned with LSB
            // natually
            cmp_flag_2[k] =
                (cmp_str_2_1.range((4 - k) * 8 - 1, (3 - k) * 8) == cmp_str_2.range((k + 1) * 8 - 1, k * 8));
        }
        ap_uint<3> predict_match_len_2 = 0;
        for (int k = 0; k < 4; ++k) {
#pragma HLS unroll
            if (!cmp_flag_2[k])
                break;
            else
                predict_match_len_2++;
        }

        //// read current character
        ap_uint<8> cur_char = cur_str.range((cur_pos + 1) * 8 - 1, cur_pos * 8);
        // pre-fetch bitset buffer
        ap_uint<10> bit_addr_1 = ((unsigned int)cur_char >> 5) + (unsigned int)operand1_1;
        bool predict_cc_cmp_1 = ((bitSetBuff[bit_addr_1] & (1u << ((unsigned int)cur_char & 0x1f))) == 0);

        ap_uint<10> bit_addr_2 = ((unsigned int)cur_char >> 5) + (unsigned int)operand1_2;
        bool predict_cc_cmp_2 = ((bitSetBuff[bit_addr_2] & (1u << ((unsigned int)cur_char & 0x1f))) == 0);

        // pre-fetch start/end offset address
        ap_uint<16> mem_start; // = offsetBuff[operand1];
        ap_uint<16> mem_end;   // = offsetBuff[operand1 + 1];

        // to avoid dependency issue, RAW
        if (operand1_1 == mem_adr_reg_1)
            mem_start = mem_dt_reg_1;
        else if (operand1_1 == mem_adr_reg_2)
            mem_start = mem_dt_reg_2;
        else
            mem_start = offsetBuff[operand1_1];

        if (operand1_1 + 1 == mem_adr_reg_1)
            mem_end = mem_dt_reg_1;
        else if (operand1_1 + 1 == mem_adr_reg_2)
            mem_end = mem_dt_reg_2;
        else
            mem_end = offsetBuff[operand1_1 + 1];

        // peek the stack in advance
        ap_uint<6> peek_stk_type = 0;
        ap_uint<16> peek_mem_id = 0;
        ap_uint<16> peek_mem_start = 0;
        ap_uint<16> peek_mem_end = 0;
        ap_uint<16> peek_opaddr = 0;
        ap_uint<16> peek_str_ptr = 0;

        // printf("stk = %d\n", (unsigned int)stk);
        // assert(stk >= 0);
        // assert(stk < STACK_SIZE);
        // if(stk == 0) {
        //    printf("debug point\n");
        //}
        ap_uint<54> stk_dt = 0;
        if (stk == stk_adr_reg)
            stk_dt = stk_dt_reg;
        else
            internal::peekStack(stack_buff, stk, stk_dt);

        internal::split(stk_dt, peek_stk_type, peek_mem_id, peek_mem_start, peek_mem_end, peek_opaddr, peek_str_ptr);

        ap_uint<16> n_peek_opaddr = peek_opaddr + 1;
        //-----------------------------------execute the current instruction-----------------------------//
        // local control signal
        ap_uint<2> nt_flag_1 = 0;
        // control signal
        bool pop_stk_1 = false;
        bool push_stk_1 = false;
        // address register for stack push
        ap_uint<16> push_addr_1 = 0;
        // push the current status
        ap_uint<16> str_ptr_1 = str_ptr;
        bool match_1 = false;
        bool end_flag_1 = false;
        unsigned int stk_1 = stk;
        unsigned int stk_a1 = stk + 1;
        unsigned int stk_m1 = stk - 1;

        ap_uint<6> stk_type_1 = 0;

        ap_uint<16> repeat_cnt_1 = repeat_cnt;
        ap_uint<16> push_mem_id_1 = 0;

        // flag to indicate if we are in OP_POP_TO_MARK
        bool in_pop_to_mark = false;
        bool update_mem_1 = false;

        bool use_str_1 = false;
        //#ifndef __SYNTHESIS__
        //        printf("opcode = %d\n", (unsigned int)opcode_1);
        //#endif
        if (str_ptr_1 + len_1 > lenMsg)
            pop_stk_1 = true;
        else {
            switch (opcode_1) {
                case (OP_END): {
                    match_1 = true;
                    end_flag_1 = true;
                    break;
                }
                case (OP_STR_N): {
                    use_str_1 = true;
                    if (predict_match_len_1 >= len_1) {
                        str_ptr_1 += len_1;
                        nt_flag_1 = 1;
                    } else {
                        pop_stk_1 = true;
                    }
                    break;
                }
                // the only difference between cclass & cclass-not is the comparision is matched or unmatched
                // OP_CCLASS_NOT is merged in OP_CCLASS in compiler
                case (OP_CCLASS): {
                    use_str_1 = true;
                    if (predict_cc_cmp_1) {
                        pop_stk_1 = true;
                    } else {
                        str_ptr_1++;
                        nt_flag_1 = 1;
                    }
                    break;
                }
                // begin-buf is contained in begin-line
                case (OP_BEGIN_BUF):
                case (OP_BEGIN_LINE): {
                    if (str_ptr_1 == 0) {
                        nt_flag_1 = 1;
                    } else if (str_ptr_1 != lenMsg && prev_char == 0x0a && opcode_1 == OP_BEGIN_LINE) {
                        nt_flag_1 = 1;
                    } else {
                        pop_stk_1 = true;
                    }
                    break;
                }
                // end-buf is contained in end-line
                case (OP_END_BUF):
                case (OP_END_LINE): {
                    if (str_ptr_1 == lenMsg) {
                        nt_flag_1 = 1;
                    } else if (cur_char == 0x0a && opcode_1 == OP_END_LINE) {
                        nt_flag_1 = 1;
                    } else {
                        pop_stk_1 = true;
                    }
                    break;
                }
                // mem-start is contained in mem-start-push
                case (OP_MEM_START):
                case (OP_MEM_START_PUSH): {
                    update_mem_1 = true;
                    nt_flag_1 = 1;
                    if (opcode_1 == OP_MEM_START_PUSH) {
                        push_stk_1 = true;
                        stk_type_1 = 1;
                        push_mem_id_1 = operand1_1;
                    }
                    break;
                }
                case (OP_MEM_END): {
                    update_mem_1 = true;
                    nt_flag_1 = 1;
                    break;
                }
                case (OP_JUMP): {
                    nt_flag_1 = 2;
                    break;
                }
                case (OP_PUSH): {
                    push_stk_1 = true;
                    stk_type_1 = 0;
                    nt_flag_1 = 1;
                    break;
                }
                case (OP_POP): {
                    // internal::popStack(stk_1);
                    stk_1 = stk_m1;
                    nt_flag_1 = 1;
                    break;
                }
                case (OP_PUSH_OR_JUMP_EXACT1): {
                    if (str_ptr_1 < lenMsg && cur_char == operand2_1.range(15, 8)) {
                        push_stk_1 = true;
                        stk_type_1 = 0;
                        nt_flag_1 = 1;
                    } else {
                        nt_flag_1 = 2;
                    }
                    break;
                }
                // repeat is the starting point of a repetition operation
                case (OP_REPEAT):
                case (OP_REPEAT_INC): {
                    if (opcode_1 == OP_REPEAT)
                        repeat_cnt_1 = 0;
                    else
                        repeat_cnt_1++;
                    // push stack
                    stk_type_1 = 0;
                    // when reach the lower bound, push to stack
                    if ((repeat_cnt_1 >= operand2_1 && repeat_cnt_1 < operand3_1)) {
                        push_stk_1 = true;
                        nt_flag_1 = 2;
                        // if not reach the lower bound, jump
                    } else if (opcode_1 == OP_REPEAT_INC && repeat_cnt_1 < operand2_1) {
                        nt_flag_1 = 2;
                    } else {
                        nt_flag_1 = 1;
                    }
                    break;
                }
                case (OP_ANYCHAR): {
                    use_str_1 = true;
                    if (cur_char == 0x0a) { // || str_ptr_1 + len_1 > lenMsg
                        pop_stk_1 = true;
                    } else {
                        str_ptr_1++;
                        nt_flag_1 = 1;
                    }
                    break;
                }
                case (OP_ANYCHAR_STAR): {
                    stk_type_1 = 0;
                    // two conditions to goto fail
                    if ((str_ptr_1 < lenMsg && cur_char == 0x0a) || str_ptr_1 + 1 > lenMsg) {
                        // if mismatch, stop and jump to next instruciton
                        nt_flag_1 = 1;
                    } else {
                        // keep exec the current instruction and push to stack until mismatch.
                        push_stk_1 = true;
                        // increase the string pointer if not failed
                        str_ptr_1++;
                    }
                    break;
                }
                // push mark-id to internal stack
                case (OP_MARK): {
                    push_stk_1 = true;
                    stk_type_1 = 2;
                    push_mem_id_1 = operand1_1;
                    nt_flag_1 = 1;
                    break;
                }
                // step back for back-tracking
                case (OP_STEP_BACK_START): {
                    ap_int<16> signed_str_ptr = (ap_int<16>)str_ptr_1;
                    signed_str_ptr -= operand2_1;
                    if (signed_str_ptr < 0) {
                        pop_stk_1 = true;
                    } else {
                        str_ptr_1 = signed_str_ptr;
                        if (operand3_1 != 0) {
                            push_stk_1 = true;
                            stk_type_1 = 0;
                            nt_flag_1 = 2;
                        } else {
                            nt_flag_1 = 1;
                        }
                    }
                    break;
                }
                // fail section just pop the stack
                case (OP_FAIL): {
                    pop_stk_1 = true;
                    break;
                }
                // pop the internal stack until mark-id found and matched
                case (OP_POP_TO_MARK): {
                    // when reach the mark_id, stop execute this instruction and go to next instruction
                    pop_stk_1 = true;
                    in_pop_to_mark = true;
                    break;
                }
                default: {
                    // the rest OPs are not supported currently
                    break;
                }
            } // end switch
        }

        //------------------------------------------------execute the pre-fetch
        // instruction------------------------------------//
        // local control signal
        ap_uint<2> nt_flag_2 = 0;
        // control signal
        bool pop_stk_2 = false;
        bool push_stk_2 = false;
        // address register for stack push
        ap_uint<16> push_addr_2 = 0;
        // push the current status
        // predict the current instruciotn exeuted
        ap_uint<16> str_ptr_2 = str_ptr + len_1;
        bool match_2 = false;
        bool end_flag_2 = false;
        unsigned int stk_2 = stk;

        ap_uint<6> stk_type_2 = 0;

        ap_uint<16> repeat_cnt_2 = repeat_cnt;
        ap_uint<16> push_mem_id_2 = 0;
        // flag indicate whether next instruction pre-exe is successfull
        bool op_exec = true;
        bool update_mem_2 = false;
        bool use_str_2 = false;

        //#ifndef __SYNTHESIS__
        //        printf("opcode = %d\n", (unsigned int)opcode_2);
        //#endif
        if (str_ptr_2 + len_2 > lenMsg)
            pop_stk_2 = true;
        else {
            switch (opcode_2) {
                case (OP_END): {
                    match_2 = true;
                    end_flag_2 = true;
                    break;
                }
                case (OP_STR_N): {
                    use_str_2 = true;
                    if (predict_match_len_2 >= len_2) {
                        str_ptr_2 += len_2;
                        nt_flag_2 = 1;
                    } else {
                        pop_stk_2 = true;
                    }
                    break;
                }
                // the only difference between cclass & cclass-not is the comparision is matched or unmatched
                case (OP_CCLASS): {
                    use_str_2 = true;
                    if (predict_cc_cmp_2) {
                        pop_stk_2 = true;
                    } else {
                        str_ptr_2++;
                        nt_flag_2 = 1;
                    }
                    break;
                }
                // begin-buf is contained in begin-line
                case (OP_BEGIN_BUF):
                case (OP_BEGIN_LINE): {
                    if (str_ptr_2 == 0) {
                        nt_flag_2 = 1;
                    } else if (str_ptr_2 != lenMsg && prev_char == 0x0a && opcode_2 == OP_BEGIN_LINE) {
                        nt_flag_2 = 1;
                    } else {
                        pop_stk_2 = true;
                    }
                    break;
                }
                // end-buf is contained in end-line
                case (OP_END_BUF):
                case (OP_END_LINE): {
                    if (str_ptr_2 == lenMsg) {
                        nt_flag_2 = 1;
                    } else if (cur_char == 0x0a && opcode_2 == OP_END_LINE) {
                        nt_flag_2 = 1;
                    } else {
                        pop_stk_2 = true;
                    }
                    break;
                }
                // mem-start is contained in mem-start-push
                case (OP_MEM_START):
                case (OP_MEM_START_PUSH): {
                    update_mem_2 = true;
                    nt_flag_2 = 1;
                    if (opcode_2 == OP_MEM_START_PUSH) {
                        op_exec = false;
                    }
                    break;
                }
                case (OP_MEM_END): {
                    update_mem_2 = true;
                    nt_flag_2 = 1;
                    break;
                }
                case (OP_JUMP): {
                    nt_flag_2 = 2;
                    break;
                }
                case (OP_PUSH): {
                    push_stk_2 = true;
                    stk_type_2 = 0;
                    nt_flag_2 = 1;
                    break;
                }
                case (OP_POP): {
                    // internal::popStack(stk_2);
                    stk_2 = stk_m1;
                    nt_flag_2 = 1;
                    break;
                }
                case (OP_PUSH_OR_JUMP_EXACT1): {
                    use_str_2 = true;
                    if (str_ptr_2 < lenMsg && cur_char == operand2_2.range(15, 8)) {
                        push_stk_2 = true;
                        stk_type_2 = 0;
                        nt_flag_2 = 1;
                    } else {
                        nt_flag_2 = 2;
                    }
                    break;
                }
                // repeat is the starting point of a repetition operation
                case (OP_REPEAT):
                case (OP_REPEAT_INC): {
                    if (opcode_2 == OP_REPEAT)
                        repeat_cnt_2 = 0;
                    else
                        repeat_cnt_2++;
                    // push stack
                    stk_type_2 = 0;
                    // when reach the lower bound, push to stack
                    if ((repeat_cnt_2 >= operand2_2 && repeat_cnt_2 < operand3_2)) {
                        push_stk_2 = true;
                        nt_flag_2 = 2;
                        // if not reach the lower bound, jump
                    } else if (repeat_cnt_2 < operand2_2 && opcode_2 == OP_REPEAT_INC) {
                        nt_flag_2 = 2;
                    } else {
                        nt_flag_2 = 1;
                    }
                    break;
                }
                case (OP_ANYCHAR): {
                    use_str_2 = true;
                    if ((cur_char == 0x0a)) { // || str_ptr_2 + 1 > lenMsg)
                        pop_stk_2 = true;
                    } else {
                        str_ptr_2++;
                        nt_flag_2 = 1;
                    }
                    break;
                }
                case (OP_ANYCHAR_STAR): {
                    op_exec = false;
                    break;
                }
                // push mark-id to internal stack
                case (OP_MARK): {
                    push_stk_2 = true;
                    stk_type_2 = 2;
                    push_mem_id_2 = operand1_2;
                    nt_flag_2 = 1;
                    break;
                }
                // step back for back-tracking
                case (OP_STEP_BACK_START): {
                    ap_int<16> signed_str_ptr = (ap_int<16>)str_ptr_2;
                    signed_str_ptr -= operand2_2;
                    if (signed_str_ptr < 0) {
                        pop_stk_2 = true;
                    } else {
                        str_ptr_2 = signed_str_ptr;
                        if (operand3_2 != 0) {
                            push_stk_2 = true;
                            stk_type_2 = 0;
                            nt_flag_2 = 2;
                        } else {
                            nt_flag_2 = 1;
                        }
                    }
                    break;
                }
                // fail section just pop the stack
                case (OP_FAIL): {
                    pop_stk_2 = true;
                    break;
                }
                // pop the internal stack until mark-id found and matched
                case (OP_POP_TO_MARK): {
                    op_exec = false;
                    break;
                }
                default: {
                    // the rest OPs are not supported currently
                    break;
                }
            } // end switch
        }
        //#endif

        // check the status to decide whether the pre-fectch instruction is exectured or withdrawed.
        bool exec_opt = true;
        // For the following scenaro is not executed
        // 1: read/write stack simutaneously
        // 2: opcode for VM2 is any_char_star or op_pop_to_mark or op_mem_start_push
        // 3: jump happend in VM1
        // 4: read/write the offsetBuffer simultaneously
        // 5: str_ptr changes in VM1 and VM2 goes into OP which needs character comparision
        // 6: write the offsetBuffer simultaneously
        if (((pop_stk_1 || push_stk_1) && (pop_stk_2 || push_stk_2)) || !op_exec || (nt_flag_1 != 1) ||
            (update_mem_1 && pop_stk_2) || (update_mem_2 && pop_stk_1) || (use_str_2 && use_str_1) ||
            (update_mem_1 && update_mem_2))

            exec_opt = false;
        // exec_opt = false;
        // printf("exec_opt %d\n", exec_opt);
        // update status
        ap_uint<2> nt_flag;
        if (exec_opt) {
            nt_flag = nt_flag_2;
            end_flag = end_flag_2;
            match_s = match_2;
            repeat_cnt = repeat_cnt_2;
        } else {
            nt_flag = nt_flag_1;
            end_flag = end_flag_1;
            match_s = match_1;
            repeat_cnt = repeat_cnt_1;
        }
        bool pop_stk = pop_stk_1 || (exec_opt && pop_stk_2);
        if (pop_stk_1)
            stk = stk_1;
        else
            stk = stk_2;

        if (!exec_opt)
            str_ptr = str_ptr_1;
        else
            str_ptr = str_ptr_2;

        ap_uint<6> stk_type;      // = stk_type_1;
        ap_uint<16> push_addr;    // = push_addr_1;
        ap_uint<16> push_str_ptr; // = push_str_ptr_1;
        ap_uint<16> push_mem_id;  //= push_mem_id_1;
        bool push_stk = push_stk_1 || (exec_opt && push_stk_2);

        // use mode to optimize
        // mode:1 op-push, op-push-or-jump-exact1
        // mode:0 others
        if (mode_1 == 1)
            push_addr_1 = predict_jump_addr_1;
        else
            push_addr_1 = predict_op_addr_1;
        if (mode_2 == 1)
            push_addr_2 = predict_jump_addr_2;
        else
            push_addr_2 = predict_op_addr_2;

        if (push_stk_1) {
            stk_type = stk_type_1;
            push_addr = push_addr_1;
            push_str_ptr = str_ptr_1;
            push_mem_id = push_mem_id_1;
        } else {
            stk_type = stk_type_2;
            push_addr = push_addr_2;
            push_str_ptr = str_ptr_2;
            push_mem_id = push_mem_id_2;
        }

        bool update_mem = update_mem_1 || (exec_opt && update_mem_2);
        ap_uint<16> mem_dt_t;
        ap_uint<16> mem_adr_t;
        if (update_mem_1) {
            mem_adr_t = operand1_1;
            mem_dt_t = str_ptr_1;
        } else {
            mem_adr_t = operand1_2;
            mem_dt_t = str_ptr_2;
        }
        if (update_mem) {
            mem_adr_reg_1 = mem_adr_t;
            mem_dt_reg_1 = mem_dt_t;
            offsetBuff[mem_adr_t] = mem_dt_t;
        } else if (peek_stk_type == 1 && pop_stk) {
            mem_dt_reg_1 = peek_mem_start;
            mem_dt_reg_2 = peek_mem_end;
            mem_adr_reg_1 = peek_mem_id;
            mem_adr_reg_2 = peek_mem_id + 1;
            offsetBuff[peek_mem_id] = peek_mem_start;
            offsetBuff[peek_mem_id + 1] = peek_mem_end;
        }
        // do real push or pop operation on stack
        if (pop_stk) {
            // finish code popped
            if (stk_m1 == 0) {
                end_flag = true;
            }
            if (peek_stk_type == 0 && !in_pop_to_mark) {
                nt_flag = 3;
                str_ptr = peek_str_ptr;
            } else if (in_pop_to_mark && peek_stk_type == 2 && peek_mem_id == operand1_1) {
                nt_flag = 1;
            }
            // internal::popStack(stk);
            stk = stk_m1;
        } else if (push_stk) {
            if (stk_a1 == STACK_SIZE - 1) {
                end_flag = true;
                stk_ovf = true;
            }
            ap_uint<54> stk_dt_n = 0;
            internal::combine(stk_dt_n, stk_type, push_mem_id, mem_start, mem_end, push_addr, push_str_ptr);
            // printf("stk_type = %d, push_mem_id = %d, mem_start = %d, mem_end = %d, push_addr = %d, push_str_ptr =
            // %d\n",
            //(unsigned int)stk_type, (unsigned int)push_mem_id, (unsigned int)mem_start, (unsigned int)mem_end,
            //(unsigned int)push_addr, (unsigned int)push_str_ptr);
            stk_adr_reg = stk_a1;
            stk_dt_reg = stk_dt_n;
            internal::pushStack_opt(stack_buff, stk, stk_dt_n);
            stk = stk_a1;
        }

        // for nt_flag == 0, stay in OP_POP_TO_MARK
        // use predict address in VM
        if (nt_flag == 1) {
            if (!exec_opt) {
                opaddr = predict_op_addr_1;
                n_opaddr = n_predict_op_addr_1;
            } else {
                opaddr = predict_op_addr_2;
                n_opaddr = n_predict_op_addr_2;
            }
            // use jump address in VM
        } else if (nt_flag == 2) {
            if (!exec_opt) {
                opaddr = predict_jump_addr_1;
                n_opaddr = n_predict_jump_addr_1;
            } else {
                opaddr = predict_jump_addr_2;
                n_opaddr = n_predict_jump_addr_2;
            }
            // use peek address in VM
        } else if (nt_flag == 3) {
            opaddr = peek_opaddr;
            n_opaddr = n_peek_opaddr;
        }
        // if(str_ptr == 109) {
        //   printf("debug point\n");
        //}
        // printf("nt_flag = %d, opaddr = %d, str_ptr = %d\n", (unsigned int)nt_flag, (unsigned int)opaddr, (unsigned
        // int)str_ptr);
        // if(str_ptr == 109) {
        //   printf("debug point\n");
        //}

    } // end while
#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
    printf("total exec time = %d\n", exec_tm);
#endif
#endif

    // if string matched, mark the whole sequence
    if (match_s) {
        match = 1;
        offsetBuff[0] = str_start;
        offsetBuff[1] = str_ptr;
    } else {
        if (stk_ovf)
            match = 2;
        else
            match = 0;
    }
} // end regexVM_opt

} // namespace text
} // namespace data_analytics
} // namespace xf

#endif // XF_SEACH_REGEX_VM_H
