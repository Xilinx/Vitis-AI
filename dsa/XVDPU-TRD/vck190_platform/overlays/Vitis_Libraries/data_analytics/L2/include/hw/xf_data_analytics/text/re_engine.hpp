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
 * @file re_engine.hpp
 * @brief RE engine implementation
 */
#ifndef XF_TEXT_RE_ENGINE_HPP
#define XF_TEXT_RE_ENGINE_HPP

#include "hls_stream.h"
#include "ap_int.h"
#include "xf_data_analytics/text/regexVM.hpp"

namespace xf {
namespace data_analytics {
namespace text {
namespace internal {

// reads configuration to internal buffer
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM>
void readCFG2Buff(unsigned int& cpgp_nm,
                  hls::stream<ap_uint<64> >& in_strm,
                  ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
                  ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8]) {
    // ignores the first row number
    ap_uint<64> head = in_strm.read();

    // second block contains cpgp_num | cclass_num | instr_num
    head = in_strm.read();
    unsigned int instr_nm = head.range(31, 0);
    unsigned int cclass_nm = head.range(47, 32);
    // passes on the cpgp_nm to processing module
    cpgp_nm = head.range(63, 48);

#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
    printf("instr_nm = %d, cclass_nm = %d, cpgp_nm = %d\n", instr_nm, cclass_nm, cpgp_nm);
#endif
#endif
    // reads instructions
    for (int i = 0; i < instr_nm; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 100 max = 100
        ap_uint<64> in = in_strm.read();
        for (int p = 0; p < PU_NM; ++p) {
            instr_buff[p][i] = in;
        }
    }
    // reads bit set map
    for (int i = 0; i < cclass_nm * 4; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
        ap_uint<64> in = in_strm.read();
        ap_uint<32> in_hp = in.range(63, 32);
        ap_uint<32> in_lp = in.range(31, 0);
        for (int p = 0; p < PU_NM; ++p) {
            bitset_buff[p][2 * i] = in_lp;
            bitset_buff[p][2 * i + 1] = in_hp;
        }
    }
}

// read data from external DDR/HBM to stream
template <int W, int NM_W>
void readFromExtMem(ap_uint<W>* ddr_buff, hls::stream<ap_uint<W> >& out_strm) {
    ap_uint<W> nm = 0; // = ddr_buff[0];
    for (int i = 0; i < NM_W / W; ++i) {
#pragma HLS pipeline II = 1
        nm = nm << W;
        nm += ddr_buff[i];
    }
#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
    printf("nm = %d\n", (unsigned int)nm);
#endif
#endif
    for (int i = 0; i < nm; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
        ap_uint<W> in = ddr_buff[i];
        out_strm.write(in);
    }
}

// read data from external DDR/HBM to stream
template <int W>
void readFromExtMem(ap_uint<W>* ddr_buff, hls::stream<ap_uint<W> >& out_strm) {
    ap_uint<W> nm = ddr_buff[0];
#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
    printf("nm = %d\n", (unsigned int)nm);
#endif
#endif
    for (int i = 0; i < nm; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
        ap_uint<W> in = ddr_buff[i];
        out_strm.write(in);
    }
}

// reads configuration from external memory and stores to internal BRAM/URAM
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM>
void readRECfg(unsigned int& cpgp_nm,
               ap_uint<64>* cfg_buff,
               ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
               ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8]) {
#pragma HLS dataflow
    hls::stream<ap_uint<64> > cfg_strm("cfg_strm");
#pragma HLS stream variable = cfg_strm depth = 32
#pragma HLS bind_storage variable = cfg_strm type = fifo impl = lutram

    // configurations from buffer to stream
    readFromExtMem(cfg_buff, cfg_strm);

    // separates configurations to opcode list and bit-set map
    readCFG2Buff<PU_NM, INSTR_DEPTH, CCLASS_NM>(cpgp_nm, cfg_strm, instr_buff, bitset_buff);
}

// pre-calcualtes the message length for each PU
template <int PU_NM, int MSG_LEN>
void dispatchMSGLen(ap_uint<32> total_msg_nm,
                    ap_uint<16> max_blk_nm,
                    ap_uint<16>& cur_len,
                    ap_uint<32>& cur_msg_nm,
                    hls::stream<ap_uint<16> >& i_len_strm,
                    hls::stream<ap_uint<16> >& o_len_strm,
                    hls::stream<bool>& o_e_strm,
                    unsigned int cpgp_nm,
                    hls::stream<ap_uint<16> >& o_nm_strm,
                    hls::stream<bool>& o_e_strm_1) {
    ap_uint<32> msg_nm = cur_msg_nm;
    // prepare the len_buf for each PU (in round-robin order)
    for (int p = 0; p < PU_NM; ++p) {
        // leave space for blk_nm and msg_len
        ap_uint<16> blk_len = 2;
        ap_uint<16> blk_nm = 0;

        // leave room for tail, initial cur_len gives in reExecMPU
        ap_uint<16> plus_1 = 0;
        if (cur_len.range(2, 0) > 0)
            plus_1 = 1;
        else
            plus_1 = 0;

        ap_uint<16> cur_len_64b = cur_len.range(15, 3) + plus_1;

        blk_len = blk_len + 2 * cur_len_64b;

        o_e_strm.write(false);
        // 3 conditions to stop dispatching message lengths,
        // 1: msg_buffer is full
        // 2: no more messages
        // 3: output offset buffer overflow
        while (blk_len < MSG_LEN * 2 && cur_msg_nm < total_msg_nm && blk_nm < max_blk_nm) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000

            o_len_strm.write(cur_len);
            o_e_strm.write(false);

            blk_nm++;

            // still have messages in external memory
            if (cur_msg_nm < total_msg_nm - 1) cur_len = i_len_strm.read();

            // leave room for tail
            if (cur_len.range(2, 0) > 0)
                plus_1 = 1;
            else
                plus_1 = 0;

            cur_len_64b = cur_len.range(15, 3) + plus_1;

            // accumulates length in msg_buff for each regexVM
            // leave 1 space for msg_len, 2 * cur_len_64b for messages
            // the multiplier 2 is because the width of input msg_buff from memory port is 64-bit,
            // where the width of msg_buff of regexVM is 32-bit
            blk_len = blk_len + 2 * cur_len_64b + 1;
            // increases current number of messages
            cur_msg_nm++;
        }
        o_len_strm.write(blk_nm);
        o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
        printf("PU[%d]'s blk_len = %d, blk_nm = %d\n", p, (unsigned int)blk_len, (unsigned int)blk_nm);
#endif
#endif
    }
    ap_uint<16> blk_nm_tot = cur_msg_nm - msg_nm;
    o_nm_strm.write(blk_nm_tot * (cpgp_nm + 1));
    o_e_strm_1.write(false);
}

// dispatchs message for each PU
template <int PU_NM, int MSG_LEN>
void dispatchMSG(hls::stream<ap_uint<64> >& msg_strm,
                 hls::stream<ap_uint<16> >& len_strm,
                 hls::stream<bool>& i_e_strm,
                 ap_uint<32> msg_buff[PU_NM][MSG_LEN * 2]) {
    for (int p = 0; p < PU_NM; ++p) {
        bool e = i_e_strm.read();

        ap_uint<16> oft = 1;
        ap_uint<16> len = 0;

        // msg_buff structure for each PU
        // blk_nm
        // msg_len0
        // msg0_0
        // ...
        // msg0_N
        // msg_len1
        // msg1_0
        // ...
        // msg_1_N
        while (!e) {
            // length in char
            len = len_strm.read();
            e = i_e_strm.read();
            // length of each message
            msg_buff[p][oft++] = len;
            // printf("len[%d] = %d\n", (unsigned int) oft, (unsigned int) len);
            // length in 64-bit
            ap_uint<16> len_64b = (len + 7) / 8;
            // printf("len_64b = %d\n", (unsigned int) len_64b);
            // dispatch 1 message
            if (!e) {
                for (int i = 0; i < len_64b; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1000 max = 1000
                    ap_uint<64> in = msg_strm.read();
                    msg_buff[p][oft + 2 * i] = in.range(31, 0);
                    msg_buff[p][oft + 2 * i + 1] = in.range(63, 32);
                }
                oft = oft + 2 * len_64b;
            }
        }
        // first element in msg_buff is blk_nm
        msg_buff[p][0] = len;
    }
}

// dispatchs a block of message for each PU
template <int PU_NM, int MSG_LEN>
void dispatch2MPU(ap_uint<32> total_msg_nm,
                  ap_uint<16> max_blk_nm,
                  ap_uint<16>& cur_len,
                  ap_uint<32>& cur_msg_nm,
                  hls::stream<ap_uint<16> >& len_strm,
                  hls::stream<ap_uint<64> >& msg_strm,
                  ap_uint<32> msg_buff[PU_NM][MSG_LEN * 2],
                  unsigned int cpgp_nm,
                  hls::stream<ap_uint<16> >& o_nm_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS dataflow
    hls::stream<ap_uint<16> > blk_len_strm("blk_len_strm");
#pragma HLS stream variable = blk_len_strm depth = 8
    hls::stream<bool> e_strm("e_strm");
#pragma HLS stream variable = e_strm depth = 8;

    // pre-calculates message length for each PU
    dispatchMSGLen<PU_NM, MSG_LEN>(total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, len_strm, blk_len_strm, e_strm,
                                   cpgp_nm, o_nm_strm, o_e_strm);

    // dispatchs messages from stream into msg_buff for each PU
    // along with blck_nm & msg_len (round-robin)
    dispatchMSG<PU_NM, MSG_LEN>(msg_strm, blk_len_strm, e_strm, msg_buff);
}

// collects the result from each PU
template <int PU_NM, int CPGP_NM>
void collectREResult(unsigned int cpgp_nm,
                     ap_uint<16> mem_oft_buff[PU_NM][CPGP_NM * 2],
                     hls::stream<ap_uint<32> >& o_strm) {
    // in round-robin order
    for (int p = 0; p < PU_NM; ++p) {
        ap_uint<16> blk_nm = mem_oft_buff[p][0];
        for (int b = 0; b < blk_nm; ++b) {
            for (int g = 0; g < cpgp_nm + 1; ++g) {
#pragma HLS pipeline II = 2
                ap_uint<32> out = 0;
                // write out match flag
                // write out position of capture group
                out.range(15, 0) = mem_oft_buff[p][b * (cpgp_nm + 1) * 2 + g * 2 + 1];
                out.range(31, 16) = mem_oft_buff[p][b * (cpgp_nm + 1) * 2 + g * 2 + 2];
                mem_oft_buff[p][b * (cpgp_nm + 1) * 2 + g * 2 + 1] = -1;
                mem_oft_buff[p][b * (cpgp_nm + 1) * 2 + g * 2 + 2] = -1;
                o_strm.write(out);
            }
        }
    }
}

// write data to external DDR/HBM
static void writeToExtMem(hls::stream<ap_uint<32> >& i_strm,
                          hls::stream<bool>& e_strm,
                          hls::stream<ap_uint<16> >& nm_strm,
                          ap_uint<32>* out_buff) {
    bool e = false;
    ap_uint<32> offt = 1;
    do {
        e = e_strm.read();
        ap_uint<16> nm = nm_strm.read();
        for (int i = 0; i < nm; ++i) {
#pragma HLS pipeline II = 1
            out_buff[offt + i] = i_strm.read();
        }
        offt += nm;
    } while (!e);
    out_buff[0] = offt;
}

// exectutes single block of message by 1 PU
template <int STACK_SIZE, int CPGP_NM>
void reExecBlock(bool exec,
                 ap_uint<16> cpgp_nm,
                 ap_uint<16>* mem_oft_buff,
                 ap_uint<64>* instr_buff,
                 ap_uint<32>* bitset_buff,
                 ap_uint<32>* msg_buff) {
    if (exec) {
        ap_uint<16> msg_oft = 2;
        ap_uint<16> out_oft = 3;
        //#ifndef __SYNTHESIS__
        //    for (int i = 0; i < CPGP_NM; ++i) {
        //    #pragma HLS pipeline II = 1
        //         printf("mem_oft_buff %d, %d\n",(int) mem_oft_buff[2 * i], (int) mem_oft_buff[2 * i + 1]);
        //    }
        //#endif

        // read block number
        ap_uint<16> blk_nm = msg_buff[0];
        // write block number to result buffer
        mem_oft_buff[0] = blk_nm;

        for (int i = 0; i < blk_nm; ++i) {
#pragma HLS loop_tripcount min = 10 max = 10
            unsigned int msg_len = msg_buff[msg_oft - 1];
            ap_uint<2> match = 3;
            //#ifndef __SYNTHESIS__
            //            printf("msg_len_pos = %d\n", (unsigned int)msg_oft - 1);
            //            printf("msg_len = %d\n", msg_len);
            //#endif
            //#ifndef __SYNTHESIS__
            //
            //             for(int j = 0; j < (msg_len+7)/8; ++j) {
            //                ap_uint<64>  tmp;
            //                tmp.range(31,0) = msg_buff[msg_oft+2*j];
            //                tmp.range(63,32) = msg_buff[msg_oft+2*j+1];
            //                for(int k = 0; k < 8; ++k) {
            //                   printf("%c", (unsigned char)(tmp.range((k+1)*8-1, 8*k)));
            //                }
            //            }
            //            printf("\n");
            //#endif
            // 0: mismatch
            // 1: match
            // 2: stack overflow
            // 3: length exceed limitation
            if (msg_len > 0)
                xf::data_analytics::text::regexVM_opt<STACK_SIZE>(bitset_buff, instr_buff, msg_buff + msg_oft, msg_len,
                                                                  match, mem_oft_buff + out_oft);
            //#ifndef __SYNTHESIS__
            //    printf("len exceed limitation\n");
            //#endif

            // message length in 64-bit
            ap_uint<16> msg_len_byte = (msg_len + 7) / 8;
            // accumulates offset with 1 message,
            // 2 * msg_len_byte 32-bit message block + 1 msg_len
            msg_oft = msg_oft + msg_len_byte * 2 + 1;

            // store match flag
            mem_oft_buff[out_oft - 2] = match;
            mem_oft_buff[out_oft - 1] = 0;

            out_oft += (cpgp_nm + 1) * 2;
        }
    }
#ifndef __SYNTHESIS__
#ifdef XF_DEBUG
    for (int i = 0; i < blk_nm; ++i) {
        printf("blk[%d] result: %d\n", i, (unsigned int)msg_oft_buff[i + 1]);
        for (int j = 0; j < cpgp_nm + 1; ++j) {
            printf("   group[%d]: [%d: %d]\n", j, (unsigned int)mem_oft_buff[i * 2 * (cpgp_nm + 1) + 2 * j + 1],
                   (unsigned int)mem_oft_buff[i * 2 * (cpgp_nm + 1) + 2 * j + 2]);
        }
    }
#endif
#endif
}

// executates multiple blocks in parallel
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN, int STACK_SIZE>
void reExecMBlock(bool exec,
                  ap_uint<16> cpgp_nm,
                  ap_uint<16> mem_oft_buff[PU_NM][CPGP_NM * 2],
                  ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
                  ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8],
                  ap_uint<32> msg_buff[PU_NM][MSG_LEN * 2]) {
#pragma HLS dataflow
    for (int p = 0; p < PU_NM; ++p) {
#pragma HLS unroll
        reExecBlock<STACK_SIZE, CPGP_NM>(exec, cpgp_nm, mem_oft_buff[p], instr_buff[p], bitset_buff[p], msg_buff[p]);
    }
}

// collects results & dispatchs message length and corresponding messages
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN>
void reExecBodyRW(bool exec_cr,
                  ap_uint<16> cpgp_nm,
                  hls::stream<ap_uint<16> >& len_strm,
                  hls::stream<ap_uint<64> >& msg_strm,
                  hls::stream<ap_uint<32> >& o_strm,
                  hls::stream<ap_uint<16> >& o_nm_strm,
                  hls::stream<bool>& o_e_strm,
                  ap_uint<32> total_msg_nm,
                  ap_uint<16> max_blk_nm,
                  ap_uint<16>& cur_len,
                  ap_uint<32>& cur_msg_nm,
                  ap_uint<32> msg_buff[PU_NM][MSG_LEN * 2],
                  ap_uint<16> mem_oft_buff[PU_NM][CPGP_NM * 2]) {
    // writes out result in serial
    if (exec_cr) collectREResult<PU_NM, CPGP_NM>(cpgp_nm, mem_oft_buff, o_strm);

    dispatch2MPU<PU_NM, MSG_LEN>(total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, len_strm, msg_strm, msg_buff, cpgp_nm,
                                 o_nm_strm, o_e_strm);
}

// executes 1 round of dispatching message, executing matching, and collecting results
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN, int STACK_SIZE>
void reExecBody(bool exec_cr,
                bool exec_re,
                ap_uint<16> cpgp_nm,
                hls::stream<ap_uint<16> >& len_strm,
                hls::stream<ap_uint<64> >& msg_strm,
                ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
                ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8],
                hls::stream<ap_uint<32> >& o_strm,
                hls::stream<ap_uint<16> >& o_nm_strm,
                hls::stream<bool>& o_e_strm,
                ap_uint<32> total_msg_nm,
                ap_uint<16> max_blk_nm,
                ap_uint<16>& cur_len,
                ap_uint<32>& cur_msg_nm,
                ap_uint<32> msg_buff_0[PU_NM][MSG_LEN * 2],
                ap_uint<32> msg_buff_1[PU_NM][MSG_LEN * 2],
                ap_uint<16> mem_oft_buff_0[PU_NM][CPGP_NM * 2],
                ap_uint<16> mem_oft_buff_1[PU_NM][CPGP_NM * 2]) {
#pragma HLS dataflow
    // 1 round read & write ops
    reExecBodyRW<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN>(exec_cr, cpgp_nm, len_strm, msg_strm, o_strm,
                                                                  o_nm_strm, o_e_strm, total_msg_nm, max_blk_nm,
                                                                  cur_len, cur_msg_nm, msg_buff_0, mem_oft_buff_0);
    // 1 round executing MPU in parallel
    reExecMBlock<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(exec_re, cpgp_nm, mem_oft_buff_1,
                                                                              instr_buff, bitset_buff, msg_buff_1);
}

// executates all message by multiple PU with multiple loop
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN, int STACK_SIZE>
void reExecMPU(ap_uint<16> cpgp_nm,
               hls::stream<ap_uint<16> >& len_strm,
               hls::stream<ap_uint<64> >& msg_strm,
               ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
               ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8],
               hls::stream<ap_uint<32> >& o_strm,
               hls::stream<ap_uint<16> >& o_nm_strm,
               hls::stream<bool>& o_e_strm) {
    ap_uint<32> msg_buff_0[PU_NM][MSG_LEN * 2];
#pragma HLS array_partition variable = msg_buff_0 dim = 1
#pragma HLS bind_storage variable = msg_buff_0 type = ram_t2p impl = bram

    ap_uint<32> msg_buff_1[PU_NM][MSG_LEN * 2];
#pragma HLS array_partition variable = msg_buff_1 dim = 1
#pragma HLS bind_storage variable = msg_buff_1 type = ram_t2p impl = bram

    ap_uint<16> mem_oft_buff_0[PU_NM][CPGP_NM * 2];
#pragma HLS array_partition variable = mem_oft_buff_0 dim = 1
#pragma HLS bind_storage variable = mem_oft_buff_0 type = ram_t2p impl = bram

    ap_uint<16> mem_oft_buff_1[PU_NM][CPGP_NM * 2];
#pragma HLS array_partition variable = mem_oft_buff_1 dim = 1
#pragma HLS bind_storage variable = mem_oft_buff_1 type = ram_t2p impl = bram

    ap_uint<16> msg_nm_h = len_strm.read();
    ap_uint<16> msg_nm_l = len_strm.read();

    ap_uint<32> total_msg_nm = 0;
    total_msg_nm.range(31, 16) = msg_nm_h;
    total_msg_nm.range(15, 0) = msg_nm_l;

    // limitation from out_buff
    // 1023 = 2 * CPGP_NM - 1)
    // 2 for begin/end offset addresses
    // +1 for matching flag
    ap_uint<16> max_blk_nm = 1023 / 2 / (cpgp_nm + 1);
    // ignores the total message length
    ap_uint<32> cur_msg_nm = 2;
    // remove the first number of message blocks
    ap_uint<64> tmp = msg_strm.read();

    // first length of message is read here and pass to dispatchMSGLen
    ap_uint<16> cur_len = len_strm.read();
    // initilizes offset buffers with -1 (mismatch)
    for (int i = 0; i < CPGP_NM; ++i) {
#pragma HLS pipeline II = 1
        for (int p = 0; p < PU_NM; ++p) {
            mem_oft_buff_0[p][2 * i] = -1;
            mem_oft_buff_0[p][2 * i + 1] = -1;
            mem_oft_buff_1[p][2 * i] = -1;
            mem_oft_buff_1[p][2 * i + 1] = -1;
        }
    }
    bool use_ping = true;
    int loop_nm = 0;
    while (cur_msg_nm < total_msg_nm) {
#pragma HLS loop_tripcount min = 10 max = 10
        // for rest rounds, do dispatching messages, executing matching, and collecting results in parallel
        bool exec_cr = true;
        bool exec_re = true;
        // for 1st round, only dispatchs messages to each PU
        if (loop_nm == 0) {
            // not exec collectResult and regex
            exec_cr = false;
            exec_re = false;
            // for 2nd round, executes the 1st block of message and dipatchs the second block of message
        } else if (loop_nm == 1) {
            // not exec collectResult
            exec_cr = false;
            exec_re = true;
        }
        if (use_ping)
            reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
                exec_cr, exec_re, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm,
                total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, msg_buff_0, msg_buff_1, mem_oft_buff_0, mem_oft_buff_1);
        else
            reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
                exec_cr, exec_re, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm,
                total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, msg_buff_1, msg_buff_0, mem_oft_buff_1, mem_oft_buff_0);

        // shift ping-pong
        use_ping = !use_ping;
        //#ifndef __SYNTHESIS__
        // printf("loop_nm = %d, cur_msg_nm = %d, total_msg_nm = %d\n", (unsigned int) loop_nm, (unsigned int)
        // cur_msg_nm, (unsigned int) total_msg_nm);
        //#endif

        //#ifndef __SYNTHESIS__
        //#ifdef HLS_DEBUG
        loop_nm++;
        //#endif
        //#endif
    }
    // for the last 2 rounds
    if (use_ping) {
        // executes matching and collecting results at first
        reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
            true, true, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm, total_msg_nm,
            max_blk_nm, cur_len, cur_msg_nm, msg_buff_0, msg_buff_1, mem_oft_buff_0, mem_oft_buff_1);
        // then collects the last round of results without executing matching process
        reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
            true, false, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm,
            total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, msg_buff_1, msg_buff_0, mem_oft_buff_1, mem_oft_buff_0);
    } else {
        bool exec_cr;
        // messages is not enough to feed 2 rounds (only exist in pong buffer senario)
        if (loop_nm < 2) {
            exec_cr = false;
        } else {
            exec_cr = true;
        }
        reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
            exec_cr, true, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm,
            total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, msg_buff_1, msg_buff_0, mem_oft_buff_1, mem_oft_buff_0);
        reExecBody<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(
            true, false, cpgp_nm, len_strm, msg_strm, instr_buff, bitset_buff, o_strm, o_nm_strm, o_e_strm,
            total_msg_nm, max_blk_nm, cur_len, cur_msg_nm, msg_buff_0, msg_buff_1, mem_oft_buff_0, mem_oft_buff_1);
    }
    o_nm_strm.write(0);
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    //#ifdef HLS_DEBUG
    printf("total loop number %d\n", loop_nm);
//#endif
#endif
}

// gets message from DDR/HBM, execuates it by multiple PU and stores the result to DDR/HBM
template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN, int STACK_SIZE>
void reExec(ap_uint<16> cpgp_nm,
            ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH],
            ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8],
            ap_uint<16>* msg_len_buff,
            ap_uint<64>* msg_in_buff,
            ap_uint<32>* out_buff) {
#pragma HLS dataflow
    hls::stream<ap_uint<64> > msg_strm("msg_strm");
#pragma HLS stream variable = msg_strm depth = 32
#pragma HLS bind_storage variable = msg_strm type = fifo impl = lutram

    hls::stream<ap_uint<16> > len_strm("len_strm");
#pragma HLS stream variable = len_strm depth = 32

    hls::stream<ap_uint<32> > out_strm("out_strm");
#pragma HLS stream variable = out_strm depth = 32

    hls::stream<ap_uint<16> > nm_strm("nm_strm");
#pragma HLS stream variable = nm_strm depth = 8

    hls::stream<bool> e_strm("e_strm");
#pragma HLS stream variable = e_strm depth = 8

    readFromExtMem(msg_in_buff, msg_strm);

    readFromExtMem<16, 32>(msg_len_buff, len_strm);

    reExecMPU<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(cpgp_nm, len_strm, msg_strm, instr_buff,
                                                                           bitset_buff, out_strm, nm_strm, e_strm);

    writeToExtMem(out_strm, e_strm, nm_strm, out_buff);
}

} // namespace internal

/**
 * @brief The reEngine executes the input messages with configured RE pattern.
 * The pattern is pre-compiled to a list of instructions and is provied by user through
 * the cfg_buff. Therefore, the reEngine which is based on the hardware regex-VM
 * is dynamically configurable. User could improve the throughput by increasing
 * the template parameter PU_NM to accelerate the matching process by sacrificing
 * the on-board resources.
 *
 * @tparam PU_NM Number of processing units in parallel.
 * @tparam INSTR_DEPTH The depth of instruction buffer in 64-bit.
 * @tparam CCLASS_NM Supported max number of character classes in regular expression pattern.
 * @tparam CPGP_NM Supported max number of capturing group in regular expression pattern.
 * @tparam MSG_LEN Supported max length for each message in 8-byte.
 * @tparam STACK_SIZE Max size of internal stack buffer in regex-VM.
 *
 * @param cfg_in_buff Input configurations which provides a list of instructions,
 * number of instructions, number of character classes, number of capturing groups, and bit set map.
 * @param msg_in_buff Input messages to be matched by the regular expression.
 * @param len_in_buff input length for each message.
 * @param out_buff Output match results.
 *
 */

template <int PU_NM, int INSTR_DEPTH, int CCLASS_NM, int CPGP_NM, int MSG_LEN, int STACK_SIZE>
void reEngine(ap_uint<64>* cfg_in_buff, ap_uint<64>* msg_in_buff, ap_uint<16>* len_in_buff, ap_uint<32>* out_buff) {
    ap_uint<64> instr_buff[PU_NM][INSTR_DEPTH];
#pragma HLS bind_storage variable = instr_buff type = ram_2p impl = uram
#pragma HLS array_partition variable = instr_buff dim = 1

    ap_uint<32> bitset_buff[PU_NM][CCLASS_NM * 8];
#pragma HLS bind_storage variable = bitset_buff type = ram_2p impl = bram
#pragma HLS array_partition variable = bitset_buff dim = 1

    unsigned int cpgp_nm = 0;
    //#ifndef __SYNTHESIS__
    //    for(int i = 0; i < 10; ++i) {
    //         printf("len[%d]= %d\n", i, (unsigned int) len_in_buff[i]);
    //    }
    //#endif

    // read cfg from external memory to stream
    internal::readRECfg<PU_NM, INSTR_DEPTH, CCLASS_NM>(cpgp_nm, cfg_in_buff, instr_buff, bitset_buff);

    // read message from external memory,
    // execute regexVM to match the message
    internal::reExec<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(cpgp_nm, instr_buff, bitset_buff,
                                                                                  len_in_buff, msg_in_buff, out_buff);
}

} // namespace text
} // namespace data_analytics
} // namespace xf

#endif //_RE_ENGINE_HPP_
