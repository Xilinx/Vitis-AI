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
 * @file hash_join_v4.hpp
 * @brief hash join implementation, targeting HBM devices.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_JOIN_V4_H
#define XF_DATABASE_HASH_JOIN_V4_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_database/combine_split_col.hpp"
#include "xf_database/bloom_filter.hpp"
#include "xf_database/hash_lookup3.hpp"
#include "xf_database/hash_join_v2.hpp"
#include "xf_database/hash_join_v3.hpp"
#include "xf_database/utils.hpp"

// control debug printer
//#define DEBUG true

#ifdef DEBUG

//#define DEBUG_BUILD true
//#define DEBUG_PROBE true
//#define DEBUG_HASH_CNT true
//#define DEBUG_HBM true
//#define DEBUG_JOIN true

#endif

// FIXME For debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

#define write_bit_vector0(i, v) \
    do {                        \
        bit_vector0[(i)] = v;   \
    } while (0)

#define read_bit_vector0(i, v) \
    do {                       \
        v = bit_vector0[(i)];  \
    } while (0)

#define write_bit_vector1(i, v) \
    do {                        \
        bit_vector1[(i)] = v;   \
    } while (0)

#define read_bit_vector1(i, v) \
    do {                       \
        v = bit_vector1[(i)];  \
    } while (0)

namespace xf {
namespace database {
namespace details {
namespace join_v4 {

// TODO update to use 256-bit port.
namespace sc_tmp {

// generate addr for read/write stb
template <int RW, int ARW>
void stb_addr_gen(hls::stream<ap_uint<ARW> >& i_addr_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<ARW> >& o_addr_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int number_of_element_per_row = (RW % 64 == 0) ? RW / 64 : RW / 64 + 1; // number of output based on 64 bit

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = number_of_element_per_row

        ap_uint<ARW> addr_in = i_addr_strm.read();
        last = i_e_strm.read();
        ap_uint<ARW> addr_base = addr_in * number_of_element_per_row;

        for (int i = 0; i < number_of_element_per_row; i++) {
            o_addr_strm.write(addr_base++);
            o_e_strm.write(false);
        }
    }

    o_e_strm.write(true);
}

// split row into several 64 bit element
template <int RW, int ARW>
void split_row(hls::stream<ap_uint<RW> >& i_row_strm,
               hls::stream<bool>& i_e_strm,

               hls::stream<ap_uint<64> >& o_row_strm,
               hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_element_per_row = (RW % 64 == 0) ? RW / 64 : RW / 64 + 1; // number of output based on 64 bit

    bool last = i_e_strm.read();
    ap_uint<64 * number_of_element_per_row> row_temp = 0;

    while (!last) {
#pragma HLS PIPELINE II = number_of_element_per_row
        row_temp = i_row_strm.read();
        last = i_e_strm.read();
        ap_uint<4> mux = 0;

        for (int i = 0; i < number_of_element_per_row; i++) {
            ap_uint<64> element = row_temp((mux + 1) * 64 - 1, mux * 64);
            mux++;

            o_row_strm.write(element);
            o_e_strm.write(false);
        }
#ifndef __SYNTHESIS__
        cnt++;
#endif
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
    std::cout << std::dec << "RW= " << RW << " II=" << number_of_element_per_row << std::endl;
    std::cout << std::dec << "STB write " << cnt << " rows" << std::endl;
#endif
#endif
}

// write row to HBM/DDR
template <int RW, int ARW>
void write_row(ap_uint<64>* stb_buf,
               hls::stream<ap_uint<ARW> >& i_addr_strm,
               hls::stream<ap_uint<64> >& i_row_strm,
               hls::stream<bool>& i_e_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<ARW> addr = i_addr_strm.read();
        ap_uint<64> row = i_row_strm.read();
        last = i_e_strm.read();

        stb_buf[addr] = row;
    }
}

/// @brief Write s-table to HBM/DDR
template <int ARW, int RW>
void write_stb(ap_uint<64>* stb_buf,

               hls::stream<ap_uint<ARW> >& i_addr_strm,
               hls::stream<ap_uint<RW> >& i_row_strm,
               hls::stream<bool>& i_e_strm) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS STREAM variable = addr_strm depth = 512
#pragma HLS bind_storage variable = addr_strm type = fifo impl = bram
    hls::stream<bool> e2_strm;
#pragma HLS STREAM variable = e2_strm depth = 512
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    hls::stream<ap_uint<64> > row_strm;
#pragma HLS STREAM variable = row_strm depth = 512
#pragma HLS bind_storage variable = row_strm type = fifo impl = bram
    hls::stream<bool> e3_strm;
#pragma HLS STREAM variable = e3_strm depth = 512
#pragma HLS bind_storage variable = e3_strm type = fifo impl = srl

    join_v3::sc::duplicate_strm_end<bool>(i_e_strm, e0_strm, e1_strm);

    stb_addr_gen<RW, ARW>(i_addr_strm, e0_strm, addr_strm, e2_strm);

    split_row<RW, ARW>(i_row_strm, e1_strm, row_strm, e3_strm);

    write_row<RW, ARW>(stb_buf, addr_strm, row_strm, e3_strm);

    join_v3::sc::eliminate_strm_end<bool>(e2_strm);
}

// read row from HBM/DDR
template <int RW, int ARW>
void read_row(ap_uint<64>* stb_buf,
              hls::stream<ap_uint<ARW> >& i_addr_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<64> >& o_row_strm,
              hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<ARW> addr = i_addr_strm.read();
        last = i_e_strm.read();

        ap_uint<64> element = stb_buf[addr];
        o_row_strm.write(element);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// combine several 64 bit stream into one row
template <int RW, int ARW>
void combine_row(hls::stream<ap_uint<64> >& i_row_strm,
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<RW> >& o_row_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_element_per_row = (RW % 64 == 0) ? RW / 64 : RW / 64 + 1; // number of output based on 64 bit

    bool last = i_e_strm.read();
    ap_uint<64 * number_of_element_per_row> row_temp = 0;
    ap_uint<4> mux = 0;

    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<64> element = i_row_strm.read();
        last = i_e_strm.read();
        row_temp((mux + 1) * 64 - 1, mux * 64) = element;

        if (mux == number_of_element_per_row - 1) {
            ap_uint<RW> row = row_temp(RW - 1, 0);
            o_row_strm.write(row);
            o_e_strm.write(false);

            mux = 0;
        } else {
            mux++;
        }

#ifndef __SYNTHESIS__
        cnt++;
#endif
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
    std::cout << std::dec << "RW= " << RW << " II=" << number_of_element_per_row << std::endl;
    std::cout << std::dec << "STB read " << cnt / number_of_element_per_row << " rows" << std::endl;
#endif
#endif
}

template <int ARW, int RW>
void read_stb(ap_uint<64>* stb_buf,

              hls::stream<ap_uint<ARW> >& i_addr_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<RW> >& o_row_strm,
              hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS STREAM variable = addr_strm depth = 512
#pragma HLS bind_storage variable = addr_strm type = fifo impl = bram
    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 512
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    hls::stream<ap_uint<64> > row_strm;
#pragma HLS STREAM variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    stb_addr_gen<RW, ARW>(i_addr_strm, i_e_strm, addr_strm, e0_strm);

    read_row<RW, ARW>(stb_buf, addr_strm, e0_strm, row_strm, e1_strm);

    combine_row<RW, ARW>(row_strm, e1_strm, o_row_strm, o_e_strm);
}
} // namespace sc_tmp

namespace sc {

// ------------------------------------Read Write HBM------------------------------------

// read hash table from HBM/DDR
template <int ARW>
void read_bitmap(ap_uint<64>* htb_buf,
                 ap_uint<64> addr_shift,
                 ap_uint<64> length,

                 hls::stream<ap_uint<64> >& o_bitmap_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<64> bit_vector_addr = 0;
    ap_uint<64> htb_start_addr = addr_shift;
    ap_uint<64> htb_end_addr = addr_shift + length;

READ_BUFF_LOOP:
    for (int i = htb_start_addr; i < htb_end_addr; i++) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<64>
        ap_uint<64> bitmap_temp = htb_buf[i];
        o_bitmap_strm.write(bitmap_temp);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// write hash table to URAM
template <int ARW>
void write_uram(hls::stream<ap_uint<64> >& i_bitmap_strm,
                hls::stream<bool>& i_e_strm,

                ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    ap_uint<72> previous_bitmap = 0;
    int cnt0 = 0;
    int cnt1 = 0;
    int cnt_depth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

    ap_uint<72> bitmap = 0;
    ap_uint<64> bitmap_temp0 = 0;
    ap_uint<8> bitmap_temp1 = 0;

    ap_uint<ARW> bit_vector_addr = 0;

    bool last = i_e_strm.read();
COMBINE_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<64>
        bitmap_temp0 = i_bitmap_strm.read();
        last = i_e_strm.read();

        // combine
        bitmap(63, 0) = bitmap_temp0(63, 0);
        bitmap(71, 64) = bitmap_temp1(7, 0);

        // write
        bit_vector[bit_vector_addr] = bitmap;

#ifndef __SYNTHESIS__
        if (bitmap != 0) {
            if (previous_bitmap != bitmap) {
#ifdef DEBUG_HBM
                std::cout << std::hex << "read htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap << std::endl;
#endif
                cnt1++;
            }

            previous_bitmap = bitmap;
            cnt0++;
        }

        for (int i = 0; i < 16; i++) {
            if (bitmap(15, 0) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(31, 16) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(47, 32) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(63, 48) == i) {
                cnt_depth[i]++;
            }
        }
#endif

        bit_vector_addr++;
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG_HASH_CNT
    for (int i = 0; i < 16; i++) {
        std::cout << std::dec << "cnt of depth " << i << " = " << cnt_depth[i] << std::endl;
    }
#endif

#ifdef DEBUG_HBM
    std::cout << std::dec << "HTB read " << cnt0 << " non-empty lines, " << cnt1 << " distinct lines" << std::endl;
#endif
#endif
}

// read hash table from HBM/DDR to URAM
template <int ARW>
void read_htb(ap_uint<64>* htb_buf, ap_uint<64> addr_shift, ap_uint<64> length, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<64> > bitmap_strm;
#pragma HLS STREAM variable = bitmap_strm depth = 512
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = bram
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 512
#pragma HLS bind_storage variable = e_strm type = fifo impl = bram

    read_bitmap<ARW>(htb_buf, addr_shift, length,

                     bitmap_strm, e_strm);

    write_uram<ARW>(bitmap_strm, e_strm,

                    bit_vector);
}

// write hash table from URAM to HBM/DDR
template <int ARW>
void read_uram(ap_uint<64> length,

               hls::stream<ap_uint<72> >& o_bitmap_strm,
               hls::stream<bool>& o_e_strm,
               ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt0 = 0;
    unsigned int cnt1 = 0;
    ap_uint<72> previous_bitmap = 0;
#endif

    bool status = false;
    ap_uint<72> bitmap = 0;
    ap_uint<ARW> bit_vector_addr = 0;

READ_URAM_LOOP:
    for (int i = 0; i < length; i++) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = bit_vector inter false

        // read
        bitmap = bit_vector[bit_vector_addr];

#ifndef __SYNTHESIS__
        if (bitmap != 0) {
            if (previous_bitmap != bitmap) {
#ifdef DEBUG_HBM
                std::cout << std::hex << "write htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap
                          << std::endl;
#endif
                cnt1++;
            }
            previous_bitmap = bitmap;
            cnt0++;
        }
#endif

        bit_vector_addr++;

        o_bitmap_strm.write(bitmap);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
    std::cout << std::dec << "HTB update " << cnt0 << " non-zero lines, " << cnt1 << " distinct lines" << std::endl;
#endif
#endif
}

// write hash table to HBM/DDR
template <int ARW>
void write_bitmap(hls::stream<ap_uint<72> >& i_bitmap_strm,
                  hls::stream<bool>& i_e_strm,

                  ap_uint<64> addr_shift,
                  ap_uint<64>* htb_buf) {
#pragma HLS INLINE off

    ap_uint<64> htb_addr = addr_shift;
    ap_uint<72> bitmap;
    ap_uint<64> bitmap_temp;

    bool last = i_e_strm.read();
SPLIT_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read
        bitmap = i_bitmap_strm.read();
        last = i_e_strm.read();

        // convert 72bit to 64bit
        bitmap_temp(63, 0) = bitmap(63, 0);

        // write strm
        htb_buf[htb_addr] = bitmap_temp;
        htb_addr++;
    }
}

// write hash table from URAM to HBM/DDR
template <int ARW>
void write_htb(ap_uint<64>* htb_buf, ap_uint<64> addr_shift, ap_uint<64> length, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<72> > bitmap_strm;
#pragma HLS STREAM variable = bitmap_strm depth = 512
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = bram
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 512
#pragma HLS bind_storage variable = e_strm type = fifo impl = bram

    read_uram<ARW>(length, bitmap_strm, e_strm, bit_vector);

    write_bitmap<ARW>(bitmap_strm, e_strm, addr_shift, htb_buf);
}

// read s-table from HBM/DDR in random addr
template <int ARW, int RW>
void read_stb(ap_uint<64>* stb_buf,

              hls::stream<ap_uint<ARW> >& i_addr_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<RW> >& o_row_strm,
              hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int number_of_input_per_row = (RW % 64 == 0) ? (RW / 64) : (RW / 64 + 1); // number of output based on 64 bit

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    bool last = false;
READ_BUFF_LOOP:
    do {
#pragma HLS pipeline II = number_of_input_per_row

        ap_uint<ARW> addr = i_addr_strm.read();
        last = i_e_strm.read();
        ap_int<64 * number_of_input_per_row> srow;

        ap_uint<64> addr_base = addr * number_of_input_per_row;
        for (int i = 0; i < number_of_input_per_row; i++) {
            srow((i + 1) * 64 - 1, i * 64) = stb_buf[addr_base + i];
        }

        if (!last) {
            o_row_strm.write(srow);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
            if (cnt < 10) std::cout << std::hex << "read_stb: read_addr=" << addr << ", row=" << srow << std::endl;
#endif
            cnt++;
#endif
        }
    } while (!last);

    o_row_strm.write(0);
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "RW= " << RW << " II=" << number_of_input_per_row << std::endl;
    std::cout << std::dec << "STB read " << cnt << " block" << std::endl;
#endif
}

// ---------------------------------------hash-----------------------------------------------

// split bloom_filter_hash and key_hash
template <int KEY_IN, int KEY_T1, int KEY_T2, int KEY_T3, int KEY_T4>
void split_unit(hls::stream<ap_uint<KEY_IN> >& kin_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<KEY_T1> >& kout1_strm,
                hls::stream<ap_uint<KEY_T2> >& kout2_strm,
                hls::stream<ap_uint<KEY_T3> >& kout3_strm,
                hls::stream<ap_uint<KEY_T4> >& kout4_strm,
                hls::stream<bool>& out_e0_strm,
                hls::stream<bool>& out_e1_strm) {
    bool e = in_e_strm.read();
    ap_uint<KEY_T1> keyout1;
    ap_uint<KEY_T2> keyout2;
    ap_uint<KEY_T3> keyout3;
    ap_uint<KEY_T4> keyout4;
    ap_uint<KEY_IN> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_t3 = keyout3.length();
    uint64_t width_t4 = keyout4.length();
    uint64_t width_in = keyin.length();
    while (!e) {
#pragma HLS pipeline II = 1
        keyin = kin_strm.read();
        e = in_e_strm.read();
        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_t1 + width_t2 - 1, width_t1);
        keyout3 = keyin.range(width_t1 + width_t2 + width_t3 - 1, width_t1 + width_t2);
        keyout4 = keyin.range(width_in - 1, width_t1 + width_t2 + width_t3);
        kout1_strm.write(keyout1);
        kout2_strm.write(keyout2);
        kout3_strm.write(keyout3);
        kout4_strm.write(keyout4);

        out_e0_strm.write(false);
        out_e1_strm.write(false);
    }
    out_e0_strm.write(true);
    out_e1_strm.write(true);
}

// calculate hash value for join and bloom filter
template <int KEYW, int HASHW, int BF_HASH_NM, int BFW>
void hash_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<HASHW + BF_HASH_NM * BFW> >& o_hash_strm,
                  hls::stream<ap_uint<KEYW> >& o_key_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    hls::stream<ap_uint<KEYW> > key_strm_in;
#pragma HLS STREAM variable = key_strm_in depth = 8
#pragma HLS bind_storage variable = key_strm_in type = fifo impl = srl
    hls::stream<ap_uint<64> > hash_strm_out;
#pragma HLS STREAM variable = hash_strm_out depth = 8
#pragma HLS bind_storage variable = hash_strm_out type = fifo impl = srl

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    bool last = i_e_strm.read();
BUILD_HASH_LOOP:
    while (!last) {
#pragma HLS loop_tripcount min = 1 max = 1000
#pragma HLS PIPELINE II = 1

        bool blk = i_e_strm.empty() || i_key_strm.empty() || o_key_strm.full();
        if (!blk) {
            last = i_e_strm.read();
            ap_uint<KEYW> key = i_key_strm.read();

            o_key_strm.write(key);
            o_e_strm.write(false);

            // radix hash
            ap_uint<HASHW> hash_val0 = key(HASHW - 1, 0);

            // lookup3 hash
            key_strm_in.write(key);
            database::hashLookup3<KEYW>(key_strm_in, hash_strm_out);
            ap_uint<64> l_hash_val = hash_strm_out.read();
            ap_uint<HASHW> hash = l_hash_val(HASHW - 1, 0);

            // bloom filter
            ap_uint<BFW> bf_hash0 = key(BFW - 1, 0);
            ap_uint<BFW> bf_hash1 = l_hash_val(BFW - 1, 0);
            ap_uint<BFW> bf_hash2 = l_hash_val(63, 63 - BFW);

            ap_uint<BF_HASH_NM * BFW> bf_hash;
            ap_uint<HASHW + BF_HASH_NM * BFW> o_hash;

            if (BF_HASH_NM == 3) {
                bf_hash = (bf_hash0, bf_hash1, bf_hash2);
            } else if (BF_HASH_NM == 2) {
                bf_hash = (bf_hash1, bf_hash2);
            } else if (BF_HASH_NM == 1) {
                bf_hash = bf_hash1;
            }

            o_hash = (hash, bf_hash);
            o_hash_strm.write(o_hash);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            if (cnt < 10) {
                std::cout << std::hex << "hash wrapper: cnt=" << cnt << " key = " << key << " hash_val = " << hash
                          << std::endl;
            }
#endif
            cnt++;
#endif
        }
    }
    o_e_strm.write(true);
}

//------------------------------------------distribute--------------------------------------

// dispatch data to multiple PU based one the hash value, every PU with different hash_value.
template <int KEYW, int PW, int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int PU>
void dispatch(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<KEYW> > o_key_strm[PU],
              hls::stream<ap_uint<PW> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<HASHWH + HASHWL + BF_HASH_NM* BFW> hash_val = i_hash_strm.read();
        ap_uint<HASHWH> idx = hash_val(HASHWH + HASHWL + BF_HASH_NM * BFW - 1, HASHWL + BF_HASH_NM * BFW);
        ap_uint<HASHWL + BF_HASH_NM* BFW> hash_out = hash_val(HASHWL + BF_HASH_NM * BFW - 1, 0);

        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        o_key_strm[idx].write(key);
        o_pld_strm[idx].write(pld);
        o_hash_strm[idx].write(hash_out);
        o_e_strm[idx].write(false);
    }

    // for do_while in merge function, why not use while() in merge function?
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        // if add merge module, need uncomment
        o_key_strm[i].write(0);
        o_pld_strm[i].write(0);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
}

// dispatch data based on hash value to multiple PU.
template <int KEYW, int PW, int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int PU>
void dispatch_unit(hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,

                   hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                   hls::stream<ap_uint<PW> > o_pld_strm[PU],
                   hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > o_hash_strm[PU],
                   hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHWH + HASHWL + BF_HASH_NM * BFW> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS STREAM variable = key_strm depth = 8
#pragma HLS bind_storage variable = key_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 8

    hash_wrapper<KEYW, HASHWH + HASHWL, BF_HASH_NM, BFW>(i_key_strm, i_e_strm, hash_strm, key_strm, e_strm);

    dispatch<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(key_strm, i_pld_strm, hash_strm, e_strm, o_key_strm,
                                                            o_pld_strm, o_hash_strm, o_e_strm);
}

// dispatch data based on hash value to multiple PU.
template <int KEYW, int PW, int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int PU>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,

                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
    // 1st:build
    // 2nd:probe
    for (int i = 0; i < 2; i++) {
        dispatch_unit<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(i_key_strm, i_pld_strm, i_e_strm, o_key_strm,
                                                                     o_pld_strm, o_hash_strm, o_e_strm);
    }
}

// -------------------------------------build------------------------------------------

// scan small table to count hash collision
template <int HASHW, int BF_HASH_NM, int BFW, int KEYW, int PW, int S_PW, int ARW>
void build_htb(
    // input
    ap_uint<32>& overflow_length,
    ap_uint<32> depth,

    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // store stb in stb_buf
    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<KEYW + S_PW> >& o_base_row_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<ARW> >& o_overflow_addr_strm,
    hls::stream<ap_uint<KEYW + S_PW> >& o_overflow_row_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif

    ap_uint<72> elem = 0;
    ap_uint<72> base_elem = 0;
    ap_uint<72> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<HASHW - 2> array_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                            0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    ap_uint<32> max_overflow = 0;
    ap_uint<ARW> overflow_base = HASH_DEPTH + BF_HASH_NM * BLOOM_FILTER_DEPTH;

    bool last = i_e_strm.read();
PRE_BUILD_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector inter false

        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<S_PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // calculate index
        ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 2);
        ap_uint<2> bit_idx = hash_val(1, 0);

        // read hash counter and ++, prevent duplicate key
        if (array_idx == array_idx_temp[0]) {
            elem = elem_temp[0];
        } else if (array_idx == array_idx_temp[1]) {
            elem = elem_temp[1];
        } else if (array_idx == array_idx_temp[2]) {
            elem = elem_temp[2];
        } else if (array_idx == array_idx_temp[3]) {
            elem = elem_temp[3];
        } else if (array_idx == array_idx_temp[4]) {
            elem = elem_temp[4];
        } else if (array_idx == array_idx_temp[5]) {
            elem = elem_temp[5];
        } else if (array_idx == array_idx_temp[6]) {
            elem = elem_temp[6];
        } else if (array_idx == array_idx_temp[7]) {
            elem = elem_temp[7];
        } else {
            elem = htb_vector[array_idx];
        }

        // update && write new hash value
        ap_uint<16> v0 = elem(15, 0);
        ap_uint<16> v1 = elem(31, 16);
        ap_uint<16> v2 = elem(47, 32);
        ap_uint<16> v3 = elem(63, 48);

        bool cnt_overflow = false;
        bool v0_overflow = v0 == 65535;
        bool v1_overflow = v1 == 65535;
        bool v2_overflow = v2 == 65535;
        bool v3_overflow = v3 == 65535;

        ap_uint<16> v0a;
        ap_uint<16> v1a;
        ap_uint<16> v2a;
        ap_uint<16> v3a;
        ap_uint<16> hash_cnt;

        if (bit_idx == 0) {
            v0a = v0_overflow ? v0 : ap_uint<16>(v0 + 1);
            v1a = v1;
            v2a = v2;
            v3a = v3;

            hash_cnt = v0;
            cnt_overflow = v0_overflow;
        } else if (bit_idx == 1) {
            v0a = v0;
            v1a = v1_overflow ? v1 : ap_uint<16>(v1 + 1);
            v2a = v2;
            v3a = v3;

            hash_cnt = v1;
            cnt_overflow = v1_overflow;
        } else if (bit_idx == 2) {
            v0a = v0;
            v1a = v1;
            v2a = v2_overflow ? v2 : ap_uint<16>(v2 + 1);
            v3a = v3;

            hash_cnt = v2;
            cnt_overflow = v2_overflow;
        } else if (bit_idx == 3) {
            v0a = v0;
            v1a = v1;
            v2a = v2;
            v3a = v3_overflow ? v3 : ap_uint<16>(v3 + 1);

            hash_cnt = v3;
            cnt_overflow = v3_overflow;
        }

        base_elem(15, 0) = v0a;
        base_elem(31, 16) = v1a;
        base_elem(47, 32) = v2a;
        base_elem(63, 48) = v3a;

        // right shift temp
        for (int i = 7; i > 0; i--) {
            elem_temp[i] = elem_temp[i - 1];
            array_idx_temp[i] = array_idx_temp[i - 1];
        }
        elem_temp[0] = base_elem;
        array_idx_temp[0] = array_idx;

        // connect key with payload
        ap_uint<KEYW + S_PW> srow = 0;
        srow = (key, pld);

        // update hash table
        htb_vector[array_idx] = base_elem;

        // generate o_addr
        ap_uint<ARW> o_addr;

        if (hash_cnt >= depth) {
            // overflow
            o_addr = overflow_base + max_overflow;
            max_overflow++;

            o_overflow_addr_strm.write(o_addr);
            o_overflow_row_strm.write(srow);
            o_e1_strm.write(false);
        } else {
            // underflow
            o_addr = depth * hash_val + hash_cnt;

            o_base_addr_strm.write(o_addr);
            o_base_row_strm.write(srow);
            o_e0_strm.write(false);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
        if (hash_cnt >= depth) {
            std::cout << std::hex << "build_stb: over key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " v0=" << v0 << " v1=" << v1 << " v2=" << v2 << " v3=" << v3
                      << " ht_addr=" << array_idx << " ht=" << base_elem << " hash_cnt=" << hash_cnt
                      << " output stb_addr=" << o_addr << std::endl;
        } else {
            std::cout << std::hex << "build_stb: base key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " v0=" << v0 << " v1=" << v1 << " v2=" << v2 << " v3=" << v3
                      << " ht_addr=" << array_idx << " ht=" << base_elem << " hash_cnt=" << hash_cnt
                      << " output stb_addr=" << o_addr << std::endl;
        }
#endif

        ap_uint<ARW> old_val = (bit_idx == 0) ? v0 : ((bit_idx == 1) ? v1 : (bit_idx == 2) ? v2 : v3);
        if (old_val > max_col) max_col = old_val;
        cnt++;
#endif
    }

    overflow_length = max_overflow;

    o_e0_strm.write(true);
    o_e1_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
    std::cout << std::dec << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << std::dec << "collision probility " << max_col << std::endl;
#endif
#endif
}

// top function of hash join build
template <int HASHW, int BF_HASH_NM, int BFW, int KEYW, int PW, int S_PW, int ARW>
void build_wrapper(
    // input
    ap_uint<32> depth,
    ap_uint<32>& overflow_length,

    hls::stream<ap_uint<HASHW + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf,

    ap_uint<72>* htb_vector0,
    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1,
    ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHW> > hash_strm;
#pragma HLS stream variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    hls::stream<ap_uint<BFW> > bf_hash0_strm;
#pragma HLS stream variable = bf_hash0_strm depth = 8
#pragma HLS bind_storage variable = bf_hash0_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash1_strm;
#pragma HLS stream variable = bf_hash1_strm depth = 8
#pragma HLS bind_storage variable = bf_hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash2_strm;
#pragma HLS stream variable = bf_hash2_strm depth = 8
#pragma HLS bind_storage variable = bf_hash2_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + S_PW> > base_row_strm;
#pragma HLS stream variable = base_row_strm depth = 8
#pragma HLS bind_storage variable = base_row_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 8
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + S_PW> > overflow_row_strm;
#pragma HLS stream variable = overflow_row_strm depth = 8
#pragma HLS bind_storage variable = overflow_row_strm type = fifo impl = srl
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8

    // split row to hash and key
    split_unit<HASHW + BF_HASH_NM * BFW, BFW, BFW, BFW, HASHW>(i_hash_strm, i_e_strm,

                                                               bf_hash0_strm, bf_hash1_strm, bf_hash2_strm, hash_strm,
                                                               e0_strm, e1_strm);

    // update hash collision cnt && calculate stb addr
    build_htb<HASHW, BF_HASH_NM, BFW, KEYW, PW, S_PW, ARW>(
        overflow_length, depth, hash_strm, i_key_strm, i_pld_strm, e0_strm,

        base_addr_strm, base_row_strm, e2_strm, overflow_addr_strm, overflow_row_strm, e3_strm,

        htb_vector0);

    // update bloom filter
    details::bv_update_uram<BFW>(bf_hash0_strm, bf_hash1_strm, bf_hash2_strm, e1_strm,

                                 bf_vector0, bf_vector1, bf_vector2);

    sc_tmp::write_stb<ARW, KEYW + S_PW>(stb_buf, base_addr_strm, base_row_strm, e2_strm);

    sc_tmp::write_stb<ARW, KEYW + S_PW>(htb_buf, overflow_addr_strm, overflow_row_strm, e3_strm);
}

//---------------------------------------------------probe------------------------------------------------

// probe the hash table and output probe address
template <int HASHW, int BF_HASH_NM, int BFW, int KEYW, int PW, int ARW>
void probe_htb(
    // input status
    ap_uint<32> depth,
    ap_uint<32> overflow_length,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output to check htb
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e0_strm,

    // output probe base stb
    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<ARW> >& o_base_nm_strm,
    hls::stream<bool>& o_e1_strm,

    // output to probe overflow stb
    hls::stream<ap_uint<ARW> >& o_overflow_pointer_addr_strm,
    hls::stream<ap_uint<ARW> >& o_overflow_nm_strm,
    hls::stream<bool>& o_e2_strm,

    // base htb
    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int probe_hit = 0;
#endif

    // htb:
    //   |=========base_hash_counter========|==========bloom_filter_vector=========|========overflow_small_table========|
    // length:
    //   |<------------HASH_DEPTH---------->|<---BF_HASH_NM*BLOOM_FILTER_DEPTH---->|<-----------overflow_lens---------->|

    // stb:
    //   |===================base_area_of_small_table===================|
    // length:
    //   |<--------------------HASH_NUMBER*depth----------------------->|

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

    ap_uint<ARW> base_addr, overflow_addr;
    ap_uint<ARW> probe_nm, base_nm, overflow_nm;

    overflow_addr = HASH_DEPTH + BF_HASH_NM * BLOOM_FILTER_DEPTH;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector inter false

        // read large table
        ap_uint<HASHW> hash = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // calculate index for HTB
        ap_uint<HASHW> array_idx = hash(HASHW - 1, 2);
        ap_uint<2> bit_idx = hash(1, 0);

        // total hash count
        ap_uint<ARW> nm;
        ap_uint<72> base_bitmap = htb_vector[array_idx];

        if (bit_idx == 0) {
            nm = base_bitmap(15, 0);
        } else if (bit_idx == 1) {
            nm = base_bitmap(31, 16);
        } else if (bit_idx == 2) {
            nm = base_bitmap(47, 32);
        } else {
            nm = base_bitmap(63, 48);
        }

        // calculate number of probe strm for controling in probe unit
        if (nm > depth) {
            base_nm = depth;
            overflow_nm = overflow_length;
            probe_nm = depth + overflow_length;
        } else {
            base_nm = nm;
            overflow_nm = 0;
            probe_nm = nm;
        }

        if (nm > 0) {
            // write large table to join
            o_t_key_strm.write(key);
            o_t_pld_strm.write(pld);
            o_nm_strm.write(probe_nm);
            o_e0_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "probe_base_htb: base_id=" << probe_hit << " nm=" << nm << " t_key=" << key
                      << " t_pld=" << pld << std::endl;
            probe_hit++;
#endif
#endif
        }

        base_addr = hash * depth;
        if (base_nm > 0) {
            o_base_addr_strm.write(base_addr);
            o_base_nm_strm.write(base_nm);
            o_e1_strm.write(false);
        }

        if (overflow_nm > 0) {
            o_overflow_pointer_addr_strm.write(overflow_addr);
            o_overflow_nm_strm.write(overflow_nm);
            o_e2_strm.write(false);
        }
    }
    o_e0_strm.write(true);

    // for do-while in probe
    o_base_addr_strm.write(0);
    o_base_nm_strm.write(0);
    o_e1_strm.write(true);

    o_overflow_pointer_addr_strm.write(0);
    o_overflow_nm_strm.write(0);
    o_e2_strm.write(true);
}

// generate probe addr
template <int ARW>
void probe_addr_gen(
    // input
    hls::stream<ap_uint<ARW> >& i_addr_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output
    hls::stream<ap_uint<ARW> >& o_read_addr_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    bool is_first_loop = true;
    ap_uint<ARW> addr;
    ap_uint<ARW> nm = 0;
    bool last = false;

    do {
#pragma HLS pipeline II = 1

        if (!last && is_first_loop) {
            addr = i_addr_strm.read();
            nm = i_nm_strm.read();
            last = i_e_strm.read();

            is_first_loop = false;
        } else if (nm > 0) {
            nm--;
            o_read_addr_strm.write(addr);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "probe_addr_gen: probe_addr=" << addr << std::endl;
#endif
#endif

            addr++;
        } else if (!last && nm == 0) {
            is_first_loop = true;
        }
    } while (!last || nm != 0);

    o_read_addr_strm.write(0);
    o_e_strm.write(true);
}

// split row to key and pld
template <int KEY_IN, int KEY_T1, int KEY_T2>
void split_unit(hls::stream<ap_uint<KEY_IN> >& kin_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<KEY_T1> >& kout1_strm,
                hls::stream<ap_uint<KEY_T2> >& kout2_strm,
                hls::stream<bool>& out_e_strm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif

    bool e = false;
    ap_uint<KEY_T1> keyout1;
    ap_uint<KEY_T2> keyout2;
    ap_uint<KEY_IN> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_in = keyin.length();

    do {
#pragma HLS pipeline II = 1
        keyin = kin_strm.read();
        e = in_e_strm.read();

        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_in - 1, width_t1);

        if (!e) {
            kout1_strm.write(keyout1);
            kout2_strm.write(keyout2);
            out_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "split: cnt=" << cnt << std::endl;
#endif
            cnt++;
#endif
        }
    } while (!e);

    out_e_strm.write(true);
}

// probe stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void probe_stb(
    // input
    hls::stream<ap_uint<ARW> >& i_base_stb_addr_strm,
    hls::stream<ap_uint<ARW> >& i_base_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,

    // HBM
    ap_uint<64>* stb_buf) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for read_stb
    hls::stream<ap_uint<ARW> > read_addr_strm;
#pragma HLS stream variable = read_addr_strm depth = 8
#pragma HLS bind_storage variable = read_addr_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    // for split probed stb
    hls::stream<ap_uint<KEYW + S_PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // eliminate end strm
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    // generate read addr from base addr
    probe_addr_gen<ARW>(i_base_stb_addr_strm, i_base_nm_strm, i_e_strm,

                        read_addr_strm, e0_strm);

    // read HBM to get base stb
    join_v4::sc::read_stb<ARW, KEYW + S_PW>(stb_buf, read_addr_strm, e0_strm, row_strm, e1_strm);

    // split base stb to key and pld
    split_unit<KEYW + S_PW, S_PW, KEYW>(row_strm, e1_strm, o_base_s_pld_strm, o_base_s_key_strm, e2_strm);

    // eleminate end signal of overflow unit
    join_v3::sc::eliminate_strm_end<bool>(e2_strm);
}

// check valid value generated by bloom filter, filter invalid large table
template <int HASHW, int KEYW, int PW, int T_PW>
void filter_invalid_ltb(hls::stream<ap_uint<HASHW> >& i_hash_strm,
                        hls::stream<ap_uint<KEYW> >& i_key_strm,
                        hls::stream<ap_uint<PW> >& i_pld_strm,
                        hls::stream<bool>& i_valid_strm,
                        hls::stream<bool>& i_e_strm,

                        hls::stream<ap_uint<HASHW> >& o_hash_strm,
                        hls::stream<ap_uint<KEYW> >& o_key_strm,
                        hls::stream<ap_uint<T_PW> >& o_pld_strm,
                        hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1

        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<T_PW> pld = i_pld_strm.read();
        bool valid = i_valid_strm.read();

        last = i_e_strm.read();

        if (valid) {
#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "pass bloom filter: hash=" << hash_val << " key=" << key << " pld=" << pld
                      << std::endl;
#endif
#endif

            o_hash_strm.write(hash_val);
            o_key_strm.write(key);
            o_pld_strm.write(pld);

            o_e_strm.write(false);
        }
    }
    o_e_strm.write(true);
}

// top function of probe
template <int HASHWL, int BF_HASH_NM, int BFW, int KEYW, int PW, int S_PW, int T_PW, int ARW>
void probe_wrapper(
    // flag
    ap_uint<32> depth,
    ap_uint<32> overflow_length,

    // input large table
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf,
    ap_uint<72>* htb_vector0,
    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1,
    ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for split key hash and bloom filter hash
    hls::stream<ap_uint<HASHWL> > hash0_strm;
#pragma HLS stream variable = hash0_strm depth = 32
#pragma HLS bind_storage variable = hash0_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash0_strm;
#pragma HLS stream variable = bf_hash0_strm depth = 8
#pragma HLS bind_storage variable = bf_hash0_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash1_strm;
#pragma HLS stream variable = bf_hash1_strm depth = 8
#pragma HLS bind_storage variable = bf_hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash2_strm;
#pragma HLS stream variable = bf_hash2_strm depth = 8
#pragma HLS bind_storage variable = bf_hash2_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    // for check bloom filter
    hls::stream<bool> valid_strm;
#pragma HLS stream variable = valid_strm depth = 8
#pragma HLS bind_storage variable = valid_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // filter invalid ltb
    hls::stream<ap_uint<HASHWL> > hash1_strm;
#pragma HLS stream variable = hash1_strm depth = 8
#pragma HLS bind_storage variable = hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS stream variable = key_strm depth = 8
#pragma HLS bind_storage variable = key_strm type = fifo impl = srl
    hls::stream<ap_uint<T_PW> > pld_strm;
#pragma HLS stream variable = pld_strm depth = 8
#pragma HLS bind_storage variable = pld_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > base_nm_strm;
#pragma HLS stream variable = base_nm_strm depth = 8
#pragma HLS bind_storage variable = base_nm_strm type = fifo impl = srl
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8
#pragma HLS bind_storage variable = e3_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 32
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > overflow_nm_strm;
#pragma HLS stream variable = overflow_nm_strm depth = 32
#pragma HLS bind_storage variable = overflow_nm_strm type = fifo impl = srl
    hls::stream<bool> e4_strm;
#pragma HLS stream variable = e4_strm depth = 32
#pragma HLS bind_storage variable = e4_strm type = fifo impl = srl

    // split bf hash and key hash
    database::splitCol<HASHWL + BF_HASH_NM * BFW, BFW, BFW, BFW, HASHWL>(i_hash_strm, i_e_strm,

                                                                         bf_hash0_strm, bf_hash1_strm, bf_hash2_strm,
                                                                         hash0_strm, e0_strm);

    // check bloom filter
    database::details::bv_check_uram<BFW>(bf_hash0_strm, bf_hash1_strm, bf_hash2_strm, e0_strm, bf_vector0, bf_vector1,
                                          bf_vector2, valid_strm, e1_strm);

    // filter invalid large table
    filter_invalid_ltb<HASHWL, KEYW, PW, T_PW>(hash0_strm, i_key_strm, i_pld_strm, valid_strm, e1_strm,

                                               hash1_strm, key_strm, pld_strm, e2_strm);

    // probe htb to get probe addr and length
    probe_htb<HASHWL, BF_HASH_NM, BFW, KEYW, T_PW, ARW>(depth, overflow_length,

                                                        hash1_strm, key_strm, pld_strm, e2_strm,

                                                        o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e0_strm,

                                                        base_addr_strm, base_nm_strm, e3_strm, overflow_addr_strm,
                                                        overflow_nm_strm, e4_strm,

                                                        htb_vector0);

    // probe stb stored in base area
    probe_stb<KEYW, S_PW, ARW>(base_addr_strm, base_nm_strm, e3_strm,

                               o_base_s_key_strm, o_base_s_pld_strm,

                               stb_buf);

    // probe stb stored in overflow area
    probe_stb<KEYW, S_PW, ARW>(overflow_addr_strm, overflow_nm_strm, e4_strm,

                               o_overflow_s_key_strm, o_overflow_s_pld_strm,

                               htb_buf);
}

//----------------------------------------------PU------------------------------------------------

// initiate htb
template <int HASHW, int ARW>
void initiate_htb(ap_uint<72>* htb_vector0) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------Initialize htb--------------------" << std::endl;
#endif
#endif

// build_id==0, the firsr time to build, initialize uram to zero
HTB0_INIT_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector0 inter false

        htb_vector0[i] = 0;
    }
}

// initiate bf vector
template <int BF_HASH_NM, int BFW, int ARW>
void initiate_bf(ap_uint<72>* bf_vector0, ap_uint<72>* bf_vector1, ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off

    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------Initialize bloom filter-----------" << std::endl;
#endif
#endif

// build_id==0, the firsr time to build, initialize uram to zero
BF_INIT_LOOP:
    for (int i = 0; i < BLOOM_FILTER_DEPTH; i++) {
        bf_vector0[i] = 0;
        bf_vector1[i] = 0;
        bf_vector2[i] = 0;
    }
}

template <int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int KEYW, int PW, int S_PW, int T_PW, int ARW>
void build_probe_wrapper(
    // input status
    ap_uint<32> depth,

    // input table
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e_strm,

    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf) {
#pragma HLS INLINE off

    // allocate uram storage
    const int HASH_DEPTH = 1 << (HASHWL - 2);
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

    ap_uint<32> overflow_length;

#ifndef __SYNTHESIS__

    ap_uint<72>* htb_vector0;
    htb_vector0 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));

    ap_uint<72>* bf_vector0;
    ap_uint<72>* bf_vector1;
    ap_uint<72>* bf_vector2;
    bf_vector0 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));
    bf_vector1 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));
    bf_vector2 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));

    //#ifdef DEBUG
    std::cout << std::hex << "HASH_DEPTH=" << HASH_DEPTH << " BLOOM_FILTER_DEPTH=" << BLOOM_FILTER_DEPTH << std::endl;
//#endif

#else

    ap_uint<72> htb_vector0[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = htb_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = htb_vector0 type = ram_2p impl = uram

    ap_uint<72> bf_vector0[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector0 type = ram_2p impl = uram

    ap_uint<72> bf_vector1[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector1 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector1 type = ram_2p impl = uram

    ap_uint<72> bf_vector2[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector2 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector2 type = ram_2p impl = uram

#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "HASH_DEPTH=" << HASH_DEPTH << " BLOOM_FILTER_DEPTH=" << BLOOM_FILTER_DEPTH << std::endl;
#endif
#endif

    // initilize htb vector
    initiate_htb<HASHWL, ARW>(htb_vector0);

    // initilize bf vector
    initiate_bf<BF_HASH_NM, BFW, ARW>(bf_vector0, bf_vector1, bf_vector2);

// build start
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "----------------------build------------------------" << std::endl;
#endif
#endif

    build_wrapper<HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, ARW>(
        // input status
        depth, overflow_length,

        // input s-table
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        // HBM/DDR
        htb_buf, stb_buf,

        // URAM
        htb_vector0, bf_vector0, bf_vector1, bf_vector2);

// probe start
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------------Probe-----------------------" << std::endl;
#endif
#endif

    probe_wrapper<HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, T_PW, ARW>(
        depth, overflow_length,

        // input large table
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        // output for join
        o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e_strm,

        o_base_s_key_strm, o_base_s_pld_strm, o_overflow_s_key_strm, o_overflow_s_pld_strm,

        htb_buf, stb_buf, htb_vector0, bf_vector0, bf_vector1, bf_vector2);

#ifndef __SYNTHESIS__

    free(htb_vector0);
    free(bf_vector0);
    free(bf_vector1);
    free(bf_vector2);

#endif
}

/// @brief compare key, if match output joined row
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit(
#ifndef __SYNTHESIS__
    int pu_id,
#endif

    ap_uint<32> depth,

    // input large table
    hls::stream<ap_uint<KEYW> >& i_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e0_strm,

    // input small table
    hls::stream<ap_uint<KEYW> >& i_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& i_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_overflow_s_pld_strm,

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o_j_strm,
    hls::stream<bool>& o_e_strm) {

#pragma HLS INLINE off

    ap_uint<KEYW> s_key;
    ap_uint<S_PW> s_pld;
    ap_uint<KEYW> t_key;
    ap_uint<T_PW> t_pld;
    ap_uint<KEYW + S_PW + T_PW> j;

#ifndef __SYNTHESIS__
    unsigned int cnt0 = 0;
    unsigned int cnt1 = 0;
    unsigned int cnt2 = 0;
    unsigned int cnt3 = 0;

    bool hit_failed;
#endif

    bool t_last = i_e0_strm.read();
JOIN_LOOP:
    while (!t_last) {
        t_key = i_t_key_strm.read();
        t_pld = i_t_pld_strm.read();
        ap_uint<ARW> nm = i_nm_strm.read();
        t_last = i_e0_strm.read();

        ap_uint<ARW> base_nm, overflow_nm;
        if (nm > depth) {
            base_nm = depth;
            overflow_nm = nm - depth;
        } else {
            base_nm = nm;
            overflow_nm = 0;
        }

#ifndef __SYNTHESIS__
        hit_failed = true;
        cnt2++;
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (t_key == 3680482 || t_key == 3691265 || t_key == 4605699 || t_key == 4987782)
            std::cout << std::hex << "Join: t_key=" << t_key << " nm=" << nm << std::endl;
#endif
#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG_JOIN
        std::cout << std::hex << "Join: t_key=" << t_key << " nm=" << nm << " base_nm=" << base_nm
                  << " overflow_nm=" << overflow_nm << std::endl;
#endif
#endif

    JOIN_COMPARE_LOOP:
        while (base_nm > 0 || overflow_nm > 0) {
#pragma HLS PIPELINE II = 1

            if (base_nm > 0) {
                s_key = i_base_s_key_strm.read();
                s_pld = i_base_s_pld_strm.read();

                base_nm--;
            } else if (overflow_nm > 0) {
                s_key = i_overflow_s_key_strm.read();
                s_pld = i_overflow_s_pld_strm.read();

                overflow_nm--;
            }

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
            if (s_key == 3680482 || s_key == 3691265 || s_key == 4605699 || s_key == 4987782)
                std::cout << std::hex << "Join: s_key=" << s_key << " s_pld" << s_pld << " base_nm=" << base_nm
                          << " overflow_nm=" << overflow_nm << std::endl;
#endif
#endif

            if (s_key == t_key) {
                // generate joion result
                j(KEYW + S_PW + T_PW - 1, S_PW + T_PW) = s_key;
                if (S_PW > 0) {
                    j(S_PW + T_PW - 1, T_PW) = s_pld;
                }
                if (T_PW > 0) {
                    j(T_PW - 1, 0) = t_pld;
                }

                o_j_strm.write(j);
                o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_JOIN
                std::cout << std::hex << "Match nm=" << nm << " key=" << s_key << " s_pld=" << s_pld
                          << " t_pld=" << t_pld << " J_res=" << j << std::endl;
#endif
                cnt0++;
                hit_failed = false;
            } else {
#ifdef DEBUG_JOIN
                std::cout << std::hex << "Mismatch nm=" << nm << " s_key=" << s_key << " t_key=" << t_key
                          << " s_pld=" << s_pld << " t_pld=" << t_pld << std::endl;
#endif
                cnt1++;
#endif
            }
        }

#ifndef __SYNTHESIS__
        if (hit_failed) {
            cnt3++;
        }
#endif
    }

    // for do while in collect
    o_j_strm.write(0);
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "Join Unit output " << cnt0 << " rows, mismatch " << cnt1 << " rows" << std::endl;
    std::cout << std::dec << "Join Unit hit " << cnt2 << " times, hit failed " << cnt3 << " times" << std::endl;
#endif
#endif
}

} // namespace sc

namespace bp {
//------------------------------------read status--------------------------------------
template <int PU>
void read_status(hls::stream<ap_uint<32> >& pu_begin_status_strms,
                 ap_uint<32>& build_id,
                 ap_uint<32>& probe_id,
                 ap_uint<32>& depth,
                 ap_uint<32> pu_start_addr[PU]) {
    // controlling parameters
    build_id = pu_begin_status_strms.read();
    probe_id = pu_begin_status_strms.read();
    depth = pu_begin_status_strms.read();

    // discard joined number of last probe
    ap_uint<32> join_number = pu_begin_status_strms.read();

    // PU status
    for (int i = 0; i < PU; i++) {
#pragma HLS pipeline II = 1
        pu_start_addr[i] = pu_begin_status_strms.read();

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::dec << "pu_start_addr[" << i << "]=" << pu_start_addr[i] << std::endl;
#endif
#endif
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "build_id=" << build_id << " probe_id=" << probe_id << " depth=" << depth
              << " join_num=" << join_number << std::endl;
#endif
#endif
}

//------------------------------------write status--------------------------------------
template <int PU>
void write_status(bool& build_probe_flag,
                  hls::stream<ap_uint<32> >& pu_end_status_strms,
                  ap_uint<32> build_id,
                  ap_uint<32> probe_id,
                  ap_uint<32> depth,
                  ap_uint<32> join_num,
                  ap_uint<32> pu_end_addr[PU]) {
    // controlling parameters
    ap_uint<32> new_build_id, new_probe_id;

    if (build_probe_flag) {
        new_build_id = build_id;
        new_probe_id = probe_id + 1;
    } else {
        new_build_id = build_id + 1;
        new_probe_id = 0;
    }

    pu_end_status_strms.write(new_build_id);
    pu_end_status_strms.write(new_probe_id);
    pu_end_status_strms.write(depth);
    pu_end_status_strms.write(join_num);

    // PU status
    for (int i = 0; i < PU; i++) {
#pragma HLS pipeline II = 1
        pu_end_status_strms.write(pu_end_addr[i]);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::dec << "pu_end_addr[" << i << "]=" << pu_end_addr[i] << std::endl;
#endif
#endif
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "new_build_id=" << new_build_id << " new_probe_id=" << new_probe_id << " depth=" << depth
              << " join_num=" << join_num << std::endl;
#endif
#endif
}

// ------------------------------------Read Write HBM------------------------------------

// read hash table from HBM/DDR
template <int ARW>
void read_bitmap(ap_uint<64>* htb_buf,
                 ap_uint<64> addr_shift,
                 ap_uint<64> length,

                 hls::stream<ap_uint<64> >& o_bitmap_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<64> bit_vector_addr = 0;
    ap_uint<64> htb_start_addr = addr_shift;
    ap_uint<64> htb_end_addr = addr_shift + length;

READ_BUFF_LOOP:
    for (int i = htb_start_addr; i < htb_end_addr; i++) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<64>
        ap_uint<64> bitmap_temp = htb_buf[i];
        o_bitmap_strm.write(bitmap_temp);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// write hash table to URAM
template <int ARW>
void write_uram(hls::stream<ap_uint<64> >& i_bitmap_strm,
                hls::stream<bool>& i_e_strm,

                ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    ap_uint<72> previous_bitmap = 0;
    int cnt0 = 0;
    int cnt1 = 0;
    int cnt_depth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

    ap_uint<72> bitmap = 0;
    ap_uint<64> bitmap_temp0 = 0;
    ap_uint<8> bitmap_temp1 = 0;

    ap_uint<ARW> bit_vector_addr = 0;

    bool last = i_e_strm.read();
COMBINE_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<64>
        bitmap_temp0 = i_bitmap_strm.read();
        last = i_e_strm.read();

        // combine
        bitmap(63, 0) = bitmap_temp0(63, 0);
        bitmap(71, 64) = bitmap_temp1(7, 0);

        // write
        bit_vector[bit_vector_addr] = bitmap;

#ifndef __SYNTHESIS__
        if (bitmap != 0) {
            if (previous_bitmap != bitmap) {
#ifdef DEBUG_HBM
                std::cout << std::hex << "read htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap << std::endl;
#endif
                cnt1++;
            }

            previous_bitmap = bitmap;
            cnt0++;
        }

        for (int i = 0; i < 16; i++) {
            if (bitmap(15, 0) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(31, 16) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(47, 32) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(63, 48) == i) {
                cnt_depth[i]++;
            }
        }
#endif

        bit_vector_addr++;
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG_HASH_CNT
    for (int i = 0; i < 16; i++) {
        std::cout << std::dec << "cnt of depth " << i << " = " << cnt_depth[i] << std::endl;
    }
#endif

#ifdef DEBUG_HBM
    std::cout << std::dec << "HTB read " << cnt0 << " non-empty lines, " << cnt1 << " distinct lines" << std::endl;
#endif
#endif
}

// read hash table from HBM/DDR to URAM
template <int ARW>
void read_htb(ap_uint<64>* htb_buf, ap_uint<64> addr_shift, ap_uint<64> length, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<64> > bitmap_strm;
#pragma HLS STREAM variable = bitmap_strm depth = 512
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = bram
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 512
#pragma HLS bind_storage variable = e_strm type = fifo impl = bram

    read_bitmap<ARW>(htb_buf, addr_shift, length,

                     bitmap_strm, e_strm);

    write_uram<ARW>(bitmap_strm, e_strm,

                    bit_vector);
}

// write hash table from URAM to HBM/DDR
template <int ARW>
void read_uram(ap_uint<64> length,

               hls::stream<ap_uint<72> >& o_bitmap_strm,
               hls::stream<bool>& o_e_strm,
               ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt0 = 0;
    unsigned int cnt1 = 0;
    ap_uint<72> previous_bitmap = 0;
#endif

    bool status = false;
    ap_uint<72> bitmap = 0;
    ap_uint<ARW> bit_vector_addr = 0;

READ_URAM_LOOP:
    for (int i = 0; i < length; i++) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = bit_vector inter false

        // read
        bitmap = bit_vector[bit_vector_addr];

#ifndef __SYNTHESIS__
        if (bitmap != 0) {
            if (previous_bitmap != bitmap) {
#ifdef DEBUG_HBM
                std::cout << std::hex << "write htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap
                          << std::endl;
#endif
                cnt1++;
            }
            previous_bitmap = bitmap;
            cnt0++;
        }
#endif

        bit_vector_addr++;

        o_bitmap_strm.write(bitmap);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_HBM
    std::cout << std::dec << "HTB update " << cnt0 << " non-zero lines, " << cnt1 << " distinct lines" << std::endl;
#endif
#endif
}

// write hash table to HBM/DDR
template <int ARW>
void write_bitmap(hls::stream<ap_uint<72> >& i_bitmap_strm,
                  hls::stream<bool>& i_e_strm,

                  ap_uint<64> addr_shift,
                  ap_uint<64>* htb_buf) {
#pragma HLS INLINE off

    ap_uint<64> htb_addr = addr_shift;
    ap_uint<72> bitmap;
    ap_uint<64> bitmap_temp;

    bool last = i_e_strm.read();
SPLIT_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read
        bitmap = i_bitmap_strm.read();
        last = i_e_strm.read();

        // convert 72bit to 64bit
        bitmap_temp(63, 0) = bitmap(63, 0);

        // write strm
        htb_buf[htb_addr] = bitmap_temp;
        htb_addr++;
    }
}

// write hash table from URAM to HBM/DDR
template <int ARW>
void write_htb(ap_uint<64>* htb_buf, ap_uint<64> addr_shift, ap_uint<64> length, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<72> > bitmap_strm;
#pragma HLS STREAM variable = bitmap_strm depth = 512
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = bram
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 512
#pragma HLS bind_storage variable = e_strm type = fifo impl = bram

    read_uram<ARW>(length, bitmap_strm, e_strm, bit_vector);

    write_bitmap<ARW>(bitmap_strm, e_strm, addr_shift, htb_buf);
}

// generate read addr for reading stb row
template <int ARW>
void read_addr_gen(
    // input
    ap_uint<32> start_addr,
    ap_uint<32> length,

    // output
    hls::stream<ap_uint<ARW> >& o_addr_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<ARW> addr_temp = start_addr;

LOOP_ADDR_GEN:
    for (int i = 0; i < length; i++) {
#pragma HLS pipeline II = 1
        o_addr_strm.write(addr_temp);
        o_e_strm.write(false);

        addr_temp++;
    }
    o_e_strm.write(true);
}

//------------------------------------------distribute--------------------------------------

// dispatch data to multiple PU based one the hash value, every PU with different hash_value.
template <int KEYW, int PW, int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int PU>
void dispatch(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<KEYW> > o_key_strm[PU],
              hls::stream<ap_uint<PW> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<HASHWH + HASHWL + BF_HASH_NM* BFW> hash_val = i_hash_strm.read();
        ap_uint<HASHWH> idx = hash_val(HASHWH + HASHWL + BF_HASH_NM * BFW - 1, HASHWL + BF_HASH_NM * BFW);
        ap_uint<HASHWL + BF_HASH_NM* BFW> hash_out = hash_val(HASHWL + BF_HASH_NM * BFW - 1, 0);

        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        o_key_strm[idx].write(key);
        o_pld_strm[idx].write(pld);
        o_hash_strm[idx].write(hash_out);
        o_e_strm[idx].write(false);
    }

    // for do_while in merge function, why not use while() in merge function?
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        // if add merge module, need uncomment
        o_key_strm[i].write(0);
        o_pld_strm[i].write(0);
        o_hash_strm[i].write(0);
        o_e_strm[i].write(true);
    }
}

// dispatch data based on hash value to multiple PU.
template <int KEYW, int PW, int HASHWH, int HASHWL, int BF_HASH_NM, int BFW, int PU>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,

                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHWH + HASHWL + BF_HASH_NM * BFW> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS STREAM variable = key_strm depth = 8
#pragma HLS bind_storage variable = key_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 8

    join_v4::sc::hash_wrapper<KEYW, HASHWH + HASHWL, BF_HASH_NM, BFW>(i_key_strm, i_e_strm, hash_strm, key_strm,
                                                                      e_strm);

    join_v4::sc::dispatch<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(key_strm, i_pld_strm, hash_strm, e_strm,
                                                                         o_key_strm, o_pld_strm, o_hash_strm, o_e_strm);
}

// -------------------------------------build------------------------------------------

// scan small table to count hash collision
template <int HASHW, int KEYW, int PW, int ARW>
void build_base_htb(
    // input
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,
    ap_uint<32> depth,

    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // store stb in stb_buf
    hls::stream<ap_uint<ARW> >& o_addr_strm,
    hls::stream<ap_uint<KEYW + PW> >& o_row_strm,
    hls::stream<bool>& o_e0_strm,

    // documenting terrible cnt overflow (>65535)
    hls::stream<ap_uint<HASHW> >& o_overflow_hash_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_NUMBER = 1 << HASHW;

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif

    ap_uint<72> elem = 0;
    ap_uint<72> base_elem = 0;
    ap_uint<72> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<HASHW - 2> array_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                            0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    ap_uint<32> max_overflow = pu_start_addr;
    ap_uint<ARW> overflow_base = HASH_NUMBER * depth;

    bool last = i_e_strm.read();
PRE_BUILD_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector inter false

        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // calculate index
        ap_uint<HASHW - 2> array_idx = hash_val(HASHW - 1, 2);
        ap_uint<2> bit_idx = hash_val(1, 0);

        // read hash counter and ++, prevent duplicate key
        if (array_idx == array_idx_temp[0]) {
            elem = elem_temp[0];
        } else if (array_idx == array_idx_temp[1]) {
            elem = elem_temp[1];
        } else if (array_idx == array_idx_temp[2]) {
            elem = elem_temp[2];
        } else if (array_idx == array_idx_temp[3]) {
            elem = elem_temp[3];
        } else if (array_idx == array_idx_temp[4]) {
            elem = elem_temp[4];
        } else if (array_idx == array_idx_temp[5]) {
            elem = elem_temp[5];
        } else if (array_idx == array_idx_temp[6]) {
            elem = elem_temp[6];
        } else if (array_idx == array_idx_temp[7]) {
            elem = elem_temp[7];
        } else {
            elem = htb_vector[array_idx];
        }

        // update && write new hash value
        ap_uint<16> v0 = elem(15, 0);
        ap_uint<16> v1 = elem(31, 16);
        ap_uint<16> v2 = elem(47, 32);
        ap_uint<16> v3 = elem(63, 48);

        bool cnt_overflow = false;
        bool v0_overflow = v0 == 65535;
        bool v1_overflow = v1 == 65535;
        bool v2_overflow = v2 == 65535;
        bool v3_overflow = v3 == 65535;

        ap_uint<16> v0a;
        ap_uint<16> v1a;
        ap_uint<16> v2a;
        ap_uint<16> v3a;
        ap_uint<16> hash_cnt;

        if (bit_idx == 0) {
            v0a = v0_overflow ? v0 : ap_uint<16>(v0 + 1);
            v1a = v1;
            v2a = v2;
            v3a = v3;

            hash_cnt = v0;
            cnt_overflow = v0_overflow;
        } else if (bit_idx == 1) {
            v0a = v0;
            v1a = v1_overflow ? v1 : ap_uint<16>(v1 + 1);
            v2a = v2;
            v3a = v3;

            hash_cnt = v1;
            cnt_overflow = v1_overflow;
        } else if (bit_idx == 2) {
            v0a = v0;
            v1a = v1;
            v2a = v2_overflow ? v2 : ap_uint<16>(v2 + 1);
            v3a = v3;

            hash_cnt = v2;
            cnt_overflow = v2_overflow;
        } else if (bit_idx == 3) {
            v0a = v0;
            v1a = v1;
            v2a = v2;
            v3a = v3_overflow ? v3 : ap_uint<16>(v3 + 1);

            hash_cnt = v3;
            cnt_overflow = v3_overflow;
        }

        base_elem(15, 0) = v0a;
        base_elem(31, 16) = v1a;
        base_elem(47, 32) = v2a;
        base_elem(63, 48) = v3a;

        // right shift temp
        for (int i = 7; i > 0; i--) {
            elem_temp[i] = elem_temp[i - 1];
            array_idx_temp[i] = array_idx_temp[i - 1];
        }
        elem_temp[0] = base_elem;
        array_idx_temp[0] = array_idx;

        // connect key with payload
        ap_uint<KEYW + PW> srow = 0;
        srow = (key, pld);

        // update hash table
        htb_vector[array_idx] = base_elem;

        // generate o_addr
        ap_uint<ARW> o_addr;

        if (hash_cnt >= depth) {
            // overflow
            o_addr = overflow_base + max_overflow;
            max_overflow++;
        } else {
            // underflow
            o_addr = depth * hash_val + hash_cnt;
        }

        o_addr_strm.write(o_addr);
        o_row_strm.write(srow);
        o_e0_strm.write(false);

        // cnt_overflow
        if (cnt_overflow) {
            o_overflow_hash_strm.write(hash_val);
            o_e1_strm.write(false);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
        if (hash_cnt >= depth) {
            std::cout << std::hex << "build_stb: over key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " v0=" << v0 << " v1=" << v1 << " v2=" << v2 << " v3=" << v3
                      << " ht_addr=" << array_idx << " ht=" << base_elem << " hash_cnt=" << hash_cnt
                      << " output stb_addr=" << o_addr << std::endl;
        } else {
            std::cout << std::hex << "build_stb: base key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " v0=" << v0 << " v1=" << v1 << " v2=" << v2 << " v3=" << v3
                      << " ht_addr=" << array_idx << " ht=" << base_elem << " hash_cnt=" << hash_cnt
                      << " output stb_addr=" << o_addr << std::endl;
        }
#endif

        ap_uint<ARW> old_val = (bit_idx == 0) ? v0 : ((bit_idx == 1) ? v1 : (bit_idx == 2) ? v2 : v3);
        if (old_val > max_col) max_col = old_val;
        cnt++;
#endif
    }

    pu_end_addr = max_overflow;

    o_e0_strm.write(true);
    o_e1_strm.write(true);

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
    std::cout << std::dec << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << std::dec << "collision probility " << max_col << std::endl;
#endif
#endif
}

template <int HASHWL, int HASHO, int ARW>
void build_overflow_htb(
    // documenting terrible cnt overflow (>65535)
    hls::stream<ap_uint<HASHWL> >& i_hash_strm,
    hls::stream<bool>& i_e_strm,

    ap_uint<72>* htb_vector) {
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;

    bool last = i_e_strm.read();
BUILD_OVERFLOW_HTB_LOOP:
    while (!last) {
#pragma HLS PIPELINE off

        ap_uint<HASHWL> hash_val = i_hash_strm.read();
        last = i_e_strm.read();

    SEARCH_LOOP:
        for (int i = 0; i < HASH_OVERFLOW_DEPTH; i++) {
#pragma HLS PIPELINE off

            // read
            ap_uint<72> elem = htb_vector[i];
            ap_uint<72> new_elem = 0;

            if (i == 511 && elem(63, 0) != 0) {
                // error occurs
            }

            // update
            if (elem(63, 0) != 0) {
                if (elem(HASHWL + ARW - 1, ARW) == hash_val) {
                    new_elem(HASHWL + ARW - 1, ARW) = hash_val;
                    new_elem(ARW - 1, 0) = elem(ARW - 1, 0) + 1;
                    htb_vector[i] = new_elem;

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
                    std::cout << std::dec << "build_overflow_htb: hash=" << hash_val << " cnt=" << elem(ARW - 1, 0)
                              << " id=" << i << std::endl;
#endif
#endif
                    break;
                }
            } else {
                new_elem(HASHWL + ARW - 1, ARW) = hash_val;
                new_elem(ARW, 0) = 65536;
                htb_vector[i] = new_elem;

#ifndef __SYNTHESIS__
#ifdef DEBUG_BUILD
                std::cout << std::dec << "build_overflow_htb: hash=" << hash_val << " cnt=" << elem(ARW - 1, 0)
                          << " id=" << i << std::endl;
#endif
#endif

                break;
            }
        }
    }
}

// top function of hash join build
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int PW, int ARW>
void build_wrapper(
    // input
    ap_uint<32> build_id,
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,

    hls::stream<ap_uint<HASHW + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    ap_uint<64>* stb_buf,

    ap_uint<72>* htb_vector0,
    ap_uint<72>* htb_vector1,
    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1,
    ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHW> > hash_strm;
#pragma HLS stream variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    hls::stream<ap_uint<BFW> > bf_hash0_strm;
#pragma HLS stream variable = bf_hash0_strm depth = 8
#pragma HLS bind_storage variable = bf_hash0_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash1_strm;
#pragma HLS stream variable = bf_hash1_strm depth = 8
#pragma HLS bind_storage variable = bf_hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash2_strm;
#pragma HLS stream variable = bf_hash2_strm depth = 8
#pragma HLS bind_storage variable = bf_hash2_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS bind_storage variable = addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8

    hls::stream<ap_uint<HASHW> > overflow_hash_strm;
#pragma HLS stream variable = overflow_hash_strm depth = 8
#pragma HLS bind_storage variable = overflow_hash_strm type = fifo impl = srl
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8

    // split row to hash and key
    join_v4::sc::split_unit<HASHW + BF_HASH_NM * BFW, BFW, BFW, BFW, HASHW>(i_hash_strm, i_e_strm,

                                                                            bf_hash0_strm, bf_hash1_strm, bf_hash2_strm,
                                                                            hash_strm, e0_strm, e1_strm);

    // update hash collision cnt && calculate stb addr
    build_base_htb<HASHW, KEYW, PW, ARW>(pu_start_addr, pu_end_addr, depth, hash_strm, i_key_strm, i_pld_strm, e0_strm,

                                         addr_strm, row_strm, e2_strm, overflow_hash_strm, e3_strm,

                                         htb_vector0);

    // cope with hash cnt overflow
    build_overflow_htb<HASHW, HASHO, ARW>(overflow_hash_strm, e3_strm, htb_vector1);

    // update bloom filter
    details::bv_update_uram<BFW>(bf_hash0_strm, bf_hash1_strm, bf_hash2_strm, e1_strm,

                                 bf_vector0, bf_vector1, bf_vector2);

    sc_tmp::write_stb<ARW, KEYW + PW>(stb_buf, addr_strm, row_strm, e2_strm);
}

//---------------------------------------------------merge hash table------------------------------------------------

// load hash collision counter to genrate bitmap
template <int HASHW, int BFW, int ARW>
void merge_base_htb(
    // output
    hls::stream<ap_uint<ARW> >& o_base_htb_strm,
    hls::stream<bool>& o_e0_etrm,
    hls::stream<ap_uint<HASHW> >& o_overflow_hash_strm,
    hls::stream<bool>& o_e1_etrm,

    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    int cnt;
#endif

    const int HASH_DEPTH = 1 << (HASHW - 2);

MERGE_BASE_HTB_LOOP:
    for (ap_uint<HASHW> i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 4
#pragma HLS dependence variable = htb_vector inter false

        // genrate bitmap from hash counter
        ap_uint<72> elem;
        ap_uint<16> bitmap[4];
        elem = htb_vector[i];

        bitmap[0] = elem(15, 0);
        bitmap[1] = elem(31, 16);
        bitmap[2] = elem(47, 32);
        bitmap[3] = elem(63, 48);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
        if (elem > 0) {
            std::cout << std::hex << "merge_base_htb: bitmap0=" << bitmap[0] << " bitmap1=" << bitmap[1]
                      << " bitmap2=" << bitmap[2] << " bitmap3=" << bitmap[3] << std::endl;
        }
#endif
#endif

        // write overflow bitmap to HBM for probe
        for (int j = 0; j < 4; j++) {
            // unroll
            o_base_htb_strm.write(bitmap[j]);
            o_e0_etrm.write(false);

            if (bitmap[j] == 65535) {
                ap_uint<HASHW> o_hash = i << 2 + j;
                o_overflow_hash_strm.write(o_hash);
                o_e1_etrm.write(false);
            }
        }
    }
    o_e0_etrm.write(true);
    o_e1_etrm.write(true);
}

// check hash count overflow
template <int HASHW, int HASHO, int ARW>
void merge_overflow_htb(
    // input status
    hls::stream<ap_uint<HASHW> >& i_overflow_hash_strm,
    hls::stream<bool>& i_e_etrm,

    // output
    hls::stream<ap_uint<ARW> >& o_overflow_htb_strm,

    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    int cnt;
#endif

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;

    bool last = i_e_etrm.read();
MERGE_OVERFLOW_HTB_LOOP:
    while (!last) {
#pragma HLS PIPELINE off

        ap_uint<HASHW> hash_val = i_overflow_hash_strm.read();
        last = i_e_etrm.read();

    SEARCH_LOOP:
        for (int i = 0; i < HASH_OVERFLOW_DEPTH; i++) {
#pragma HLS PIPELINE off
#pragma HLS dependence variable = htb_vector inter false

            // genrate bitmap from hash counter
            ap_uint<72> elem;
            elem = htb_vector[i];

            ap_uint<HASHW> hash = elem(HASHW + ARW - 1, HASHW);
            ap_uint<ARW> hash_cnt = elem(ARW - 1, 0);

            if (i == 0 && elem == 0) {
                o_overflow_htb_strm.write(0);
                break;
            } else if (hash_val == hash) {
                o_overflow_htb_strm.write(hash_cnt);
                break;
            } else if (i == 511) {
                o_overflow_htb_strm.write(0);
                break;
            }
        }
    }
}

// load hash collision counter to genrate overflow pointer for overflow stb
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int ARW>
void bitmap_addr_gen(
    // input
    hls::stream<ap_uint<ARW> >& i_base_htb_strm,
    hls::stream<ap_uint<ARW> >& i_overflow_htb_strm,
    hls::stream<bool>& i_e_etrm,

    // output
    hls::stream<ap_uint<ARW> >& o_addr_strm,
    hls::stream<ap_uint<64> >& o_bitmap_strm,
    hls::stream<bool>& o_e_etrm,

    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_NUMBER = 1 << HASHW;
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

    // srow addr
    ap_uint<ARW> overflow_pointer_addr = HASH_DEPTH + HASH_OVERFLOW_DEPTH + BF_HASH_NM * BLOOM_FILTER_DEPTH;
    ap_uint<64> overflow_pointer = HASH_DEPTH + HASH_OVERFLOW_DEPTH + BF_HASH_NM * BLOOM_FILTER_DEPTH + HASH_NUMBER;

    // htb:
    //   |=========base_hash_counter========|===overflow_hash_counter===|==========bloom_filter_vector=========|======overflow_pointer=====|========overflow_small_table========|
    // length:
    //   |<------------HASH_DEPTH---------->|<---HASH_OVERFLOW_DEPTH--->|<---BF_HASH_NM*BLOOM_FILTER_DEPTH---->|<-------HASH_NUMBER------->|<-----------pu_start_addr---------->|

    // stb:
    //   |===================base_area_of_small_table===================|========temp_of_overflow_small_table========|
    // length:
    //   |<--------------------HASH_NUMBER*depth----------------------->|<---------------pu_start_addr-------------->|

    ap_uint<64> bitmap_temp = 0;
    ap_uint<HASHW> cnt = 0;

    bool last = i_e_etrm.read();
BITMAP_ADDR_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bf_vector0 inter false
#pragma HLS dependence variable = bf_vector1 inter false

        // genrate bitmap from hash counter
        ap_uint<ARW> base_nm = i_base_htb_strm.read();
        last = i_e_etrm.read();

        ap_uint<ARW> overflow_nm;
        ap_uint<ARW> nm;

        if (base_nm == 65535) {
            overflow_nm = i_overflow_htb_strm.read();
            nm = overflow_nm;
        } else {
            nm = base_nm;
        }

        // output to htb as overflow pointer
        o_addr_strm.write(overflow_pointer_addr);
        o_bitmap_strm.write(overflow_pointer);
        o_e_etrm.write(false);

        // store bitmap temp in URAM for merge unit
        ap_uint<2> bit_idx = cnt(1, 0);
        ap_uint<HASHW - 2> array_idx = cnt(HASHW - 1, 2);

        if (bit_idx == 0) {
            bitmap_temp(ARW - 1, 0) = overflow_pointer(ARW - 1, 0);
        } else if (bit_idx == 1) {
            bitmap_temp(2 * ARW - 1, ARW) = overflow_pointer(ARW - 1, 0);

            bf_vector0[array_idx] = bitmap_temp;
        } else if (bit_idx == 2) {
            bitmap_temp(ARW - 1, 0) = overflow_pointer(ARW - 1, 0);
        } else {
            bitmap_temp(2 * ARW - 1, ARW) = overflow_pointer(ARW - 1, 0);

            bf_vector1[array_idx] = bitmap_temp;
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
        if (nm) {
            std::cout << std::hex << "bitmap_addr_gen: overflow_pointer_addr=" << overflow_pointer_addr
                      << " overflow_stb_pointer=" << overflow_pointer << " nm=" << nm << std::endl;

            if (bit_idx == 1) {
                std::cout << std::hex << "bitmap_addr_gen: array_idx=" << array_idx << " bf_vector0=" << bitmap_temp
                          << std::endl;
            }
            if (bit_idx == 3) {
                std::cout << std::hex << "bitmap_addr_gen: array_idx=" << array_idx << " bf_vector1=" << bitmap_temp
                          << std::endl;
            }
        }
#endif
#endif

        overflow_pointer += nm;
        overflow_pointer_addr++;
        cnt++;
    }
    o_e_etrm.write(true);
}

// load hash collision counter to genrate bitmap
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int ARW>
void merge_htb(ap_uint<64>* htb_buf,
               ap_uint<72>* htb_vector0,
               ap_uint<72>* htb_vector1,
               ap_uint<72>* bf_vector0,
               ap_uint<72>* bf_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<ARW> > base_htb_strm;
#pragma HLS stream variable = base_htb_strm depth = 8
#pragma HLS bind_storage variable = base_htb_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > overflow_htb_strm;
#pragma HLS stream variable = overflow_htb_strm depth = 8
#pragma HLS bind_storage variable = overflow_htb_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    hls::stream<ap_uint<HASHW> > overflow_hash_strm;
#pragma HLS stream variable = overflow_hash_strm depth = 8
#pragma HLS bind_storage variable = overflow_hash_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    hls::stream<ap_uint<64> > bitmap_strm;
#pragma HLS stream variable = bitmap_strm depth = 8
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > write_addr_strm;
#pragma HLS stream variable = write_addr_strm depth = 8
#pragma HLS bind_storage variable = write_addr_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8

    merge_base_htb<HASHW, BFW, ARW>(base_htb_strm, e0_strm, overflow_hash_strm, e1_strm,

                                    htb_vector0);

    merge_overflow_htb<HASHW, HASHO, ARW>(overflow_hash_strm, e1_strm,

                                          overflow_htb_strm,

                                          htb_vector1);

    // generate overflow bitmap
    bitmap_addr_gen<HASHW, HASHO, BF_HASH_NM, BFW, ARW>(base_htb_strm, overflow_htb_strm, e0_strm,

                                                        write_addr_strm, bitmap_strm, e_strm,

                                                        bf_vector0, bf_vector1);

    // write merged htb to htb_buf
    sc_tmp::write_stb<ARW, 64>(htb_buf, write_addr_strm, bitmap_strm, e_strm);
}

// merge overflow srow
template <int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_unit(hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,

                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<KEYW + PW> >& o_row_strm,
                hls::stream<bool>& o_e_strm,

                ap_uint<72>* bf_vector0,
                ap_uint<72>* bf_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    ap_uint<72> elem0 = 0;
    ap_uint<72> elem1 = 0;
    ap_uint<72> old_elem = 0;
    ap_uint<72> new_elem = 0;

    ap_uint<72> elem_temp0[4] = {0, 0, 0, 0};
    ap_uint<72> elem_temp1[4] = {0, 0, 0, 0};
    ap_uint<ARW> array_idx_temp[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    bool last = i_e_strm.read();
LOOP_BUILD_UNIT:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bf_vector0 inter false
#pragma HLS dependence variable = bf_vector1 inter false

        if (!i_e_strm.empty()) {
            // read
            ap_uint<KEYW> key = i_key_strm.read();
            ap_uint<PW> pld = i_pld_strm.read();
            ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
            last = i_e_strm.read();

            // connect key with payload
            ap_uint<KEYW + PW> stb_row = 0;
            stb_row = (key, pld);

            // mod 3 to calculate index for 24bit address
            ap_uint<HASHWL> hash = hash_val(HASHWL - 1, 0);
            ap_uint<HASHWL - 2> array_idx = hash(HASHWL - 1, 2);
            ap_uint<2> bit_idx = hash(1, 0);
            ap_uint<1> bf_mux;

            /*
            if(key==0xC1){
  #ifndef __SYNTHESIS__
            std::cout <<std::dec<< "error here"<<std::endl;
  #endif
            }
            */

            // read previous temp, prevent duplicate key
            if (array_idx == array_idx_temp[0]) {
                elem0 = elem_temp0[0];
                elem1 = elem_temp1[0];
            } else if (array_idx == array_idx_temp[1]) {
                elem0 = elem_temp0[1];
                elem1 = elem_temp1[1];
            } else if (array_idx == array_idx_temp[2]) {
                elem0 = elem_temp0[2];
                elem1 = elem_temp1[2];
            } else if (array_idx == array_idx_temp[3]) {
                elem0 = elem_temp0[3];
                elem1 = elem_temp1[3];
            } else {
                elem0 = bf_vector0[array_idx];
                elem1 = bf_vector1[array_idx];
            }

            if (bit_idx == 0 || bit_idx == 1) {
                old_elem = elem0;
            } else {
                old_elem = elem1;
            }

            ap_uint<ARW> v0 = old_elem(ARW - 1, 0);
            ap_uint<ARW> v1 = old_elem(2 * ARW - 1, ARW);

            ap_uint<ARW> v0a, v1a;
            if (bit_idx == 0 || bit_idx == 2) {
                v0a = v0 + 1;
                v1a = v1;
            } else {
                v0a = v0;
                v1a = v1 + 1;
            }
            new_elem = (v1a, v0a);

            // right shift temp
            for (int i = 3; i > 0; i--) {
                elem_temp0[i] = elem_temp0[i - 1];
                elem_temp1[i] = elem_temp1[i - 1];
                array_idx_temp[i] = array_idx_temp[i - 1];
            }

            if (bit_idx == 0 || bit_idx == 1) {
                elem_temp0[0] = new_elem;
                elem_temp1[0] = elem1;
            } else {
                elem_temp0[0] = elem0;
                elem_temp1[0] = new_elem;
            }

            array_idx_temp[0] = array_idx;

            // update Uram
            if (bit_idx == 0 || bit_idx == 1) {
                bf_vector0[array_idx] = new_elem;
            } else {
                bf_vector1[array_idx] = new_elem;
            }

            ap_uint<ARW> o_addr = (bit_idx == 0 || bit_idx == 2) ? v0 : v1;

            // write stb
            o_addr_strm.write(o_addr);
            o_row_strm.write(stb_row);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "merge_unit: cnt=" << cnt << " array_idx=" << array_idx << " bit_idx=" << bit_idx
                      << " overflow_ht_addr=" << array_idx << " old_elem=" << old_elem << " new_elem=" << new_elem
                      << " stb_addr=" << o_addr << " stb_row=" << stb_row << std::endl;
#endif
            cnt++;
#endif
        }
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
    std::cout << std::dec << "merge_unit: merge " << cnt << " lines" << std::endl;
#endif
#endif

    o_e_strm.write(true);
}

// merge srow from htb to stb
template <int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_stb(ap_uint<32> depth,
               ap_uint<32> pu_start_addr,
               ap_uint<64>* htb_buf,
               ap_uint<64>* stb_buf,

               ap_uint<72>* bf_vector0,
               ap_uint<72>* bf_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<ARW> > read_addr_strm;
#pragma HLS stream variable = read_addr_strm depth = 8
#pragma HLS bind_storage variable = read_addr_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    hls::stream<ap_uint<KEYW + PW> > row0_strm;
#pragma HLS stream variable = row0_strm depth = 8
#pragma HLS bind_storage variable = row0_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    hls::stream<ap_uint<KEYW> > key0_strm;
#pragma HLS stream variable = key0_strm depth = 8
#pragma HLS bind_storage variable = key0_strm type = fifo impl = srl
    hls::stream<ap_uint<PW> > pld_strm;
#pragma HLS stream variable = pld_strm depth = 8
#pragma HLS bind_storage variable = pld_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8

    hls::stream<ap_uint<KEYW> > key1_strm;
#pragma HLS stream variable = key1_strm depth = 8
#pragma HLS bind_storage variable = key1_strm type = fifo impl = srl
    hls::stream<ap_uint<HASHWH + HASHWL> > hash_strm;
#pragma HLS stream variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8

    hls::stream<ap_uint<KEYW + PW> > row1_strm;
#pragma HLS stream variable = row1_strm depth = 8
#pragma HLS bind_storage variable = row1_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > write_addr_strm;
#pragma HLS stream variable = write_addr_strm depth = 8
#pragma HLS bind_storage variable = write_addr_strm type = fifo impl = srl
    hls::stream<bool> e4_strm;
#pragma HLS stream variable = e4_strm depth = 8

    // generate read addr
    read_addr_gen<ARW>(depth << HASHWL, pu_start_addr,

                       read_addr_strm, e0_strm);

    // read srow in htb_buf
    sc_tmp::read_stb<ARW, KEYW + PW>(stb_buf,

                                     read_addr_strm, e0_strm,

                                     row0_strm, e1_strm);

    // split row to hash and key
    database::splitCol<KEYW + PW, PW, KEYW>(row0_strm, e1_strm,

                                            pld_strm, key0_strm, e2_strm);

    // calculate hash
    join_v3::sc::hash_wrapper<1, KEYW, HASHWH + HASHWL>(key0_strm, e2_strm,

                                                        hash_strm, key1_strm, e3_strm);

    // build srow with its hash addr
    merge_unit<HASHWH, HASHWL, KEYW, PW, ARW>(hash_strm, key1_strm, pld_strm, e3_strm, write_addr_strm, row1_strm,
                                              e4_strm,

                                              bf_vector0, bf_vector1);

    // write srow to stb_buf
    sc_tmp::write_stb<ARW, KEYW + PW>(htb_buf, write_addr_strm, row1_strm, e4_strm);
}

template <int HASHWH, int HASHWL, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int PW, int ARW>
void merge_wrapper(
    // input status
    ap_uint<32> depth,
    ap_uint<32> pu_start_addr,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf,

    ap_uint<72>* htb_vector0,
    ap_uint<72>* htb_vector1,
    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1) {
#pragma HLS INLINE off

    // generate bitmap addr by hash counter
    merge_htb<HASHWL, HASHO, BF_HASH_NM, BFW, ARW>(htb_buf, htb_vector0, htb_vector1, bf_vector0, bf_vector1);

    // build srow with its hash addr
    merge_stb<HASHWH, HASHWL, KEYW, PW, ARW>(depth, pu_start_addr, htb_buf, stb_buf, bf_vector0, bf_vector1);
}

//---------------------------------------------------probe------------------------------------------------

// probe the hash table and output address which hash same hash_value
template <int HASHW, int KEYW, int PW, int ARW>
void probe_base_htb(
    // input status
    ap_uint<32> depth,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output to check htb
    hls::stream<ap_uint<HASHW> >& o_t_hash_strm,
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_base_nm_strm,
    hls::stream<bool>& o_e0_strm,

    // check hash_cnt overflow
    hls::stream<ap_uint<HASHW> >& o_overflow_hash_strm,
    hls::stream<bool>& o_e1_strm,

    // base htb
    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int probe_hit = 0;
    unsigned int terrible_cnt = 0;
#endif

    // htb:
    //   |=========base_hash_counter========|===overflow_hash_counter===|==========bloom_filter_vector=========|======overflow_pointer=====|========overflow_small_table========|
    // length:
    //   |<------------HASH_DEPTH---------->|<---HASH_OVERFLOW_DEPTH--->|<---BF_HASH_NM*BLOOM_FILTER_DEPTH---->|<-------HASH_NUMBER------->|<-----------pu_start_addr---------->|

    // stb:
    //   |===================base_area_of_small_table===================|========temp_of_overflow_small_table========|
    // length:
    //   |<--------------------HASH_NUMBER*depth----------------------->|<---------------pu_start_addr-------------->|

    ap_uint<ARW> base_addr, base_nm, overflow_pointer_addr;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector inter false

        // read large table
        ap_uint<HASHW> hash = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // calculate index for HTB
        ap_uint<HASHW> array_idx = hash(HASHW - 1, 2);
        ap_uint<2> bit_idx = hash(1, 0);

        // total hash count
        ap_uint<ARW> nm;
        ap_uint<72> base_bitmap = htb_vector[array_idx];

        if (bit_idx == 0) {
            nm = base_bitmap(15, 0);
        } else if (bit_idx == 1) {
            nm = base_bitmap(31, 16);
        } else if (bit_idx == 2) {
            nm = base_bitmap(47, 32);
        } else {
            nm = base_bitmap(63, 48);
        }

        if (nm > 0) {
            // write large table to join
            o_t_hash_strm.write(hash);
            o_t_key_strm.write(key);
            o_t_pld_strm.write(pld);

            // write controlling strm for probe base stb
            o_base_nm_strm.write(nm);
            o_e0_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "probe_base_htb: base_id=" << probe_hit << " nm=" << nm << " t_key=" << key
                      << " t_pld=" << pld << std::endl;
            probe_hit++;
#endif
#endif

            // search htb_vector1 to get overflow htb
            if (nm == 65535) {
                o_overflow_hash_strm.write(hash);
                o_e1_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
                std::cout << std::hex << "probe_base_htb: terrible_id=" << terrible_cnt << " hash=" << hash
                          << std::endl;
                terrible_cnt++;
#endif
#endif
            }
        }
    }

    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

// get the overflow hash count from overflow hash table
template <int HASHW, int HASHO, int ARW>
void probe_overflow_htb(
    // input
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<bool>& i_e_strm,

    // output overflow hash count
    hls::stream<ap_uint<ARW> >& o_nm_strm,

    // overflow htb
    ap_uint<72>* htb_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int overflow_cnt = 0;
#endif

    const int OVERFLOW_HASH_DEPTH = 1 << HASHO;

    ap_uint<72> bitmap;
    ap_uint<ARW> nm;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE off

        ap_uint<HASHW> hash = i_hash_strm.read();
        last = i_e_strm.read();

        for (int i = 0; i < OVERFLOW_HASH_DEPTH; i++) {
#pragma HLS PIPELINE off

            bitmap = htb_vector[i];
            ap_uint<ARW> nm = bitmap(ARW - 1, 0);
            ap_uint<HASHW> hash_val = bitmap(HASHW + ARW - 1, ARW);

            if (hash == hash_val) {
                o_nm_strm.write(nm);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
                std::cout << std::hex << "probe_overflow_htb: overflow_hash=" << hash << " overflow_cnt=" << nm
                          << std::endl;
#endif
#endif

                break;
            } else if (i == 511) {
                // error occurs
            }
        }
    }
}

// get read hash count from base/overflow hash count
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int PW, int ARW>
void check_htb(
    // input status
    ap_uint<32> depth,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,

    // input base hash count and overflow hash count
    hls::stream<ap_uint<ARW> >& i_base_nm_strm,
    hls::stream<ap_uint<ARW> >& i_overflow_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probe base stb
    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<ARW> >& o_base_nm_strm,
    hls::stream<bool>& o_e0_strm,

    // output to probe overflow stb
    hls::stream<ap_uint<ARW> >& o_overflow_pointer_addr_strm,
    hls::stream<ap_uint<ARW> >& o_overflow_nm_strm,
    hls::stream<bool>& o_e1_strm,

    // output to join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_total_nm_strm,
    hls::stream<bool>& o_e2_strm) {
#pragma HLS INLINE off

    ap_uint<ARW> nm, real_nm;
    ap_uint<ARW> base_nm, overflow_nm;
    ap_uint<ARW> base_addr, overflow_pointer_addr;

#ifndef __SYNTHESIS__
    unsigned int base_cnt = 0;
    unsigned int overflow_cnt = 0;
#endif

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

    // htb:
    //   |=========base_hash_counter========|===overflow_hash_counter===|==========bloom_filter_vector=========|======overflow_pointer=====|========overflow_small_table========|
    // length:
    //   |<------------HASH_DEPTH---------->|<---HASH_OVERFLOW_DEPTH--->|<---BF_HASH_NM*BLOOM_FILTER_DEPTH---->|<-------HASH_NUMBER------->|<-----------pu_start_addr---------->|

    // stb:
    //   |===================base_area_of_small_table===================|========temp_of_overflow_small_table========|
    // length:
    //   |<--------------------HASH_NUMBER*depth----------------------->|<---------------pu_start_addr-------------->|

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
        // read large table
        ap_uint<HASHW> hash = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        nm = i_base_nm_strm.read();
        last = i_e_strm.read();

        // calculate addr
        base_addr = hash * depth;
        overflow_pointer_addr = HASH_DEPTH + HASH_OVERFLOW_DEPTH + BF_HASH_NM * BLOOM_FILTER_DEPTH + hash;

        // calculate base && overflow number
        if (nm == 65535) {
            real_nm = i_overflow_nm_strm.read();
        } else {
            real_nm = nm;
        }

        if (real_nm > depth) {
            base_nm = depth;
            overflow_nm = real_nm - depth;

#ifndef __SYNTHESIS__
            overflow_cnt++;
#endif
        } else {
            base_nm = real_nm;
            overflow_nm = 0;

#ifndef __SYNTHESIS__
            base_cnt++;
#endif
        }

        // write output
        if (base_nm > 0) {
            o_base_addr_strm.write(base_addr);
            o_base_nm_strm.write(base_nm);
            o_e0_strm.write(false);
        }

        if (overflow_nm > 0) {
            o_overflow_pointer_addr_strm.write(overflow_pointer_addr);
            o_overflow_nm_strm.write(overflow_nm);
            o_e1_strm.write(false);
        }

        o_t_key_strm.write(key);
        o_t_pld_strm.write(pld);
        o_total_nm_strm.write(real_nm);
        o_e2_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
        std::cout << std::hex << "check_htb: base_cnt=" << base_cnt << " overflow_cnt=" << overflow_cnt
                  << " base_nm=" << base_nm << " overflow_nm=" << overflow_nm << std::endl;
#endif
#endif
    }

    o_e0_strm.write(true);
    o_e2_strm.write(true);

    // for do while in probe overflow stb
    o_overflow_pointer_addr_strm.write(overflow_pointer_addr);
    o_overflow_nm_strm.write(0);
    o_e1_strm.write(true);
}

// probe the hash table and output address which hash same hash_value
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int PW, int ARW>
void probe_htb(
    // input status
    ap_uint<32> depth,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output to generate base area probe addr
    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<ARW> >& o_base_nm_strm,
    hls::stream<bool>& o_e0_strm,
    hls::stream<ap_uint<ARW> >& o_overflow_pointer_addr_strm,
    hls::stream<ap_uint<ARW> >& o_overflow_nm_strm,
    hls::stream<bool>& o_e1_strm,

    // output to join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_total_nm_strm,
    hls::stream<bool>& o_e2_strm,

    // base htb
    ap_uint<72>* htb_vector0,
    ap_uint<72>* htb_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHW> > t_hash_strm;
#pragma HLS stream variable = t_hash_strm depth = 8
#pragma HLS bind_storage variable = t_hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > t_key_strm;
#pragma HLS stream variable = t_key_strm depth = 8
#pragma HLS bind_storage variable = t_key_strm type = fifo impl = srl
    hls::stream<ap_uint<PW> > t_pld_strm;
#pragma HLS stream variable = t_pld_strm depth = 8
#pragma HLS bind_storage variable = t_pld_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > base_nm_strm;
#pragma HLS stream variable = base_nm_strm depth = 8
#pragma HLS bind_storage variable = base_nm_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    hls::stream<ap_uint<HASHW> > overflow_hash_strm;
#pragma HLS stream variable = overflow_hash_strm depth = 8
#pragma HLS bind_storage variable = overflow_hash_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > overflow_nm_strm;
#pragma HLS stream variable = overflow_nm_strm depth = 8
#pragma HLS bind_storage variable = overflow_nm_strm type = fifo impl = srl

    // probe the hash table and output address which hash same hash_value
    probe_base_htb<HASHW, KEYW, PW, ARW>(
        // input
        depth,

        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        t_hash_strm, t_key_strm, t_pld_strm, base_nm_strm, e0_strm,

        overflow_hash_strm, e1_strm,

        htb_vector0);

    // get the overflow hash count from overflow hash table
    probe_overflow_htb<HASHW, HASHO, ARW>(overflow_hash_strm, e1_strm,

                                          overflow_nm_strm, htb_vector1);

    // get read hash count from base/overflow hash count
    check_htb<HASHW, HASHO, BF_HASH_NM, BFW, KEYW, PW, ARW>(depth,

                                                            t_hash_strm, t_key_strm, t_pld_strm, base_nm_strm,
                                                            overflow_nm_strm, e0_strm,

                                                            o_base_addr_strm, o_base_nm_strm, o_e0_strm,

                                                            o_overflow_pointer_addr_strm, o_overflow_nm_strm, o_e1_strm,

                                                            o_t_key_strm, o_t_pld_strm, o_total_nm_strm, o_e2_strm);
}

// generate base probe addr
template <int ARW>
void probe_addr_gen(
    // input
    hls::stream<ap_uint<ARW> >& i_base_stb_addr_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output
    hls::stream<ap_uint<ARW> >& o_read_addr_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    bool is_first_loop = true;
    ap_uint<ARW> addr;
    ap_uint<ARW> nm = 0;

    bool last = i_e_strm.read();
    while (!last || nm != 0) {
#pragma HLS pipeline II = 1

        if (!last && is_first_loop) {
            addr = i_base_stb_addr_strm.read();
            nm = i_nm_strm.read();
            last = i_e_strm.read();

            is_first_loop = false;
        } else if (nm > 0) {
            nm--;
            o_read_addr_strm.write(addr);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
            std::cout << std::hex << "probe_addr_gen: probe_addr=" << addr << std::endl;
#endif
#endif

            addr++;
        } else if (!last && nm == 0) {
            is_first_loop = true;
        }
    }
    o_e_strm.write(true);
}

// probe stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void probe_base_stb(
    // input
    ap_uint<32> depth,
    hls::stream<ap_uint<ARW> >& i_base_stb_addr_strm,
    hls::stream<ap_uint<ARW> >& i_base_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,

    // HBM
    ap_uint<64>* stb_buf) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for read_stb
    hls::stream<ap_uint<ARW> > read_addr_strm;
#pragma HLS stream variable = read_addr_strm depth = 8
#pragma HLS bind_storage variable = read_addr_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    // for split probed stb
    hls::stream<ap_uint<KEYW + S_PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // eliminate end strm
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    // generate read addr from base addr
    probe_addr_gen<ARW>(i_base_stb_addr_strm, i_base_nm_strm, i_e_strm,

                        read_addr_strm, e0_strm);

    // read HBM to get base stb
    sc_tmp::read_stb<ARW, KEYW + S_PW>(stb_buf, read_addr_strm, e0_strm, row_strm, e1_strm);

    // split base stb to key and pld
    database::splitCol<KEYW + S_PW, S_PW, KEYW>(row_strm, e1_strm, o_base_s_pld_strm, o_base_s_key_strm, e2_strm);

    // eleminate end signal of overflow unit
    join_v3::sc::eliminate_strm_end<bool>(e2_strm);
}

// probe stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void read_overflow_stb(
    // input
    ap_uint<ARW> overflow_addr,
    ap_uint<ARW> overflow_nm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    // HBM
    ap_uint<64>* htb_buf) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for read_stb
    hls::stream<ap_uint<ARW> > read_addr_strm;
#pragma HLS stream variable = read_addr_strm depth = 8
#pragma HLS bind_storage variable = read_addr_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    // for split probed stb
    hls::stream<ap_uint<KEYW + S_PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // eliminate end strm
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    // generate read addr
    read_addr_gen<ARW>(overflow_addr, overflow_nm,

                       read_addr_strm, e0_strm);

    // read HBM to get base stb
    sc_tmp::read_stb<ARW, KEYW + S_PW>(htb_buf, read_addr_strm, e0_strm, row_strm, e1_strm);

    // split base stb to key and pld
    database::splitCol<KEYW + S_PW, S_PW, KEYW>(row_strm, e1_strm, o_overflow_s_pld_strm, o_overflow_s_key_strm,
                                                e2_strm);

    // eleminate end signal of overflow unit
    join_v3::sc::eliminate_strm_end<bool>(e2_strm);
}

// probe stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void probe_overflow_stb(
    // input base addr
    hls::stream<ap_uint<ARW> >& i_overflow_pointer_addr_strm,
    hls::stream<ap_uint<ARW> >& i_overflow_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    // HBM
    ap_uint<64>* htb_buf) {
#pragma HLS INLINE off

    ap_uint<ARW> overflow_stb_addr, nm;
    ap_uint<64> overflow_pointer;

    bool last = false;
PROBE_OVERFLOW_LOOP:
    do {
#pragma HLS PIPELINE off

        overflow_stb_addr = i_overflow_pointer_addr_strm.read();
        nm = i_overflow_nm_strm.read();
        last = i_e_strm.read();

        // read overflow pointer
        overflow_pointer = htb_buf[overflow_stb_addr];

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
        std::cout << std::hex << "probe_overflow_stb: overflow_stb_addr=" << overflow_stb_addr
                  << " overflow_pointer=" << overflow_pointer << " nm=" << nm << std::endl;
#endif
#endif

        read_overflow_stb<KEYW, S_PW, ARW>(overflow_pointer, nm, o_overflow_s_key_strm, o_overflow_s_pld_strm, htb_buf);
    } while (!last);
}

// top function of probe
template <int HASHWL, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int S_PW, int T_PW, int ARW>
void probe_wrapper(
    // flag
    ap_uint<32> depth,
    ap_uint<32> pu_start_addr,

    // input large table
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<T_PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf,
    ap_uint<72>* htb_vector0,
    ap_uint<72>* htb_vector1,
    ap_uint<72>* bf_vector0,
    ap_uint<72>* bf_vector1,
    ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for split key hash and bloom filter hash
    hls::stream<ap_uint<HASHWL> > hash0_strm;
#pragma HLS stream variable = hash0_strm depth = 512
#pragma HLS bind_storage variable = hash0_strm type = fifo impl = bram
    hls::stream<ap_uint<BFW> > bf_hash0_strm;
#pragma HLS stream variable = bf_hash0_strm depth = 8
#pragma HLS bind_storage variable = bf_hash0_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash1_strm;
#pragma HLS stream variable = bf_hash1_strm depth = 8
#pragma HLS bind_storage variable = bf_hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<BFW> > bf_hash2_strm;
#pragma HLS stream variable = bf_hash2_strm depth = 8
#pragma HLS bind_storage variable = bf_hash2_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    // for check bloom filter
    hls::stream<bool> valid_strm;
#pragma HLS stream variable = valid_strm depth = 8
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    // filter invalid ltb
    hls::stream<ap_uint<HASHWL> > hash1_strm;
#pragma HLS stream variable = hash1_strm depth = 8
#pragma HLS bind_storage variable = hash1_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS stream variable = key_strm depth = 8
#pragma HLS bind_storage variable = key_strm type = fifo impl = srl
    hls::stream<ap_uint<T_PW> > pld_strm;
#pragma HLS stream variable = pld_strm depth = 8
#pragma HLS bind_storage variable = pld_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8

    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > base_nm_strm;
#pragma HLS stream variable = base_nm_strm depth = 8
#pragma HLS bind_storage variable = base_nm_strm type = fifo impl = srl
    hls::stream<bool> e3_strm;
#pragma HLS stream variable = e3_strm depth = 8

    hls::stream<ap_uint<ARW> > overflow_pointer_addr_strm;
#pragma HLS stream variable = overflow_pointer_addr_strm depth = 32
#pragma HLS bind_storage variable = overflow_pointer_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > overflow_nm_strm;
#pragma HLS stream variable = overflow_nm_strm depth = 32
#pragma HLS bind_storage variable = overflow_nm_strm type = fifo impl = srl
    hls::stream<bool> e4_strm;
#pragma HLS stream variable = e4_strm depth = 32

    // split bf hash and key hash
    database::splitCol<HASHWL + BF_HASH_NM * BFW, BFW, BFW, BFW, HASHWL>(i_hash_strm, i_e_strm,

                                                                         bf_hash0_strm, bf_hash1_strm, bf_hash2_strm,
                                                                         hash0_strm, e0_strm);

    // check bloom filter
    database::details::bv_check_uram<BFW>(bf_hash0_strm, bf_hash1_strm, bf_hash2_strm, e0_strm, bf_vector0, bf_vector1,
                                          bf_vector2, valid_strm, e1_strm);

    // filter invalid large table
    join_v4::sc::filter_invalid_ltb<HASHWL, KEYW, T_PW>(hash0_strm, i_key_strm, i_pld_strm, valid_strm, e1_strm,

                                                        hash1_strm, key_strm, pld_strm, e2_strm);

    // probe htb to get probe addr and length
    probe_htb<HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, T_PW, ARW>(depth,

                                                               hash1_strm, key_strm, pld_strm, e2_strm,

                                                               base_addr_strm, base_nm_strm, e3_strm,
                                                               overflow_pointer_addr_strm, overflow_nm_strm, e4_strm,

                                                               o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e0_strm,

                                                               htb_vector0, htb_vector1);

    // probe stb stored in base area
    probe_base_stb<KEYW, S_PW, ARW>(depth, base_addr_strm, base_nm_strm, e3_strm,

                                    o_base_s_key_strm, o_base_s_pld_strm,

                                    stb_buf);

    // probe stb stored in overflow area
    probe_overflow_stb<KEYW, S_PW, ARW>(overflow_pointer_addr_strm, overflow_nm_strm, e4_strm,

                                        o_overflow_s_key_strm, o_overflow_s_pld_strm,

                                        htb_buf);
}

//----------------------------------------------build+merge+probe------------------------------------------------

// initiate htb
template <int HASHW, int HASHO, int ARW>
void initiate_htb(ap_uint<32> build_id, ap_uint<64>* htb_buf, ap_uint<72>* htb_vector0, ap_uint<72>* htb_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------Initialize htb--------------------" << std::endl;
#endif
#endif

    if (build_id == 0) {
    // build_id==0, the firsr time to build, initialize uram to zero
    HTB0_INIT_LOOP:
        for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector0 inter false

            htb_vector0[i] = 0;
        }

    HTB1_INIT_LOOP:
        for (int i = 0; i < HASH_OVERFLOW_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = htb_vector1 inter false

            htb_vector1[i] = 0;
        }
    } else {
        // read hash table from previous built to initialize URAM
        read_htb<ARW>(htb_buf, 0, HASH_DEPTH, htb_vector0);

        read_htb<ARW>(htb_buf, HASH_DEPTH, HASH_OVERFLOW_DEPTH, htb_vector1);
    }
}

// update hash vector
template <int HASHW, int HASHO, int ARW>
void update_htb(ap_uint<64>* htb_buf, ap_uint<72>* htb_vector0, ap_uint<72>* htb_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "---------------------Update htb--------------------" << std::endl;
#endif
#endif

    // output hash table
    write_htb<ARW>(htb_buf, 0, HASH_DEPTH, htb_vector0);

    write_htb<ARW>(htb_buf, HASH_DEPTH, HASH_OVERFLOW_DEPTH, htb_vector1);
}

// initiate bf vector
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int ARW>
void initiate_bf(ap_uint<32> build_id,
                 ap_uint<64>* htb_buf,
                 ap_uint<72>* bf_vector0,
                 ap_uint<72>* bf_vector1,
                 ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------Initialize bloom filter-----------" << std::endl;
#endif
#endif

    if (build_id == 0) {
    // build_id==0, the firsr time to build, initialize uram to zero
    BF_INIT_LOOP:
        for (int i = 0; i < BLOOM_FILTER_DEPTH; i++) {
            bf_vector0[i] = 0;
            bf_vector1[i] = 0;
            bf_vector2[i] = 0;
        }
    } else {
        // read bloom-filter table from previous built to initialize URAM
        read_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH, BLOOM_FILTER_DEPTH, bf_vector0);

        read_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH + BLOOM_FILTER_DEPTH, BLOOM_FILTER_DEPTH, bf_vector1);

        read_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH + 2 * BLOOM_FILTER_DEPTH, BLOOM_FILTER_DEPTH,
                      bf_vector2);
    }
}

// update bf vector
template <int HASHW, int HASHO, int BF_HASH_NM, int BFW, int ARW>
void update_bf(ap_uint<64>* htb_buf, ap_uint<72>* bf_vector0, ap_uint<72>* bf_vector1, ap_uint<72>* bf_vector2) {
#pragma HLS INLINE off

    const int HASH_DEPTH = 1 << (HASHW - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "-----------------Update bloom filter---------------" << std::endl;
#endif
#endif

    // output bloom filter table
    write_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH, BLOOM_FILTER_DEPTH, bf_vector0);

    write_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH + BLOOM_FILTER_DEPTH, BLOOM_FILTER_DEPTH, bf_vector1);

    write_htb<ARW>(htb_buf, HASH_DEPTH + HASH_OVERFLOW_DEPTH + 2 * BLOOM_FILTER_DEPTH, BLOOM_FILTER_DEPTH, bf_vector2);
}

template <int HASHWH, int HASHWL, int HASHO, int BF_HASH_NM, int BFW, int KEYW, int S_PW, int T_PW, int ARW>
void build_merge_probe_wrapper(
    // input status
    ap_uint<32> build_id,
    ap_uint<32> probe_id,
    const ap_uint<1>& build_probe_flag,
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,

    // input table
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<T_PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e_strm,

    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    ap_uint<64>* htb_buf,
    ap_uint<64>* stb_buf) {
#pragma HLS INLINE off

    // allocate uram storage
    const int HASH_DEPTH = 1 << (HASHWL - 2);
    const int HASH_OVERFLOW_DEPTH = 1 << HASHO;
    const int BLOOM_FILTER_DEPTH = 1 << (BFW - 6);

#ifndef __SYNTHESIS__

    ap_uint<72>* htb_vector0;
    htb_vector0 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));
    ap_uint<72>* htb_vector1;
    htb_vector1 = (ap_uint<72>*)malloc(HASH_OVERFLOW_DEPTH * sizeof(ap_uint<72>));

    ap_uint<72>* bf_vector0;
    ap_uint<72>* bf_vector1;
    ap_uint<72>* bf_vector2;
    bf_vector0 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));
    bf_vector1 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));
    bf_vector2 = (ap_uint<72>*)malloc(BLOOM_FILTER_DEPTH * sizeof(ap_uint<72>));

    //#ifdef DEBUG
    std::cout << std::hex << "HASH_DEPTH=" << HASH_DEPTH << " HASH_OVERFLOW_DEPTH=" << HASH_OVERFLOW_DEPTH
              << " BLOOM_FILTER_DEPTH=" << BLOOM_FILTER_DEPTH << std::endl;
//#endif

#else

    ap_uint<72> htb_vector0[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = htb_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = htb_vector0 type = ram_2p impl = uram

    ap_uint<72> htb_vector1[HASH_OVERFLOW_DEPTH];
#pragma HLS bind_storage variable = htb_vector1 type = ram_2p impl = uram

    ap_uint<72> bf_vector0[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector0 type = ram_2p impl = uram

    ap_uint<72> bf_vector1[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector1 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector1 type = ram_2p impl = uram

    ap_uint<72> bf_vector2[BLOOM_FILTER_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bf_vector2 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bf_vector2 type = ram_2p impl = uram

#endif

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "pu_start_addr=" << pu_start_addr << " build_id=" << build_id << " probe_id=" << probe_id
              << std::endl;
    std::cout << std::dec << "HASH_DEPTH=" << HASH_DEPTH << " HASH_OVERFLOW_DEPTH=" << HASH_OVERFLOW_DEPTH
              << " BLOOM_FILTER_DEPTH=" << BLOOM_FILTER_DEPTH << std::endl;
#endif
#endif

    // initilize htb vector
    initiate_htb<HASHWL, HASHO, ARW>(build_id, htb_buf, htb_vector0, htb_vector1);

    if (build_probe_flag && probe_id == 0 && pu_start_addr > 0) {
// the first time to probe, need to fully build bitmap
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "----------------------merge------------------------" << std::endl;
#endif
#endif

        merge_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, ARW>(
            // input status
            depth, pu_start_addr,

            htb_buf, stb_buf,

            htb_vector0, htb_vector1, bf_vector0, bf_vector1);
    }

    // initilize bf vector
    initiate_bf<HASHWL, HASHO, BF_HASH_NM, BFW, ARW>(build_id, htb_buf, bf_vector0, bf_vector1, bf_vector2);

    if (!build_probe_flag) {
// build start
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "----------------------build------------------------" << std::endl;
#endif
#endif

        build_wrapper<HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, ARW>(
            // input status
            build_id, depth, pu_start_addr, pu_end_addr,

            // input s-table
            i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

            // HBM/DDR
            stb_buf,

            // URAM
            htb_vector0, htb_vector1, bf_vector0, bf_vector1, bf_vector2);

        // update uram vector to hbm
        update_htb<HASHWL, HASHO, ARW>(htb_buf, htb_vector0, htb_vector1);

        update_bf<HASHWL, HASHO, BF_HASH_NM, BFW, ARW>(htb_buf, bf_vector0, bf_vector1, bf_vector2);
    } else {
// probe start
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "-----------------------Probe-----------------------" << std::endl;
#endif
#endif

        probe_wrapper<HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, T_PW, ARW>(
            depth, pu_start_addr,

            // input large table
            i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

            // output for join
            o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e_strm,

            o_base_s_key_strm, o_base_s_pld_strm, o_overflow_s_key_strm, o_overflow_s_pld_strm,

            htb_buf, stb_buf, htb_vector0, htb_vector1, bf_vector0, bf_vector1, bf_vector2);
    }

#ifndef __SYNTHESIS__

    free(htb_vector0);
    free(htb_vector1);
    free(bf_vector0);
    free(bf_vector1);
    free(bf_vector2);

#endif
}

//-----------------------------------------------join-----------------------------------------------
/// @brief compare key, if match output joined row
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit(
#ifndef __SYNTHESIS__
    int pu_id,
#endif

    const ap_uint<1>& build_probe_flag,
    ap_uint<32> depth,

    // input large table
    hls::stream<ap_uint<KEYW> >& i_t_key_strm,
    hls::stream<ap_uint<T_PW> >& i_t_pld_strm,
    hls::stream<ap_uint<ARW> >& i_nm_strm,
    hls::stream<bool>& i_e0_strm,

    // input small table
    hls::stream<ap_uint<KEYW> >& i_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_base_s_pld_strm,
    hls::stream<ap_uint<KEYW> >& i_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& i_overflow_s_pld_strm,

    // output join result
    hls::stream<ap_uint<KEYW + S_PW + T_PW> >& o_j_strm,
    hls::stream<bool>& o_e_strm) {

#pragma HLS INLINE off

    ap_uint<KEYW> s_key;
    ap_uint<S_PW> s_pld;
    ap_uint<KEYW> t_key;
    ap_uint<T_PW> t_pld;
    ap_uint<KEYW + S_PW + T_PW> j;

#ifndef __SYNTHESIS__
    unsigned int cnt0 = 0;
    unsigned int cnt1 = 0;
    unsigned int cnt2 = 0;
    unsigned int cnt3 = 0;

    bool hit_failed;
#endif

    if (build_probe_flag) {
        // do join if it is probe
        bool t_last = i_e0_strm.read();

    JOIN_LOOP:
        while (!t_last) {
            t_key = i_t_key_strm.read();
            t_pld = i_t_pld_strm.read();
            ap_uint<ARW> nm = i_nm_strm.read();
            t_last = i_e0_strm.read();

            ap_uint<ARW> base_nm, overflow_nm;
            if (nm > depth) {
                base_nm = depth;
                overflow_nm = nm - depth;
            } else {
                base_nm = nm;
                overflow_nm = 0;
            }

#ifndef __SYNTHESIS__
            hit_failed = true;
            cnt2++;
#endif

        JOIN_COMPARE_LOOP:
            while (base_nm > 0 || overflow_nm > 0) {
#pragma HLS PIPELINE II = 1

                if (base_nm > 0) {
                    s_key = i_base_s_key_strm.read();
                    s_pld = i_base_s_pld_strm.read();

                    base_nm--;
                } else if (overflow_nm > 0) {
                    s_key = i_overflow_s_key_strm.read();
                    s_pld = i_overflow_s_pld_strm.read();

                    overflow_nm--;
                }

                if (s_key == t_key) {
                    // generate joion result
                    j(KEYW + S_PW + T_PW - 1, S_PW + T_PW) = s_key;
                    if (S_PW > 0) {
                        j(S_PW + T_PW - 1, T_PW) = s_pld;
                    }
                    if (T_PW > 0) {
                        j(T_PW - 1, 0) = t_pld;
                    }

                    o_j_strm.write(j);
                    o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
                    std::cout << std::hex << "Match nm=" << nm << " key=" << s_key << " s_pld=" << s_pld
                              << " t_pld=" << t_pld << " J_res=" << j << std::endl;
#endif
                    cnt0++;
                    hit_failed = false;
                } else {
#ifdef DEBUG_PROBE
                    std::cout << std::hex << "Mismatch nm=" << nm << " s_key=" << s_key << " t_key=" << t_key
                              << " s_pld=" << s_pld << " t_pld=" << t_pld << std::endl;
#endif
                    cnt1++;
#endif
                }
            }

#ifndef __SYNTHESIS__
            if (hit_failed) {
                cnt3++;
            }
#endif
        }

        // for do while in collect
        o_j_strm.write(0);
        o_e_strm.write(true);
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << std::dec << "Join Unit output " << cnt0 << " rows, mismatch " << cnt1 << " rows" << std::endl;
    std::cout << std::dec << "Join Unit hit " << cnt2 << " times, hit failed " << cnt3 << " times" << std::endl;
#endif
#endif
}

} // namespace bp

} // namespace v4
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Hash-Join v4 primitive, using bloom filter to enhance performance of hash join.
 *
 * The build and probe procedure is similar to which in ``hashJoinV3``, and this primitive
 * adds a bloom filter to reduce the redundant access to HBM.
 *
 * The maximum size of small table is 256MBx8=2GB in this design. The total hash entries
 * is equal to 1<<(HASHWH + HASHWL), and it is limitied to maximum of 1M entries because
 * of the size of URAM in a single SLR.
 *
 * This module can accept more than 1 input row per cycle, via multiple input channels.
 * The small table and the big table shares the same input ports, so the width of the
 * payload should be the max of both, while the data should be aligned to the little-end.
 * Similar to ``hashJoinV3``, small table and big table should be fed only once.
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam ARW width of address, log2(small table max num of rows).
 * @tparam CH_NM number of input channels, 1,2,4.
 * @tparam BF_HASH_NM number of bloom filter, 1,2,3.
 * @tparam BF_W bloom-filter hash width.
 * @tparam EN_BF bloom-filter switch, 0 for off, 1 for on.
 *
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 *
 * @param htb0_buf HBM/DDR buffer of hash_table0
 * @param htb1_buf HBM/DDR buffer of hash_table1
 * @param htb2_buf HBM/DDR buffer of hash_table2
 * @param htb3_buf HBM/DDR buffer of hash_table3
 * @param htb4_buf HBM/DDR buffer of hash_table4
 * @param htb5_buf HBM/DDR buffer of hash_table5
 * @param htb6_buf HBM/DDR buffer of hash_table6
 * @param htb7_buf HBM/DDR buffer of hash_table7
 *
 * @param stb0_buf HBM/DDR buffer of PU0
 * @param stb1_buf HBM/DDR buffer of PU1
 * @param stb2_buf HBM/DDR buffer of PU2
 * @param stb3_buf HBM/DDR buffer of PU3
 * @param stb4_buf HBM/DDR buffer of PU4
 * @param stb5_buf HBM/DDR buffer of PU5
 * @param stb6_buf HBM/DDR buffer of PU6
 * @param stb7_buf HBM/DDR buffer of PU7
 *
 * @param pu_begin_status_strms contains build id, fixed hash depth
 * @param pu_end_status_strms returns next build id, fixed hash depth, joined number
 *
 * @param j_strm output of joined result
 * @param j_e_strm end flag of joined result
 */
template <int HASH_MODE,
          int KEYW,
          int PW,
          int S_PW,
          int B_PW,
          int HASHWH,
          int HASHWL,
          int ARW,
          int CH_NM,
          int BF_HASH_NM,
          int BFW,
          bool EN_BF>
static void hashJoinV4(
    // input
    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output hash table
    ap_uint<64>* htb0_buf,
    ap_uint<64>* htb1_buf,
    ap_uint<64>* htb2_buf,
    ap_uint<64>* htb3_buf,
    ap_uint<64>* htb4_buf,
    ap_uint<64>* htb5_buf,
    ap_uint<64>* htb6_buf,
    ap_uint<64>* htb7_buf,

    // output
    ap_uint<64>* stb0_buf,
    ap_uint<64>* stb1_buf,
    ap_uint<64>* stb2_buf,
    ap_uint<64>* stb3_buf,
    ap_uint<64>* stb4_buf,
    ap_uint<64>* stb5_buf,
    ap_uint<64>* stb6_buf,
    ap_uint<64>* stb7_buf,

    hls::stream<ap_uint<32> >& pu_begin_status_strms,
    hls::stream<ap_uint<32> >& pu_end_status_strms,

    hls::stream<ap_uint<KEYW + S_PW + B_PW> >& j_strm,
    hls::stream<bool>& j_e_strm) {
    enum { PU = (1 << HASHWH) }; // high hash for distribution.

#pragma HLS DATAFLOW

    // dispatch k0_strm_arry, p0_strm_arry, e0strm_arry to channel1-4
    // Channel1
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c0 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 1

    // Channel2
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c1 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 1

    // Channel3
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c2 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 1

    // Channel4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c3 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 1

    // pu status
    ap_uint<32> depth;
    ap_uint<32> join_num;

    // merge channel1-channel4 to here, then perform build or probe
    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 512
#pragma HLS array_partition variable = k1_strm_arry dim = 1
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 512
#pragma HLS array_partition variable = p1_strm_arry dim = 1
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 1
#pragma HLS bind_storage variable = hash_strm_arry type = fifo impl = srl
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 1
#pragma HLS bind_storage variable = e1_strm_arry type = fifo impl = srl

    // output of probe for join
    hls::stream<ap_uint<KEYW> > t_key_strm_arry[PU];
#pragma HLS stream variable = t_key_strm_arry depth = 512
#pragma HLS array_partition variable = t_key_strm_arry dim = 1
#pragma HLS bind_storage variable = t_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<B_PW> > t_pld_strm_arry[PU];
#pragma HLS stream variable = t_pld_strm_arry depth = 512
#pragma HLS array_partition variable = t_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = t_pld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<ARW> > nm_strm_arry[PU];
#pragma HLS stream variable = nm_strm_arry depth = 512
#pragma HLS array_partition variable = nm_strm_arry dim = 1
#pragma HLS bind_storage variable = nm_strm_arry type = fifo impl = bram
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 512
#pragma HLS array_partition variable = e2_strm_arry dim = 1
#pragma HLS bind_storage variable = e2_strm_arry type = fifo impl = bram

    hls::stream<ap_uint<KEYW> > s_base_key_strm_arry[PU];
#pragma HLS stream variable = s_base_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<S_PW> > s_base_pld_strm_arry[PU];
#pragma HLS stream variable = s_base_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_pld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<KEYW> > s_overflow_key_strm_arry[PU];
#pragma HLS stream variable = s_overflow_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<S_PW> > s_overflow_pld_strm_arry[PU];
#pragma HLS stream variable = s_overflow_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_pld_strm_arry type = fifo impl = bram

    // output of join for collect
    hls::stream<ap_uint<KEYW + S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 9
#pragma HLS array_partition variable = j0_strm_arry dim = 1
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS array_partition variable = e3_strm_arry dim = 1
#pragma HLS stream variable = e3_strm_arry depth = 9

    //---------------------------------dispatch PU-------------------------------
    if (CH_NM >= 1) {
        details::join_v4::sc::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }

    if (CH_NM >= 2) {
        details::join_v4::sc::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }

    if (CH_NM >= 4) {
        details::join_v4::sc::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);

        details::join_v4::sc::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

    //---------------------------------merge PU---------------------------------
    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v3::sc::merge1_1_wrapper<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge2_1_wrapper<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        // CH_NM == 4
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge4_1_wrapper<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

//------------------------------read status-----------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------read status---------------" << std::endl;
#endif
#endif
    details::join_v3::sc::read_status<PU>(pu_begin_status_strms, depth);

    //-------------------------------build----------------------------------------
    if (PU >= 1) {
#ifndef __SYNTHESIS__
        std::cout << "-----------------PU0-----------------" << std::endl;
#endif

        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0], e1_strm_arry[0],

            // output for join
            t_key_strm_arry[0], t_pld_strm_arry[0], nm_strm_arry[0], e2_strm_arry[0], s_base_key_strm_arry[0],
            s_base_pld_strm_arry[0], s_overflow_key_strm_arry[0], s_overflow_pld_strm_arry[0],

            // HBM/DDR
            htb0_buf, stb0_buf);
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU1------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1], e1_strm_arry[1],

            // output for join
            t_key_strm_arry[1], t_pld_strm_arry[1], nm_strm_arry[1], e2_strm_arry[1], s_base_key_strm_arry[1],
            s_base_pld_strm_arry[1], s_overflow_key_strm_arry[1], s_overflow_pld_strm_arry[1],

            // HBM/DDR
            htb1_buf, stb1_buf);
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU2------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2], e1_strm_arry[2],

            // output for join
            t_key_strm_arry[2], t_pld_strm_arry[2], nm_strm_arry[2], e2_strm_arry[2], s_base_key_strm_arry[2],
            s_base_pld_strm_arry[2], s_overflow_key_strm_arry[2], s_overflow_pld_strm_arry[2],

            // HBM/DDR
            htb2_buf, stb2_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU3------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3], e1_strm_arry[3],

            // output for join
            t_key_strm_arry[3], t_pld_strm_arry[3], nm_strm_arry[3], e2_strm_arry[3], s_base_key_strm_arry[3],
            s_base_pld_strm_arry[3], s_overflow_key_strm_arry[3], s_overflow_pld_strm_arry[3],

            // HBM/DDR
            htb3_buf, stb3_buf);
    }

    if (PU >= 8) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU4------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[4], k1_strm_arry[4], p1_strm_arry[4], e1_strm_arry[4],

            // output for join
            t_key_strm_arry[4], t_pld_strm_arry[4], nm_strm_arry[4], e2_strm_arry[4], s_base_key_strm_arry[4],
            s_base_pld_strm_arry[4], s_overflow_key_strm_arry[4], s_overflow_pld_strm_arry[4],

            // HBM/DDR
            htb4_buf, stb4_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU5------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[5], k1_strm_arry[5], p1_strm_arry[5], e1_strm_arry[5],

            // output for join
            t_key_strm_arry[5], t_pld_strm_arry[5], nm_strm_arry[5], e2_strm_arry[5], s_base_key_strm_arry[5],
            s_base_pld_strm_arry[5], s_overflow_key_strm_arry[5], s_overflow_pld_strm_arry[5],

            // HBM/DDR
            htb5_buf, stb5_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU6------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[6], k1_strm_arry[6], p1_strm_arry[6], e1_strm_arry[6],

            // output for join
            t_key_strm_arry[6], t_pld_strm_arry[6], nm_strm_arry[6], e2_strm_arry[6], s_base_key_strm_arry[6],
            s_base_pld_strm_arry[6], s_overflow_key_strm_arry[6], s_overflow_pld_strm_arry[6],

            // HBM/DDR
            htb6_buf, stb6_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU7------------------------" << std::endl;
#endif
#endif
        details::join_v4::sc::build_probe_wrapper<HASHWH, HASHWL, BF_HASH_NM, BFW, KEYW, PW, S_PW, B_PW, ARW>(
            // input status
            depth,

            // input t-table
            hash_strm_arry[7], k1_strm_arry[7], p1_strm_arry[7], e1_strm_arry[7],

            // output for join
            t_key_strm_arry[7], t_pld_strm_arry[7], nm_strm_arry[7], e2_strm_arry[7], s_base_key_strm_arry[7],
            s_base_pld_strm_arry[7], s_overflow_key_strm_arry[7], s_overflow_pld_strm_arry[7],

            // HBM/DDR
            htb7_buf, stb7_buf);
    }

    //-----------------------------------join--------------------------------------
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v4::sc::join_unit<KEYW, S_PW, B_PW, ARW>(
#ifndef __SYNTHESIS__
            i,
#endif

            depth, t_key_strm_arry[i], t_pld_strm_arry[i], nm_strm_arry[i], e2_strm_arry[i], s_base_key_strm_arry[i],
            s_base_pld_strm_arry[i], s_overflow_key_strm_arry[i], s_overflow_pld_strm_arry[i], j0_strm_arry[i],
            e3_strm_arry[i]);
    }

    //-----------------------------------Collect-----------------------------------
    details::join_v3::sc::collect_unit<PU, KEYW + S_PW + B_PW>(j0_strm_arry, e3_strm_arry, join_num, j_strm, j_e_strm);

    //------------------------------Write status-----------------------------------
    details::join_v3::sc::write_status<PU>(pu_end_status_strms, depth, join_num);

} // hash_join_v4

/**
 * @brief Hash-Build-Probe v4 primitive. Compared with ``HashBuildProbeV3``, it enables bloom filter to
 * reduce redundant access to HBM which can further reduce run-time of hash join. Build and probe are
 * separately performed and controlled by a boolean flag. Mutiple build and probe are also provided, and
 * it should make sure all rows in build phase can be stored temporarily in HBM to maintain correctness.
 *
 * The maximum size of small table is 256MBx8=2GB in this design. The total hash entries is equal to
 * 1<<(HASHWH + HASHWL), and it is limitied to maximun of 1M entries because of the size of URAM in a
 * single SLR.
 *
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam HASHO number of hash bits used for overflow hash counter, 8-12.
 * @tparam ARW width of address, log2(small table max num of rows).
 * @tparam CH_NM number of input channels, 1,2,4.
 * @tparam BF_HASH_NM number of hash functions in bloom filter, 1,2,3.
 * @tparam BFW bloom-filter hash width.
 * @tparam EN_BF bloom-filter switch, 0 for off, 1 for on.
 *
 * @param build_probe_flag 0:build 1:probe
 * @param k0_strm_arry input of key columns of both tables.
 * @param p0_strm_arry input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 *
 * @param htb0_buf HBM/DDR buffer of hash_table0
 * @param htb1_buf HBM/DDR buffer of hash_table1
 * @param htb2_buf HBM/DDR buffer of hash_table2
 * @param htb3_buf HBM/DDR buffer of hash_table3
 * @param htb4_buf HBM/DDR buffer of hash_table4
 * @param htb5_buf HBM/DDR buffer of hash_table5
 * @param htb6_buf HBM/DDR buffer of hash_table6
 * @param htb7_buf HBM/DDR buffer of hash_table7
 *
 * @param stb0_buf HBM/DDR buffer of PU0
 * @param stb1_buf HBM/DDR buffer of PU1
 * @param stb2_buf HBM/DDR buffer of PU2
 * @param stb3_buf HBM/DDR buffer of PU3
 * @param stb4_buf HBM/DDR buffer of PU4
 * @param stb5_buf HBM/DDR buffer of PU5
 * @param stb6_buf HBM/DDR buffer of PU6
 * @param stb7_buf HBM/DDR buffer of PU7
 *
 * @param pu_begin_status_strms contains build ID, probe ID, fixed hash depth, joined number of last probe
 * and start addr of unused stb_buf for each PU
 * @param pu_end_status_strms returns next build ID, next probe ID, fixed hash depth, joined number of
 * current probe and end addr of stb_buf for each PU
 *
 * @param j_strm output of joined rows.
 * @param j_e_strm is the end flag of joined result.
 */
template <int KEYW,
          int PW,
          int S_PW,
          int B_PW,
          int HASHWH,
          int HASHWL,
          int HASHO,
          int ARW,
          int CH_NM,
          int BF_HASH_NM,
          int BFW,
          int EN_BF>
static void hashBuildProbeV4(
    // input
    bool& build_probe_flag,
    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output hash table
    ap_uint<64>* htb0_buf,
    ap_uint<64>* htb1_buf,
    ap_uint<64>* htb2_buf,
    ap_uint<64>* htb3_buf,
    ap_uint<64>* htb4_buf,
    ap_uint<64>* htb5_buf,
    ap_uint<64>* htb6_buf,
    ap_uint<64>* htb7_buf,

    // output
    ap_uint<64>* stb0_buf,
    ap_uint<64>* stb1_buf,
    ap_uint<64>* stb2_buf,
    ap_uint<64>* stb3_buf,
    ap_uint<64>* stb4_buf,
    ap_uint<64>* stb5_buf,
    ap_uint<64>* stb6_buf,
    ap_uint<64>* stb7_buf,

    hls::stream<ap_uint<32> >& pu_begin_status_strms,
    hls::stream<ap_uint<32> >& pu_end_status_strms,

    hls::stream<ap_uint<KEYW + S_PW + B_PW> >& j_strm,
    hls::stream<bool>& j_e_strm) {
    enum { PU = (1 << HASHWH) }; // high hash for distribution.

#pragma HLS DATAFLOW

    // dispatch k0_strm_arry, p0_strm_arry, e0strm_arry to channel1-4
    // Channel1
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c0 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 1

    // Channel2
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c1 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 1

    // Channel3
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c2 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 1

    // Channel4
    hls::stream<ap_uint<KEYW> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = k1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c3 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 1

    // pu status
    ap_uint<32> pu_start_addr[PU];
#pragma HLS array_partition variable = pu_start_addr
    ap_uint<32> pu_end_addr[PU];
#pragma HLS array_partition variable = pu_end_addr

    ap_uint<32> build_id;
    ap_uint<32> probe_id;
    ap_uint<32> depth;
    ap_uint<32> join_num;

    // merge channel1-channel4 to here, then perform build or probe
    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 512
#pragma HLS array_partition variable = k1_strm_arry dim = 1
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 512
#pragma HLS array_partition variable = p1_strm_arry dim = 1
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<HASHWL + BF_HASH_NM * BFW> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 512
#pragma HLS array_partition variable = hash_strm_arry dim = 1
#pragma HLS bind_storage variable = hash_strm_arry type = fifo impl = bram
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 512
#pragma HLS array_partition variable = e1_strm_arry dim = 1
#pragma HLS bind_storage variable = e1_strm_arry type = fifo impl = srl

    // output of probe for join
    hls::stream<ap_uint<KEYW> > t_key_strm_arry[PU];
#pragma HLS stream variable = t_key_strm_arry depth = 512
#pragma HLS array_partition variable = t_key_strm_arry dim = 1
#pragma HLS bind_storage variable = t_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > t_pld_strm_arry[PU];
#pragma HLS stream variable = t_pld_strm_arry depth = 512
#pragma HLS array_partition variable = t_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = t_pld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<ARW> > nm_strm_arry[PU];
#pragma HLS stream variable = nm_strm_arry depth = 512
#pragma HLS array_partition variable = nm_strm_arry dim = 1
#pragma HLS bind_storage variable = nm_strm_arry type = fifo impl = bram
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 512
#pragma HLS array_partition variable = e2_strm_arry dim = 1
#pragma HLS bind_storage variable = e2_strm_arry type = fifo impl = srl

    hls::stream<ap_uint<KEYW> > s_base_key_strm_arry[PU];
#pragma HLS stream variable = s_base_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > s_base_pld_strm_arry[PU];
#pragma HLS stream variable = s_base_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_base_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_base_pld_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<KEYW> > s_overflow_key_strm_arry[PU];
#pragma HLS stream variable = s_overflow_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > s_overflow_pld_strm_arry[PU];
#pragma HLS stream variable = s_overflow_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_overflow_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_overflow_pld_strm_arry type = fifo impl = bram

    // output of join for collect
    hls::stream<ap_uint<KEYW + S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 9
#pragma HLS array_partition variable = j0_strm_arry dim = 1
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS array_partition variable = e3_strm_arry dim = 1
#pragma HLS stream variable = e3_strm_arry depth = 9

    //------------------------------read status--------------------------------
    details::join_v4::bp::read_status<PU>(pu_begin_status_strms, build_id, probe_id, depth, pu_start_addr);

    //------------------------------dispatch PU--------------------------------
    if (CH_NM >= 1) {
        details::join_v4::bp::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }

    if (CH_NM >= 2) {
        details::join_v4::bp::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }

    if (CH_NM >= 4) {
        details::join_v4::bp::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);

        details::join_v4::bp::dispatch_wrapper<KEYW, PW, HASHWH, HASHWL, BF_HASH_NM, BFW, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

    //---------------------------------merge PU---------------------------------
    if (CH_NM == 1) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge1_1<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge2_1<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 4) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge4_1<KEYW, PW, HASHWL + BF_HASH_NM * BFW>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    //-------------------------------PU----------------------------------------
    if (PU >= 1) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU0------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[0], pu_end_addr[0],

            // input table
            hash_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0], e1_strm_arry[0],

            // output for join
            t_key_strm_arry[0], t_pld_strm_arry[0], nm_strm_arry[0], e2_strm_arry[0], s_base_key_strm_arry[0],
            s_base_pld_strm_arry[0], s_overflow_key_strm_arry[0], s_overflow_pld_strm_arry[0],

            // HBM/DDR
            htb0_buf, stb0_buf);
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU1------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[1], pu_end_addr[1],

            // input t-table
            hash_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1], e1_strm_arry[1],

            // output for join
            t_key_strm_arry[1], t_pld_strm_arry[1], nm_strm_arry[1], e2_strm_arry[1], s_base_key_strm_arry[1],
            s_base_pld_strm_arry[1], s_overflow_key_strm_arry[1], s_overflow_pld_strm_arry[1],

            // HBM/DDR
            htb1_buf, stb1_buf);
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU2------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[2], pu_end_addr[2],

            // input t-table
            hash_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2], e1_strm_arry[2],

            // output for join
            t_key_strm_arry[2], t_pld_strm_arry[2], nm_strm_arry[2], e2_strm_arry[2], s_base_key_strm_arry[2],
            s_base_pld_strm_arry[2], s_overflow_key_strm_arry[2], s_overflow_pld_strm_arry[2],

            // HBM/DDR
            htb2_buf, stb2_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU3------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[3], pu_end_addr[3],

            // input t-table
            hash_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3], e1_strm_arry[3],

            // output for join
            t_key_strm_arry[3], t_pld_strm_arry[3], nm_strm_arry[3], e2_strm_arry[3], s_base_key_strm_arry[3],
            s_base_pld_strm_arry[3], s_overflow_key_strm_arry[3], s_overflow_pld_strm_arry[3],

            // HBM/DDR
            htb3_buf, stb3_buf);
    }

    if (PU >= 8) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU4------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[4], pu_end_addr[4],

            // input t-table
            hash_strm_arry[4], k1_strm_arry[4], p1_strm_arry[4], e1_strm_arry[4],

            // output for join
            t_key_strm_arry[4], t_pld_strm_arry[4], nm_strm_arry[4], e2_strm_arry[4], s_base_key_strm_arry[4],
            s_base_pld_strm_arry[4], s_overflow_key_strm_arry[4], s_overflow_pld_strm_arry[4],

            // HBM/DDR
            htb4_buf, stb4_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU5------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[5], pu_end_addr[5],

            // input t-table
            hash_strm_arry[5], k1_strm_arry[5], p1_strm_arry[5], e1_strm_arry[5],

            // output for join
            t_key_strm_arry[5], t_pld_strm_arry[5], nm_strm_arry[5], e2_strm_arry[5], s_base_key_strm_arry[5],
            s_base_pld_strm_arry[5], s_overflow_key_strm_arry[5], s_overflow_pld_strm_arry[5],

            // HBM/DDR
            htb5_buf, stb5_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU6------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[6], pu_end_addr[6],

            // input t-table
            hash_strm_arry[6], k1_strm_arry[6], p1_strm_arry[6], e1_strm_arry[6],

            // output for join
            t_key_strm_arry[6], t_pld_strm_arry[6], nm_strm_arry[6], e2_strm_arry[6], s_base_key_strm_arry[6],
            s_base_pld_strm_arry[6], s_overflow_key_strm_arry[6], s_overflow_pld_strm_arry[6],

            // HBM/DDR
            htb6_buf, stb6_buf);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "------------------------PU7------------------------" << std::endl;
#endif
#endif
        details::join_v4::bp::build_merge_probe_wrapper<HASHWH, HASHWL, HASHO, BF_HASH_NM, BFW, KEYW, S_PW, B_PW, ARW>(
            // input status
            build_id, probe_id, build_probe_flag, depth, pu_start_addr[7], pu_end_addr[7],

            // input t-table
            hash_strm_arry[7], k1_strm_arry[7], p1_strm_arry[7], e1_strm_arry[7],

            // output for join
            t_key_strm_arry[7], t_pld_strm_arry[7], nm_strm_arry[7], e2_strm_arry[7], s_base_key_strm_arry[7],
            s_base_pld_strm_arry[7], s_overflow_key_strm_arry[7], s_overflow_pld_strm_arry[7],

            // HBM/DDR
            htb7_buf, stb7_buf);
    }

    //-----------------------------------join---------------------------------
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v4::bp::join_unit<KEYW, S_PW, B_PW, ARW>(
#ifndef __SYNTHESIS__
            i,
#endif

            build_probe_flag, depth, t_key_strm_arry[i], t_pld_strm_arry[i], nm_strm_arry[i], e2_strm_arry[i],
            s_base_key_strm_arry[i], s_base_pld_strm_arry[i], s_overflow_key_strm_arry[i], s_overflow_pld_strm_arry[i],
            j0_strm_arry[i], e3_strm_arry[i]);
    }

    //----------------------------collect_join_unit---------------------------
    details::join_v3::bp::collect_unit<PU, KEYW + S_PW + B_PW>(build_probe_flag, j0_strm_arry, e3_strm_arry, join_num,
                                                               j_strm, j_e_strm);

    //------------------------------Write status------------------------------
    details::join_v4::bp::write_status<PU>(build_probe_flag, pu_end_status_strms, build_id, probe_id, depth, join_num,
                                           pu_end_addr);

} // hash_build_probe_v4

} // namespace database
} // namespace xf

#undef write_bit_vector0
#undef read_bit_vector0

#undef write_bit_vector1
#undef read_bit_vector1

#endif // !defined(XF_DATABASE_HASH_JOIN_V4_H)
