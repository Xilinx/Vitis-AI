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
 * @file hash_join_v3.hpp
 * @brief hash join implementation, targeting HBM devices.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_JOIN_V3_H
#define XF_DATABASE_HASH_JOIN_V3_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_database/combine_split_col.hpp"
#include "xf_database/hash_lookup3.hpp"
#include "xf_database/hash_join_v2.hpp"
#include "xf_database/utils.hpp"

//#define DEBUG true

#ifdef DEBUG

//#define DEBUG_BUILD true
//#define DEBUG_PROBE true
//#define DEBUG_JOIN true
//#define DEBUG_MISS true
//#define DEBUG_HBM true

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
namespace join_v3 {

namespace sc {
//----------------------------read status-------------------------------------
template <int PU>
void read_status(hls::stream<ap_uint<32> >& pu_begin_status_strms, ap_uint<32>& depth) {
    // get depth
    depth = pu_begin_status_strms.read();
    // disgard join number
    pu_begin_status_strms.read();
}

//---------------------------write status-------------------------------------
template <int PU>
void write_status(hls::stream<ap_uint<32> >& pu_end_status_strms, ap_uint<32> depth, ap_uint<32> join_num) {
    // write back depth
    pu_end_status_strms.write(depth);

    // write back join number
    pu_end_status_strms.write(join_num);
}

// --------------------------read write HBM------------------------------------

// read 256bit hash table from HBM/DDR
template <int HASHW>
void read_bitmap(ap_uint<256>* htb_buf,
                 ap_uint<64> addr_shift,

                 hls::stream<ap_uint<256> >& o_bitmap_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    ap_uint<64> bit_vector_addr = 0;
    ap_uint<64> htb_start_addr = addr_shift;
    ap_uint<64> htb_end_addr = addr_shift + HASH_DEPTH;

READ_BUFF_LOOP:
    for (int i = htb_start_addr; i < htb_end_addr; i++) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<256>
        ap_uint<256> bitmap_temp = htb_buf[i];
        o_bitmap_strm.write(bitmap_temp);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// load 256bit hash table into 72bit
template <int ARW>
void combine_bitmap(hls::stream<ap_uint<256> >& i_bitmap_strm,
                    hls::stream<bool>& i_e_strm,

                    hls::stream<ap_uint<72> >& o_bitmap_strm,
                    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    int cnt = 0;
#endif

    ap_uint<64> status = false;
    ap_uint<72> bitmap = 0;
    ap_uint<64> bitmap_temp0 = 0;
    ap_uint<64> bitmap_temp1 = 0;
    ap_uint<256> bitmap_temp = 0;

    bool last = i_e_strm.read();
COMBINE_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read based on width of ap_uint<256>
        bitmap_temp = i_bitmap_strm.read();
        last = i_e_strm.read();

        bitmap(71, 0) = bitmap_temp(71, 0);

        o_bitmap_strm.write(bitmap);
        o_e_strm.write(false);

#ifndef __SYNTHESIS__
        cnt++;
#endif
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "HTB read " << cnt << " lines" << std::endl;
#endif
}

// write hash table stream to URAM
template <int ARW>
void write_uram(hls::stream<ap_uint<72> >& i_bitmap_strm,
                hls::stream<bool>& i_e_strm,

                ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    ap_uint<72> previous_bitmap = 0;
    int cnt = 0;
    int cnt_depth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

    ap_uint<ARW> bit_vector_addr = 0;

    bool last = i_e_strm.read();
WRITE_URAM_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = bit_vector inter false

        ap_uint<72> bitmap = i_bitmap_strm.read();
        last = i_e_strm.read();

        bit_vector[bit_vector_addr] = bitmap;

#ifndef __SYNTHESIS__
        if (previous_bitmap != bitmap && bitmap != 0) {
#ifdef DEBUG
            std::cout << std::hex << "read htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap << std::endl;
#endif
            previous_bitmap = bitmap;
            cnt++;
        }

        for (int i = 0; i < 16; i++) {
            if (bitmap(23, 0) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(47, 24) == i) {
                cnt_depth[i]++;
            }
            if (bitmap(71, 48) == i) {
                cnt_depth[i]++;
            }
        }
#endif

        bit_vector_addr++;
    }

#ifndef __SYNTHESIS__
#ifdef DEBUG
    for (int i = 0; i < 16; i++) {
        std::cout << std::dec << "cnt of depth " << i << " = " << cnt_depth[i] << std::endl;
    }
#endif

    std::cout << std::dec << "HTB have " << cnt << " non-empty lines" << std::endl;
#endif
}

/// @brief Read hash table from HBM/DDR to URAM
template <int HASHW, int ARW>
void read_htb(ap_uint<256>* htb_buf, ap_uint<64> addr_shift, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<256> > bitmap0_strm;
#pragma HLS STREAM variable = bitmap0_strm depth = 512
#pragma HLS bind_storage variable = bitmap0_strm type = fifo impl = bram
    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 512
#pragma HLS bind_storage variable = e0_strm type = fifo impl = bram

    hls::stream<ap_uint<72> > bitmap1_strm;
#pragma HLS STREAM variable = bitmap1_strm depth = 8
#pragma HLS bind_storage variable = bitmap1_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8

    read_bitmap<HASHW>(htb_buf, addr_shift, bitmap0_strm, e0_strm);

    combine_bitmap<ARW>(bitmap0_strm, e0_strm,

                        bitmap1_strm, e1_strm);

    write_uram<ARW>(bitmap1_strm, e1_strm,

                    bit_vector);
}

// read 72bit hash table from URAM
template <int HASHW, int ARW>
void read_uram(hls::stream<ap_uint<72> >& o_bitmap_strm, hls::stream<bool>& o_e_strm, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    ap_uint<72> previous_bitmap = 0;
#endif

    bool status = false;
    ap_uint<72> bitmap = 0;
    ap_uint<ARW> bit_vector_addr = 0;

READ_URAM_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = bit_vector inter false

        // read
        bitmap = bit_vector[bit_vector_addr];

#ifndef __SYNTHESIS__
        if (previous_bitmap != bitmap && bitmap != 0) {
#ifdef DEBUG
            std::cout << std::hex << "write htb: addr=" << bit_vector_addr << ", hash_table=" << bitmap << std::endl;
#endif
            previous_bitmap = bitmap;
            cnt++;
        }
#endif

        bit_vector_addr++;

        o_bitmap_strm.write(bitmap);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "HTB update " << cnt << " lines" << std::endl;
#endif
}

// load 72bit hash table into two 256bit hash table
template <int HASHW, int ARW>
void split_bitmap(hls::stream<ap_uint<72> >& i_bitmap_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<256> >& o_bitmap_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    ap_uint<72> bitmap = 0;
    ap_uint<256> bitmap_temp = 0;

    bool last = i_e_strm.read();
SPLIT_BITMAP_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        // read
        bitmap = i_bitmap_strm.read();
        last = i_e_strm.read();

        bitmap_temp(71, 0) = bitmap(71, 0);

        // write strm
        o_bitmap_strm.write(bitmap_temp);
        o_e_strm.write(false);
    }

    o_e_strm.write(true);
}

// write hash table strm to HBM/DDR
template <int HASHW, int ARW>
void write_bitmap(hls::stream<ap_uint<256> >& i_bitmap_strm,
                  hls::stream<bool>& i_e_strm,

                  ap_uint<64> addr_shift,
                  ap_uint<256>* htb_buf) {
#pragma HLS INLINE off

    ap_uint<64> htb_addr = addr_shift;

    bool last = i_e_strm.read();
WRITE_BUFF_LOOP:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<256> bitmap_temp = i_bitmap_strm.read();
        last = i_e_strm.read();

        htb_buf[htb_addr] = bitmap_temp;
        htb_addr++;
    }
}

/// @brief Write hash table from URAM to HBM/DDR
template <int HASHW, int ARW>
void write_htb(ap_uint<256>* htb_buf, ap_uint<64> addr_shift, ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<72> > bitmap0_strm;
#pragma HLS STREAM variable = bitmap0_strm depth = 8
#pragma HLS bind_storage variable = bitmap0_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS STREAM variable = e0_strm depth = 8

    hls::stream<ap_uint<256> > bitmap1_strm;
#pragma HLS STREAM variable = bitmap1_strm depth = 512
#pragma HLS bind_storage variable = bitmap1_strm type = fifo impl = bram
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 512
#pragma HLS bind_storage variable = e1_strm type = fifo impl = bram

    read_uram<HASHW, ARW>(bitmap0_strm, e0_strm, bit_vector);

    split_bitmap<HASHW, ARW>(bitmap0_strm, e0_strm, bitmap1_strm, e1_strm);

    write_bitmap<HASHW, ARW>(bitmap1_strm, e1_strm, addr_shift, htb_buf);
}

// generate addr for read/write stb
template <int RW, int ARW>
void stb_addr_gen(hls::stream<ap_uint<ARW> >& i_addr_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<ARW> >& o_addr_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int number_of_element_per_row =
        (RW % 256 == 0) ? RW / 256 : RW / 256 + 1; // number of output based on 256 bit

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

// read row from HBM/DDR
template <int RW, int ARW>
void read_row(ap_uint<256>* stb_buf,
              hls::stream<ap_uint<ARW> >& i_addr_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<256> >& o_row_strm,
              hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<ARW> addr = i_addr_strm.read();
        last = i_e_strm.read();

        ap_uint<256> element = stb_buf[addr];
        o_row_strm.write(element);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}

// combine several 256 bit stream into one row
template <int RW, int ARW>
void combine_row(hls::stream<ap_uint<256> >& i_row_strm,
                 hls::stream<bool>& i_e_strm,

                 hls::stream<ap_uint<RW> >& o_row_strm,
                 hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_element_per_row =
        (RW % 256 == 0) ? RW / 256 : RW / 256 + 1; // number of output based on 256 bit

    bool last = i_e_strm.read();
    ap_uint<256 * number_of_element_per_row> row_temp = 0;
    ap_uint<4> mux = 0;

    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<256> element = i_row_strm.read();
        last = i_e_strm.read();
        row_temp((mux + 1) * 256 - 1, mux * 256) = element;

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

/// @brief Read s-table from HBM/DDR
template <int ARW, int RW>
void read_stb(ap_uint<256>* stb_buf,

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

    hls::stream<ap_uint<256> > row_strm;
#pragma HLS STREAM variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS STREAM variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    stb_addr_gen<RW, ARW>(i_addr_strm, i_e_strm,

                          addr_strm, e0_strm);

    read_row<RW, ARW>(stb_buf, addr_strm, e0_strm,

                      row_strm, e1_strm);

    combine_row<RW, ARW>(row_strm, e1_strm,

                         o_row_strm, o_e_strm);
}

// duplicate strm end
template <typename type_t>
void duplicate_strm_end(hls::stream<type_t>& i_e_strm, hls::stream<type_t>& o_e0_strm, hls::stream<type_t>& o_e1_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        last = i_e_strm.read();

        o_e0_strm.write(false);
        o_e1_strm.write(false);
    }
    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

// eleminate temporary strm_end
template <typename type_t>
void eliminate_strm_end(hls::stream<type_t>& strm_end) {
#pragma HLS INLINE off

    bool end = strm_end.read();
    while (!end) {
        end = strm_end.read();
    }
}

// write row to HBM/DDR
template <int RW, int ARW>
void write_row(ap_uint<256>* stb_buf,
               hls::stream<ap_uint<ARW> >& i_addr_strm,
               hls::stream<ap_uint<256> >& i_row_strm,
               hls::stream<bool>& i_e_strm) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
    while (!last) {
#pragma HLS PIPELINE II = 1
        ap_uint<ARW> addr = i_addr_strm.read();
        ap_uint<256> row = i_row_strm.read();
        last = i_e_strm.read();

        stb_buf[addr] = row;
    }
}

// split row into several 256 bit element
template <int RW, int ARW>
void split_row(hls::stream<ap_uint<RW> >& i_row_strm,
               hls::stream<bool>& i_e_strm,

               hls::stream<ap_uint<256> >& o_row_strm,
               hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_element_per_row =
        (RW % 256 == 0) ? RW / 256 : RW / 256 + 1; // number of output based on 256 bit

    bool last = i_e_strm.read();
    ap_uint<256 * number_of_element_per_row> row_temp = 0;

    while (!last) {
#pragma HLS PIPELINE II = number_of_element_per_row
        row_temp = i_row_strm.read();
        last = i_e_strm.read();
        ap_uint<4> mux = 0;

        for (int i = 0; i < number_of_element_per_row; i++) {
            ap_uint<256> element = row_temp((mux + 1) * 256 - 1, mux * 256);
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

/// @brief Write s-table to HBM/DDR
template <int ARW, int RW>
void write_stb(ap_uint<256>* stb_buf,

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

    hls::stream<ap_uint<256> > row_strm;
#pragma HLS STREAM variable = row_strm depth = 512
#pragma HLS bind_storage variable = row_strm type = fifo impl = bram
    hls::stream<bool> e3_strm;
#pragma HLS STREAM variable = e3_strm depth = 512
#pragma HLS bind_storage variable = e3_strm type = fifo impl = srl

    duplicate_strm_end<bool>(i_e_strm, e0_strm, e1_strm);

    stb_addr_gen<RW, ARW>(i_addr_strm, e0_strm, addr_strm, e2_strm);

    split_row<RW, ARW>(i_row_strm, e1_strm, row_strm, e3_strm);

    write_row<RW, ARW>(stb_buf, addr_strm, row_strm, e3_strm);

    eliminate_strm_end<bool>(e2_strm);
}

// ---------------------------------hash---------------------------------------

/// @brief Calculate hash value based on key
template <int HASH_MODE, int KEYW, int HASHW>
void hash_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<ap_uint<HASHW> >& o_hash_strm,
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

            if (HASH_MODE == 0) {
                // radix hash
                ap_uint<HASHW> s_hash_val = key(HASHW - 1, 0);
                o_hash_strm.write(s_hash_val);

            } else {
                // Jekins lookup3 hash
                key_strm_in.write(key);
                database::hashLookup3<KEYW>(key_strm_in, hash_strm_out);
                ap_uint<64> l_hash_val = hash_strm_out.read();

                ap_uint<HASHW> s_hash_val = l_hash_val(HASHW - 1, 0);
                o_hash_strm.write(s_hash_val);

#ifndef __SYNTHESIS__

#ifdef DEBUG_MISS
                if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782)
                    std::cout << std::hex << "hashwrapper: key=" << key << " hash=" << s_hash_val << std::endl;
#endif

#ifdef DEBUG
                if (cnt < 10) {
                    std::cout << std::hex << "hash wrapper: cnt=" << cnt << " key = " << key
                              << " hash_val = " << s_hash_val << std::endl;
                }
#endif
                cnt++;
#endif
            }
        }
    }
    o_e_strm.write(true);
}

//-------------------------------dispatch--------------------------------------

/// @brief Dispatch data to multiple PU based one the hash value, every PU with
/// different high bit hash_value.
template <int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch(hls::stream<ap_uint<KEYW> >& i_key_strm,
              hls::stream<ap_uint<PW> >& i_pld_strm,
              hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
              hls::stream<bool>& i_e_strm,

              hls::stream<ap_uint<KEYW> > o_key_strm[PU],
              hls::stream<ap_uint<PW> > o_pld_strm[PU],
              hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
              hls::stream<bool> o_e_strm[PU]) {
#pragma HLS INLINE off

    bool last = i_e_strm.read();
LOOP_DISPATCH:
    while (!last) {
#pragma HLS pipeline II = 1

        ap_uint<HASHWH + HASHWL> hash_val = i_hash_strm.read();
        ap_uint<HASHWL> hash_out = hash_val(HASHWL - 1, 0);
        ap_uint<4> idx;
        if (HASHWH > 0)
            idx = hash_val(HASHWH + HASHWL - 1, HASHWL);
        else
            idx = 0;

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
template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_unit(hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,

                   hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                   hls::stream<ap_uint<PW> > o_pld_strm[PU],
                   hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                   hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHWH + HASHWL> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS STREAM variable = key_strm depth = 32
#pragma HLS resource variable = key_strm core = FIFO_SRL
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 32

    hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(i_key_strm, i_e_strm, hash_strm, key_strm, e_strm);

    dispatch<KEYW, PW, HASHWH, HASHWL, PU>(key_strm, i_pld_strm, hash_strm, e_strm, o_key_strm, o_pld_strm, o_hash_strm,
                                           o_e_strm);
}

template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,

                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
    // 1st:build
    // 2nd:probe
    for (int r = 0; r < 2; r++) {
        dispatch_unit<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(i_key_strm, i_pld_strm, i_e_strm, o_key_strm, o_pld_strm,
                                                               o_hash_strm, o_e_strm);
    }
}

template <int KEYW, int PW, int HASHW>
void merge1_1_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i_hash_strm,
                      hls::stream<bool>& i_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    // 1st:build
    // 2nd:probe
    for (int r = 0; r < 2; r++) {
        details::join_v2::merge1_1<KEYW, PW, HASHW>(i_key_strm, i_pld_strm, i_hash_strm, i_e_strm, o_key_strm,
                                                    o_pld_strm, o_hash_strm, o_e_strm);
    }
}

template <int KEYW, int PW, int HASHW>
void merge2_1_wrapper(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                      hls::stream<ap_uint<KEYW> >& i1_key_strm,
                      hls::stream<ap_uint<PW> >& i0_pld_strm,
                      hls::stream<ap_uint<PW> >& i1_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i0_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i1_hash_strm,
                      hls::stream<bool>& i0_e_strm,
                      hls::stream<bool>& i1_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    // 1st:build
    // 2nd:probe
    for (int r = 0; r < 2; r++) {
        details::join_v2::merge2_1<KEYW, PW, HASHW>(i0_key_strm, i1_key_strm, i0_pld_strm, i1_pld_strm, i0_hash_strm,
                                                    i1_hash_strm, i0_e_strm, i1_e_strm, o_key_strm, o_pld_strm,
                                                    o_hash_strm, o_e_strm);
    }
}

template <int KEYW, int PW, int HASHW>
void merge4_1_wrapper(hls::stream<ap_uint<KEYW> >& i0_key_strm,
                      hls::stream<ap_uint<KEYW> >& i1_key_strm,
                      hls::stream<ap_uint<KEYW> >& i2_key_strm,
                      hls::stream<ap_uint<KEYW> >& i3_key_strm,
                      hls::stream<ap_uint<PW> >& i0_pld_strm,
                      hls::stream<ap_uint<PW> >& i1_pld_strm,
                      hls::stream<ap_uint<PW> >& i2_pld_strm,
                      hls::stream<ap_uint<PW> >& i3_pld_strm,
                      hls::stream<ap_uint<HASHW> >& i0_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i1_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i2_hash_strm,
                      hls::stream<ap_uint<HASHW> >& i3_hash_strm,
                      hls::stream<bool>& i0_e_strm,
                      hls::stream<bool>& i1_e_strm,
                      hls::stream<bool>& i2_e_strm,
                      hls::stream<bool>& i3_e_strm,
                      hls::stream<ap_uint<KEYW> >& o_key_strm,
                      hls::stream<ap_uint<PW> >& o_pld_strm,
                      hls::stream<ap_uint<HASHW> >& o_hash_strm,
                      hls::stream<bool>& o_e_strm) {
    // 1st:build
    // 2nd:probe
    for (int r = 0; r < 2; r++) {
        details::join_v2::merge4_1<KEYW, PW, HASHW>(i0_key_strm, i1_key_strm, i2_key_strm, i3_key_strm, i0_pld_strm,
                                                    i1_pld_strm, i2_pld_strm, i3_pld_strm, i0_hash_strm, i1_hash_strm,
                                                    i2_hash_strm, i3_hash_strm, i0_e_strm, i1_e_strm, i2_e_strm,
                                                    i3_e_strm, o_key_strm, o_pld_strm, o_hash_strm, o_e_strm);
    }
}

// -------------------------------------build------------------------------------------

/// @brief Scan small table to count hash collision, build small table to its
/// hash addr
template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_unit(ap_uint<32>& depth,
                ap_uint<32>& overflow_length,

                hls::stream<ap_uint<HASHW> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,

                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<KEYW + S_PW> >& o_row_strm,
                hls::stream<bool>& o_e_strm,

                ap_uint<72>* bit_vector0,
                ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;
    const int HASH_NUMBER = 1 << HASHW;

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif

    ap_uint<72> elem = 0;
    ap_uint<72> base_elem = 0;
    ap_uint<72> overflow_elem = 0;
    ap_uint<72> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<HASHW> array_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    ap_uint<32> overflow_length_temp = 0;
    ap_uint<ARW> overflow_base = HASH_NUMBER * depth;

    bool last = i_e_strm.read();
PRE_BUILD_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<S_PW> pld = i_pld_strm.read(); // XXX trun
        last = i_e_strm.read();

        // mod 3 to calculate index for 24bit address
        ap_uint<HASHW> array_idx = hash_val / 3;
        ap_uint<HASHW> temp = array_idx * 3;
        ap_uint<2> bit_idx = hash_val - temp;

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782)
            std::cout << std::hex << "build_stb debug: key=" << key << " hash=" << hash_val
                      << " error bitmap=" << bit_vector0[0x21e] << " array_idx=" << array_idx << " bit_idx=" << bit_idx
                      << std::endl;
#endif
#endif

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
            elem = bit_vector0[array_idx];
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782) {
            for (int i = 0; i < 8; i++) {
                std::cout << std::hex << "build_stb: ele_temp[" << i << "]" << elem_temp[i] << " array_idx_temp[" << i
                          << "]=" << array_idx_temp[i] << std::endl;
            }
        }

        if (array_idx == 0x21e) {
            std::cout << std::hex << "build_stb debug: key=" << key << " hash=" << hash_val
                      << " error bitmap=" << bit_vector0[0x21e] << " elem=" << elem << std::endl;
        }

#endif
#endif

        // update && write new hash value
        ap_uint<24> v0 = elem(23, 0);
        ap_uint<24> v1 = elem(47, 24);
        ap_uint<24> v2 = elem(71, 48);

        ap_uint<24> v0a, v0b;
        ap_uint<24> v1a, v1b;
        ap_uint<24> v2a, v2b;

        ap_uint<ARW> hash_cnt;

        if (bit_idx == 0 || bit_idx == 3) {
            v0a = v0 + 1;
            v1a = v1;
            v2a = v2;

            v0b = ((v0 + 1) > depth) ? (v0 - depth + 1) : (ap_uint<24>*)0;
            v1b = (v1 > depth) ? (v1 - depth) : (ap_uint<24>*)0;
            v2b = (v2 > depth) ? (v2 - depth) : (ap_uint<24>*)0;

            hash_cnt = v0;
        } else if (bit_idx == 1) {
            v0a = v0;
            v1a = v1 + 1;
            v2a = v2;

            v0b = (v0 > depth) ? (v0 - depth) : (ap_uint<24>*)0;
            v1b = ((v1 + 1) > depth) ? (v1 - depth + 1) : (ap_uint<24>*)0;
            v2b = (v2 > depth) ? (v2 - depth) : (ap_uint<24>*)0;

            hash_cnt = v1;
        } else if (bit_idx == 2) {
            v0a = v0;
            v1a = v1;
            v2a = v2 + 1;

            v0b = (v0 > depth) ? (v0 - depth) : (ap_uint<24>*)0;
            v1b = (v1 > depth) ? (v1 - depth) : (ap_uint<24>*)0;
            v2b = ((v2 + 1) > depth) ? (v2 - depth + 1) : (ap_uint<24>*)0;

            hash_cnt = v2;
        }

        base_elem(23, 0) = v0a;
        base_elem(47, 24) = v1a;
        base_elem(71, 48) = v2a;

        overflow_elem(23, 0) = v0b;
        overflow_elem(47, 24) = v1b;
        overflow_elem(71, 48) = v2b;

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

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (array_idx == 0x21e) {
            std::cout << std::hex << "build_stb debug: key=" << key << " hash=" << hash_val
                      << " error bitmap=" << bit_vector0[0x21e] << " elem=" << elem << std::endl;
        }
#endif
#endif

        // update hash table
        bit_vector0[array_idx] = base_elem;
        bit_vector1[array_idx] = overflow_elem;

        // generate o_addr
        ap_uint<ARW> o_addr;

        if (hash_cnt >= depth) {
            // overflow
            o_addr = overflow_base + overflow_length_temp;
            overflow_length_temp++;
        } else {
            // underflow
            o_addr = depth * hash_val + hash_cnt;
        }

        o_addr_strm.write(o_addr);
        o_row_strm.write(srow);
        o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782 || array_idx == 0x21e)
            std::cout << std::hex << "build_stb: key=" << key << " hash=" << hash_val << " hash_cnt=" << hash_cnt
                      << " depth=" << depth << " array_idx=" << array_idx << " bit_idx=" << bit_idx << " elem=" << elem
                      << " base_ht_addr=" << array_idx << " base_ht=" << bit_vector0[array_idx]
                      << " overflow_ht_addr=" << array_idx + HASH_DEPTH << " overflow_ht=" << overflow_elem
                      << " output htb_addr=" << o_addr << std::endl;
#endif

#ifdef DEBUG_BUILD
        if (hash_cnt >= depth) {
            std::cout << std::hex << "build_stb: over key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " base_ht_addr=" << array_idx << " base_ht=" << base_elem
                      << " overflow_ht_addr=" << array_idx + HASH_DEPTH << " overflow_ht=" << overflow_elem
                      << " output htb_addr=" << o_addr << std::endl;
        } else {
            std::cout << std::hex << "build_stb: base key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " base_ht_addr=" << array_idx << " base_ht=" << base_elem
                      << " overflow_ht_addr=" << array_idx + HASH_DEPTH << " overflow_ht=" << overflow_elem
                      << " output stb_addr=" << o_addr << std::endl;
        }
#endif

        ap_uint<ARW> old_val = (bit_idx == 0 || bit_idx == 3) ? v0 : ((bit_idx == 1) ? v1 : v2);
        if (old_val > max_col) max_col = old_val;
        cnt++;
#endif
    }

    overflow_length = overflow_length_temp;
    o_e_strm.write(true);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << std::dec << "collision probility " << max_col << std::endl;
#endif
}

/// @brief Top function of hash join build
template <int HASHW, int KEYW, int PW, int S_PW, int ARW>
void build_wrapper(ap_uint<32>& depth,
                   ap_uint<32>& overflow_length,

                   hls::stream<ap_uint<HASHW> >& i_hash_strm,
                   hls::stream<ap_uint<KEYW> >& i_key_strm,
                   hls::stream<ap_uint<PW> >& i_pld_strm,
                   hls::stream<bool>& i_e_strm,

                   ap_uint<256>* stb_buf,
                   ap_uint<72>* bit_vector0,
                   ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<ARW> > addr_strm;
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS bind_storage variable = addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + S_PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8

    build_unit<HASHW, KEYW, PW, S_PW, ARW>( //
        depth, overflow_length, i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        addr_strm, row_strm, e_strm,

        bit_vector0, bit_vector1);

    write_stb<ARW, KEYW + S_PW>(stb_buf, addr_strm, row_strm, e_strm);
}

//----------------------------------------merge_hash_table------------------------------------

/// @brief Load overflow hash collision counter to genrate overflow hash bitmap
template <int HASHW, int ARW>
void bitmap_addr_gen(ap_uint<32>& depth,
                     ap_uint<32>& overflow_length,

                     ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    // base addr for overflow srow
    ap_uint<ARW> base_addr = 0;

BITMAP_ADDR_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector1 inter false

        // genrate bitmap from hash counter
        ap_uint<72> elem;
        read_bit_vector1(i, elem);

        ap_uint<24> v0 = elem(23, 0);
        ap_uint<24> v1 = elem(47, 24);
        ap_uint<24> v2 = elem(71, 48);

        ap_uint<ARW> sum_0 = v0;
        ap_uint<ARW> sum_1 = v0 + v1;
        ap_uint<ARW> sum_2 = v0 + v1 + v2;

        ap_uint<72> head;
        head(23, 0) = base_addr;
        head(47, 24) = base_addr + sum_0;
        head(71, 48) = base_addr + sum_1;

#ifndef __SYNTHESIS__
#ifdef DEBUG_PROBE
        if (v0 > 0 || v1 > 0 || v2 > 0) {
            std::cout << std::hex << std::hex << "bitmap_addr_gen: ht_addr=" << base_addr1 << " bitmap=" << elem
                      << " head=" << head << " srow_addr=" << base_addr0 << " v0=" << v0 << " v1=" << v1 << " v2=" << v2
                      << std::endl;
        }
#endif
#endif

        base_addr = base_addr + sum_2;

        // write bitmap addr to URAM for fully build
        write_bit_vector1(i, head);
        ap_uint<128> bitmap = head;
    }
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

// merge overflow srow
template <int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_unit(hls::stream<ap_uint<HASHWH + HASHWL> >& i_hash_strm,
                hls::stream<ap_uint<KEYW> >& i_key_strm,
                hls::stream<ap_uint<PW> >& i_pld_strm,
                hls::stream<bool>& i_e_strm,

                hls::stream<ap_uint<ARW> >& o_addr_strm,
                hls::stream<ap_uint<KEYW + PW> >& o_row_strm,
                hls::stream<bool>& o_e_strm,

                ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_stb_per_row = (KEYW + PW) / 256; // number of output based on 256 bit

    ap_uint<72> elem = 0;
    ap_uint<72> new_elem = 0;
    ap_uint<72> elem_temp[4] = {0, 0, 0, 0};
    ap_uint<HASHWL> array_idx_temp[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    bool last = i_e_strm.read();
LOOP_BUILD_UNIT:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector1 inter false

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
            ap_uint<HASHWL> array_idx = hash / 3;
            ap_uint<HASHWL> temp = array_idx * 3;
            ap_uint<2> bit_idx = hash - temp;

            // read hash counter and ++, prevent duplicate key
            if (array_idx == array_idx_temp[0]) {
                elem = elem_temp[0];
            } else if (array_idx == array_idx_temp[1]) {
                elem = elem_temp[1];
            } else if (array_idx == array_idx_temp[2]) {
                elem = elem_temp[2];
            } else if (array_idx == array_idx_temp[3]) {
                elem = elem_temp[3];
            } else {
                read_bit_vector1(array_idx, elem);
            }

            ap_uint<24> v0 = elem(23, 0);
            ap_uint<24> v1 = elem(47, 24);
            ap_uint<24> v2 = elem(71, 48);

            ap_uint<24> v0a = (bit_idx == 0 || bit_idx == 3) ? ap_uint<24>(v0 + 1) : v0;
            ap_uint<24> v1a = (bit_idx == 1) ? ap_uint<24>(v1 + 1) : v1;
            ap_uint<24> v2a = (bit_idx == 2) ? ap_uint<24>(v2 + 1) : v2;

            new_elem(23, 0) = v0a;
            new_elem(47, 24) = v1a;
            new_elem(71, 48) = v2a;

            // right shift temp
            for (int i = 3; i > 0; i--) {
                elem_temp[i] = elem_temp[i - 1];
                array_idx_temp[i] = array_idx_temp[i - 1];
            }
            elem_temp[0] = new_elem;
            array_idx_temp[0] = array_idx;

            write_bit_vector1(array_idx, new_elem);

            ap_uint<ARW> o_addr = (bit_idx == 0 || bit_idx == 3) ? v0 : ((bit_idx == 1) ? v1 : v2);

            // write stb
            o_addr_strm.write(o_addr);
            o_row_strm.write(stb_row);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
            if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782)
                std::cout << std::hex << "merge_unit: key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                          << " bit_idx=" << bit_idx << " overflow_ht_addr=" << array_idx << " old_elem=" << elem
                          << " new_elem=" << new_elem << " stb_addr=" << o_addr << " stb_row=" << stb_row << std::endl;
#endif

#ifdef DEBUG_PROBE
            std::cout << std::hex << "merge_unit: cnt=" << cnt << " array_idx=" << array_idx << " bit_idx=" << bit_idx
                      << " overflow_ht_addr=" << array_idx << " old_elem=" << elem << " new_elem=" << new_elem
                      << " stb_addr=" << o_addr << " stb_row=" << stb_row << std::endl;
#endif
            cnt++;
#endif
        }
    }

#ifndef __SYNTHESIS__
    std::cout << std::dec << "merge_unit: merge " << cnt << " lines" << std::endl;
#endif

    o_e_strm.write(true);
}

/// @brief Merge overflow srow from htb to stb
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_stb(ap_uint<32>& depth,
               ap_uint<32>& overflow_length,
               ap_uint<256>* htb_buf,
               ap_uint<256>* stb_buf,
               ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    const int HASH_NUMBER = 1 << HASHWL;

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
    read_addr_gen<ARW>(HASH_NUMBER * depth, overflow_length,

                       read_addr_strm, e0_strm);

    // read overflow srow in stb_buf
    read_stb<ARW, KEYW + PW>(stb_buf,

                             read_addr_strm, e0_strm,

                             row0_strm, e1_strm);

    // split row to hash and key
    splitCol<KEYW + PW, PW, KEYW>(row0_strm, e1_strm,

                                  pld_strm, key0_strm, e2_strm);

    // calculate hash
    hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(key0_strm, e2_strm,

                                                   hash_strm, key1_strm, e3_strm);

    // build overflow srow with its hash addr
    merge_unit<HASHWH, HASHWL, KEYW, PW, ARW>(hash_strm, key1_strm, pld_strm, e3_strm, write_addr_strm, row1_strm,
                                              e4_strm,

                                              bit_vector);

    // write shuffled overflow srow to htb_buf
    write_stb<ARW, KEYW + PW>(htb_buf, write_addr_strm, row1_strm, e4_strm);
}

/// @brief Top function of hash join merge
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_wrapper(
    // input status
    ap_uint<32>& depth,
    ap_uint<32>& overflow_length,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

    // generate overflow bitmap by overflow hash counter
    bitmap_addr_gen<HASHWL, ARW>(depth, overflow_length, bit_vector);

    // build overflow srow with its hash addr
    merge_stb<HASH_MODE, HASHWH, HASHWL, KEYW, PW, ARW>(depth, overflow_length, htb_buf, stb_buf, bit_vector);
}

//----------------------------------------probe------------------------------------------------

/// @brief Probe the hash table and output address which hash same hash_value
template <int HASHW, int KEYW, int PW, int B_PW, int ARW>
void probe_htb(ap_uint<32>& depth,

               // input large table
               hls::stream<ap_uint<HASHW> >& i_hash_strm,
               hls::stream<ap_uint<KEYW> >& i_key_strm,
               hls::stream<ap_uint<PW> >& i_pld_strm,
               hls::stream<bool>& i_e_strm,

               // output to generate probe addr
               hls::stream<ap_uint<ARW> >& o_base_addr_strm,
               hls::stream<ap_uint<ARW> >& o_nm0_strm,
               hls::stream<bool>& o_e0_strm,
               hls::stream<ap_uint<ARW> >& o_overflow_addr_strm,
               hls::stream<ap_uint<ARW> >& o_nm1_strm,
               hls::stream<bool>& o_e1_strm,

               // output to join
               hls::stream<ap_uint<KEYW> >& o_key_strm,
               hls::stream<ap_uint<B_PW> >& o_pld_strm,
               hls::stream<ap_uint<ARW> >& o_nm2_strm,
               hls::stream<bool>& o_e2_strm,

               ap_uint<72>* bit_vector0,
               ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int base_cnt = 0;
    unsigned int overflow_cnt = 0;
#endif

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;
    const int HASH_NUMBER = 1 << HASHW;

    ap_uint<72> base_bitmap;
    ap_uint<72> overflow_bitmap;
    ap_uint<ARW> base_ht_addr;
    ap_uint<ARW> overflow_ht_addr;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        // read select field from stream and store them on local ram.
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<B_PW> pld = i_pld_strm.read(); // XXX trunc
        last = i_e_strm.read();

        // mod 3 to calculate index for 24bit address
        ap_uint<HASHW> array_idx = hash_val / 3;
        ap_uint<HASHW> temp = array_idx * 3;
        ap_uint<2> bit_idx = hash_val - temp;

        // total hash count
        ap_uint<ARW> nm;
        read_bit_vector0(array_idx, base_bitmap);

        if (bit_idx == 0 || bit_idx == 3) {
            nm = base_bitmap(23, 0);
        } else if (bit_idx == 1) {
            nm = base_bitmap(47, 24);
        } else {
            nm = base_bitmap(71, 48);
        }

        // calculate addr
        base_ht_addr = hash_val * depth;

        if ((bit_idx == 0 || bit_idx == 3) && array_idx > 0)
            read_bit_vector1(array_idx - 1, overflow_bitmap);
        else
            read_bit_vector1(array_idx, overflow_bitmap);

        if (bit_idx == 0 || bit_idx == 3) {
            if (array_idx > 0) {
                overflow_ht_addr = overflow_bitmap(71, 48);
            } else {
                overflow_ht_addr = 0;
            }
        } else if (bit_idx == 1) {
            overflow_ht_addr = overflow_bitmap(23, 0);
        } else {
            overflow_ht_addr = overflow_bitmap(47, 24);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG_MISS
        if (key == 3680482 || key == 3691265 || key == 4605699 || key == 4987782)
            std::cout << std::hex << "probe_ahead: key=" << key << " hash_val=" << hash_val
                      << " array_idx=" << array_idx << " bit_idx=" << bit_idx << " nm=" << nm
                      << " base_ht_addr=" << base_ht_addr << " base_bitmap=" << base_bitmap
                      << " overflow_addr=" << overflow_ht_addr << " overflow_bitmap=" << overflow_bitmap << std::endl;
#endif

#ifdef DEBUG_PROBE
        std::cout << std::hex << "probe_ahead: key=" << key << " hash_val=" << hash_val << " array_idx=" << array_idx
                  << " bit_idx=" << bit_idx << " nm=" << nm << " base_ht_addr=" << base_ht_addr
                  << " base_bitmap=" << base_bitmap << " overflow_addr=" << overflow_ht_addr
                  << " overflow_bitmap=" << overflow_bitmap << std::endl;
#endif
#endif

        // optimization: add bloom filter to filter out more row
        if (nm > 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif

            ap_uint<ARW> nm0; // base number
            ap_uint<ARW> nm1; // overflow number

            if (nm > depth) {
                nm0 = depth;
                nm1 = nm - depth;

#ifndef __SYNTHESIS__
                overflow_cnt++;
#endif
            } else {
                nm0 = nm;
                nm1 = 0;

#ifndef __SYNTHESIS__
                base_cnt++;
#endif
            }

            o_base_addr_strm.write(base_ht_addr);
            o_nm0_strm.write(nm0);
            o_e0_strm.write(false);
            if (nm1 > 0) {
                o_overflow_addr_strm.write(overflow_ht_addr);
                o_nm1_strm.write(nm1);
                o_e1_strm.write(false);
            }

            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_nm2_strm.write(nm);
            o_e2_strm.write(false);
        }
    }

#ifndef __SYNTHESIS__
    std::cout << std::dec << "probe will read " << cnt << " block from stb, including " << base_cnt << " base block, "
              << overflow_cnt << " overflow block" << std::endl;
#endif

    o_e0_strm.write(true);
    o_e2_strm.write(true);

    // for do-while in probe overflow stb
    o_overflow_addr_strm.write(0);
    o_nm1_strm.write(0);
    o_e1_strm.write(true);
}

// generate probe addr
template <int ARW>
void probe_addr_gen(
    // input
    ap_uint<32>& depth,
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

/// @brief Probe stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void probe_base_stb(
    // input
    ap_uint<32>& depth,
    hls::stream<ap_uint<ARW> >& i_base_stb_addr_strm,
    hls::stream<ap_uint<ARW> >& i_base_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_base_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_base_s_pld_strm,

    // HBM
    ap_uint<256>* stb_buf) {
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
    probe_addr_gen<ARW>(depth, i_base_stb_addr_strm, i_base_nm_strm, i_e_strm,

                        read_addr_strm, e0_strm);

    // read HBM to get base stb
    read_stb<ARW, KEYW + S_PW>(stb_buf, read_addr_strm, e0_strm, row_strm, e1_strm);

    // split base stb to key and pld
    splitCol<KEYW + S_PW, S_PW, KEYW>(row_strm, e1_strm, o_base_s_pld_strm, o_base_s_key_strm, e2_strm);

    // eleminate end signal of overflow unit
    eliminate_strm_end<bool>(e2_strm);
}

// generate overflow probe addr
template <int ARW>
void probe_addr_gen(
    // input
    ap_uint<ARW> overflow_addr,
    ap_uint<ARW> overflow_nm,

    // output
    hls::stream<ap_uint<ARW> >& o_read_addr_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    ap_uint<ARW> addr = overflow_addr;
    ap_uint<ARW> nm = overflow_nm;

    while (nm--) {
#pragma HLS pipeline II = 1

        o_read_addr_strm.write(addr++);
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
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
    ap_uint<256>* htb_buf) {
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
    probe_addr_gen<ARW>(overflow_addr, overflow_nm,

                        read_addr_strm, e0_strm);

    // read HBM to get base stb
    read_stb<ARW, KEYW + S_PW>(htb_buf, read_addr_strm, e0_strm, row_strm, e1_strm);

    // split base stb to key and pld
    splitCol<KEYW + S_PW, S_PW, KEYW>(row_strm, e1_strm, o_overflow_s_pld_strm, o_overflow_s_key_strm, e2_strm);

    // eleminate end signal of overflow unit
    eliminate_strm_end<bool>(e2_strm);
}

/// @brief probe overflow stb which temporarily stored in HBM
template <int KEYW, int S_PW, int ARW>
void probe_overflow_stb(
    // input base addr
    hls::stream<ap_uint<ARW> >& i_overflow_addr_strm,
    hls::stream<ap_uint<ARW> >& i_overflow_nm_strm,
    hls::stream<bool>& i_e_strm,

    // output probed small table
    hls::stream<ap_uint<KEYW> >& o_overflow_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_overflow_s_pld_strm,

    // HBM
    ap_uint<256>* htb_buf) {
#pragma HLS INLINE off

    ap_uint<ARW> overflow_addr, nm;

    bool last = false;
PROBE_OVERFLOW_LOOP:
    do {
#pragma HLS PIPELINE off

        overflow_addr = i_overflow_addr_strm.read();
        nm = i_overflow_nm_strm.read();
        last = i_e_strm.read();

        read_overflow_stb<KEYW, S_PW, ARW>(overflow_addr, nm, o_overflow_s_key_strm, o_overflow_s_pld_strm, htb_buf);
    } while (!last);
}

/// @brief Top function of hash join probe
template <int HASHW, int KEYW, int PW, int S_PW, int T_PW, int ARW>
void probe_wrapper(ap_uint<32>& depth,

                   // input large table
                   hls::stream<ap_uint<HASHW> >& i_hash_strm,
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

                   ap_uint<256>* htb_buf,
                   ap_uint<256>* stb_buf,
                   ap_uint<72>* bit_vector0,
                   ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for probe_addr_gen
    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm0_strm;
#pragma HLS stream variable = nm0_strm depth = 8
#pragma HLS bind_storage variable = nm0_strm type = fifo impl = bram
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 8
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm1_strm;
#pragma HLS stream variable = nm1_strm depth = 8
#pragma HLS bind_storage variable = nm1_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8
#pragma HLS bind_storage variable = e1_strm type = fifo impl = srl

    // calculate number of srow need to probe in HBM/DDR
    probe_htb<HASHW, KEYW, PW, T_PW, ARW>(depth, i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                          base_addr_strm, nm0_strm, e0_strm, overflow_addr_strm, nm1_strm, e1_strm,
                                          o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e0_strm,

                                          bit_vector0, bit_vector1);

    probe_base_stb<KEYW, S_PW, ARW>(depth, base_addr_strm, nm0_strm, e0_strm,

                                    o_base_s_key_strm, o_base_s_pld_strm,

                                    stb_buf);

    probe_overflow_stb<KEYW, S_PW, ARW>(overflow_addr_strm, nm1_strm, e1_strm,

                                        o_overflow_s_key_strm, o_overflow_s_pld_strm,

                                        htb_buf);
}

//----------------------------------------------build+merge+probe------------------------------------------------

/// @brief Initiate uram to zero or read previous stored hash counter
template <int HASHW, int ARW>
void initiate_uram(ap_uint<72>* bit_vector0, ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

// id==0, the firsr time to build, initialize uram to zero
INIT_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        write_bit_vector0(i, 0);
        write_bit_vector1(i, 0);
    }
}

/// @brief Top function of hash join PU
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int S_PW, int T_PW, int ARW>
void build_merge_probe_wrapper(
    // input status
    ap_uint<32>& depth,

    // input table
    hls::stream<ap_uint<HASHWL> >& i_hash_strm,
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

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf) {
#pragma HLS INLINE off

    const int PW = (S_PW > T_PW) ? S_PW : T_PW;
    // alllocate uram storage
    const int HASH_DEPTH = (1 << HASHWL) / 3 + 1;

#ifndef __SYNTHESIS__

    ap_uint<72>* bit_vector0;
    ap_uint<72>* bit_vector1;
    bit_vector0 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));
    bit_vector1 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));

#else

    ap_uint<72> bit_vector0[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
    ap_uint<72> bit_vector1[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector1 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram

#endif

    ap_uint<32> overflow_length = 0;

    // initilize uram by previous hash build or probe
    initiate_uram<HASHWL, ARW>(bit_vector0, bit_vector1);

// build
#ifndef __SYNTHESIS__
    std::cout << "----------------------build------------------------" << std::endl;
#endif

    build_wrapper<HASHWL, KEYW, PW, S_PW, ARW>(
        // input status
        depth, overflow_length,

        // input s-table
        i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        // HBM/DDR
        stb_buf, bit_vector0, bit_vector1);

    // merge
    if (overflow_length > 0) {
// the first time to probe, need to fully build bitmap
#ifndef __SYNTHESIS__
        std::cout << "----------------------merge------------------------" << std::endl;
#endif

        merge_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, ARW>(
            // input status
            depth, overflow_length,

            htb_buf, stb_buf, bit_vector1);
    }

// probe
#ifndef __SYNTHESIS__
    std::cout << "-----------------------Probe------------------------" << std::endl;
#endif

    probe_wrapper<HASHWL, KEYW, PW, S_PW, T_PW, ARW>(depth,

                                                     // input large table
                                                     i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                     // output for join
                                                     o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e_strm, o_base_s_key_strm,
                                                     o_base_s_pld_strm, o_overflow_s_key_strm, o_overflow_s_pld_strm,

                                                     htb_buf, stb_buf, bit_vector0, bit_vector1);

#ifndef __SYNTHESIS__

    free(bit_vector0);
    free(bit_vector1);

#endif
}

//-----------------------------------------------join-----------------------------------------------
/// @brief compare key, if match output joined row
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit(
#ifndef __SYNTHESIS__
    int pu_id,
#endif

    ap_uint<32>& depth,

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

//---------------------------------------------collect-------------------------------------------

// collect join result of mutiple PU
template <int PU, int JW>
void collect_unit(hls::stream<ap_uint<JW> > i_jrow_strm[PU],
                  hls::stream<bool> i_e_strm[PU],

                  ap_uint<32>& join_num,
                  hls::stream<ap_uint<JW> >& o_jrow_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int MAX = (1 << PU) - 1;
    ap_uint<JW> jrow_arr[PU];
#pragma HLS array_partition variable = jrow_arr dim = 1

    ap_uint<32> cnt = 0;
    ap_uint<PU> empty_e = 0;
    ap_uint<PU> last = 0;
    ap_uint<PU> rd_e = 0;

#ifndef __SYNTHESIS__
    std::cout << std::dec << "PU=" << PU << std::endl;
#endif

    do {
#pragma HLS pipeline II = 1
        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            empty_e[i] = !i_e_strm[i].empty() && !last[i];
        }

        rd_e = join_v2::mul_ch_read(empty_e);

        for (int i = 0; i < PU; i++) {
#pragma HLS unroll
            if (rd_e[i]) {
                jrow_arr[i] = i_jrow_strm[i].read();
                last[i] = i_e_strm[i].read();
            }
        }

        ap_uint<3> id = join_v2::mux<PU>(rd_e);
        ap_uint<JW> j = jrow_arr[id];
        bool valid_n = last[id];

        if (!valid_n && rd_e != 0) {
            o_jrow_strm.write(j);
            o_e_strm.write(false);

            cnt++;
        }
    } while (last != MAX);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "Collect " << cnt << " rows" << std::endl;
#endif

    join_num = cnt;
    o_e_strm.write(true);
}

} // namespace sc

namespace bp {

//------------------------------------read status--------------------------------------
template <int PU>
void read_status(hls::stream<ap_uint<32> >& pu_begin_status_strms,
                 ap_uint<32>& id,
                 ap_uint<32>& depth,
                 ap_uint<32> pu_start_addr[PU]) {
    // PU status
    for (int i = 0; i < PU; i++) {
#pragma HLS pipeline II = 1
        pu_start_addr[i] = pu_begin_status_strms.read();
    }

    // build id
    id = pu_begin_status_strms.read();

    // fixed hash depth
    depth = pu_begin_status_strms.read();

    // discard joined number of last probe
    ap_uint<32> join_number = pu_begin_status_strms.read();
}

//------------------------------------write status--------------------------------------
template <int PU>
void write_status(hls::stream<ap_uint<32> >& pu_end_status_strms,
                  ap_uint<32> id,
                  ap_uint<32> depth,
                  ap_uint<32> join_num,
                  ap_uint<32> pu_end_addr[PU]) {
    // PU status
    for (int i = 0; i < PU; i++) {
#pragma HLS pipeline II = 1
        pu_end_status_strms.write(pu_end_addr[i]);
    }

    // build id
    pu_end_status_strms.write(id + 1);

    // fixed hash depth
    pu_end_status_strms.write(depth);

    // joined number of probe
    pu_end_status_strms.write(join_num);
}

//------------------------------------------dispatch--------------------------------------

// dispatch data based on hash value to multiple PU.
template <int HASH_MODE, int KEYW, int PW, int HASHWH, int HASHWL, int PU>
void dispatch_wrapper(hls::stream<ap_uint<KEYW> >& i_key_strm,
                      hls::stream<ap_uint<PW> >& i_pld_strm,
                      hls::stream<bool>& i_e_strm,

                      hls::stream<ap_uint<KEYW> > o_key_strm[PU],
                      hls::stream<ap_uint<PW> > o_pld_strm[PU],
                      hls::stream<ap_uint<HASHWL> > o_hash_strm[PU],
                      hls::stream<bool> o_e_strm[PU]) {
#pragma HLS DATAFLOW

    hls::stream<ap_uint<HASHWH + HASHWL> > hash_strm;
#pragma HLS STREAM variable = hash_strm depth = 8
#pragma HLS bind_storage variable = hash_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW> > key_strm;
#pragma HLS STREAM variable = key_strm depth = 32
#pragma HLS resource variable = key_strm core = FIFO_SRL
    hls::stream<bool> e_strm;
#pragma HLS STREAM variable = e_strm depth = 32

    join_v3::sc::hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(i_key_strm, i_e_strm, hash_strm, key_strm, e_strm);

    join_v3::sc::dispatch<KEYW, PW, HASHWH, HASHWL, PU>(key_strm, i_pld_strm, hash_strm, e_strm, o_key_strm, o_pld_strm,
                                                        o_hash_strm, o_e_strm);
}

// -------------------------------------build------------------------------------------

// scan small table to count hash collision, build small table to its hash addr
template <int HASHW, int KEYW, int PW, int ARW>
void build_unit(
    // input
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,
    ap_uint<32>& depth,

    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<KEYW + PW> >& o_row0_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<ARW> >& overflow_addr_strm,
    hls::stream<ap_uint<KEYW + PW> >& o_row1_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int max_col = 0;
#endif

    ap_uint<72> elem = 0;
    ap_uint<72> base_elem = 0;
    ap_uint<72> overflow_elem = 0;
    ap_uint<72> elem_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ap_uint<HASHW> array_idx_temp[8] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    ap_uint<32> max_overflow = pu_start_addr;
    ap_uint<ARW> overflow_base = 2 * HASH_DEPTH;

    bool last = i_e_strm.read();
PRE_BUILD_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // mod 3 to calculate index for 24bit address
        ap_uint<HASHW> array_idx = hash_val / 3;
        ap_uint<HASHW> temp = array_idx * 3;
        ap_uint<2> bit_idx = hash_val - temp;

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
            elem = bit_vector0[array_idx];
        }

        // update && write new hash value
        ap_uint<24> v0 = elem(23, 0);
        ap_uint<24> v1 = elem(47, 24);
        ap_uint<24> v2 = elem(71, 48);

        ap_uint<24> v0a, v0b;
        ap_uint<24> v1a, v1b;
        ap_uint<24> v2a, v2b;

        ap_uint<ARW> hash_cnt;

        if (bit_idx == 0 || bit_idx == 3) {
            v0a = v0 + 1;
            v1a = v1;
            v2a = v2;

            v0b = ((v0 + 1) > depth) ? (v0 - depth + 1) : (ap_uint<24>*)0;
            v1b = (v1 > depth) ? (v1 - depth) : (ap_uint<24>*)0;
            v2b = (v2 > depth) ? (v2 - depth) : (ap_uint<24>*)0;

            hash_cnt = v0;
        } else if (bit_idx == 1) {
            v0a = v0;
            v1a = v1 + 1;
            v2a = v2;

            v0b = (v0 > depth) ? (v0 - depth) : (ap_uint<24>*)0;
            v1b = ((v1 + 1) > depth) ? (v1 - depth + 1) : (ap_uint<24>*)0;
            v2b = (v2 > depth) ? (v2 - depth) : (ap_uint<24>*)0;

            hash_cnt = v1;
        } else if (bit_idx == 2) {
            v0a = v0;
            v1a = v1;
            v2a = v2 + 1;

            v0b = (v0 > depth) ? (v0 - depth) : (ap_uint<24>*)0;
            v1b = (v1 > depth) ? (v1 - depth) : (ap_uint<24>*)0;
            v2b = ((v2 + 1) > depth) ? (v2 - depth + 1) : (ap_uint<24>*)0;

            hash_cnt = v2;
        }

        base_elem(23, 0) = v0a;
        base_elem(47, 24) = v1a;
        base_elem(71, 48) = v2a;

        overflow_elem(23, 0) = v0b;
        overflow_elem(47, 24) = v1b;
        overflow_elem(71, 48) = v2b;

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
        bit_vector0[array_idx] = base_elem;
        bit_vector1[array_idx] = overflow_elem;

        // generate o_addr
        ap_uint<ARW> base_addr;
        ap_uint<ARW> overflow_addr;

        if (hash_cnt >= depth) {
            // overflow
            overflow_addr = overflow_base + max_overflow;
            max_overflow++;

            overflow_addr_strm.write(overflow_addr);
            o_row1_strm.write(srow);
            o_e1_strm.write(false);
        } else {
            // underflow
            base_addr = depth * hash_val + hash_cnt;

            o_base_addr_strm.write(base_addr);
            o_row0_strm.write(srow);
            o_e0_strm.write(false);
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        if (hash_cnt >= depth) {
            std::cout << std::hex << "build_stb: over key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " base_ht_addr=" << array_idx << " base_ht=" << base_elem
                      << " overflow_ht_addr=" << array_idx + HASH_DEPTH << " overflow_ht=" << overflow_elem
                      << " output htb_addr=" << overflow_addr << std::endl;
        } else {
            std::cout << std::hex << "build_stb: base key=" << key << " hash=" << hash_val << " array_idx=" << array_idx
                      << " bit_idx=" << bit_idx << " base_ht_addr=" << array_idx << " base_ht=" << base_elem
                      << " overflow_ht_addr=" << array_idx + HASH_DEPTH << " overflow_ht=" << overflow_elem
                      << " output stb_addr=" << base_addr << std::endl;
        }
#endif

        ap_uint<ARW> old_val = (bit_idx == 0 || bit_idx == 3) ? v0 : ((bit_idx == 1) ? v1 : v2);
        if (old_val > max_col) max_col = old_val;
        cnt++;
#endif
    }

    pu_end_addr = max_overflow;

    o_e0_strm.write(true);
    o_e1_strm.write(true);

#ifndef __SYNTHESIS__
    std::cout << std::dec << "Get " << cnt << " to build bitmap" << std::endl;
    std::cout << std::dec << "collision probility " << max_col << std::endl;
#endif
}

// build small table
template <int HASHW, int KEYW, int PW, int ARW>
void build_stb(
    // input
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,
    ap_uint<32> depth,

    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + PW> > row0_strm;
#pragma HLS stream variable = row0_strm depth = 8
#pragma HLS bind_storage variable = row0_strm type = fifo impl = srl
    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 8
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<KEYW + PW> > row1_strm;
#pragma HLS stream variable = row1_strm depth = 8
#pragma HLS bind_storage variable = row1_strm type = fifo impl = srl
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 8

    build_unit<HASHW, KEYW, PW, ARW>(
        // input
        pu_start_addr, pu_end_addr, depth, i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        base_addr_strm, row0_strm, e0_strm, overflow_addr_strm, row1_strm, e1_strm,

        bit_vector0, bit_vector1);

    join_v3::sc::write_stb<ARW, KEYW + PW>(stb_buf, base_addr_strm, row0_strm, e0_strm);

    join_v3::sc::write_stb<ARW, KEYW + PW>(htb_buf, overflow_addr_strm, row1_strm, e1_strm);
}

// top function of hash join build
template <int HASHW, int KEYW, int PW, int ARW>
void build_wrapper(
    // input
    ap_uint<32>& id,
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,

    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    // build small table
    build_stb<HASHW, KEYW, PW, ARW>(
        // input
        pu_start_addr, pu_end_addr, depth, i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

        // internal stream to build unit
        htb_buf, stb_buf, bit_vector0, bit_vector1);

    // output base hash table
    join_v3::sc::write_htb<HASHW, ARW>(htb_buf, 0, bit_vector0);

    // output overflow hash table
    join_v3::sc::write_htb<HASHW, ARW>(htb_buf, HASH_DEPTH, bit_vector1);
}

//---------------------------------------------------merge hash table-------------------------

// load overflow hash collision counter to genrate overflow hash bitmap
template <int HASHW, int ARW>
void bitmap_addr_gen(
    // input status
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,

    // output
    hls::stream<ap_uint<ARW> >& o_addr_strm,
    hls::stream<ap_uint<256> >& o_bitmap_strm,
    hls::stream<bool>& o_e_etrm,

    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;
    const int HASH_NUMBER = 1 << HASHW;

    // base addr for overflow srow
    ap_uint<ARW> base_addr0 = depth * HASH_NUMBER;

    // base addr for overflow ht
    ap_uint<64> base_addr1 = HASH_DEPTH;

BITMAP_ADDR_LOOP:
    for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector1 inter false

        // genrate bitmap from hash counter
        ap_uint<72> elem;
        read_bit_vector1(i, elem);

        ap_uint<24> v0 = elem(23, 0);
        ap_uint<24> v1 = elem(47, 24);
        ap_uint<24> v2 = elem(71, 48);

        ap_uint<ARW> sum_0 = v0;
        ap_uint<ARW> sum_1 = v0 + v1;
        ap_uint<ARW> sum_2 = v0 + v1 + v2;

        ap_uint<72> head;
        head(23, 0) = base_addr0;
        head(47, 24) = base_addr0 + sum_0;
        head(71, 48) = base_addr0 + sum_1;

#ifndef __SYNTHESIS__
#ifdef DEBUG
        if (v0 > 0 || v1 > 0 || v2 > 0) {
            std::cout << std::hex << std::hex << "bitmap_addr_gen: ht_addr=" << base_addr1 << " bitmap=" << elem
                      << " head=" << head << " srow_addr=" << base_addr0 << " v0=" << v0 << " v1=" << v1 << " v2=" << v2
                      << std::endl;
        }
#endif
#endif

        base_addr0 = base_addr0 + sum_2;

        // write bitmap addr to URAM for fully build
        write_bit_vector1(i, head);
        ap_uint<256> bitmap = head;

        o_addr_strm.write(base_addr1);
        o_bitmap_strm.write(bitmap);
        o_e_etrm.write(false);

        base_addr1++;
    }
    o_e_etrm.write(true);
}

// load overflow hash collision counter to genrate overflow bitmap, update overflow bitmap to HBM
template <int HASHW, int ARW>
void merge_htb(ap_uint<32>& depth,
               ap_uint<32>& pu_start_addr,

               ap_uint<256>* htb_buf,
               ap_uint<72>* bit_vector) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ap_uint<256> > bitmap_strm;
#pragma HLS stream variable = bitmap_strm depth = 8
#pragma HLS bind_storage variable = bitmap_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > write_addr_strm;
#pragma HLS stream variable = write_addr_strm depth = 8
#pragma HLS bind_storage variable = write_addr_strm type = fifo impl = srl
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 8

    // generate overflow bitmap
    bitmap_addr_gen<HASHW, ARW>(depth, pu_start_addr, write_addr_strm, bitmap_strm, e_strm, bit_vector);

    // write merged overflow bitmap to HBM
    join_v3::sc::write_stb<ARW, 256>(htb_buf, write_addr_strm, bitmap_strm, e_strm);
}

// generate read addr for reading overflwo stb row
template <int HASHW, int ARW>
void read_stb_addr_gen(
    // input status
    ap_uint<32> pu_start_addr,

    // output
    hls::stream<ap_uint<ARW> >& o_addr_strm,
    hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    ap_uint<ARW> addr_temp = 2 * HASH_DEPTH;

LOOP_ADDR_GEN:
    for (int i = 0; i < pu_start_addr; i++) {
#pragma HLS pipeline II = 1
        o_addr_strm.write(addr_temp);
        o_e_strm.write(false);

        addr_temp++;
    }
    o_e_strm.write(true);
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

                ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    const int number_of_stb_per_row = (KEYW + PW) / 256; // number of output based on 256 bit

    ap_uint<72> elem = 0;
    ap_uint<72> new_elem = 0;
    ap_uint<72> elem_temp[4] = {0, 0, 0, 0};
    ap_uint<HASHWL> array_idx_temp[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

    bool last = i_e_strm.read();
LOOP_BUILD_UNIT:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector1 inter false

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
            ap_uint<HASHWL> array_idx = hash / 3;
            ap_uint<HASHWL> temp = array_idx * 3;
            ap_uint<2> bit_idx = hash - temp;

            // read hash counter and ++, prevent duplicate key
            if (array_idx == array_idx_temp[0]) {
                elem = elem_temp[0];
            } else if (array_idx == array_idx_temp[1]) {
                elem = elem_temp[1];
            } else if (array_idx == array_idx_temp[2]) {
                elem = elem_temp[2];
            } else if (array_idx == array_idx_temp[3]) {
                elem = elem_temp[3];
            } else {
                read_bit_vector1(array_idx, elem);
            }

            ap_uint<24> v0 = elem(23, 0);
            ap_uint<24> v1 = elem(47, 24);
            ap_uint<24> v2 = elem(71, 48);

            ap_uint<24> v0a = (bit_idx == 0 || bit_idx == 3) ? ap_uint<24>(v0 + 1) : v0;
            ap_uint<24> v1a = (bit_idx == 1) ? ap_uint<24>(v1 + 1) : v1;
            ap_uint<24> v2a = (bit_idx == 2) ? ap_uint<24>(v2 + 1) : v2;

            new_elem(23, 0) = v0a;
            new_elem(47, 24) = v1a;
            new_elem(71, 48) = v2a;

            // right shift temp
            for (int i = 3; i > 0; i--) {
                elem_temp[i] = elem_temp[i - 1];
                array_idx_temp[i] = array_idx_temp[i - 1];
            }
            elem_temp[0] = new_elem;
            array_idx_temp[0] = array_idx;

            write_bit_vector1(array_idx, new_elem);

            ap_uint<ARW> o_addr = (bit_idx == 0 || bit_idx == 3) ? v0 : ((bit_idx == 1) ? v1 : v2);

            // write stb
            o_addr_strm.write(o_addr);
            o_row_strm.write(stb_row);
            o_e_strm.write(false);

#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << std::hex << "merge_unit: cnt=" << cnt << " array_idx=" << array_idx << " bit_idx=" << bit_idx
                      << " overflow_ht_addr=" << array_idx << " old_elem=" << elem << " new_elem=" << new_elem
                      << " stb_addr=" << o_addr << " stb_row=" << stb_row << std::endl;
#endif
            cnt++;
#endif
        }
    }

#ifndef __SYNTHESIS__
    std::cout << std::dec << "merge_unit: merge " << cnt << " lines" << std::endl;
#endif

    o_e_strm.write(true);
}

// merge overflow srow from htb to stb
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_stb(
    // input status
    ap_uint<32>& pu_start_addr,
    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector) {
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
    read_stb_addr_gen<HASHWL, ARW>(pu_start_addr,

                                   read_addr_strm, e0_strm);

    // read srow in htb_buf
    join_v3::sc::read_stb<ARW, KEYW + PW>(htb_buf,

                                          read_addr_strm, e0_strm,

                                          row0_strm, e1_strm);

    // split row to hash and key
    splitCol<KEYW + PW, PW, KEYW>(row0_strm, e1_strm,

                                  pld_strm, key0_strm, e2_strm);

    // calculate hash
    join_v3::sc::hash_wrapper<HASH_MODE, KEYW, HASHWH + HASHWL>(key0_strm, e2_strm,

                                                                hash_strm, key1_strm, e3_strm);

    // build overflow srow with its hash addr
    merge_unit<HASHWH, HASHWL, KEYW, PW, ARW>(hash_strm, key1_strm, pld_strm, e3_strm, write_addr_strm, row1_strm,
                                              e4_strm,

                                              bit_vector);

    // write srow to stb_buf
    join_v3::sc::write_stb<ARW, KEYW + PW>(stb_buf, write_addr_strm, row1_strm, e4_strm);
}

template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int PW, int ARW>
void merge_wrapper(
    // input status
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector) {
#pragma HLS INLINE off

    // generate overflow bitmap by overflow hash counter
    merge_htb<HASHWL, ARW>(depth, pu_start_addr, htb_buf, bit_vector);

    // build overflow srow with its hash addr
    merge_stb<HASH_MODE, HASHWH, HASHWL, KEYW, PW, ARW>(pu_start_addr, htb_buf, stb_buf, bit_vector);
}

//---------------------------------------------------probe------------------------------------------------

// probe the hash table and output address which hash same hash_value
template <int HASHW, int KEYW, int PW, int ARW>
void probe_htb(
    // flag
    bool is_first_probe,
    ap_uint<32> depth,
    ap_uint<32> pu_start_addr,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output to generate probe addr
    hls::stream<ap_uint<ARW> >& o_base_addr_strm,
    hls::stream<ap_uint<ARW> >& o_nm0_strm,
    hls::stream<ap_uint<ARW> >& o_overflow_addr_strm,
    hls::stream<ap_uint<ARW> >& o_nm1_strm,
    hls::stream<bool>& o_e0_strm,

    // output to join
    hls::stream<ap_uint<KEYW> >& o_key_strm,
    hls::stream<ap_uint<PW> >& o_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm2_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
    unsigned int base_cnt = 0;
    unsigned int overflow_cnt = 0;
#endif

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;
    const int HASH_NUMBER = 1 << HASHW;

    ap_uint<72> base_bitmap;
    ap_uint<72> overflow_bitmap;
    ap_uint<ARW> base_ht_addr;
    ap_uint<ARW> overflow_ht_addr;

    bool last = i_e_strm.read();
LOOP_PROBE:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

        // read select field from stream and store them on local ram.
        ap_uint<HASHW> hash_val = i_hash_strm.read();
        ap_uint<KEYW> key = i_key_strm.read();
        ap_uint<PW> pld = i_pld_strm.read();
        last = i_e_strm.read();

        // mod 3 to calculate index for 24bit address
        ap_uint<HASHW> array_idx = hash_val / 3;
        ap_uint<HASHW> temp = array_idx * 3;
        ap_uint<2> bit_idx = hash_val - temp;

        // total hash count
        ap_uint<ARW> nm;
        read_bit_vector0(array_idx, base_bitmap);

        if (bit_idx == 0 || bit_idx == 3) {
            nm = base_bitmap(23, 0);
        } else if (bit_idx == 1) {
            nm = base_bitmap(47, 24);
        } else {
            nm = base_bitmap(71, 48);
        }

        // calculate addr
        base_ht_addr = hash_val * depth;

        if (is_first_probe && (bit_idx == 0 || bit_idx == 3) && array_idx > 0)
            read_bit_vector1(array_idx - 1, overflow_bitmap);
        else
            read_bit_vector1(array_idx, overflow_bitmap);

        if (is_first_probe) {
            if (bit_idx == 0 || bit_idx == 3) {
                if (array_idx > 0) {
                    overflow_ht_addr = overflow_bitmap(71, 48);
                } else {
                    overflow_ht_addr = depth * HASH_NUMBER;
                }
            } else if (bit_idx == 1) {
                overflow_ht_addr = overflow_bitmap(23, 0);
            } else {
                overflow_ht_addr = overflow_bitmap(47, 24);
            }
        } else {
            if (bit_idx == 0 || bit_idx == 3) {
                overflow_ht_addr = overflow_bitmap(23, 0);
            } else if (bit_idx == 1) {
                overflow_ht_addr = overflow_bitmap(47, 24);
            } else {
                overflow_ht_addr = overflow_bitmap(71, 48);
            }
        }

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << std::hex << "probe_ahead: key=" << key << " hash_val=" << hash_val << " array_idx=" << array_idx
                  << " bit_idx=" << bit_idx << " nm=" << nm << " base_ht_addr=" << base_ht_addr
                  << " base_bitmap=" << base_bitmap << " overflow_addr=" << overflow_ht_addr
                  << " overflow_bitmap=" << overflow_bitmap << std::endl;
#endif
#endif

        // optimization: add bloom filter to filter out more row
        if (nm > 0) {
#ifndef __SYNTHESIS__
            cnt++;
#endif

            ap_uint<ARW> nm0; // base number
            ap_uint<ARW> nm1; // overflow number

            if (nm > depth) {
                nm0 = depth;
                nm1 = nm - depth;

#ifndef __SYNTHESIS__
                overflow_cnt++;
#endif
            } else {
                nm0 = nm;
                nm1 = 0;

#ifndef __SYNTHESIS__
                base_cnt++;
#endif
            }

            o_base_addr_strm.write(base_ht_addr);
            o_nm0_strm.write(nm0);
            o_overflow_addr_strm.write(overflow_ht_addr);
            o_nm1_strm.write(nm1);
            o_e0_strm.write(false);

            o_key_strm.write(key);
            o_pld_strm.write(pld);
            o_nm2_strm.write(nm);
            o_e1_strm.write(false);
        }
    }

#ifndef __SYNTHESIS__
    std::cout << std::dec << "probe will read " << cnt << " block from stb, including " << base_cnt << " base block, "
              << overflow_cnt << " overflow block" << std::endl;
#endif

    o_e0_strm.write(true);
    o_e1_strm.write(true);
}

// generate read addr of small table which need probe
template <int KEYW, int PW, int ARW>
void probe_stb(hls::stream<ap_uint<ARW> >& i_base_addr_strm,
               hls::stream<ap_uint<ARW> >& i_nm0_strm,
               hls::stream<ap_uint<ARW> >& i_overflow_addr_strm,
               hls::stream<ap_uint<ARW> >& i_nm1_strm,
               hls::stream<bool>& i_e_strm,

               hls::stream<ap_uint<ARW> >& o_addr_strm,
               hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif

    bool last = i_e_strm.read();
LOOP_PROBE_ADDR_GEN:
    while (!last) {
        ap_uint<ARW> base_addr = i_base_addr_strm.read();
        ap_uint<ARW> nm0 = i_nm0_strm.read();
        ap_uint<ARW> overflow_addr = i_overflow_addr_strm.read();
        ap_uint<ARW> nm1 = i_nm1_strm.read();
        last = i_e_strm.read();

        ap_uint<ARW> o_addr;

    PROBE:
        while (nm0 > 0 || nm1 > 0) {
#pragma HLS pipeline II = 1

            if (nm0) {
                o_addr = base_addr;
                base_addr++;
                nm0--;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << std::hex << "read_base_stb: read_addr=" << base_addr << std::endl;
#endif
                cnt++;
#endif
            } else if (nm1) {
                o_addr = overflow_addr;
                overflow_addr++;
                nm1--;

#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << std::hex << "read_overflow_stb: read_addr=" << overflow_addr << std::endl;
#endif
                cnt++;
#endif
            }

            o_addr_strm.write(o_addr);
            o_e_strm.write(false);
        }
    }
#ifndef __SYNTHESIS__
    std::cout << std::dec << "STB read " << cnt << " unit" << std::endl;
#endif
    o_e_strm.write(true);
}

// top function of probe
template <int HASHW, int KEYW, int S_PW, int T_PW, int ARW>
void probe_wrapper(
    // flag
    bool is_first_probe,
    ap_uint<32> depth,
    ap_uint<32> pu_start_addr,

    // input large table
    hls::stream<ap_uint<HASHW> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<T_PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<KEYW> >& o_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_s_pld_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<256>* stb_buf,
    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    // for probe_addr_gen
    hls::stream<ap_uint<ARW> > base_addr_strm;
#pragma HLS stream variable = base_addr_strm depth = 8
#pragma HLS bind_storage variable = base_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm0_strm;
#pragma HLS stream variable = nm0_strm depth = 8
#pragma HLS bind_storage variable = nm0_strm type = fifo impl = bram

    hls::stream<ap_uint<ARW> > overflow_addr_strm;
#pragma HLS stream variable = overflow_addr_strm depth = 8
#pragma HLS bind_storage variable = overflow_addr_strm type = fifo impl = srl
    hls::stream<ap_uint<ARW> > nm1_strm;
#pragma HLS stream variable = nm1_strm depth = 8
#pragma HLS bind_storage variable = nm1_strm type = fifo impl = srl

    hls::stream<bool> e0_strm;
#pragma HLS stream variable = e0_strm depth = 8
#pragma HLS bind_storage variable = e0_strm type = fifo impl = srl

    // for read_stb
    hls::stream<ap_uint<ARW> > read_addr_strm;
#pragma HLS stream variable = read_addr_strm depth = 1024
#pragma HLS bind_storage variable = read_addr_strm type = fifo impl = bram
    hls::stream<bool> e1_strm;
#pragma HLS stream variable = e1_strm depth = 1024
#pragma HLS bind_storage variable = e1_strm type = fifo impl = bram

    // for split
    hls::stream<ap_uint<KEYW + S_PW> > row_strm;
#pragma HLS stream variable = row_strm depth = 8
#pragma HLS bind_storage variable = row_strm type = fifo impl = srl
    hls::stream<bool> e2_strm;
#pragma HLS stream variable = e2_strm depth = 8
#pragma HLS bind_storage variable = e2_strm type = fifo impl = srl

    // calculate number of srow need to probe in HBM/DDR
    probe_htb<HASHW, KEYW, T_PW, ARW>(is_first_probe, depth, pu_start_addr, i_hash_strm, i_key_strm, i_pld_strm,
                                      i_e_strm,

                                      base_addr_strm, nm0_strm, overflow_addr_strm, nm1_strm, e0_strm, o_t_key_strm,
                                      o_t_pld_strm, o_nm_strm, o_e0_strm,

                                      bit_vector0, bit_vector1);

    // generate probe addr
    probe_stb<KEYW, S_PW, ARW>(base_addr_strm, nm0_strm, overflow_addr_strm, nm1_strm, e0_strm, read_addr_strm,
                               e1_strm);

    // read stb to get probed srow
    join_v3::sc::read_stb<ARW, KEYW + S_PW>(stb_buf, read_addr_strm, e1_strm, row_strm, e2_strm);

    // split srow to key and pld
    splitCol<KEYW + S_PW, S_PW, KEYW>(row_strm, e2_strm, o_s_pld_strm, o_s_key_strm, o_e1_strm);
}

//----------------------------------------------build+merge+probe------------------------------

// initiate uram to zero or read previous stored hash counter
template <int HASHW, int ARW>
void initiate_uram(
    // input
    ap_uint<32> id,
    ap_uint<256>* htb_buf,
    ap_uint<72>* bit_vector0,
    ap_uint<72>* bit_vector1) {
#pragma HLS INLINE

    const int HASH_DEPTH = (1 << HASHW) / 3 + 1;

    if (id == 0) {
    // id==0, the firsr time to build, initialize uram to zero
    INIT_LOOP:
        for (int i = 0; i < HASH_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = bit_vector0 inter false
#pragma HLS dependence variable = bit_vector1 inter false

            write_bit_vector0(i, 0);
            write_bit_vector1(i, 0);
        }
    } else {
        // read hash table from previous built to initialize URAM
        join_v3::sc::read_htb<HASHW, ARW>(htb_buf, 0, bit_vector0);

        join_v3::sc::read_htb<HASHW, ARW>(htb_buf, HASH_DEPTH, bit_vector1);
    }
}

// top function of hash join PU
template <int HASH_MODE, int HASHWH, int HASHWL, int KEYW, int S_PW, int T_PW, int ARW>
void build_merge_probe_wrapper(
    // input status
    ap_uint<32> id,
    bool& build_probe_flag,
    ap_uint<32> depth,
    ap_uint<32>& pu_start_addr,
    ap_uint<32>& pu_end_addr,

    // input table
    hls::stream<ap_uint<HASHWL> >& i_hash_strm,
    hls::stream<ap_uint<KEYW> >& i_key_strm,
    hls::stream<ap_uint<T_PW> >& i_pld_strm,
    hls::stream<bool>& i_e_strm,

    // output for join
    hls::stream<ap_uint<KEYW> >& o_t_key_strm,
    hls::stream<ap_uint<T_PW> >& o_t_pld_strm,
    hls::stream<ap_uint<ARW> >& o_nm_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<KEYW> >& o_s_key_strm,
    hls::stream<ap_uint<S_PW> >& o_s_pld_strm,
    hls::stream<bool>& o_e1_strm,

    ap_uint<256>* htb_buf,
    ap_uint<256>* stb_buf) {
#pragma HLS INLINE off

    // alllocate uram storage
    const int HASH_DEPTH = (1 << HASHWL) / 3 + 1;

#ifndef __SYNTHESIS__

    ap_uint<72>* bit_vector0;
    ap_uint<72>* bit_vector1;
    bit_vector0 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));
    bit_vector1 = (ap_uint<72>*)malloc(HASH_DEPTH * sizeof(ap_uint<72>));

#else

    ap_uint<72> bit_vector0[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector0 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
    ap_uint<72> bit_vector1[HASH_DEPTH];
#pragma HLS ARRAY_PARTITION variable = bit_vector1 block factor = 4 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram

#endif

    // initilize uram by previous hash build or probe
    initiate_uram<HASHWL, ARW>(id, htb_buf, bit_vector0, bit_vector1);

    if (!build_probe_flag) {
// build
#ifndef __SYNTHESIS__
        std::cout << "----------------------build------------------------" << std::endl;
#endif

        build_wrapper<HASHWL, KEYW, S_PW, ARW>(
            // input status
            id, depth, pu_start_addr, pu_end_addr,

            // input s-table
            i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

            // HBM/DDR
            htb_buf, stb_buf, bit_vector0, bit_vector1);
    } else {
        // merge && probe
        bool is_first_probe;
        ap_uint<72> max_bitmap;
        max_bitmap = bit_vector1[HASH_DEPTH - 1];
        is_first_probe = max_bitmap(71, 48) < pu_start_addr && pu_start_addr > 0;

#ifndef __SYNTHESIS__
        std::cout << std::dec << "pu_start_addr=" << pu_start_addr << " max_htb_addr=" << HASH_DEPTH - 1
                  << " max_bitmap=" << max_bitmap << std::endl;
#endif

        if (is_first_probe) {
// the first time to probe, need to fully build bitmap
#ifndef __SYNTHESIS__
            std::cout << "----------------------merge------------------------" << std::endl;
#endif

            merge_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, ARW>(
                // input status
                depth, pu_start_addr,

                htb_buf, stb_buf, bit_vector1);
        }

// probe
#ifndef __SYNTHESIS__
        std::cout << "-----------------------Probe------------------------" << std::endl;
#endif

        probe_wrapper<HASHWL, KEYW, S_PW, T_PW, ARW>(is_first_probe, depth, pu_start_addr,

                                                     // input large table
                                                     i_hash_strm, i_key_strm, i_pld_strm, i_e_strm,

                                                     // output for join
                                                     o_t_key_strm, o_t_pld_strm, o_nm_strm, o_e0_strm, o_s_key_strm,
                                                     o_s_pld_strm, o_e1_strm,

                                                     stb_buf, bit_vector0, bit_vector1);
    }

#ifndef __SYNTHESIS__

    free(bit_vector0);
    free(bit_vector1);

#endif
}

//-----------------------------------------------join-----------------------------------------------
/// @brief compare key, if match output joined row
template <int KEYW, int S_PW, int T_PW, int ARW>
void join_unit(bool& build_probe_flag,
#ifndef __SYNTHESIS__
               int pu_id,
#endif

               hls::stream<ap_uint<KEYW> >& i_t_key_strm,
               hls::stream<ap_uint<T_PW> >& i_t_pld_strm,
               hls::stream<ap_uint<ARW> >& i_nm_strm,
               hls::stream<bool>& i_e0_strm,

               hls::stream<ap_uint<KEYW> >& i_s_key_strm,
               hls::stream<ap_uint<S_PW> >& i_s_pld_strm,
               hls::stream<bool>& i_e1_strm,

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
        bool s_last = i_e1_strm.read();
    JOIN_LOOP:
        while (!t_last) {
            t_key = i_t_key_strm.read();
            t_pld = i_t_pld_strm.read();
            ap_uint<ARW> nm = i_nm_strm.read();
            t_last = i_e0_strm.read();

#ifndef __SYNTHESIS__
            hit_failed = true;
            cnt2++;
#endif

            while (nm--) {
#pragma HLS PIPELINE II = 1
                s_key = i_s_key_strm.read();
                s_pld = i_s_pld_strm.read();
                s_last = i_e1_strm.read();

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
#ifdef DEBUG
                    std::cout << std::hex << "Join Unit=" << j << " nm=" << nm << " key=" << s_key << " s_pld=" << s_pld
                              << " t_pld=" << t_pld << std::endl;
#endif
                    cnt0++;
                    hit_failed = false;
                } else {
#ifdef DEBUG
                    std::cout << std::hex << "nm=" << nm << " s_key=" << s_key << " t_key=" << t_key
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
    std::cout << std::dec << "Join Unit output " << cnt0 << " rows, mismatch " << cnt1 << " rows" << std::endl;
    std::cout << std::dec << "Join Unit hit " << cnt2 << " times, hit failed " << cnt3 << " times" << std::endl;
#endif
}

//---------------------------------------------collect-----------------------------------------

// collect join result of mutiple PU
template <int PU, int JW>
void collect_unit(bool& build_probe_flag,
                  hls::stream<ap_uint<JW> > i_jrow_strm[PU],
                  hls::stream<bool> i_e_strm[PU],

                  ap_uint<32>& join_num,
                  hls::stream<ap_uint<JW> >& o_jrow_strm,
                  hls::stream<bool>& o_e_strm) {
#pragma HLS INLINE off

    const int MAX = (1 << PU) - 1;
    ap_uint<JW> jrow_arr[PU];
#pragma HLS array_partition variable = jrow_arr dim = 1

    ap_uint<PU> empty_e = 0;
    ap_uint<PU> last = 0;
    ap_uint<PU> rd_e = 0;
    ap_uint<32> cnt = 0;

#ifndef __SYNTHESIS__
    std::cout << std::dec << "PU=" << PU << std::endl;
    std::cout << "build_probe_flag" << build_probe_flag << std::endl;
#endif
    if (build_probe_flag) {
        // do collect if it is probe
        do {
#pragma HLS pipeline II = 1
            for (int i = 0; i < PU; i++) {
#pragma HLS unroll
                empty_e[i] = !i_e_strm[i].empty() && !last[i];
            }

            rd_e = join_v2::mul_ch_read(empty_e);

            for (int i = 0; i < PU; i++) {
#pragma HLS unroll
                if (rd_e[i]) {
                    jrow_arr[i] = i_jrow_strm[i].read();
                    last[i] = i_e_strm[i].read();
                }
            }

            ap_uint<3> id = join_v2::mux<PU>(rd_e);
            ap_uint<JW> j = jrow_arr[id];
            bool valid_n = last[id];

            if (!valid_n && rd_e != 0) {
                o_jrow_strm.write(j);
                o_e_strm.write(false);
                cnt++;
            }
        } while (last != MAX);

#ifndef __SYNTHESIS__
        std::cout << std::dec << "Collect " << cnt << " rows" << std::endl;
#endif

        join_num = cnt;
        o_e_strm.write(true);
    }
}

} // namespace bp

} // namespace join_v3
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Hash-Join v3 primitive, it takes more resourse than ``hashJoinMPU`` and promises a better
 * performance in large size of table
 *
 * The maximum size of small table is 256MBx8(HBM number)=2GB in this design. The total hash entries
 * is equal to 1<<(HASHWH + HASHWL), and it is limitied to maximum of 1M entries because of the size
 * of URAM in a single SLR.
 *
 * This module can accept more than 1 input row per cycle, via multiple input channels. The small
 * table and the big table shares the same input ports, so the width of the payload should be the
 * max of both, while the data should be aligned to the little-end. To be different with ``hashJoinMPU``,
 * the small table and big table should be fed only once.
 *
 * @tparam HASH_MODE 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam KEYW width of key, in bit.
 * @tparam PW width of max payload, in bit.
 * @tparam S_PW width of payload of small table.
 * @tparam B_PW width of payload of big table.
 * @tparam HASHWH number of hash bits used for PU/buffer selection, 1~3.
 * @tparam HASHWL number of hash bits used for hash-table in PU.
 * @tparam ARW width of address, larger than 24 is suggested.
 * @tparam CH_NM number of input channels, 1,2,4.
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
 * @param pu_begin_status_strms contains hash depth, row number of join result
 * @param pu_end_status_strms contains hash depth, row number of join result
 *
 * @param j_strm output of joined result
 * @param j_e_strm end flag of joined result
 */
template <int HASH_MODE, int KEYW, int PW, int S_PW, int B_PW, int HASHWH, int HASHWL, int ARW, int CH_NM>
void hashJoinV3(
    // input
    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output hash table
    ap_uint<256>* htb0_buf,
    ap_uint<256>* htb1_buf,
    ap_uint<256>* htb2_buf,
    ap_uint<256>* htb3_buf,
    ap_uint<256>* htb4_buf,
    ap_uint<256>* htb5_buf,
    ap_uint<256>* htb6_buf,
    ap_uint<256>* htb7_buf,

    // output
    ap_uint<256>* stb0_buf,
    ap_uint<256>* stb1_buf,
    ap_uint<256>* stb2_buf,
    ap_uint<256>* stb3_buf,
    ap_uint<256>* stb4_buf,
    ap_uint<256>* stb5_buf,
    ap_uint<256>* stb6_buf,
    ap_uint<256>* stb7_buf,

    hls::stream<ap_uint<32> >& pu_begin_status_strms,
    hls::stream<ap_uint<32> >& pu_end_status_strms,

    hls::stream<ap_uint<KEYW + S_PW + B_PW> >& j_strm,
    hls::stream<bool>& j_e_strm) {
    enum { PU = (1 << HASHWH) }; // high hash for distribution.

#pragma HLS DATAFLOW

    ap_uint<32> depth;
    ap_uint<32> join_num;

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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c0[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c1[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c2[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 1
#pragma HLS bind_storage variable = hash_strm_arry_c3 type = fifo impl = srl
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 1

    // merge channel1-channel4 to here, then perform build or probe
    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 1
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 1
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 1
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 1

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
#pragma HLS stream variable = j0_strm_arry depth = 8
#pragma HLS array_partition variable = j0_strm_arry dim = 1
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS array_partition variable = e3_strm_arry dim = 1
#pragma HLS stream variable = e3_strm_arry depth = 8

//------------------------------read status-----------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------read status------------------------" << std::endl;
#endif
#endif
    details::join_v3::sc::read_status<PU>(pu_begin_status_strms, depth);

//---------------------------------dispatch PU-------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------dispatch PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM >= 1) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }

    if (CH_NM >= 2) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }

    if (CH_NM >= 4) {
        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);

        details::join_v3::sc::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

//---------------------------------merge PU---------------------------------
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "------------------------merge PU------------------------" << std::endl;
#endif
#endif

    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v3::sc::merge1_1_wrapper<KEYW, PW, HASHWL>(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge2_1_wrapper<KEYW, PW, HASHWL>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        // CH_NM == 4
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v3::sc::merge4_1_wrapper<KEYW, PW, HASHWL>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            depth,

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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
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
        details::join_v3::sc::join_unit<KEYW, S_PW, B_PW, ARW>(
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

} // hash_join_v3

/**
 * @brief Hash-Build-Probe v3 primitive, it can perform hash build and hash probe separately. It needs
 * two call of kernel to perform build and probe seperately. There is a control flag to decide buld or
 * probe. This primitive supports multiple build and mutiple probe, for example, you can scadule a workflow
 * as: build0->build1->probe0->probe1->build2->build3->probe3...
 *
 * The maximum size of small table is 256MBx8=2GB in this design. The total hash entries is equal
 * to 1<<(HASHWH + HASHWL), and it is limitied to maximum of 1M entries because of the size of URAM
 * in a single SLR.
 *
 * This module can accept more than 1 input row per cycle, via multiple input channels. The small
 * table and the big table shares the same input ports, so the width of the payload should be the
 * max of both, while the data should be aligned to the little-end. The small table and big table
 * should be fed only ONCE.
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
 * @tparam BF_W bloom-filter hash width.
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
 * @param pu_begin_status_strms contains build id, fixed hash depth, joined number of last probe and start addr of
 * unused stb_buf for each PU
 * @param pu_end_status_strms returns next build id, fixed hash depth, joined number of current probe and end addr of
 * used stb_buf for each PU
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
          int BF_W,
          int EN_BF>
static void hashBuildProbeV3(
    // input
    bool& build_probe_flag,
    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output hash table
    ap_uint<256>* htb0_buf,
    ap_uint<256>* htb1_buf,
    ap_uint<256>* htb2_buf,
    ap_uint<256>* htb3_buf,
    ap_uint<256>* htb4_buf,
    ap_uint<256>* htb5_buf,
    ap_uint<256>* htb6_buf,
    ap_uint<256>* htb7_buf,

    // output
    ap_uint<256>* stb0_buf,
    ap_uint<256>* stb1_buf,
    ap_uint<256>* stb2_buf,
    ap_uint<256>* stb3_buf,
    ap_uint<256>* stb4_buf,
    ap_uint<256>* stb5_buf,
    ap_uint<256>* stb6_buf,
    ap_uint<256>* stb7_buf,

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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c0[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c1[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c2[PU];
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
    hls::stream<ap_uint<HASHWL> > hash_strm_arry_c3[PU];
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
    ap_uint<32> id;
    ap_uint<32> depth;
    ap_uint<32> join_num;

    // merge channel1-channel4 to here, then perform build or probe
    hls::stream<ap_uint<KEYW> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 1
#pragma HLS bind_storage variable = k1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 1
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<HASHWL> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 1
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 1

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
#pragma HLS bind_storage variable = e2_strm_arry type = fifo impl = bram

    hls::stream<ap_uint<KEYW> > s_key_strm_arry[PU];
#pragma HLS stream variable = s_key_strm_arry depth = 512
#pragma HLS array_partition variable = s_key_strm_arry dim = 1
#pragma HLS bind_storage variable = s_key_strm_arry type = fifo impl = bram
    hls::stream<ap_uint<PW> > s_pld_strm_arry[PU];
#pragma HLS stream variable = s_pld_strm_arry depth = 512
#pragma HLS array_partition variable = s_pld_strm_arry dim = 1
#pragma HLS bind_storage variable = s_pld_strm_arry type = fifo impl = bram
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS stream variable = e3_strm_arry depth = 512
#pragma HLS array_partition variable = e3_strm_arry dim = 1
#pragma HLS bind_storage variable = e3_strm_arry type = fifo impl = bram

    // output of join for collect
    hls::stream<ap_uint<KEYW + S_PW + B_PW> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 8
#pragma HLS array_partition variable = j0_strm_arry dim = 1
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl
    hls::stream<bool> e4_strm_arry[PU];
#pragma HLS array_partition variable = e4_strm_arry dim = 1
#pragma HLS stream variable = e4_strm_arry depth = 8

    //---------------------------------dispatch PU-------------------------------
    if (CH_NM >= 1) {
        details::join_v3::bp::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[0], p0_strm_arry[0], e0_strm_arry[0], k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
            e1_strm_arry_c0);
    }

    if (CH_NM >= 2) {
        details::join_v3::bp::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[1], p0_strm_arry[1], e0_strm_arry[1], k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
            e1_strm_arry_c1);
    }

    if (CH_NM >= 4) {
        details::join_v3::bp::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[2], p0_strm_arry[2], e0_strm_arry[2], k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
            e1_strm_arry_c2);

        details::join_v3::bp::dispatch_wrapper<HASH_MODE, KEYW, PW, HASHWH, HASHWL, PU>(
            k0_strm_arry[3], p0_strm_arry[3], e0_strm_arry[3], k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
            e1_strm_arry_c3);
    }

    //---------------------------------merge PU---------------------------------
    if (CH_NM == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v2::merge1_1<KEYW, PW, HASHWL>(
                k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p], e1_strm_arry_c0[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else if (CH_NM == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge2_1<KEYW, PW, HASHWL>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p], p1_strm_arry_c1[p], hash_strm_arry_c0[p],
                hash_strm_arry_c1[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    } else {
        // CH_NM == 4
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge4_1<KEYW, PW, HASHWL>(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p],

                k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    //------------------------------read status-----------------------------------
    details::join_v3::bp::read_status<PU>(pu_begin_status_strms, id, depth, pu_start_addr);
#ifndef __SYNTHESIS__
    std::cout << std::dec << "PU start status: id=" << id << " depth=" << depth << std::endl;
    for (int i = 0; i < PU; i++) {
        std::cout << std::hex << "PU start addr" << i << "=" << pu_start_addr[i] << std::endl;
    }
#endif

    //-------------------------------build----------------------------------------
    if (PU >= 1) {
#ifndef __SYNTHESIS__
        std::cout << "-----------------PU0-----------------" << std::endl;
#endif

        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[0], pu_end_addr[0],

            // input table
            hash_strm_arry[0], k1_strm_arry[0], p1_strm_arry[0], e1_strm_arry[0],

            // output for join
            t_key_strm_arry[0], t_pld_strm_arry[0], nm_strm_arry[0], e2_strm_arry[0], s_key_strm_arry[0],
            s_pld_strm_arry[0], e3_strm_arry[0],

            // HBM/DDR
            htb0_buf, stb0_buf);
    }

    if (PU >= 2) {
#ifndef __SYNTHESIS__
        std::cout << "-----------------PU1-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[1], pu_end_addr[1],

            // input t-table
            hash_strm_arry[1], k1_strm_arry[1], p1_strm_arry[1], e1_strm_arry[1],

            // output for join
            t_key_strm_arry[1], t_pld_strm_arry[1], nm_strm_arry[1], e2_strm_arry[1], s_key_strm_arry[1],
            s_pld_strm_arry[1], e3_strm_arry[1],

            // HBM/DDR
            htb1_buf, stb1_buf);
    }

    if (PU >= 4) {
#ifndef __SYNTHESIS__
        std::cout << "-----------------PU2-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[2], pu_end_addr[2],

            // input t-table
            hash_strm_arry[2], k1_strm_arry[2], p1_strm_arry[2], e1_strm_arry[2],

            // output for join
            t_key_strm_arry[2], t_pld_strm_arry[2], nm_strm_arry[2], e2_strm_arry[2], s_key_strm_arry[2],
            s_pld_strm_arry[2], e3_strm_arry[2],

            // HBM/DDR
            htb2_buf, stb2_buf);

#ifndef __SYNTHESIS__
        std::cout << "-----------------PU3-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[3], pu_end_addr[3],

            // input t-table
            hash_strm_arry[3], k1_strm_arry[3], p1_strm_arry[3], e1_strm_arry[3],

            // output for join
            t_key_strm_arry[3], t_pld_strm_arry[3], nm_strm_arry[3], e2_strm_arry[3], s_key_strm_arry[3],
            s_pld_strm_arry[3], e3_strm_arry[3],

            // HBM/DDR
            htb3_buf, stb3_buf);
    }

    if (PU >= 8) {
#ifndef __SYNTHESIS__
        std::cout << "-----------------PU4-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[4], pu_end_addr[4],

            // input t-table
            hash_strm_arry[4], k1_strm_arry[4], p1_strm_arry[4], e1_strm_arry[4],

            // output for join
            t_key_strm_arry[4], t_pld_strm_arry[4], nm_strm_arry[4], e2_strm_arry[4], s_key_strm_arry[4],
            s_pld_strm_arry[4], e3_strm_arry[4],

            // HBM/DDR
            htb4_buf, stb4_buf);

#ifndef __SYNTHESIS__
        std::cout << "-----------------PU5-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[5], pu_end_addr[5],

            // input t-table
            hash_strm_arry[5], k1_strm_arry[5], p1_strm_arry[5], e1_strm_arry[5],

            // output for join
            t_key_strm_arry[5], t_pld_strm_arry[5], nm_strm_arry[5], e2_strm_arry[5], s_key_strm_arry[5],
            s_pld_strm_arry[5], e3_strm_arry[5],

            // HBM/DDR
            htb5_buf, stb5_buf);

#ifndef __SYNTHESIS__
        std::cout << "-----------------PU6-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[6], pu_end_addr[6],

            // input t-table
            hash_strm_arry[6], k1_strm_arry[6], p1_strm_arry[6], e1_strm_arry[6],

            // output for join
            t_key_strm_arry[6], t_pld_strm_arry[6], nm_strm_arry[6], e2_strm_arry[6], s_key_strm_arry[6],
            s_pld_strm_arry[6], e3_strm_arry[6],

            // HBM/DDR
            htb6_buf, stb6_buf);

#ifndef __SYNTHESIS__
        std::cout << "-----------------PU7-----------------" << std::endl;
#endif
        details::join_v3::bp::build_merge_probe_wrapper<HASH_MODE, HASHWH, HASHWL, KEYW, S_PW, B_PW, ARW>(
            // input status
            id, build_probe_flag, depth, pu_start_addr[7], pu_end_addr[7],

            // input t-table
            hash_strm_arry[7], k1_strm_arry[7], p1_strm_arry[7], e1_strm_arry[7],

            // output for join
            t_key_strm_arry[7], t_pld_strm_arry[7], nm_strm_arry[7], e2_strm_arry[7], s_key_strm_arry[7],
            s_pld_strm_arry[7], e3_strm_arry[7],

            // HBM/DDR
            htb7_buf, stb7_buf);
    }

    //-----------------------------------join--------------------------------------
    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v3::bp::join_unit<KEYW, S_PW, B_PW, ARW>(build_probe_flag,

#ifndef __SYNTHESIS__
                                                               i,
#endif

                                                               t_key_strm_arry[i], t_pld_strm_arry[i], nm_strm_arry[i],
                                                               e2_strm_arry[i], s_key_strm_arry[i], s_pld_strm_arry[i],
                                                               e3_strm_arry[i], j0_strm_arry[i], e4_strm_arry[i]);
    }

    //-----------------------------------Collect-----------------------------------
    details::join_v3::bp::collect_unit<PU, KEYW + S_PW + B_PW>(build_probe_flag, j0_strm_arry, e4_strm_arry, join_num,
                                                               j_strm, j_e_strm);

    //------------------------------Write status----------------------------------
    details::join_v3::bp::write_status<PU>(pu_end_status_strms, id, depth, join_num, pu_end_addr);

} // hash_build_probe_v3

} // namespace database
} // namespace xf

#undef write_bit_vector0
#undef read_bit_vector0

#undef write_bit_vector1
#undef read_bit_vector1

#endif // !defined(XF_DATABASE_HASH_JOIN_v3)
