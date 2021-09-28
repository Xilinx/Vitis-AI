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
 * @file hash_semi_join.hpp
 * @brief hash join template function implementation, targeting HBM devices.
 *
 * The limitations are:
 * (1) less than 2M entries from inner table.
 * (2) max number of key with same hash is less than 256K.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_HASH_SEMI_JOIN_MPU_H
#define XF_DATABASE_HASH_SEMI_JOIN_MPU_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_database/hash_join_v2.hpp"

// FIXME For debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace details {
namespace hash_semi_join {
// ------------------------------------------------------------

/// @brief compare key, if match, output the joined row from outer table  only once.
// that is to say, if a key in outer table matches with many keys in inner table,
// the corresonding row of outer table is output once.
template <int WPayload, int WKey>
void semi_join_unit(hls::stream<ap_uint<WKey> >& inner_row_istrm,
                    hls::stream<ap_uint<WKey> >& outer_key_istrm,
                    hls::stream<ap_uint<WPayload> >& outer_playload_istrm,
                    hls::stream<ap_uint<18> >& nm_istrm,
                    hls::stream<bool>& outer_end_istrm,
                    hls::stream<ap_uint<WPayload> >& join_ostrm,
                    hls::stream<bool>& join_end_ostrm) {
    ap_uint<WKey> inner_key = 0;
    ap_uint<WKey> outer_key = 0;
    ap_uint<WPayload> outer_payload = 0;
    bool last = outer_end_istrm.read();
    ap_uint<18> nm = 0;
    bool matched = false;
#ifndef __SYNTHESIS__
    unsigned int cnt = 0;
#endif
JOIN_LOOP:
    while (!last) {
#pragma HLS PIPELINE II = 1
        if (nm == 0) {
            last = outer_end_istrm.read();
            outer_payload = outer_playload_istrm.read();
            outer_key = outer_key_istrm.read();
            nm = nm_istrm.read();
            matched = false;
        } else {
            nm--;
            ap_uint<WKey> stb_row = inner_row_istrm.read();
            inner_key = stb_row(WKey - 1, 0);
            if (inner_key == outer_key && false == matched) {
                matched = true; // if match, set the flag to true in case of output more times.
                ap_uint<WPayload> j = 0;
                j(WPayload - 1, 0) = outer_payload;
                join_ostrm.write(j);
                join_end_ostrm.write(false);
#ifndef __SYNTHESIS__
                cnt++;
#endif
            }
        }
    }
    if (nm != 0) {
    JOIN_CLEAR_LOOP:
        for (int i = 0; i < nm; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<WKey> stb_row = inner_row_istrm.read();
            inner_key = stb_row(WKey - 1, 0);
            if (inner_key == outer_key && false == matched) {
                matched = true;
                ap_uint<WPayload> j = 0;
                j(WPayload - 1, 0) = outer_payload;
                join_ostrm.write(j);
                join_end_ostrm.write(false);
#ifndef __SYNTHESIS__
                cnt++;
#endif
            }
        }
    }
    join_ostrm.write(0);
    join_end_ostrm.write(true);
#ifndef __SYNTHESIS__
    std::cout << " Semi Join Unit output " << cnt << " rows" << std::endl;
#endif
} // semi_join_unit

} // namespace hash_semi_join
} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Multi-PU Hash-Semi-Join primitive, using multiple DDR/HBM buffers.
 *
 * The max number of lines of inner table is 2M in this design.
 * It is assumed that the hash-conflict is within 256K per bin.
 *
 * This module can accept more than 1 input row per cycle, via multiple
 * input channels.
 * The outer table and the inner table share the same input ports,
 * so the width of the payload should be the max of both, while the data
 * should be aligned to the little-end.
 * The inner table should be fed TWICE, followed by the outer table ONCE.
 *
 * @tparam HashMode 0 for radix and 1 for Jenkin's Lookup3 hash.
 * @tparam WKey width of key, in bit.
 * @tparam WPayload width of payload of outer table.
 * @tparam WHashHigh number of hash bits used for PU/buffer selection, 1~3.
 * @tparam WhashLow number of hash bits used for hash-table in PU.
 * @tparam WTmpBufferAddress width of address, log2(inner table max num of rows).
 * @tparam WTmpBuffer width of buffer.
 * @tparam NChannels number of input channels, 1,2,4.
 * @tparam WBloomFilter bloom-filter hash width.
 * @tparam EnBloomFilter bloom-filter switch, 0 for off, 1 for on.
 *
 * @param key_istrms input of key columns of both tables.
 * @param payload_istrms input of payload columns of both tables.
 * @param e0_strm_arry input of end signal of both tables.
 * @param pu0_tmp_rwtpr HBM/DDR buffer of PU0
 * @param pu1_tmp_rwptr HBM/DDR buffer of PU1
 * @param pu2_tmp_rwptr HBM/DDR buffer of PU2
 * @param pu3_tmp_rwptr HBM/DDR buffer of PU3
 * @param pu4_tmp_rwptr HBM/DDR buffer of PU4
 * @param pu5_tmp_rwptr HBM/DDR buffer of PU5
 * @param pu6_tmp_rwptr HBM/DDR buffer of PU6
 * @param pu7_tmp_rwptr HBM/DDR buffer of PU7
 * @param join_ostrm output of joined rows.
 * @param end_ostrm end signal of joined rows.
 */
template <int HashMode,
          int WKey,
          int WPayload,
          int WHashHigh,
          int WhashLow,
          int WTmpBufferAddress,
          int WTmpBuffer,
          int NChannels,
          int WBloomFilter,
          int EnBloomFilter>
static void hashSemiJoin(hls::stream<ap_uint<WKey> > key_istrms[NChannels],
                         hls::stream<ap_uint<WPayload> > payload_istrms[NChannels],
                         hls::stream<bool> e0_strm_arry[NChannels],
                         ap_uint<WTmpBuffer>* pu0_tmp_rwtpr,
                         ap_uint<WTmpBuffer>* pu1_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu2_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu3_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu4_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu5_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu6_tmp_rwptr,
                         ap_uint<WTmpBuffer>* pu7_tmp_rwptr,
                         hls::stream<ap_uint<WPayload> >& join_ostrm,
                         hls::stream<bool>& end_ostrm) {
    enum { HDP_J = (1 << (WhashLow - 2)) }; // 4 entries per slot, so -2.
    enum { PU = (1 << WHashHigh) };         // high hash for distribution.

#pragma HLS dataflow

    hls::stream<ap_uint<WKey> > k1_strm_arry[PU];
#pragma HLS stream variable = k1_strm_arry depth = 8
#pragma HLS array_partition variable = k1_strm_arry dim = 0
    hls::stream<ap_uint<WPayload> > p1_strm_arry[PU];
#pragma HLS stream variable = p1_strm_arry depth = 8
#pragma HLS array_partition variable = p1_strm_arry dim = 0
#pragma HLS bind_storage variable = p1_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<WhashLow> > hash_strm_arry[PU];
#pragma HLS stream variable = hash_strm_arry depth = 8
#pragma HLS array_partition variable = hash_strm_arry dim = 0
    hls::stream<bool> e1_strm_arry[PU];
#pragma HLS stream variable = e1_strm_arry depth = 8
#pragma HLS array_partition variable = e1_strm_arry dim = 0

    // NChannels >= 1
    hls::stream<ap_uint<WKey> > k1_strm_arry_c0[PU];
#pragma HLS stream variable = k1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c0 dim = 0
    hls::stream<ap_uint<WPayload> > p1_strm_arry_c0[PU];
#pragma HLS stream variable = p1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c0 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c0 type = fifo impl = srl
    hls::stream<ap_uint<WhashLow> > hash_strm_arry_c0[PU];
#pragma HLS stream variable = hash_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c0 dim = 0
    hls::stream<bool> e1_strm_arry_c0[PU];
#pragma HLS stream variable = e1_strm_arry_c0 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c0 dim = 0
    // NChannels >= 2
    hls::stream<ap_uint<WKey> > k1_strm_arry_c1[PU];
#pragma HLS stream variable = k1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c1 dim = 0
    hls::stream<ap_uint<WPayload> > p1_strm_arry_c1[PU];
#pragma HLS stream variable = p1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c1 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c1 type = fifo impl = srl
    hls::stream<ap_uint<WhashLow> > hash_strm_arry_c1[PU];
#pragma HLS stream variable = hash_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c1 dim = 0
    hls::stream<bool> e1_strm_arry_c1[PU];
#pragma HLS stream variable = e1_strm_arry_c1 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c1 dim = 0
    // NChannels >= 4
    hls::stream<ap_uint<WKey> > k1_strm_arry_c2[PU];
#pragma HLS stream variable = k1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<WPayload> > p1_strm_arry_c2[PU];
#pragma HLS stream variable = p1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c2 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c2 type = fifo impl = srl
    hls::stream<ap_uint<WhashLow> > hash_strm_arry_c2[PU];
#pragma HLS stream variable = hash_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c2 dim = 0
    hls::stream<bool> e1_strm_arry_c2[PU];
#pragma HLS stream variable = e1_strm_arry_c2 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c2 dim = 0
    hls::stream<ap_uint<WKey> > k1_strm_arry_c3[PU];
#pragma HLS stream variable = k1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = k1_strm_arry_c3 dim = 0
    hls::stream<ap_uint<WPayload> > p1_strm_arry_c3[PU];
#pragma HLS stream variable = p1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = p1_strm_arry_c3 dim = 0
#pragma HLS bind_storage variable = p1_strm_arry_c3 type = fifo impl = srl
    hls::stream<ap_uint<WhashLow> > hash_strm_arry_c3[PU];
#pragma HLS stream variable = hash_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = hash_strm_arry_c3 dim = 0
    hls::stream<bool> e1_strm_arry_c3[PU];
#pragma HLS stream variable = e1_strm_arry_c3 depth = 8
#pragma HLS array_partition variable = e1_strm_arry_c3 dim = 0

    hls::stream<ap_uint<WKey> > w_row_strm[PU];
#pragma HLS stream variable = w_row_strm depth = 8
#pragma HLS array_partition variable = w_row_strm dim = 0
    hls::stream<ap_uint<WKey> > r_row_strm[PU];
#pragma HLS stream variable = r_row_strm depth = 8
    hls::stream<ap_uint<WKey> > k2_strm_arry[PU];
#pragma HLS stream variable = k2_strm_arry depth = 32
#pragma HLS array_partition variable = k2_strm_arry dim = 0
#pragma HLS bind_storage variable = k2_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<WPayload> > p2_strm_arry[PU];
#pragma HLS stream variable = p2_strm_arry depth = 32
#pragma HLS array_partition variable = p2_strm_arry dim = 0
#pragma HLS bind_storage variable = p2_strm_arry type = fifo impl = srl
    hls::stream<bool> e2_strm_arry[PU];
#pragma HLS stream variable = e2_strm_arry depth = 8
#pragma HLS array_partition variable = e2_strm_arry dim = 0
    hls::stream<bool> e3_strm_arry[PU];
#pragma HLS stream variable = e3_strm_arry depth = 32
#pragma HLS array_partition variable = e3_strm_arry dim = 0
    hls::stream<bool> e4_strm_arry[PU];
#pragma HLS array_partition variable = e4_strm_arry dim = 0
#pragma HLS stream variable = e4_strm_arry depth = 8
    hls::stream<ap_uint<WTmpBufferAddress> > addr_strm[PU];
#pragma HLS stream variable = addr_strm depth = 8
#pragma HLS array_partition variable = addr_strm dim = 0
    hls::stream<ap_uint<18> > nm0_strm_arry[PU];
#pragma HLS stream variable = nm0_strm_arry depth = 32
#pragma HLS array_partition variable = nm0_strm_arry dim = 0
    hls::stream<ap_uint<WPayload> > j0_strm_arry[PU];
#pragma HLS stream variable = j0_strm_arry depth = 8
#pragma HLS array_partition variable = j0_strm_arry dim = 0
#pragma HLS bind_storage variable = j0_strm_arry type = fifo impl = srl

#ifndef __SYNTHESIS__
    ap_uint<72>* bit_vector0[PU];
    ap_uint<72>* bit_vector1[PU];
    ap_uint<72>* bit_vector2[PU];
    ap_uint<72>* bit_vector3[PU];
    for (int i = 0; i < PU; i++) {
        bit_vector0[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector1[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector2[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
        bit_vector3[i] = (ap_uint<72>*)malloc((HDP_J >> 2) * sizeof(ap_uint<72>));
    }
#else
    ap_uint<72> bit_vector0[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector1[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector2[PU][(HDP_J >> 2)];
    ap_uint<72> bit_vector3[PU][(HDP_J >> 2)];
#pragma HLS array_partition variable = bit_vector0 dim = 1
#pragma HLS bind_storage variable = bit_vector0 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector1 dim = 1
#pragma HLS bind_storage variable = bit_vector1 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector2 dim = 1
#pragma HLS bind_storage variable = bit_vector2 type = ram_2p impl = uram
#pragma HLS array_partition variable = bit_vector3 dim = 1
#pragma HLS bind_storage variable = bit_vector3 type = ram_2p impl = uram
#endif

    // clang-format off
      // -------------|----------|------------|------|------|------|--------
      //   Dispatch   |          |Bloom Filter|Bitmap|Build |Probe |        
      // -------------|          |------------|------|------|------|        
      //   Dispatch   | switcher |Bloom filter|Bitmap|Build |Probe |Collect 
      // -------------|          |------------|------|------|------|        
      //   Dispatch   |          |Bloom filter|Bitmap|Build |Porbe |        
      // -------------|----------|------------|------|------|------|--------
    // clang-format on
    ;

    if (NChannels >= 1) {
        details::join_v2::dispatch_wrapper<HashMode, WKey, WPayload, WHashHigh, WhashLow, PU, WBloomFilter,
                                           EnBloomFilter>(key_istrms[0], payload_istrms[0], e0_strm_arry[0],
                                                          k1_strm_arry_c0, p1_strm_arry_c0, hash_strm_arry_c0,
                                                          e1_strm_arry_c0);
    }
    if (NChannels >= 2) {
        details::join_v2::dispatch_wrapper<HashMode, WKey, WPayload, WHashHigh, WhashLow, PU, WBloomFilter,
                                           EnBloomFilter>(key_istrms[1], payload_istrms[1], e0_strm_arry[1],
                                                          k1_strm_arry_c1, p1_strm_arry_c1, hash_strm_arry_c1,
                                                          e1_strm_arry_c1);
    }
    if (NChannels >= 4) {
        details::join_v2::dispatch_wrapper<HashMode, WKey, WPayload, WHashHigh, WhashLow, PU, WBloomFilter,
                                           EnBloomFilter>(key_istrms[2], payload_istrms[2], e0_strm_arry[2],
                                                          k1_strm_arry_c2, p1_strm_arry_c2, hash_strm_arry_c2,
                                                          e1_strm_arry_c2);
        details::join_v2::dispatch_wrapper<HashMode, WKey, WPayload, WHashHigh, WhashLow, PU, WBloomFilter,
                                           EnBloomFilter>(key_istrms[3], payload_istrms[3], e0_strm_arry[3],
                                                          k1_strm_arry_c3, p1_strm_arry_c3, hash_strm_arry_c3,
                                                          e1_strm_arry_c3);
    }

    if (NChannels == 1) {
        for (int p = 0; p < PU; ++p) {
#pragma HLS unroll
            details::join_v2::merge1_1_wrapper(k1_strm_arry_c0[p], p1_strm_arry_c0[p], hash_strm_arry_c0[p],
                                               e1_strm_arry_c0[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p],
                                               e1_strm_arry[p]);
        }
    }
    if (NChannels == 2) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge2_1_wrapper(k1_strm_arry_c0[p], k1_strm_arry_c1[p], p1_strm_arry_c0[p],
                                               p1_strm_arry_c1[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                                               e1_strm_arry_c0[p], e1_strm_arry_c1[p], k1_strm_arry[p], p1_strm_arry[p],
                                               hash_strm_arry[p], e1_strm_arry[p]);
        }
    }
    if (NChannels == 4) {
        for (int p = 0; p < PU; p++) {
#pragma HLS unroll
            details::join_v2::merge4_1_wrapper(
                k1_strm_arry_c0[p], k1_strm_arry_c1[p], k1_strm_arry_c2[p], k1_strm_arry_c3[p], p1_strm_arry_c0[p],
                p1_strm_arry_c1[p], p1_strm_arry_c2[p], p1_strm_arry_c3[p], hash_strm_arry_c0[p], hash_strm_arry_c1[p],
                hash_strm_arry_c2[p], hash_strm_arry_c3[p], e1_strm_arry_c0[p], e1_strm_arry_c1[p], e1_strm_arry_c2[p],
                e1_strm_arry_c3[p], k1_strm_arry[p], p1_strm_arry[p], hash_strm_arry[p], e1_strm_arry[p]);
        }
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::join_v2::build_probe_wrapper<WhashLow, WKey, WPayload, 0, WTmpBufferAddress>(
            hash_strm_arry[i], k1_strm_arry[i], p1_strm_arry[i], e1_strm_arry[i], w_row_strm[i], k2_strm_arry[i],
            p2_strm_arry[i], addr_strm[i], nm0_strm_arry[i], e2_strm_arry[i], e3_strm_arry[i], bit_vector0[i],
            bit_vector1[i], bit_vector2[i], bit_vector3[i]);
    }

    if (PU >= 4) {
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu0_tmp_rwtpr, addr_strm[0], w_row_strm[0],
                                                                           e2_strm_arry[0], r_row_strm[0]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu1_tmp_rwptr, addr_strm[1], w_row_strm[1],
                                                                           e2_strm_arry[1], r_row_strm[1]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu2_tmp_rwptr, addr_strm[2], w_row_strm[2],
                                                                           e2_strm_arry[2], r_row_strm[2]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu3_tmp_rwptr, addr_strm[3], w_row_strm[3],
                                                                           e2_strm_arry[3], r_row_strm[3]);
    }
    if (PU >= 8) {
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu4_tmp_rwptr, addr_strm[4], w_row_strm[4],
                                                                           e2_strm_arry[4], r_row_strm[4]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu5_tmp_rwptr, addr_strm[5], w_row_strm[5],
                                                                           e2_strm_arry[5], r_row_strm[5]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu6_tmp_rwptr, addr_strm[6], w_row_strm[6],
                                                                           e2_strm_arry[6], r_row_strm[6]);
        details::join_v2::access_srow<WTmpBuffer, WTmpBufferAddress, WKey>(pu7_tmp_rwptr, addr_strm[7], w_row_strm[7],
                                                                           e2_strm_arry[7], r_row_strm[7]);
    }

    for (int i = 0; i < PU; i++) {
#pragma HLS unroll
        details::hash_semi_join::semi_join_unit<WPayload, WKey>(r_row_strm[i], k2_strm_arry[i], p2_strm_arry[i],
                                                                nm0_strm_arry[i], e3_strm_arry[i], j0_strm_arry[i],
                                                                e4_strm_arry[i]);
    }

    // Collect
    details::join_v2::collect_unit<PU, WPayload>(j0_strm_arry, e4_strm_arry, join_ostrm, end_ostrm);
} // hash_semi_join

} // namespace database
} // namespace xf

#endif // !defined(XF_DATABASE_HASH_SEMI_JOIN_MPU_H)
