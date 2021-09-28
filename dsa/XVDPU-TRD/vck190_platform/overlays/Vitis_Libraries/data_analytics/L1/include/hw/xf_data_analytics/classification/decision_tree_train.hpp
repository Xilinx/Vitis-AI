
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
 * @file axi_to_multi_stream.hpp
 * @brief This file is a template implement of loading data from AXI master to multi stream.
 * Xilinx.
 *
 * This file is part of Vitis Utility Library.
 */

#ifndef _XF_DATA_ANALYTICS_L1_CLASSIFICATION_DECISIONTREE_TRAIN_HPP_
#define _XF_DATA_ANALYTICS_L1_CLASSIFICATION_DECISIONTREE_TRAIN_HPP_

#include "xf_utils_hw/common.hpp"
#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/enums.hpp"

namespace xf {
namespace data_analytics {
namespace classification {

//--------------------------- APIs -------------------------------

/**
 * @brief Loading table from AXI master to stream.
 * Table should be row based storage of identical datawidth.
 *
 * @tparam _BurstLen burst length of AXI buffer, default is 32.
 * @tparam _WAxi width of AXI port, must be multiple of datawidth, default is 512.
 * @tparam _WData datawith, default is 64.
 *
 * @param ddr input AXI port
 * @param offset offset(in _WAxi bits) to load table.
 * @param rows Row number of table
 * @param cols Column number of table
 * @dataStrm Output streams of _WAxi/_WData channels
 * @eDataStrm end flag of output stream.
 */

template <int _BurstLen = 32, int _WAxi = 512, int _WData = 64>
void axiVarColToStreams(ap_uint<_WAxi>* ddr,
                        const ap_uint<32> offset,
                        const ap_uint<32> rows,
                        ap_uint<32> cols,
                        hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                        hls::stream<bool>& eDataStrm);

namespace internal {

template <int _WAxi, int _BurstLen, int _WData>
void read_raw(ap_uint<_WAxi>* ddr,
              const ap_uint<32> offset,
              const ap_uint<32> rows,
              const ap_uint<32> cols,
              hls::stream<ap_uint<_WAxi> >& vec_strm) {
    const ap_uint<64> nread = (rows * cols + (_WAxi / _WData - 1)) / (_WAxi / _WData);
READ_RAW:
    for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
#pragma HLS pipeline II = 1
        vec_strm.write(ddr[offset + i]);
        //   int len = ((i + _BurstLen) > nread) ? (int(nread - i)) : _BurstLen;
        /* for (int j = 0; j < len; j++) {
 #pragma HLS loop_tripcount min=64 max=64 avg=64
 #pragma HLS pipeline II = 1
             vec_strm.write(ddr[offset + i + j]);
         }*/
    }
}

template <int _WAxi, int _WData>
void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
    if (s >= 0 && s <= _WAxi / _WData) {
        source >>= s * _WData;
    }
}

template <int _WAxi, int _WData>
void var_split(hls::stream<ap_uint<_WAxi> >& vec_strm,
               const ap_uint<32> rows,
               const ap_uint<32> cols,
               hls::stream<ap_uint<_WData> > data[_WAxi / _WData],
               hls::stream<bool>& eData) {
    const int full_batch = (_WAxi / _WData);
    const int tmp_tail_batch = cols % full_batch;
    const int tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
    const int batch_num = (cols + full_batch - 1) / full_batch;

    int reserve = 0;
    ap_uint<_WAxi> inventory = 0;

LOOP1:
    for (int i = 0; i < rows; i++) {
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
    LOOP2:
        for (int j = 0; j < batch_num; j++) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
#pragma HLS pipeline II = 1
            int output;
            if (j == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }

            ap_uint<_WAxi> new_come;
            int tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vec_strm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right<_WAxi, _WData>(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right<_WAxi, _WData>(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                ap_uint<_WData> tmp;
                if (k < output) {
                    tmp = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp = 0;
                }
                data[k].write(tmp);
            }
            if (j == 0) {
                eData.write(false);
            }
        }
    }
    eData.write(true);
}

} // internal

template <int _BurstLen, int _WAxi, int _WData>
void axiVarColToStreams(ap_uint<_WAxi>* ddr,
                        const ap_uint<32> offset,
                        const ap_uint<32> rows,
                        const ap_uint<32> cols,
                        hls::stream<ap_uint<_WData> > data[_WAxi / _WData],
                        hls::stream<bool>& eData) {
    static const int fifo_depth = _BurstLen * 2;
    hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS bind_storage variable = vec_strm type = fifo impl = lutram
#pragma HLS stream variable = vec_strm depth = fifo_depth
#pragma HLS dataflow
    internal::read_raw<_WAxi, _BurstLen, _WData>(ddr, offset, rows, cols, vec_strm);
    internal::var_split(vec_strm, rows, cols, data, eData);
}

} // classification
} // data_analytics
} // xf

#endif //_XF_DATA_ANALYTICS_DECISIONTREE_L1_TRAIN_HPP_
