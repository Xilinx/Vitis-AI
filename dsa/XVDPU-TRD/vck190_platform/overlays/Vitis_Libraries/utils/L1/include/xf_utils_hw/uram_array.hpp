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
 * @file uram_array.hpp
 * @brief This file provides implementation of uram array helper class.
 *
 * This file is part of XF Common Utils Library.
 */

#ifndef XF_UTILS_HW_URAM_ARRAY_H
#define XF_UTILS_HW_URAM_ARRAY_H

#include <ap_int.h>

namespace xf {
namespace common {
namespace utils_hw {

namespace details {
/// @brief the structure of compute available uram block numbers.
///
/// @tparam width width of one data.
/// @tparam depth size of one array.
/// @tparam B     condition of W <= 72.
template <int total, int one, bool B = (total <= 72)>
struct need_num {
   private:
    static const int elem_per_line = 72 / total;

   public:
    static const int value_x = ((one + (elem_per_line * 4096) - 1) / (elem_per_line * 4096));
    static const int value_y = ((total + 71) / 72);
};

template <int total, int one>
struct need_num<total, one, false> {
   public:
    static const int value_x = (one + 4095) / 4096;
    static const int value_y = (total + 71) / 72;
};
} // details

/**
 * @class UramArray uram_array.hpp "xf_utils_hw/uram_array.hpp"
 *
 * @brief Helper class to create URAM array that can be updated every cycle
 * with forwarding regs.
 *
 * Forwarding regs keeps a history of address and values, and when acess to URAM
 * hits the "cache", read from URAM is skipped. This addresses the
 * Read-After-Write dependency across iterations. The depth of the cache,
 * specified with the ``_NCache`` variable, dependes on the latency of actual
 * write. It is suggested to be set after initial sythesis attempts.
 *
 * When element can be held in one URAM row (72-bit), this helper would try to
 * pack multiple elements to one row. If it is beyond 72-bits, multiple URAMs
 * will be used to ensure elements can be fetched in one cycle.
 *
 * To make the cache really functional, HLS needs to be instructed to ignore
 * inter-iteration dependencies on ``blocks`` of ``UramArray`` objects.
 * Please refer to the test of this module for an example.
 *
 * \rst
 * .. ATTENTION::
 *     This module requires HLS 2019.1 or above.
 * \endrst
 *
 * @tparam _WData  the width of every element.
 * @tparam _NData  the number of elements in the array.
 * @tparam _NCache the number of cache.
 */
template <int _WData, int _NData, int _NCache>
class UramArray {
   public:
    UramArray() {
#pragma HLS inline
#pragma HLS resource variable = blocks core = RAM_2P_URAM
#pragma HLS array_partition variable = blocks complete dim = 1
#pragma HLS array_partition variable = blocks complete dim = 2
#pragma HLS array_partition variable = _index complete dim = 1
#pragma HLS array_partition variable = _state complete dim = 1
        initialize();
    }

   private:
    void initialize() {
#ifndef __SYNTHESIS__
        for (int i = 0; i < _num_uram_x; i++) {
            for (int j = 0; j < _num_uram_y; j++) {
                blocks[i][j] = (ap_uint<72>*)malloc(sizeof(ap_uint<72>) * 4096);
            }
        }
#endif
        for (int i = 0; i < _NCache; i++) {
#pragma HLS unroll
            _index[i] = -1;
            _state[i] = 0;
        }
    }

   public:
    ~UramArray() {
#ifndef __SYNTHESIS__
        for (int i = 0; i < _num_uram_x; i++) {
            for (int j = 0; j < _num_uram_y; j++) {
                free(blocks[i][j]);
            }
        }
#endif
    }

    /// @brief  initialization for uram.
    /// @param  d value for initialization.
    /// @return number of block which had been initialize.
    int memSet(const ap_uint<_WData>& d);

    /// @brief write to uram.
    /// @param index the index which you want to write.
    /// @param d     the value what you want to write.
    void write(int index, const ap_uint<_WData>& d);

    /// @brief  read from uram.
    /// @param  index the index which you want to read.
    /// @return value you had read.
    ap_uint<_WData> read(int index);

   private:
    /// number elements per line, used with _WData<=72. For example, when _WData =
    /// 16, _elem_per_line is 4.
    static const int _elem_per_line;

    /// piece of block using one time.
    static const int _num_parallel_block;

    /// index numbers of one block.
    static const int _elem_per_block;

    /// the numbers of need to access.
    static const int _num_uram_block;
    static const int _num_uram_x;
    static const int _num_uram_y;

    /// the cache for saving latest index.
    int _index[_NCache];

    /// the cache for saving latest value.
    ap_uint<_WData> _state[_NCache];

   public:
// XXX this has to be public, otherwise `depencency false` cannot be
// specified.
#ifndef __SYNTHESIS__
    ap_uint<72>* blocks[details::need_num<_WData, _NData>::value_x][details::need_num<_WData, _NData>::value_y];
#else
    ap_uint<72> blocks[details::need_num<_WData, _NData>::value_x][details::need_num<_WData, _NData>::value_y][4096];
#endif
};

// Const member variables

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_elem_per_line = (72 / _WData);

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_elem_per_block = (_elem_per_line * 4096);

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_num_parallel_block = ((_WData + 71) / 72);

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_num_uram_block = (details::need_num<_WData, _NData>::value_x *
                                                                 details::need_num<_WData, _NData>::value_y);

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_num_uram_x = (details::need_num<_WData, _NData>::value_x);

template <int _WData, int _NData, int _NCache>
const int UramArray<_WData, _NData, _NCache>::_num_uram_y = (details::need_num<_WData, _NData>::value_y);

// Methods
template <int _WData, int _NData, int _NCache>
int UramArray<_WData, _NData, _NCache>::memSet(const ap_uint<_WData>& d) {
    if (_num_uram_block == 0) return 0;

    ap_uint<72> t;
l_init_value:
    for (int i = 0; i < 4096; i++) {
#pragma HLS PIPELINE II = 1
        for (int nx = 0; nx < _num_uram_x; nx++) {
#pragma HLS unroll
            for (int ny = 0; ny < _num_uram_y; ny++) {
#pragma HLS unroll
                if (_WData <= 72) {
                    for (int j = 0; j < _elem_per_line; j++) {
#pragma HLS unroll
                        t(j * _WData + _WData - 1, j * _WData) = d(_WData - 1, 0);
                    }
                    blocks[nx][ny][i] = t;
                } else {
                    int begin = ny * 72;
                    if (ny == (_num_parallel_block - 1))
                        blocks[nx][ny][i] = d(_WData - 1, begin);
                    else
                        blocks[nx][ny][i] = d(begin + 71, begin);
                }
            }
        }
    }

// update d for cache
init_cache:
    for (int i = 0; i < _NCache; i++) {
#pragma HLS unroll
        _state[i] = d;
        _index[i] = 0;
    }

    return _num_uram_block;
}

template <int _WData, int _NData, int _NCache>
void UramArray<_WData, _NData, _NCache>::write(int index, const ap_uint<_WData>& d) {
#pragma HLS inline
    int div_block = 0, div_index = 0;
    int dec_block = 0, dec, begin;

Write_Inner:
    for (int i = 0; i < _num_parallel_block; i++) {
        if (_WData <= 72) {
            div_block = index / _elem_per_block;
            dec_block = index % _elem_per_block;
            div_index = dec_block / _elem_per_line;
            dec = dec_block % _elem_per_line;
            begin = dec * _WData;
            ap_uint<72> tmp = blocks[div_block][0][div_index];
            tmp.range(begin + _WData - 1, begin) = d;
            blocks[div_block][0][div_index] = tmp;
        } else {
            div_block = index / 4096;
            dec_block = index % 4096;
            begin = i * 72;
            if (i == (_num_parallel_block - 1))
                blocks[div_block][i][dec_block] = d.range(_WData - 1, begin);
            else
                blocks[div_block][i][dec_block] = d.range(begin + 71, begin);
        }
    }

Write_Cache:
    for (int i = _NCache - 1; i >= 1; i--) {
        _state[i] = _state[i - 1];
        _index[i] = _index[i - 1];
    }
    _state[0] = d;
    _index[0] = index;
}

template <int _WData, int _NData, int _NCache>
ap_uint<_WData> UramArray<_WData, _NData, _NCache>::read(int index) {
#pragma HLS inline
    ap_uint<_WData> value;
    int div_block = 0, div_index = 0;
    int dec_block = 0, dec, begin;

Read_Cache:
    for (int i = 0; i < _NCache; i++) {
        if (index == _index[i]) {
            value = _state[i];
            return value;
        }
    }

Read_Inner:
    for (int i = 0; i < _num_parallel_block; i++) {
        if (_WData <= 72) {
            div_block = index / _elem_per_block;
            dec_block = index % _elem_per_block;
            div_index = dec_block / _elem_per_line;
            dec = dec_block % _elem_per_line;
            begin = dec * _WData;
            ap_uint<72> tmp = blocks[div_block][0][div_index];
            value = tmp.range(begin + _WData - 1, begin);
        } else {
            div_block = index / 4096;
            dec_block = index % 4096;
            begin = i * 72;

            if (i == (_num_parallel_block - 1))
                value.range(_WData - 1, begin) = blocks[div_block][i][dec_block];
            else {
                value.range(begin + 71, begin) = blocks[div_block][i][dec_block];
            }
        }
    }
    return value;
}

} // utils_hw
} // common
} // xf

#endif
