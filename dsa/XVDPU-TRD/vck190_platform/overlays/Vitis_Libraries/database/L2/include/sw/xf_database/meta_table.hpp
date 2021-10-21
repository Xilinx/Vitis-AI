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
#ifndef META_TABLE_HPP
#define META_TABLE_HPP

#include "ap_int.h"
// L2
#include "xf_database/gqe_utils.hpp"

namespace xf {
namespace database {
namespace gqe {

//! the MetaTable class the overall layout is as follows:
//---------------------------------------------------------------------------------------
// 511 ... 136 135       104  103        80  79       72  71  8  7  0                   |
// xxxxxxxxxxx partition_size buffer_id[1-3] buffer_id[0] length vnum      <--- meta[0] |
// xxxxxxxxxxx partition_size buffer_id[1-3] buffer_id[0] length vnum      <--- meta[1] |
// ...                                                                           ...    |
// xxxxxxxxxxx partition_size buffer_id[1-3] buffer_id[0] length vnum      <--- meta[7] |
//                                                                                      |
// 511     480 479     448                                 31       0                   |
// part15_nrow part14_nrow                ...              part0_nrow      <--- meta[8] |
// part31_nrow part30_nrow                ...             part16_nrow      <--- meta[9] |
//                                        ...                                           |
// part255_nrow                           ...            part240_nrow      <--- meta[23]|
//---------------------------------------------------------------------------------------
//
class MetaTable {
   private:
    // the total number of meta info:
    //- 8 cols, each 512 bit
    //- 16 cols, only used by gqePart kernel, recording the row number of each partition, supports maximum 256
    // partitions.
    static const int _metaDepth = 24;

    // valid input cols
    int8_t _vnum;

    // sec_id, used for gen_row_id
    int32_t _secID;

    // only used in gqePart, indicate the size of each parition in each col output
    int32_t _partition_size = 0;
    // only used in gqePart, the num of partition
    int32_t _partition_num;
    // only used in gqePart, the output row num of each partition result
    int32_t _part_nrow[256];

    // the meta info of each input column
    struct vector {
        int64_t _data_type_enum; // reserved
        int64_t _length;         // row number
        int64_t _null_count;     // reserved
        int8_t _buffer_ids[4];   // real buffer id, reserved for now
        int8_t _dictionary;      // reserved
        int8_t _layout_enum;     // reserved
    } _vecs[_metaDepth];

    // kernel used meta
    ap_uint<512>* _mbuf;

   public:
    MetaTable() {
        _secID = -1;
        _mbuf = gqe::utils::aligned_alloc<ap_uint<512> >(_metaDepth);
        memset(_mbuf, 0, sizeof(ap_uint<512>) * _metaDepth);
    };
    ~MetaTable() { free(_mbuf); };

    //! convert meta struct data to ap_uint, which can be transferred to kernel
    ap_uint<512>* meta() {
        for (int m = 0; m < _vnum; ++m) {
            _mbuf[m].range(7, 0) = _vnum;
            _mbuf[m].range(71, 8) = _vecs[0]._length;
            _mbuf[m].range(79, 72) = _vecs[m]._buffer_ids[0];
            _mbuf[m].range(87, 80) = _vecs[m]._buffer_ids[1];
            _mbuf[m].range(95, 88) = _vecs[m]._buffer_ids[2];
            _mbuf[m].range(103, 96) = _vecs[m]._buffer_ids[3];
            _mbuf[m].range(135, 104) = _partition_size;
            _mbuf[m].range(167, 136) = _secID;
        }
        for (int m = _vnum; m < 8; ++m) {
            _mbuf[m].range(7, 0) = _vnum;
            _mbuf[m].range(71, 8) = 0;
            _mbuf[m].range(79, 72) = -1;
        }

        return _mbuf;
    };
    //! set the sec id in meta
    void setSecID(int32_t _id) { _secID = _id; };

    //! set the col num of input table
    void setColNum(int8_t num) {
        assert(num < 17);
        _vnum = num;
    };

    //! set partition num and size, only vaild for gqePart
    void setPartition(int num, int size) {
        assert(num < 257);
        _partition_num = num;
        _partition_size = size;
    };

    //! set the buf id and row num of each col
    void setCol(int col_id, int buf_id, int64_t len) {
        _vecs[col_id]._buffer_ids[0] = buf_id;
        _vecs[col_id]._buffer_ids[1] = buf_id;
        _vecs[col_id]._buffer_ids[2] = buf_id;
        _vecs[col_id]._buffer_ids[3] = buf_id;
        _vecs[col_id]._length = len;
    }

    //! get the row num of results
    int64_t getColLen() {
        for (int m = 0; m < _vnum; ++m) {
            _vecs[m]._length = _mbuf[0].range(71, 8);
        }
        return _vecs[0]._length;
    }

    //! get the row num of each partition
    int32_t* getPartLen() {
        for (int i = 0; i < _partition_num; ++i) {
            int _row_id = i / 16;
            int _row_id_ex = i % 16;
            _part_nrow[i] = _mbuf[8 + _row_id].range(32 * _row_id_ex + 31, 32 * _row_id_ex);
        }
        return _part_nrow;
    }

    //! convert kernel mbuf back to meta struct, refresh all data
    // reserved for now
    void reload() {
        for (int m = 0; m < _vnum; ++m) {
            _vecs[m]._length = _mbuf[0].range(71, 8);
        }
    };
};
} /* gqe */
} /* database */
} /* xf */

#endif // META_BUFF_HPP
