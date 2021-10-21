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
 * @file write_to_dataframe.hpp
 * @brief write Object stream format data into FPGA DDR, saved as Dataframe format
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef XF_DATA_ANALYTICS_L1_DATAFRAME_WRITE_TO_DATAFRAME_HPP
#define XF_DATA_ANALYTICS_L1_DATAFRAME_WRITE_TO_DATAFRAME_HPP

#include "xf_data_analytics/common/obj_interface.hpp"
#include "xf_data_analytics/dataframe/df_utils.hpp"

//#define _DF_DEBUG 1

namespace xf {
namespace data_analytics {
namespace dataframe {

namespace internal {
/**
 *
 * @brief process input data, record data count, null count,
 * flag each data whether is null
 *
 *  @param i_field_id_strm input index for each value.
 *  @param i_valid_strm input valid info for each value
 *  @param i_e_strm end flag of i_strm.
 *  @param n_buff buffer to record null count .
 *  @param l_buff buffer to store length.
 *  @param bit_map buffer to store value.
 *
 **/
template <int W>
void processNull(hls::stream<ap_uint<4> >& i_field_id_strm,
                 hls::stream<ap_uint<4> >& i_valid_strm,
                 hls::stream<bool>& i_e_strm,
                 ap_uint<32>* n_buff,
                 ap_uint<32>* l_buff,
                 ap_uint<64> bit_map[4096][1 << W]) {
    ap_uint<32> null_count_buff[1 << W];
#pragma HLS array_partition variable = null_count_buff
    ap_uint<32> len_buff[1 << W];
#pragma HLS array_partition variable = len_buff

    ap_uint<64> pre_elem[8];
#pragma HLS array_partition variable = pre_elem
    ap_uint<12 + 1> pre_addr[8] = {0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF};
#pragma HLS array_partition variable = pre_addr
    ap_uint<W + 1> pre_idx[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
#pragma HLS array_partition variable = pre_idx

NULLINIT:
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
        null_count_buff[i] = 0;
        len_buff[i] = 0;
    }
    bool e = i_e_strm.read();
NULLW:
    while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE false inter variable = bit_map
#pragma HLS LATENCY min = 8
        e = i_e_strm.read();
        ap_uint<W> idx = i_field_id_strm.read();
        ap_uint<32> len = len_buff[idx];
        len_buff[idx] = len + 1;

        ap_uint<6> low_addr = len.range(5, 0);
        ap_uint<12> high_addr = len.range(17, 6) + 1;
        ap_uint<4> valid = i_valid_strm.read();

        bool is_null = (valid == 0) ? true : false;

        ap_uint<64> elem = 0;

        if (is_null) null_count_buff[idx]++;
        // shift register to implement II=1

        if (high_addr == pre_addr[0] && idx == pre_idx[0]) {
            elem = pre_elem[0];
        } else if (high_addr == pre_addr[1] && idx == pre_idx[1]) {
            elem = pre_elem[1];
        } else if (high_addr == pre_addr[2] && idx == pre_idx[2]) {
            elem = pre_elem[2];
        } else if (high_addr == pre_addr[3] && idx == pre_idx[3]) {
            elem = pre_elem[3];
        } else if (high_addr == pre_addr[4] && idx == pre_idx[4]) {
            elem = pre_elem[4];
        } else if (high_addr == pre_addr[5] && idx == pre_idx[5]) {
            elem = pre_elem[5];
        } else if (high_addr == pre_addr[6] && idx == pre_idx[6]) {
            elem = pre_elem[6];
        } else if (high_addr == pre_addr[7] && idx == pre_idx[7]) {
            elem = pre_elem[7];
        } else {
            elem = bit_map[high_addr][idx];
        }
        elem[low_addr] = !is_null;
        for (int i = 7; i > 0; i--) {
            pre_elem[i] = pre_elem[i - 1];
            pre_idx[i] = pre_idx[i - 1];
            pre_addr[i] = pre_addr[i - 1];
        }
        pre_elem[0] = elem;
        pre_idx[0] = idx;
        pre_addr[0] = high_addr;
        bit_map[high_addr][idx] = elem;
    }
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        n_buff[i] = null_count_buff[i];
        l_buff[i] = len_buff[i];

        ap_uint<32> len = len_buff[i];
        ap_uint<6> low_addr = len.range(5, 0);
        ap_uint<12> high_addr = len.range(17, 6);
        if (low_addr != 0) high_addr++;
        // bit_map[0][i] is used to save the current high_addr
        bit_map[0][i] = high_addr + 1;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        printf("field: %d, length: %d, null_count: %d\n", i, (int)len_buff[i], (int)null_count_buff[i]);
#endif
#endif
    }
}
/**
 *
 * @brief process bool data, save the data in bool_buff
 *
 *  @param i_field_id_strm input index for each value.
 *  @param i_dat_strm input data
 *  @param i_e_strm end flag of i_strm.
 *  @param bool_buff buffer to store value.
 *
 **/
template <int W>
void processBoolean(hls::stream<ap_uint<4> >& i_field_id_strm,
                    hls::stream<bool>& i_dat_strm,
                    hls::stream<bool>& i_e_strm,
                    ap_uint<64> bool_buff[4096][1 << W]) {
    ap_uint<32> len_buff[1 << W];
#pragma HLS array_partition variable = len_buff
    ap_uint<64> pre_elem[8];
#pragma HLS array_partition variable = pre_elem
    ap_uint<12 + 1> pre_addr[8] = {0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF, 0xFFFFF};
#pragma HLS array_partition variable = pre_addr
    ap_uint<W + 1> pre_idx[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
#pragma HLS array_partition variable = pre_idx

// initialize
BOOLINIT:
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
        len_buff[i] = 0;
    }
    bool e = i_e_strm.read();
BOOLW:
    while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE false inter variable = bool_buff
#pragma HLS LATENCY min = 8
        e = i_e_strm.read();
        ap_uint<W> idx = i_field_id_strm.read();
        ap_uint<32> len = len_buff[idx];
        len_buff[idx] = len + 1;

        ap_uint<6> low_addr = len.range(5, 0);
        ap_uint<12> high_addr = len.range(17, 6) + 1;
        bool val = i_dat_strm.read();
        ap_uint<64> elem = 0;
        // use shift reg to implement II=1
        if (high_addr == pre_addr[0] && idx == pre_idx[0]) {
            elem = pre_elem[0];
        } else if (high_addr == pre_addr[1] && idx == pre_idx[1]) {
            elem = pre_elem[1];
        } else if (high_addr == pre_addr[2] && idx == pre_idx[2]) {
            elem = pre_elem[2];
        } else if (high_addr == pre_addr[3] && idx == pre_idx[3]) {
            elem = pre_elem[3];
        } else if (high_addr == pre_addr[4] && idx == pre_idx[4]) {
            elem = pre_elem[4];
        } else if (high_addr == pre_addr[5] && idx == pre_idx[5]) {
            elem = pre_elem[5];
        } else if (high_addr == pre_addr[6] && idx == pre_idx[6]) {
            elem = pre_elem[6];
        } else if (high_addr == pre_addr[7] && idx == pre_idx[7]) {
            elem = pre_elem[7];
        } else {
            elem = bool_buff[high_addr][idx];
        }
        elem[low_addr] = val;
        for (int i = 7; i > 0; i--) {
            pre_elem[i] = pre_elem[i - 1];
            pre_idx[i] = pre_idx[i - 1];
            pre_addr[i] = pre_addr[i - 1];
        }
        pre_elem[0] = elem;
        pre_idx[0] = idx;
        pre_addr[0] = high_addr;
        bool_buff[high_addr][idx] = elem;
    }
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ap_uint<32> len = len_buff[i];
        ap_uint<6> low_addr = len.range(5, 0);
        ap_uint<12> high_addr = len.range(17, 6);
        if (low_addr != 0) high_addr++;
        bool_buff[0][i] = high_addr + 1;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        printf("boolean buffer field: %d, number: %d\n", i, (int)high_addr);
#endif
#endif
    }
}
/**
 *
 * @brief string offset read and issue to different field_id channels
 *
 *  @param i_str_field_id_strm input field with strign data type
 *  @param i_str_offset_strm input string offset
 *  @param i_e_strm end flag of i_str_*_strm.
 *  @param o_offset_strm output offset strm for different field
 *  @param o_field_id_strm the output field index
 *  @param o_e_strm end flag of o_offset_strm.
 *
 **/
template <int W>
void collectData(hls::stream<ap_uint<4> >& i_str_field_id_strm,
                 hls::stream<ap_uint<32> >& i_str_offset_strm,
                 hls::stream<bool>& i_str_e_strm,

                 hls::stream<ap_uint<32> > o_offset_strm[1 << W],
                 hls::stream<bool>& o_e_strm,
                 hls::stream<ap_uint<W> >& o_field_id_strm) {
    bool e = i_str_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        e = i_str_e_strm.read();
        ap_uint<W> field_id = i_str_field_id_strm.read();
        ap_uint<32> offset = i_str_offset_strm.read();
        o_offset_strm[field_id].write(offset);
        o_e_strm.write(false);
        o_field_id_strm.write(field_id);
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// printf("data colloector o_str size :%d\n", o_offset_strm[0].size());
#endif
#endif
    o_e_strm.write(true);
    o_field_id_strm.write(0);
}

/**
 *
 * @brief process 64-bits data types, double, int64, date, float32, string data,
 * issue to different field_id channels
 *
 *  @param i_str_field_id_strm input field with strign data type
 *  @param i_str_offset_strm input string offset
 *  @param i_e_strm end flag of i_str_*_strm.
 *  @param o_offset_strm output offset strm for different field
 *  @param o_field_id_strm the output field index
 *  @param o_e_strm end flag of o_offset_strm.
 *
 **/
template <int W>
void collectData(hls::stream<ap_uint<4> >& i_w64_field_id_strm,
                 hls::stream<ap_uint<64> >& i_w64_dat_strm,
                 hls::stream<ap_uint<4> >& i_w64_dt_strm,
                 hls::stream<bool>& i_w64_e_strm,

                 hls::stream<ap_uint<32> > o_strm[1 << W],
                 hls::stream<bool>& o_e_strm,
                 hls::stream<ap_uint<W> >& o_field_id_strm) {
    bool e = i_w64_e_strm.read();
    ap_uint<16> len = 0;
    ap_uint<W> idx;
    ap_uint<4> dt;
    bool last = true;
    ap_uint<64> in;
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<32> out;
        if (last) {
            e = i_w64_e_strm.read();
            dt = i_w64_dt_strm.read();
            idx = i_w64_field_id_strm.read();
            in = i_w64_dat_strm.read();
            out = in.range(31, 0);
            if (dt == TFloat32) {
                last = true;
            } else {
                last = false;
            }

        } else {
            last = true;
            out = in.range(63, 32);
        }
        o_strm[idx].write(out);
        o_e_strm.write(false);
        o_field_id_strm.write(idx);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "out data: " << out << ", e: 0, hash: " << idx << std::endl;
#endif
#endif
    }
    if (!last) {
        ap_uint<32> out = in.range(63, 32);
        o_strm[idx].write(out);
        o_e_strm.write(false);
        o_field_id_strm.write(idx);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "out data: " << out << ", e: 0, hash: " << idx << std::endl;
#endif
#endif
    }
    o_e_strm.write(true);
    o_field_id_strm.write(0);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "out data:, e: 1, hash: 0" << std::endl;
#endif
#endif
}
/**
 *
 * @brief read data from object stream,
 * analyze and send to different channels according to the data type
 *
 *  @param obj_strm the input stream data that packed as Object Struct
 *
 *  @param o_null_field_id_strm null strm index of each data
 *  @param o_null_valid_strm null flag of each data
 *  @param o_null_e_strm end flag of each null data
 *
 *  @param o_bool_field_id_strm bool strm index of each data
 *  @param o_bool_dat_strm bool value of each data
 *  @param o_bool_e_strm end flag of each boolean data
 *
 *  @param o_w64_field_id_strm output index of each w64 data
 *  @param o_w64_dat_strm output data value of types: Int64, Float32, Double, Date, String data
 *  @param o_w64_dt_strm output data type
 *  @param o_w64_e_strm end flag of output w64 data
 *
 *  @param o_strlen_strm output string length of string fields in each json line
 *  @param o_str_field_id_strm output index of each string field
 *  @param o_str_e_strm end flag of output string length
 *
 **/
template <int W>
void readObjStrm(hls::stream<Object>& obj_strm,
                 // null output
                 hls::stream<ap_uint<4> >& o_null_field_id_strm,
                 hls::stream<ap_uint<4> >& o_null_valid_strm,
                 hls::stream<bool>& o_null_e_strm,

                 // boolean output
                 hls::stream<ap_uint<4> >& o_bool_field_id_strm,
                 hls::stream<bool>& o_bool_dat_strm,
                 hls::stream<bool>& o_bool_e_strm,

                 // 64 bit type, int64, double, date, string
                 hls::stream<ap_uint<4> >& o_w64_field_id_strm,
                 hls::stream<ap_uint<64> >& o_w64_dat_strm,
                 // hls::stream<ap_uint<4> >& o_w64_valid_strm,
                 hls::stream<ap_uint<4> >& o_w64_dt_strm,
                 hls::stream<bool>& o_w64_e_strm,

                 hls::stream<ap_uint<32> >& o_strlen_strm,
                 hls::stream<ap_uint<4> >& o_str_field_id_strm,
                 hls::stream<bool>& o_str_e_strm,

                 ap_uint<32>* buff) {
    ap_uint<32> size_buff[1 << W];
#pragma HLS array_partition variable = size_buff
    ap_uint<32> offset_size_buff[1 << W];
#pragma HLS array_partition variable = offset_size_buff
    // internal used only
    ap_uint<32> offset_buff[1 << W];
#pragma HLS array_partition variable = offset_buff

    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
        size_buff[i] = 0;
        offset_size_buff[i] = 0;
        offset_buff[i] = 0;
    }

    Object obj_data;
    obj_data = obj_strm.read();
    ap_uint<4> type = obj_data.get_type();
    ap_uint<64> dat = obj_data.get_data();
    ap_uint<16> field_id_tmp = obj_data.get_id();
    ap_uint<4> field_id = field_id_tmp.range(3, 0);
    ap_uint<4> valid = obj_data.get_valid();

    // ap_uint<16> str_len = 0;
    int str_len = 0;

    while (type != FEOF) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::string tf_str =
            (type == FEOF) ? "EOF" : (type == FEOC ? "EOC" : (type == FEOL) ? "EOL" : type.to_string());
        std::cout << tf_str << ",";
        std::cout << valid << ",";
        std::cout << field_id << ",";
        std::cout << dat << "; ";
#endif
#endif

        if (type == FEOC) {
// end of col
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << std::endl << std::endl;
#endif
#endif
        } else if (type == FEOL) {
// end of json line
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << std::endl;
#endif
#endif
            // output the string strlen for each end of line
            if (str_len != 0) {
                // ap_uint<32> strlen_tmp = offset_buff[field_id];
                // ap_uint<32> offset_n = offset_tmp + str_len;
                // o_str_offset_strm.write(offset_n);
                o_strlen_strm.write(str_len);

                o_str_field_id_strm.write(field_id);
                o_str_e_strm.write(false);

                // offset_buff[field_id] = offset_n;
                offset_size_buff[field_id] += 4;
            }
            str_len = 0;

        } else {
            // each line data
            // for processNull
            //  if (valid == 0) {
            o_null_field_id_strm.write(field_id);
            o_null_valid_strm.write(valid);
            o_null_e_strm.write(false);
            //} else {
            if (type == TBoolean) {
                // data is boolean
                o_bool_field_id_strm.write(field_id);
                o_bool_dat_strm.write(dat[0]);
                o_bool_e_strm.write(false);
            } else if (type == TInt64 || type == TDouble || type == TDate) {
                // data is 64bits width
                o_w64_field_id_strm.write(field_id);
                o_w64_dat_strm.write(dat);
                o_w64_dt_strm.write(type);
                o_w64_e_strm.write(false);
                size_buff[field_id] += 8;
            } else if (type == TFloat32) {
                o_w64_field_id_strm.write(field_id);
                o_w64_dat_strm.write(dat);
                o_w64_dt_strm.write(type);
                o_w64_e_strm.write(false);
                size_buff[field_id] += 4;
            } else if (type == TString) {
                // parseSting xxx
                // input: field_id, dat, type, valid
                // output: field_id, dat, offset,type,e

                size_buff[field_id] += valid;
                str_len += valid;

                o_w64_field_id_strm.write(field_id);
                o_w64_dat_strm.write(dat);
                o_w64_dt_strm.write(type);
                // o_w64_valid_strm.write(valid);
                o_w64_e_strm.write(false);
            }
            //}
        }

        obj_data = obj_strm.read();

        type = obj_data.get_type();
        dat = obj_data.get_data();
        field_id = obj_data.get_id();
        valid = obj_data.get_valid();
    }

    // last data, end of file
    o_null_e_strm.write(true);
    o_bool_e_strm.write(true);
    o_w64_e_strm.write(true);
    o_str_e_strm.write(true);

    // write size_buff, offset_size_buff to buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        buff[i] = size_buff[i];
    }
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        buff[i + (1 << W)] = offset_size_buff[i];
    }

#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::string tf_str = (type == FEOF) ? "EOF" : (type == FEOC ? "EOC" : (type == FEOL) ? "EOL" : type.to_string());
    std::cout << tf_str << ",";
    std::cout << valid << ",";
    std::cout << field_id << ",";
    std::cout << dat << "; " << std::endl;
    std::cout << std::endl;
#endif
#endif
}
/**
 *
 * @brief genearte the addr of data write requests, every 32 x 32-bit requests output 1 addr of burst write req
 *
 **/
template <int W>
void memManage(hls::stream<ap_uint<W> > i_field_id_strm[2],
               hls::stream<bool> e_strm[2],

               hls::stream<ap_uint<15> >& o_low_addr_strm,  // DDR address
               hls::stream<ap_uint<12> >& o_high_addr_strm, // DDR address

               hls::stream<ap_uint<8> > o_rnm_strm[2], // number of row: ap_uint<256>
               hls::stream<ap_uint<W> > o_field_id_strm[2],
               hls::stream<bool> o_e_strm_a[2],
               hls::stream<ap_uint<2> >& o_ch_strm,
               hls::stream<bool>& o_e_strm,

               Node linkTable[1024],
               FieldInfo f_buff[2 * (1 << W)]) {
    // Node linkTable[1024];
    FieldInfo buff_info[1 << W][2];
#pragma HLS array_partition variable = buff_info dim = 1

    bool is_head[1 << W][2];
#pragma HLS array_partition variable = is_head dim = 1
    // current block size
    ap_uint<12> cur_addr = 1;

    ap_uint<12> link_tb_addr = 0;
    // initialize
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
            buff_info[i][ch].head = 0;
            buff_info[i][ch].tail = 0;
            buff_info[i][ch].size = 0;
            is_head[i][ch] = true;
        }
    }
    bool new_block_flag = false;
    ap_uint<2> empty_e;
    ap_uint<2> last = 0;
    // write the same data with end flag because of do while loop
    do {
#pragma HLS pipeline II = 2
        // round-robin read non-empty input.
        for (int i = 0; i < 2; ++i) {
            empty_e[i] = !e_strm[i].empty() && !last[i];
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "empty_e = " << empty_e << std::endl;
#endif
#endif
        // select one channel, for reading later.
        // If both channel has data, channel 0 has higher priority
        // empty_e	rd_e
        // 00		00
        // 01		01
        // 10		10
        // 11		01
        ap_uint<2> rd_e;
        for (int i = 0; i < 2; ++i) {
            ap_uint<2> t_e = 0;
            if (i > 0) t_e = empty_e(i - 1, 0);
            if (t_e > 0)
                rd_e[i] = 0;
            else
                rd_e[i] = empty_e[i];
        }

        // read the selected channel field_id and e flag
        ap_uint<W> idx_arr[2];
        for (int i = 0; i < 2; ++i) {
            if (rd_e[i]) {
                idx_arr[i] = i_field_id_strm[i].read();
                last[i] = e_strm[i].read();
            }
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "rd_e = " << rd_e << std::endl;
// std::cout << "idx_arr[0] = " << idx_arr[0] << std::endl;
// std::cout << "idx_arr[1] = " << idx_arr[1] << std::endl;
// std::cout << "last = " << last << std::endl;
#endif
#endif

        ap_uint<2> ch = rd_e[1];
        ap_uint<W> idx = idx_arr[ch];
        bool valid_n = last[ch];

#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "ch = " << ch << std::endl;
// std::cout << "idx = " << idx << std::endl;
// std::cout << "valid_n= " << valid_n << std::endl;
#endif
#endif

        if (!valid_n && rd_e != 0) {
            // record the size, when it reaches the 4MB. allocate a new node.
            ap_uint<20> size = buff_info[idx][ch].size; // one 256;
            ap_uint<20> size_new;                       // = size + 1;
            // when reach 4MB, need to create a new block
            if (size == 0) {
                new_block_flag = true;
                size_new = 1;
                // size_new -= size;
            } else {
                new_block_flag = false;
                size_new = size + 1;
            }
            buff_info[idx][ch].size = size_new;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "buff_info[" << idx << "][" << ch << "].size = " << buff_info[idx][ch].size << std::endl;
#endif
#endif

            ap_uint<12> tail = buff_info[idx][ch].tail;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "buff_info[" << idx << "][" << ch << "].tail= " << buff_info[idx][ch].tail << std::endl;
#endif
#endif
            // burst len = 256 * 32;
            ap_uint<5> low_bit = size_new.range(4, 0);
            ap_uint<15> low_addr = size.range(19, 5);
            ap_uint<12> high_addr = 0; // linkTable[tail].base_addr;
            if (new_block_flag) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                printf("Create a new block\n");
#endif
#endif
                high_addr = cur_addr;
            } else {
                high_addr = linkTable[tail].base_addr;
            }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "high_addr: " << high_addr << ", low_addr: " << low_addr << ", bit: " << low_bit << std::endl;
#endif
#endif
            // when it reaches 32, burst write-out them.
            if (low_bit == 0) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                std::cout << "generate burst write request" << std::endl;
#endif
#endif
                o_rnm_strm[ch].write(32);
                o_low_addr_strm.write(low_addr);
                o_high_addr_strm.write(high_addr);
                o_ch_strm.write(ch);

                o_field_id_strm[ch].write(idx);
                o_e_strm_a[ch].write(false);
                o_e_strm.write(false);
            }
            // allocate a new block
            if (new_block_flag) {
                if (is_head[idx][ch]) {
                    buff_info[idx][ch].head = link_tb_addr;
                    is_head[idx][ch] = false;
                    tail = link_tb_addr;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                    std::cout << "buff_info[" << idx << "][" << ch << "].head= " << buff_info[idx][ch].head
                              << std::endl;
#endif
#endif
                }
                // when it reaches 32, burst write-out them.
                ap_uint<12> new_node_base = cur_addr++;
                new_block_flag = false;
                // update the next of node
                linkTable[tail].next = link_tb_addr;
                Node new_node;
                // new_node.before = tail;
                new_node.base_addr = new_node_base;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                std::cout << "new_node.base_addr = " << new_node.base_addr << std::endl;
#endif
#endif
                // new_node.next = 0;
                // add the new node and before pointer to the last node.
                linkTable[link_tb_addr] = new_node;
                // the content of buff_info is the address/index of linkTable
                buff_info[idx][ch].tail = link_tb_addr;
                link_tb_addr++;
            }
        }
    } while (last != 3);
    // output the last non-full block
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
            FieldInfo last_field = buff_info[i][ch];
            ap_uint<20> size = last_field.size;
            ap_uint<5> low_bit = size.range(4, 0);
            ap_uint<15> low_addr = size.range(19, 5);
            ap_uint<12> tail = last_field.tail;
            ap_uint<12> high_addr = linkTable[tail].base_addr;
            if (low_bit != 0) {
                o_rnm_strm[ch].write(low_bit);
                o_low_addr_strm.write(low_addr);
                o_high_addr_strm.write(high_addr);
                o_ch_strm.write(ch);
                o_field_id_strm[ch].write(i);
                o_e_strm_a[ch].write(false);
                o_e_strm.write(false);
            }
        }
    }
    o_e_strm_a[0].write(true);
    o_e_strm_a[1].write(true);
    o_e_strm.write(true);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
            f_buff[i + ch * (1 << W)] = buff_info[i][ch];
        }
    }
}

// select field_id, convert 32 x 32bit data to 16 x 64 bit, prepare for the burst write
template <int W>
void combine(hls::stream<ap_uint<32> > i_strm[1 << W],
             hls::stream<ap_uint<8> >& i_rnm_strm,      // number of row: ap_uint<256>
             hls::stream<ap_uint<W> >& i_field_id_strm, // number of row: ap_uint<256>
             hls::stream<bool>& i_e_strm,
             hls::stream<ap_uint<64> >& o_strm) {
    bool e = i_e_strm.read();
    while (!e) {
        ap_uint<8> rnm = i_rnm_strm.read();
        ap_uint<W> field_id = i_field_id_strm.read();
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "rnm: " << rnm << ", field_id: " << field_id << std::endl;
#endif
#endif
        for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline II = 2
            ap_uint<64> tmp = 0;
            ap_uint<32> in = 0;
            if (2 * i < rnm) {
                in = i_strm[field_id].read();
                tmp.range(31, 0) = in.range(31, 0);
            }
            if (2 * i + 1 < rnm) {
                in = i_strm[field_id].read();
                tmp.range(63, 32) = in.range(31, 0);
            }
            o_strm.write(tmp);
        }
        e = i_e_strm.read();
    }
}

// burst write to DDR, each burst writes 16 x 64-bit data
void writeData(hls::stream<ap_uint<64> > i_strm[2],
               hls::stream<ap_uint<15> >& i_low_addr_strm,  // DDR address
               hls::stream<ap_uint<12> >& i_high_addr_strm, // DDR address
               hls::stream<ap_uint<2> >& i_ch_strm,
               hls::stream<bool>& i_e_strm,
               ap_uint<64>* ddr_buff) {
    bool e = i_e_strm.read();
    while (!e) {
        ap_uint<2> ch = i_ch_strm.read();
        ap_uint<15> low_addr = i_low_addr_strm.read();
        ap_uint<12> high_addr = i_high_addr_strm.read();
        ap_uint<32> base_addr = 0;
        base_addr.range(18, 4) = low_addr;
        base_addr.range(30, 19) = high_addr;
        for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline II = 1
            ap_uint<64> tmp = i_strm[ch].read();
            ddr_buff[base_addr + i] = tmp;
            // std::cout << "write_addr: " << std::hex << base_addr + i << std::endl;
        }
        e = i_e_strm.read();
    }
}

template <int W>
void writeToMem(hls::stream<Object>& obj_strm,
                ap_uint<64>* ddr_buff,
                ap_uint<32>* s_buff,
                ap_uint<32>* n_buff,
                ap_uint<32>* l_buff,
                ap_uint<64> bit_map[4096][1 << W],
                ap_uint<64> bool_buff[4096][1 << W],
                Node linkTable[1024],
                FieldInfo f_buff[2 * (1 << W)]) {
    hls::stream<ap_uint<W> > null_field_id_strm("null filed id strm");
#pragma HLS stream variable = null_field_id_strm depth = 64
    hls::stream<ap_uint<W> > null_valid_strm("null valid strm");
#pragma HLS stream variable = null_valid_strm depth = 64
    hls::stream<bool> null_e_strm("null e strm");
#pragma HLS stream variable = null_e_strm depth = 64

    hls::stream<ap_uint<W> > bool_field_id_strm("bool field id strm");
#pragma HLS stream variable = bool_field_id_strm depth = 64
    hls::stream<bool> bool_dat_strm("bool dat strm");
#pragma HLS stream variable = bool_dat_strm depth = 64
    hls::stream<bool> bool_e_strm("bool e strm");
#pragma HLS stream variable = bool_e_strm depth = 64

    hls::stream<ap_uint<W> > w64_field_id_strm("w64 field id strm");
#pragma HLS stream variable = w64_field_id_strm depth = 64
    hls::stream<ap_uint<64> > w64_dat_strm("w64 dat strm");
#pragma HLS stream variable = w64_dat_strm depth = 64
    // hls::stream<ap_uint<4> > w64_valid_strm("w64 valid strm");
    //#pragma HLS stream variable = w64_valid_strm depth = 8
    hls::stream<ap_uint<4> > w64_dt_strm("w64 dt strm");
#pragma HLS stream variable = w64_dt_strm depth = 64
    hls::stream<bool> w64_e_strm("w64 e strm");
#pragma HLS stream variable = w64_e_strm depth = 64

    hls::stream<ap_uint<W> > str_field_id_strm("str field id strm");
#pragma HLS stream variable = str_field_id_strm depth = 64
    hls::stream<ap_uint<32> > str_offset_strm("str offset strm");
#pragma HLS stream variable = str_offset_strm depth = 64
    hls::stream<bool> str_e_strm("str e strm");
#pragma HLS stream variable = str_e_strm depth = 64

    hls::stream<ap_uint<32> > w32_dat_strm[2][1 << W];
#pragma HLS stream variable = w32_dat_strm depth = 64
    hls::stream<bool> w32_e_strm[2];
#pragma HLS stream variable = w32_e_strm depth = 64
    hls::stream<ap_uint<W> > w32_field_id_strm[2];
#pragma HLS stream variable = w32_field_id_strm depth = 64

    hls::stream<ap_uint<15> > l_addr_strm;
#pragma HLS stream variable = l_addr_strm depth = 64
    hls::stream<ap_uint<12> > h_addr_strm;
#pragma HLS stream variable = h_addr_strm depth = 64
    hls::stream<ap_uint<8> > rnm_strm[2];
#pragma HLS stream variable = rnm_strm depth = 64
    hls::stream<ap_uint<W> > c_field_id_strm[2];
#pragma HLS stream variable = c_field_id_strm depth = 64
    hls::stream<bool> c_e_strm[2];
#pragma HLS stream variable = c_e_strm depth = 64
    hls::stream<ap_uint<2> > ch_strm;
#pragma HLS stream variable = ch_strm depth = 64
    hls::stream<bool> wr_e_strm;
#pragma HLS stream variable = wr_e_strm depth = 64
    hls::stream<ap_uint<64> > m_strm[2];
#pragma HLS stream variable = m_strm depth = 32
#pragma HLS BIND_STORAGE variable = m_strm type = fifo impl = lutram

#pragma HLS dataflow

    // read in object stream format data, issue to different channels according to the data type
    readObjStrm<W>(obj_strm, null_field_id_strm, null_valid_strm, null_e_strm, bool_field_id_strm, bool_dat_strm,
                   bool_e_strm, w64_field_id_strm, w64_dat_strm,
                   /*w64_valid_strm,*/ w64_dt_strm, w64_e_strm, str_offset_strm, str_field_id_strm, str_e_strm, s_buff);

    // update the bit_map for each data, indicate the data is null or not
    processNull<W>(null_field_id_strm, null_valid_strm, null_e_strm, n_buff, l_buff, bit_map);

    // update the bool_buff for Boolean data
    processBoolean<W>(bool_field_id_strm, bool_dat_strm, bool_e_strm, bool_buff);

    // 1x64-bit ==> 16x32-bit: change 64-bit data(Int64/Double/String data/Date) into 32-bit, issue to different field
    // id channel
    collectData<W>(w64_field_id_strm, w64_dat_strm, w64_dt_strm, w64_e_strm, w32_dat_strm[0], w32_e_strm[0],
                   w32_field_id_strm[0]);
    // 1x32-bit ==> 16x32-bit: 32-bit offset data read, issue to different field id channel
    collectData<W>(str_field_id_strm, str_offset_strm, str_e_strm, w32_dat_strm[1], w32_e_strm[1],
                   w32_field_id_strm[1]);

    // generate the addr according to writing request
    memManage<W>(w32_field_id_strm, w32_e_strm, l_addr_strm, h_addr_strm, rnm_strm, c_field_id_strm, c_e_strm, ch_strm,
                 wr_e_strm, linkTable, f_buff);

    combine(w32_dat_strm[0], rnm_strm[0], c_field_id_strm[0], c_e_strm[0], m_strm[0]);
    combine(w32_dat_strm[1], rnm_strm[1], c_field_id_strm[1], c_e_strm[1], m_strm[1]);

    // burst write to DDR
    writeData(m_strm, l_addr_strm, h_addr_strm, ch_strm, wr_e_strm, ddr_buff);
}

/**
 *
 * @brief save the on-chip mem data to DDR
 *
 * @param s_buff size buffer
 * @param n_buff null count buffer
 * @param l_buff len count buffer
 * @param bit_map detailed null data for each input value
 * @param bool_buff detailed bool data for each input bool value
 * @param lineTable detailed link info of each mem block node
 * @param f_buff the link info between nodes
 * @param ddr_out DDR buffer to save local mem data
 *
 **/
template <int W>
void localRamToMem(ap_uint<32>* s_buff,
                   ap_uint<32>* n_buff,
                   ap_uint<32>* l_buff,
                   ap_uint<64> bit_map[4096][1 << W],
                   ap_uint<64> bool_buff[4096][1 << W],
                   Node linkTable[1024],
                   FieldInfo f_buff[2 * (1 << W)],
                   ap_uint<64>* ddr_out) {
    int addr = 1;
    // write size buff
    for (int i = 0; i < 2 * (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = s_buff[i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "s_buff[" << i << "] = " << s_buff[i] << std::endl;
#endif
#endif
    }
    addr += 2 * (1 << W);
    // write length buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = l_buff[i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "l_buff[" << i << "] = " << l_buff[i] << std::endl;
#endif
#endif
    }
    addr += (1 << W);
    // write null count buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = n_buff[i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "n_buff[" << i << "] = " << n_buff[i] << std::endl;
#endif
#endif
    }
    addr += (1 << W);
    // bitmap_address
    ap_uint<32> tmp = 129;
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = tmp;
        tmp += bit_map[0][i];
    }
    addr += (1 << W);
    // bool address
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = tmp;
        tmp += bool_buff[0][i];
    }
    addr += (1 << W);
    // value address and offset address
    for (int i = 0; i < 2 * (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = tmp;
        ap_uint<32> sz = s_buff[i] + 7;
        ap_uint<32> sz_32 = sz.range(31, 3);
        tmp += sz_32;
    }
    addr += 2 * (1 << W);
    // write bitmap buffer
    for (int i = 0; i < (1 << W); ++i) {
        // get the number of buffer
        int nm = bit_map[0][i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "bit_map " << i << ", nm: " << nm << "data: " << std::endl;
#endif
#endif
        for (int j = 0; j < nm; ++j) {
#pragma HLS pipeline II = 1
            ddr_out[addr + j] = bit_map[j][i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << bit_map[j][i] << ",";
#endif
#endif
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << std::endl;
#endif
#endif
        addr += nm;
    }
    // write boolean buffer
    for (int i = 0; i < (1 << W); ++i) {
        // get the number of buffer
        int nm = bool_buff[0][i];
        for (int j = 0; j < nm; ++j) {
#pragma HLS pipeline II = 1
            ddr_out[addr + j] = bool_buff[j][i];
        }
        addr += nm;
    }

#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "-----------field info buffer data-----------" << std::endl;
#endif
#endif
    // write field info buffer
    for (int i = 0; i < 2 * (1 << W); ++i) {
#pragma HLS pipeline II = 1
        FieldInfo fld_info = f_buff[i];
        ap_uint<64> out = 0;
        out.range(11, 0) = fld_info.head;
        out.range(23, 12) = fld_info.tail;
        out.range(43, 24) = fld_info.size;
        ddr_out[addr + i] = out;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "field " << i << ", head: " << fld_info.head;
        std::cout << ", tail: " << fld_info.tail;
        std::cout << ", size: " << fld_info.size;
        std::cout << std::endl;
#endif
#endif
    }
    addr += 2 * (1 << W);
    // write link table
    for (int i = 0; i < 1024; ++i) {
#pragma HLS pipeline II = 1
        Node nd = linkTable[i];
        ap_uint<64> out = 0;
        out.range(11, 0) = nd.next;
        out.range(23, 12) = nd.base_addr;
        ddr_out[addr + i] = out;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        if (i < 20) {
            std::cout << "linkTable[" << i << "].base_addr= " << linkTable[i].base_addr;
            std::cout << ", linkTable[" << i << "].next = " << linkTable[i].next << std::endl;
        }
#endif
#endif
    }
    addr += 1024;
    ddr_out[0] = addr;
}

} // end of internal namespace

/**
 *
 * @brief write object stream data to DDR
 *
 * @param obj_strm input stream data with type Object Struct
 * @param ddr_buff DDR buffer to save parserd data in DataFrame format
 *
 * all field_id in each json line is unique
 * maximum supported field_id number is 16
 *
 **/
void writeToDataFrame(hls::stream<Object>& obj_strm, ap_uint<64>* ddr_buff) {
    // define the filed_id number by 1 << W
    const int W = 4;

    // the total data size and offset size info of each field_id
    // w64 data size of each field_id:		s_buff[0]  s_buff[1]  ... s_buff[15]
    // string offset size of each field_id: s_buff[16] s_buff[17] ... s_buff[31]
    ap_uint<32> s_buff[32];
#pragma HLS BIND_STORAGE variable = s_buff type = ram_1p impl = lutram

    // processNull results buffers
    // null count of each filed
    ap_uint<32> n_buff[16];
#pragma HLS BIND_STORAGE variable = n_buff type = ram_1p impl = lutram
    // the length of each field, input data count, all data types
    ap_uint<32> l_buff[16];
#pragma HLS BIND_STORAGE variable = l_buff type = ram_1p impl = lutram
    // indicate each data is null or not
    ap_uint<64> bit_map[4096][16];
#pragma HLS BIND_STORAGE variable = bit_map type = ram_2p impl = uram
    // processBoolean result buffer
    ap_uint<64> bool_buff[4096][16];
#pragma HLS BIND_STORAGE variable = bool_buff type = ram_2p impl = uram
    // mem management buffers
    // the link info between mem block node
    Node linkTable[1024];
    // the nodes info for each field
    FieldInfo f_buff[32];
#pragma HLS BIND_STORAGE variable = f_buff type = ram_1p impl = lutram

    // Init
    for (int i = 0; i < 4096; ++i) {
        for (int j = 0; j < (1 << W); ++j) {
#pragma HLS pipeline II = 1
            bit_map[i][j] = 0;
            bool_buff[i][j] = 0;
        }
    }
    for (int i = 0; i < 1024; i++) {
        linkTable[i].next = 0;
        linkTable[i].base_addr = 0;
    }

    // read 64-bit data of different type to DDR
    internal::writeToMem<W>(obj_strm, ddr_buff, s_buff, n_buff, l_buff, bit_map, bool_buff, linkTable, f_buff);
    // save on-chip mem data(Null info, boolean info, 4MB block link info) to DDR header section
    internal::localRamToMem<W>(s_buff, n_buff, l_buff, bit_map, bool_buff, linkTable, f_buff, ddr_buff);
}

} // end of dataframe namespace
} // end of data_analytics namespace
} // end of xf namespace

#endif
