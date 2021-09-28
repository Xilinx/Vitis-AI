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
//#define _DF_DEBUG_V3 1
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
#include <iostream>
#endif
#endif
#include "xf_data_analytics/common/obj_interface.hpp"
#include "xf_data_analytics/dataframe/df_utils.hpp"

namespace xf {
namespace data_analytics {
namespace dataframe {

namespace internal {

const int L_BIT = 6;
const int BURST_LEN = (1 << L_BIT);
const int H_BIT = 20 - L_BIT;
const int H_BIT_PAGE = 19;
const int PAGE_SIZE_BYTE = 8 << 20;
const int PAGE_SIZE_BIT(8 << 23);
const int PAGE_VNUM_BIT64 = (1 << 20);
const int PAGE_VNUM_BIT32 = (2 << 20);

template <int W>
void readDDR(ap_uint<88>* ddr_obj, hls::stream<Object>& obj_strm) {
    int obj_num = ddr_obj[0].range(31, 0);
    Object obj_dat;
    for (int i = 1; i < obj_num + 1; i++) {
        ap_uint<88> all_data = ddr_obj[i];
        obj_dat.set_data(all_data.range(63, 0));
        obj_dat.set_id(all_data.range(79, 64));
        obj_dat.set_valid(all_data.range(83, 80));
        obj_dat.set_type(all_data.range(87, 84));
        obj_strm.write(obj_dat);
    }
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
    std::cout << "final stream elem: " << obj_num << std::endl;
#endif
#endif
}
template <int W>
void readObjStrm(hls::stream<Object>& obj_strm,

                 hls::stream<ap_uint<4> >& o_field_id_strm,
                 hls::stream<ap_uint<64> >& o_w64_dat_strm,
                 hls::stream<ap_uint<4> >& o_dt_strm,
                 hls::stream<ap_uint<8> >& o_w64_vld_strm,
                 hls::stream<ap_uint<32> >& o_strlen_strm,
                 hls::stream<bool>& o_e_strm,

                 ap_uint<32>* buff) {
    ap_uint<32> size_buff[1 << W];
#pragma HLS array_partition variable = size_buff

    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
        size_buff[i] = 0;
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
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "start in readObj" << std::endl;
    int counter = 0;
#endif
#endif

    while (type != FEOF) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::string tf_str =
            (type == FEOF) ? "EOF" : (type == FEOC ? "EOC" : (type == FEOL) ? "EOL" : type.to_string());
        std::cout << tf_str << ",";
        std::cout << field_id << ",";
        std::cout << dat << "..; ";
        counter++;
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
                o_field_id_strm.write(field_id);
                o_dt_strm.write(TCount);
                // o_w64_vld_strm.write(32);
                o_strlen_strm.write(str_len);
                o_e_strm.write(false);
            }
            str_len = 0;

        } else {
            if (type == TBoolean) {
                size_buff[field_id] += 1;
                o_field_id_strm.write(field_id);
                o_dt_strm.write(type);
                o_w64_vld_strm.write(1);
                o_w64_dat_strm.write(dat);
                o_e_strm.write(false);
            } else if (type == TInt64 || type == TDouble || type == TDate) {
                size_buff[field_id] += 8;
                o_field_id_strm.write(field_id);
                o_dt_strm.write(type);
                o_w64_vld_strm.write(64);
                o_w64_dat_strm.write(dat);
                o_e_strm.write(false);
            } else if (type == TFloat32) {
                size_buff[field_id] += 4;
                o_field_id_strm.write(field_id);
                o_dt_strm.write(type);
                o_w64_vld_strm.write(32);
                o_w64_dat_strm.write(dat);
                o_e_strm.write(false);
            } else if (type == TString) {
                size_buff[field_id] += valid;
                str_len += valid;
                o_field_id_strm.write(field_id);
                o_dt_strm.write(type);
                o_w64_vld_strm.write(valid * 8);
                o_w64_dat_strm.write(dat);
                o_e_strm.write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                std::cout << "debug in readObj, Tstrig, vld: " << valid << " Bytes, " << (valid * 8) << " bits"
                          << std::endl;
#endif
#endif
                // o_field_id_strm.write(field_id);
                // o_e_strm.write(false);
            }
        }

        obj_data = obj_strm.read();
        type = obj_data.get_type();
        dat = obj_data.get_data();
        field_id = obj_data.get_id();
        valid = obj_data.get_valid();
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "debug in readObj stream, counter:" << counter << std::endl;
#endif
#endif

    // last data, end of file
    o_e_strm.write(true);

    // write size_buff, offset_size_buff to buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        buff[i] = size_buff[i];
    }
}
template <int W>
void dispatch(hls::stream<ap_uint<W> >& i_w64_field_id_strm,
              hls::stream<ap_uint<64> >& i_w64_dat_strm,
              hls::stream<ap_uint<4> >& i_w64_dt_strm,
              hls::stream<ap_uint<8> >& i_w64_vld_strm,
              hls::stream<ap_uint<32> >& i_str_len_strm,
              hls::stream<bool>& i_w64_e_strm,
              ap_uint<8> schema[1 << 16],
              // for collectDataForCombine
              hls::stream<ap_uint<64> > o_w64_dat_strm[1 << W],
              hls::stream<ap_uint<8> > o_w64_vld_strm[1 << W],
              hls::stream<ap_uint<4> > o_w64_dt_strm[1 << W],
              hls::stream<bool> o_str_len_strm[1 << W],
              hls::stream<bool> o_w64_e_strm[1 << W],
              // for collectDataForMem
              hls::stream<ap_uint<4> >& o1_w64_field_id_strm,
              hls::stream<ap_uint<8> >& o1_w64_vld_strm,
              hls::stream<ap_uint<32> >& o1_w64_len_strm,
              hls::stream<bool>& o1_str_len_strm,
              hls::stream<bool>& o1_w64_e_strm) {
    bool e = i_w64_e_strm.read();
    ap_uint<W> idx;
    ap_uint<4> dt;
    ap_uint<8> vld;
    ap_uint<64> in;
    ap_uint<64> len;

    bool first = true;
    for (int i = 0; i < (1 << W); i++) {
        o_w64_dt_strm[i].write(schema[i]);
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    int counter[16] = {0, 0, 0, 0, 0};
#endif
#endif
    while (!e) {
#pragma HLS pipeline II = 1
        idx = i_w64_field_id_strm.read();
        dt = i_w64_dt_strm.read();
        e = i_w64_e_strm.read();
        if (dt == TCount) {
            vld = i_w64_vld_strm.read();
            in = i_w64_dat_strm.read();
            // len = i_str_len_strm.read();
            o_w64_dat_strm[idx].write(in);
            o_w64_vld_strm[idx].write(vld);
            o_str_len_strm[idx].write(false);
            o_w64_e_strm[idx].write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            counter[idx]++;
            std::cout << "debug in dispatch, Tstrig, idx: " << idx << ", vld: " << vld << std::endl;
#endif
#endif

            o1_w64_field_id_strm.write(idx);
            o1_w64_vld_strm.write(vld);
            o1_str_len_strm.write(false);
            o1_w64_e_strm.write(false);
            first = true;
        } else if (dt == TString) {
            if (first) {
                // in = i_w64_dat_strm.read();
                vld = 32;
                in = i_str_len_strm.read();
                o_str_len_strm[idx].write(true);
                o1_str_len_strm.write(true);
                o1_w64_len_strm.write(32 + (in << 3));
                first = false;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                counter[idx]++;
                std::cout << "debug in dispatch, Tcount,idx: " << idx << ", len: " << in << std::endl;
#endif
#endif
            } else {
                vld = i_w64_vld_strm.read();
                in = i_w64_dat_strm.read();
                o_str_len_strm[idx].write(false);
                o1_str_len_strm.write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                counter[idx]++;
                std::cout << "debug in dispatch, Tstrig, idx: " << idx << ", vld: " << vld << std::endl;
#endif
#endif
            }
            o_w64_dat_strm[idx].write(in);
            o_w64_vld_strm[idx].write(vld);
            o_w64_e_strm[idx].write(false);

            o1_w64_field_id_strm.write(idx);
            o1_w64_vld_strm.write(vld);
            // o1_w64_len_strm.write(4 + len);
            o1_w64_e_strm.write(false);
        } else if (dt == TBoolean || dt == TFloat32 || dt == TDouble || dt == TDate || dt == TInt64) {
            vld = i_w64_vld_strm.read();
            in = i_w64_dat_strm.read();

            o_w64_dat_strm[idx].write(in);
            o_w64_vld_strm[idx].write(vld);
            o_str_len_strm[idx].write(false);
            o_w64_e_strm[idx].write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            counter[idx]++;
            std::cout << "debug in dispatch, idx: " << idx << ", vld: " << vld << std::endl;
#endif
#endif

            o1_w64_field_id_strm.write(idx);
            o1_w64_vld_strm.write(vld);
            o1_w64_len_strm.write(vld);
            o1_str_len_strm.write(true);
            o1_w64_e_strm.write(false);
        }
        // std::cout << "debug in dispatch, E: " << e << std::endl;
    }
    // std::cout << "debug in dispatch, E: " << e << std::endl;
    for (int i = 0; i < (1 << W); i++) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "debug in dispatch, Finish " << idx << ", counter rows: " << counter[i] << std::endl;
#endif
#endif

        o_w64_e_strm[i].write(true);
    }
    o1_w64_e_strm.write(true);
}

template <int W>
void collectDataForMem(hls::stream<ap_uint<4> >& i_w64_field_id_strm,
                       hls::stream<ap_uint<8> >& i_w64_vld_strm,
                       // hls::stream<ap_uint<4> >& i_w64_dt_strm,
                       hls::stream<ap_uint<32> >& i_w64_len_strm,
                       hls::stream<bool>& i_str_len_strm,
                       hls::stream<bool>& i_w64_e_strm,

                       hls::stream<ap_uint<4> >& o_field_id_strm,
                       hls::stream<bool>& o_e_strm,
                       hls::stream<bool>& o_chunk_end) {
    bool e = i_w64_e_strm.read();
    ap_uint<4> idx;
    ap_uint<8> vld;
    ap_uint<32> len;
    bool if_str_len;
    ap_uint<4> dt;

    ap_uint<32> line_counter[1 << W];
#pragma HLS array_partition variable = line_counter
    // for string
    ap_uint<8> reserve[1 << W];
#pragma HLS array_partition variable = reserve
    ap_uint<32> reserve_page_size[1 << W];
#pragma HLS array_partition variable = reserve_page_size
    bool first_elem[1 << W];
#pragma HLS array_partition variable = reserve_page_size
    for (int i = 0; i < (1 << W); i++) {
#pragma HLS unroll
        reserve[i] = 0;
        reserve_page_size[i] = 0;
        line_counter[i] = 0;
        first_elem[i] = true;
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    int counter[5] = {0, 0, 0, 0, 0};
#endif
#endif

    while (!e) {
#pragma HLS pipeline II = 1
        idx = i_w64_field_id_strm.read();
        vld = i_w64_vld_strm.read();
        if_str_len = i_str_len_strm.read();
        // dt = i_w64_dt_strm.read();
        e = i_w64_e_strm.read();
        bool if_new_page = false;

        // bool invalid_dat = false;
        if (if_str_len) {
            len = i_w64_len_strm.read();
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << "debug in collectDataForMem,idx: " << idx << ", len: " << len << ", vld: " << vld << std::endl;
#endif
#endif
            if (line_counter[idx] == (1 << 30) || reserve_page_size[idx] + len > (PAGE_SIZE_BIT)) {
// if (reserve_page_size[idx] + 32 + len > (PAGE_SIZE_BIT)) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
                std::cout << "debug in collectDataForMem, a new page for " << std::dec << idx
                          << ", counter: " << line_counter[idx] << ", size:" << reserve_page_size[idx] << std::endl;
#endif
#endif

                if (reserve[idx] != 0) {
                    reserve[idx] = 64 + vld;
                    if_new_page = true;
                } else if (reserve_page_size[idx] != PAGE_SIZE_BIT) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
                    std::cout << "debug in collectDataForMem,filed" << idx << ": " << reserve_page_size[idx] << ", "
                              << PAGE_SIZE_BIT << std::endl;
#endif
#endif
                    reserve[idx] = 64 + vld;
                    if_new_page = true;
                } else {
                    reserve[idx] = vld;
                }
                line_counter[idx] = 1;
                reserve_page_size[idx] = vld;

                // reserve[idx] = vld;
                // else reserve[idx] = 64 + vld;
            } else {
                line_counter[idx]++;
                reserve[idx] += vld;
                reserve_page_size[idx] += vld;
            }
        } else {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << "debug in collectDataForMem, idx: " << idx << ", vld: " << vld << std::endl;
#endif
#endif
            reserve[idx] += vld;
            reserve_page_size[idx] += vld;
        }

        if (reserve[idx] >= 64) {
            reserve[idx] -= 64;
            o_field_id_strm.write(idx);
            o_chunk_end.write(if_new_page);
            o_e_strm.write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            counter[idx]++;
            std::cout << "debug in collectDataForMem, ouput: " << idx << ", " << if_new_page
                      << ", reserve: " << reserve[idx] << std::endl;
#endif
#endif
        }
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "debug in collectDataForMem, Finish step1" << std::endl;
#endif
#endif
    for (int i = 0; i < (1 << W); i++) {
#pragma HLS pipeline
        if (reserve[i] > 0) {
            o_field_id_strm.write(i);
            o_chunk_end.write(false);
            o_e_strm.write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            std::cout << "Last reserve: " << reserve[i] << std::endl;
            counter[i]++;
#endif
#endif
        }
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    for (int i = 0; i < (1 << W); i++) {
        std::cout << "debug in collectDataForMem, Finish step2, write out counter rows " << counter[i] << std::endl;
    }
#endif
#endif
    o_e_strm.write(true);
    // o_field_id_strm.write(0);
}
template <int W>
void collectDataForCombine(hls::stream<ap_uint<64> >& i_w64_dat_strm,
                           hls::stream<ap_uint<8> >& i_w64_vld_strm,
                           hls::stream<ap_uint<4> >& i_w64_dt_strm,
                           hls::stream<bool>& i_str_len_strm,
                           hls::stream<bool>& i_w64_e_strm,

                           hls::stream<ap_uint<64> >& o_dat_strm,
                           ap_uint<32> page_size[1024],
                           ap_uint<32> page_vnum[1024]) {
    // hls::stream<bool>& page_e_tag) {
    bool e = i_w64_e_strm.read();
    bool check = false;

    ap_uint<64> in;
    ap_uint<8> vld;
    ap_uint<4> dt;
    dt = i_w64_dt_strm.read();
    bool is_str_len;

    ap_uint<32> line_counter = 0;

    // for 32-bit dt
    const int full_batch = 64; // each output
    int tail_output = 0;
    int batch_num = 0;
    int batch_counter = 0;
    // for string
    ap_uint<64 * 2> cage_str = 0;
    ap_uint<64> inventory_str = 0;
    ap_uint<32> reserve = 0;
    ap_uint<32> reserve_page_size = 0;

    // ap_uint<32> page_size[1024];
    // ap_uint<32> page_vnum[1024];
    for (int i = 0; i < 1024; i++) {
#pragma HLS pipeline II = 1
        page_size[i] = (PAGE_SIZE_BIT);
        if (dt == TFloat32)
            page_vnum[i] = PAGE_VNUM_BIT32;
        else
            page_vnum[i] = PAGE_VNUM_BIT64;
    }
    int page_counter = 0;
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
    std::cout << "--------------------------start debugging in collectDataForCombine--------------------------"
              << std::endl;
    int counter = 0;

#endif
#endif

    while (!e) {
#pragma HLS pipeline II = 1
        in = i_w64_dat_strm.read();
        vld = i_w64_vld_strm.read();
        is_str_len = i_str_len_strm.read();
        e = i_w64_e_strm.read();

        ap_uint<32> tmp_reserve = reserve;
        if (is_str_len) {
            // reserver for row group
            // if (line_counter == (1 << 30) || reserve_page_size + 32 + (in << 3) > (PAGE_SIZE_BIT)) {
            if (reserve_page_size + 32 + (in << 3) > (PAGE_SIZE_BIT)) {
// check = true;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                std::cout << "debug in collectDataForCombine, a new page"
                          << ", counter: " << line_counter << ", size:" << reserve_page_size << std::endl;
                std::cout << "First elem in str page: " << in << "tmp_reserve: " << tmp_reserve << std::endl;
#endif
#endif
                page_size[page_counter] = (reserve_page_size);
                page_vnum[page_counter] = line_counter;
                page_counter++;
                // if (tmp_reserve != 0) {
                if (reserve_page_size != PAGE_SIZE_BIT) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
                    std::cout << "debug in collectDataForCombine" << std::endl;
#endif
#endif
                    tmp_reserve = 64;
                    reserve = 64;
                }

                reserve_page_size = 0;
                line_counter = 0;
            }
            line_counter++;
        } else {
            if (dt != TString) {
                // reserver for row group
                /*
                if (line_counter == (1 << 30)) {
                    // check = true;
                    page_size[page_counter] = (reserve_page_size);
                    page_vnum[page_counter] = line_counter;
                    page_counter++;
                    tmp_reserve = 64;
                    reserve = 64;
                    reserve_page_size = 0;
                    line_counter = 0;
                }
                */
                line_counter++;
            }
        }
        reserve += vld;
        reserve_page_size += vld;

        cage_str.range(127, 64) = in;
        // if (!check)
        cage_str >>= ((full_batch - tmp_reserve));
        cage_str.range(63, 0) = cage_str.range(63, 0) ^ inventory_str.range(63, 0);

        if (reserve >= 64) {
            reserve -= 64;
            ap_uint<64> out = cage_str.range(63, 0);
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
            // std::cout << "debug in collectDataForCombine, vld: " << vld << std::endl;
            std::cout << std::dec << counter << ": ";
            std::cout << std::hex << out << std::endl;
            counter++;

#endif
#endif
            o_dat_strm.write(out);
            cage_str >>= 64;
        }
        inventory_str = cage_str.range(63, 0);
        cage_str.range(63, 0) = 0;
    }
    if (reserve != 0) {
        o_dat_strm.write(inventory_str);
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
        //        std::cout << "debug in collectDataForCombine, vld: " << vld << std::endl;
        std::cout << std::dec << counter << ": ";
        std::cout << std::hex << inventory_str << std::endl;
        counter++;

#endif
#endif
    }
    if ((reserve_page_size % (PAGE_SIZE_BIT)) != 0) {
        page_size[page_counter] = reserve_page_size % (PAGE_SIZE_BIT);
        if (dt == TFloat32) {
            page_vnum[page_counter] = line_counter % (PAGE_VNUM_BIT64);
        } else if (dt == TInt64 || dt == TDouble || dt == TDate) {
            page_vnum[page_counter] = line_counter % PAGE_VNUM_BIT32;
        } else if (dt == TString) {
            page_vnum[page_counter] = line_counter;
        }
        page_counter++;
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "In collecData3 write to combine, counter rows:" << counter << std::endl;
#endif
#endif

// page_e_tag.write(true);
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
    std::cout << "Finish collectData" << std::endl;

#endif
#endif
}

/**
 *
 * @brief genearte the addr of data write requests, every 32 x 32-bit requests output 1 addr of burst write req
 *
 **/
template <int W>
void memManage(hls::stream<ap_uint<W> >& i_field_id_strm,
               hls::stream<bool>& i_chunk_end,
               hls::stream<bool>& e_strm,

               hls::stream<ap_uint<H_BIT> >& o_low_addr_strm, // DDR address
               hls::stream<ap_uint<12> >& o_high_addr_strm,   // DDR address

               hls::stream<ap_uint<8> >& o_rnm_strm,
               hls::stream<ap_uint<W> >& o_field_id_strm,
               hls::stream<bool>& o_e_strm,
               hls::stream<bool>& wr_e_strm,

               hls::stream<ap_uint<12> >& o_pre_nid_strm,
               hls::stream<ap_uint<12> >& o_cur_nid_strm,
               hls::stream<ap_uint<12> >& o_naddr_strm,
               hls::stream<bool>& o_node_e_strm,

               FieldInfo f_buff[(1 << W)]) {
    // Node linkTable[1024];
    FieldInfo buff_info[1 << W];
#pragma HLS array_partition variable = buff_info

    ap_uint<12> tail_high_addr[1 << W];
#pragma HLS array_partition variable = tail_high_addr

    bool is_head[1 << W];
#pragma HLS array_partition variable = is_head
    // current block size
    ap_uint<12> cur_addr = 1;

    ap_uint<12> link_tb_addr = 0;
    // initialize
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS unroll
        buff_info[i].head = 0;
        buff_info[i].tail = 0;
        buff_info[i].size = 0;
        tail_high_addr[i] = 0;
        is_head[i] = true;
    }
    bool new_block_flag = false;
    bool e = e_strm.read();
    bool is_new_chunk = false;
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
    int debug_counter0[5] = {0, 0, 0, 0, 0};
    int debug_counter1[5] = {0, 0, 0, 0, 0};
#endif
#endif
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<W> idx = i_field_id_strm.read();
        bool line_id = i_chunk_end.read();
        e = e_strm.read();

#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
        if (line_id) std::cout << idx << ", Line id is true: " << debug_counter1[idx] << std::endl;
        //   printf("Create a new block\n");
        debug_counter0[idx]++;
        debug_counter1[idx]++;
#endif
#endif
        // record the size, when it reaches the 8MB. allocate a new node.
        ap_uint<20> size = buff_info[idx].size; //;
        ap_uint<20> size_new;                   // = size + 1;
        // when reach 4MB, need to create a new block
        if (size.range(H_BIT_PAGE, 0) == 0 || is_new_chunk == true) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
            std::cout << "Generate new page addr in memManage modules, idx: " << idx
                      << ", counter: " << debug_counter0[idx] << std::endl;
            debug_counter0[idx] = 0;
#endif
#endif
            new_block_flag = true;
            size_new = 1;
            size = 0;
        } else {
            new_block_flag = false;
            size_new = size + 1;
        }

        ap_uint<12> tail = buff_info[idx].tail;

        // burst len = BURST_LEN * 64;
        ap_uint<L_BIT> low_bit = size_new.range(L_BIT - 1, 0);
        ap_uint<H_BIT> low_addr = size.range(H_BIT_PAGE, L_BIT);
        ap_uint<12> high_addr = 0;
        if (new_block_flag) {
            high_addr = cur_addr;
        } else {
            high_addr = tail_high_addr[idx];
        }
        buff_info[idx].size = size_new;

        if (line_id) {
            is_new_chunk = true;
        } else {
            is_new_chunk = false;
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
// std::cout << "high_addr: " << high_addr << ", low_addr: " << low_addr << ", bit: " << low_bit << std::endl;
#endif
#endif
        // when it reaches 32, burst write-out them.
        if (low_bit == 0) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
            std::cout << "generate burst write request BURST_LEN for field " << idx << std::endl;
#endif
#endif
            o_rnm_strm.write(1 << L_BIT);
            o_low_addr_strm.write(low_addr);
            o_high_addr_strm.write(high_addr);

            o_field_id_strm.write(idx);
            o_e_strm.write(false);
            wr_e_strm.write(false);
        } else if (line_id == true) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
            std::cout << "generate burst write request " << low_bit << " for field " << idx << std::endl;
#endif
#endif
            ap_uint<L_BIT> low_bit_ = 0;
            buff_info[idx].size = (low_addr, low_bit_);
            o_rnm_strm.write(low_bit);
            o_low_addr_strm.write(low_addr);
            o_high_addr_strm.write(high_addr);

            o_field_id_strm.write(idx);
            o_e_strm.write(false);
            wr_e_strm.write(false);
        }
        // allocate a new block
        if (new_block_flag) {
            if (is_head[idx]) {
                buff_info[idx].head = link_tb_addr;
                is_head[idx] = false;
                tail = link_tb_addr;
            }
            o_pre_nid_strm.write(tail);
            o_cur_nid_strm.write(link_tb_addr);
            o_naddr_strm.write(cur_addr);

            o_node_e_strm.write(false);

            buff_info[idx].tail = link_tb_addr;
            tail_high_addr[idx] = cur_addr++;
            new_block_flag = false;

            link_tb_addr++;
        }
    }
    o_node_e_strm.write(true);
    // output the last non-full block
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        FieldInfo last_field = buff_info[i];
        ap_uint<20> size = last_field.size;
        ap_uint<L_BIT> low_bit = size.range(L_BIT - 1, 0);
        ap_uint<H_BIT> low_addr = size.range(H_BIT_PAGE, L_BIT);
        ap_uint<12> tail = last_field.tail;
        ap_uint<12> high_addr = tail_high_addr[i];
        if (low_bit != 0) {
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
            std::cout << "DEBUG memManage:generate burst write request " << low_bit << " for field " << i << std::endl;
#endif
#endif
            o_rnm_strm.write(low_bit);
            o_low_addr_strm.write(low_addr);
            o_high_addr_strm.write(high_addr);
            o_field_id_strm.write(i);
            o_e_strm.write(false);
            wr_e_strm.write(false);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__

#endif
#endif
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "All couter rows in memManage: " << debug_counter1[i] << std::endl;
#endif
#endif
    }
    o_e_strm.write(true);
    wr_e_strm.write(true);

    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        f_buff[i] = buff_info[i];
    }
}
template <int W>
void createNode(hls::stream<ap_uint<12> >& i_pre_nid_strm,
                hls::stream<ap_uint<12> >& i_cur_nid_strm,
                hls::stream<ap_uint<12> >& i_naddr_strm,
                hls::stream<bool>& i_e_strm,
                Node linkTable[1024]) {
    bool e = i_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<12> pre_nid = i_pre_nid_strm.read();
        ap_uint<12> cur_nid = i_cur_nid_strm.read();
        ap_uint<12> cur_addr = i_naddr_strm.read();

        // update the next of node
        linkTable[pre_nid].next = cur_nid;

        Node new_node;
        new_node.base_addr = cur_addr;
        new_node.next = 1024;
        // add the new node and before pointer to the last node.
        linkTable[cur_nid] = new_node;
        e = i_e_strm.read();
    }
}

// select field_id, convert 32 x 32bit data to 16 x 64 bit, prepare for the burst write
template <int W>
void combine(hls::stream<ap_uint<64> > i_strm[1 << W],
             hls::stream<ap_uint<8> >& i_rnm_strm,      // number of row: ap_uint<256>
             hls::stream<ap_uint<W> >& i_field_id_strm, // number of row: ap_uint<256>
             hls::stream<bool>& i_e_strm,
             hls::stream<ap_uint<64> >& o_strm) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    int counter[5] = {0, 0, 0, 0, 0};
#endif
#endif
    bool e = i_e_strm.read();
    while (!e) {
        ap_uint<8> rnm = i_rnm_strm.read();
        ap_uint<W> field_id = i_field_id_strm.read();
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "read " << rnm << " datas form idx " << field_id << std::endl;
#endif
#endif
        for (int i = 0; i < BURST_LEN; ++i) {
#pragma HLS pipeline II = 1
            ap_uint<64> in = 0;
            if (i < rnm) {
                in = i_strm[field_id].read();
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                counter[field_id]++;
// std::cout << "debug in combin, " << counter[field_id] << ": " << in << std::endl;
#endif
#endif
            }
            o_strm.write(in);
        }
        e = i_e_strm.read();
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    // std::cout << "Finish combine" << std::endl;
    for (int i = 0; i < 5; i++) std::cout << "field_id: " << i << ", counter rows: " << counter[i] << std::endl;
#endif
#endif
}

// burst write to DDR, each burst writes 16 x 64-bit data
void writeData(hls::stream<ap_uint<64> >& i_strm,
               hls::stream<ap_uint<H_BIT> >& i_low_addr_strm, // DDR address
               hls::stream<ap_uint<12> >& i_high_addr_strm,   // DDR address
               hls::stream<bool>& i_e_strm,
               ap_uint<64>* ddr_buff) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    int count = 0;
#endif
#endif
    bool e = i_e_strm.read();
    while (!e) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "writeData: " << count << std::endl;
        count++;
#endif
#endif
        ap_uint<H_BIT> low_addr = i_low_addr_strm.read();
        ap_uint<12> high_addr = i_high_addr_strm.read();
        ap_uint<32> base_addr = 0;
        base_addr.range(19, L_BIT) = low_addr;
        base_addr.range(31, 20) = high_addr;
        for (int i = 0; i < BURST_LEN; ++i) {
#pragma HLS pipeline II = 1
            ap_uint<64> tmp = i_strm.read();
            ddr_buff[base_addr + i] = tmp;
#ifdef _DF_DEBUG_V1
#ifndef __SYNTHESIS__
            std::cout << "writeData-base_addr: " << base_addr << ", i:" << i << std::endl;
#endif
#endif
        }
        e = i_e_strm.read();
    }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "writeData end" << std::endl;
#endif
#endif
}
template <int W>
void localRamToMem(ap_uint<32>* s_buff,
                   ap_uint<32>* n_buff,
                   ap_uint<32>* l_buff,
                   Node linkTable[1024],
                   FieldInfo f_buff[2 * (1 << W)],
                   ap_uint<32> page_size[16][1024],
                   ap_uint<32> page_vnum[16][1024],
                   ap_uint<64>* ddr_out) {
    int addr = 1;
    // write size buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ddr_out[addr + i] = s_buff[i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "s_buff[" << i << "] = " << s_buff[i] << std::endl;
#endif
#endif
    }
    addr += (1 << W);
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
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "-----------field info buffer data-----------" << std::endl;
#endif
#endif
    // write field info buffer
    for (int i = 0; i < (1 << W); ++i) {
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
    addr += (1 << W);
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
            std::cout << ", linkTable[" << i << "].next = " << nd.next << std::endl;
        }
#endif
#endif
    }
    addr += 1024;
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 1024; i++) {
            ap_uint<32> p_sz = page_size[j][i];
            ap_uint<32> p_vn = page_vnum[j][i];
            ddr_out[addr + i] = ((p_sz / 8), p_vn);
        }
        addr += 1024;
    }
    ddr_out[0] = addr;
}
void writePageInfo(hls::stream<ap_uint<32> >& page_size,
                   hls::stream<ap_uint<32> >& page_vnum,
                   hls::stream<bool>& page_e_tag,
                   ap_uint<64>* page_info) {
    bool e = page_e_tag.read();
    int index = 1;
    while (!e) {
#pragma HLS pipeline II = 1
        e = page_e_tag.read();
        ap_uint<32> psize = page_size.read();
        ap_uint<32> pvnum = page_vnum.read();

        page_info[index++] = (psize, pvnum);
    }
    page_info[0] = index - 1;
}

template <int W>
void writeToMem(ap_uint<88>* ddr_obj,
                //(hls::stream<Object>& obj_strm,
                ap_uint<8>* schema,
                ap_uint<64>* ddr_buff,
                ap_uint<32>* s_buff,
                ap_uint<32>* n_buff,
                ap_uint<32>* l_buff,
                Node linkTable[1024],
                FieldInfo f_buff[2 * (1 << W)],
                ap_uint<32> page_size[16][1024],
                ap_uint<32> page_vnum[16][1024]) {
    // ap_uint<64> page_info[16][1024]) {
    hls::stream<Object> obj_strm("obj stream write");
#pragma HLS stream variable = obj_strm depth = 64
    hls::stream<ap_uint<W> > s1_field_id_strm("w64 field id strm");
#pragma HLS stream variable = s1_field_id_strm depth = 64
    hls::stream<ap_uint<64> > s1_dat_strm("w64 dat strm");
#pragma HLS stream variable = s1_dat_strm depth = 64
    hls::stream<ap_uint<4> > s1_dt_strm("w64 dt strm");
#pragma HLS stream variable = s1_dt_strm depth = 64
    hls::stream<ap_uint<8> > s1_valid_strm("w64 valid strm");
#pragma HLS stream variable = s1_valid_strm depth = 8
    hls::stream<ap_uint<32> > s1_strlen_strm("str offset strm");
#pragma HLS stream variable = s1_strlen_strm depth = 64
    hls::stream<bool> s1_e_strm("w64 e strm");
#pragma HLS stream variable = s1_e_strm depth = 64

    hls::stream<ap_uint<64> > s2_0_dat_strm[1 << W];
#pragma HLS stream variable = s2_0_dat_strm depth = 64
    hls::stream<ap_uint<8> > s2_0_vld_strm[1 << W];
#pragma HLS stream variable = s2_0_vld_strm depth = 64
    hls::stream<ap_uint<4> > s2_0_dt_strm[1 << W];
#pragma HLS stream variable = s2_0_dt_strm depth = 64
    hls::stream<bool> s2_0_strlen_strm[1 << W];
#pragma HLS stream variable = s2_0_strlen_strm depth = 64
    hls::stream<bool> s2_0_e_strm[1 << W];
#pragma HLS stream variable = s2_0_e_strm depth = 64

    hls::stream<ap_uint<4> > s2_1_field_id_strm;
#pragma HLS stream variable = s2_1_field_id_strm depth = 64
    hls::stream<ap_uint<8> > s2_1_vld_strm;
#pragma HLS stream variable = s2_1_vld_strm depth = 64
    hls::stream<ap_uint<32> > s2_1_len_strm;
#pragma HLS stream variable = s2_1_len_strm depth = 64
    hls::stream<bool> s2_1_strlen_strm;
#pragma HLS stream variable = s2_1_strlen_strm depth = 64
    hls::stream<bool> s2_1_e_strm;
#pragma HLS stream variable = s2_1_e_strm depth = 64

    hls::stream<ap_uint<64> > s3_0_dat_strm[1 << W];
#pragma HLS stream variable = s3_0_dat_strm depth = 96
    /*
        hls::stream<ap_uint<32> > s3_0_page_size_strm[1 << W];
    #pragma HLS stream variable = s3_0_page_size_strm depth = 64
        hls::stream<ap_uint<32> > s3_0_page_vnum_strm[1 << W];
    #pragma HLS stream variable = s3_0_page_vnum_strm depth = 64
        hls::stream<bool> s3_0_page_e_tag_strm[1 << W];
    #pragma HLS stream variable = s3_0_page_e_tag_strm depth = 64
    */

    hls::stream<ap_uint<W> > s3_1_field_id_strm;
#pragma HLS stream variable = s3_1_field_id_strm depth = 64
    hls::stream<bool> s3_1_e_strm;
#pragma HLS stream variable = s3_1_e_strm depth = 64
    hls::stream<bool> s3_1_chunk_end;
#pragma HLS stream variable = s3_1_chunk_end depth = 64

    hls::stream<ap_uint<H_BIT> > l_addr_strm;
#pragma HLS stream variable = l_addr_strm depth = 64
    hls::stream<ap_uint<12> > h_addr_strm;
#pragma HLS stream variable = h_addr_strm depth = 64
    hls::stream<ap_uint<8> > rnm_strm;
#pragma HLS stream variable = rnm_strm depth = 64
    hls::stream<ap_uint<W> > c_field_id_strm;
#pragma HLS stream variable = c_field_id_strm depth = 64
    hls::stream<bool> c_e_strm;
#pragma HLS stream variable = c_e_strm depth = 64
    hls::stream<bool> wr_e_strm;
#pragma HLS stream variable = wr_e_strm depth = 64

    hls::stream<ap_uint<12> > node_pre_nid_strm;
#pragma HLS stream variable = node_pre_nid_strm depth = 8
    hls::stream<ap_uint<12> > node_cur_nid_strm;
#pragma HLS stream variable = node_cur_nid_strm depth = 8
    hls::stream<ap_uint<12> > node_naddr_strm;
#pragma HLS stream variable = node_naddr_strm depth = 8
    hls::stream<bool> node_e_strm;
#pragma HLS stream variable = node_e_strm depth = 8

    hls::stream<ap_uint<64> > m_strm;
#pragma HLS stream variable = m_strm depth = 32
#pragma HLS BIND_STORAGE variable = m_strm type = fifo impl = lutram

#pragma HLS dataflow
    readDDR<W>(ddr_obj, obj_strm);
    readObjStrm<W>(obj_strm, s1_field_id_strm, s1_dat_strm, s1_dt_strm, s1_valid_strm, s1_strlen_strm, s1_e_strm,
                   s_buff);

    dispatch<W>(s1_field_id_strm, s1_dat_strm, s1_dt_strm, s1_valid_strm, s1_strlen_strm, s1_e_strm, schema,
                s2_0_dat_strm, s2_0_vld_strm, s2_0_dt_strm, s2_0_strlen_strm, s2_0_e_strm, //
                s2_1_field_id_strm, s2_1_vld_strm, s2_1_len_strm, s2_1_strlen_strm, s2_1_e_strm);

    for (int i = 0; i < (1 << W); i++) {
#pragma HLS unroll
        collectDataForCombine<W>(s2_0_dat_strm[i], s2_0_vld_strm[i], s2_0_dt_strm[i], s2_0_strlen_strm[i],
                                 s2_0_e_strm[i], s3_0_dat_strm[i], page_size[i], page_vnum[i]);
    }
    collectDataForMem<W>(s2_1_field_id_strm, s2_1_vld_strm, s2_1_len_strm, s2_1_strlen_strm, s2_1_e_strm,
                         s3_1_field_id_strm, s3_1_e_strm, s3_1_chunk_end);

    memManage<W>(s3_1_field_id_strm, s3_1_chunk_end, s3_1_e_strm, l_addr_strm, h_addr_strm, rnm_strm, c_field_id_strm,
                 c_e_strm, wr_e_strm, node_pre_nid_strm, node_cur_nid_strm, node_naddr_strm, node_e_strm, f_buff);

    createNode<W>(node_pre_nid_strm, node_cur_nid_strm, node_naddr_strm, node_e_strm, linkTable);

    combine(s3_0_dat_strm, rnm_strm, c_field_id_strm, c_e_strm, m_strm);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "Kernel end" << std::endl;
#endif
#endif

    writeData(m_strm, l_addr_strm, h_addr_strm, wr_e_strm, ddr_buff);
}

} // end of internal

void writeToParquetSupport(/*hls::stream<Object>& obj_strm,*/ ap_uint<88>* ddr_obj,
                           ap_uint<8> schema[16],
                           ap_uint<64>* ddr_buff) {
    ap_uint<32> s_buff[16];
#pragma HLS BIND_STORAGE variable = s_buff type = ram_1p impl = lutram
    ap_uint<32> n_buff[16];
#pragma HLS BIND_STORAGE variable = n_buff type = ram_1p impl = lutram
    ap_uint<32> l_buff[16];
#pragma HLS BIND_STORAGE variable = l_buff type = ram_1p impl = lutram
    ap_uint<64> page_info[16][1024];
    ap_uint<32> page_size[16][1024];
    ap_uint<32> page_vnum[16][1024];
#pragma HLS array_partition variable = page_size dim = 1
#pragma HLS array_partition variable = page_vnum dim = 1

    // define the filed_id number by 1 << W
    const int W = 4;
    const int file_nm = 1 << W;

    // string offset size of each field_id: s_buff[16] s_buff[17] ... s_buff[31]
    Node linkTable[1024];
    //#pragma HLS BIND_STORAGE variable = linkTable type = ram_1p impl = lutram
    // the nodes info for each field
    FieldInfo f_buff[file_nm];
#pragma HLS BIND_STORAGE variable = f_buff type = ram_1p impl = lutram

    // Init
    for (int i = 0; i < 1024; i++) {
        linkTable[i].next = 0;
        linkTable[i].base_addr = 0;
    }

    // read 64-bit data of different type to DDR
    internal::writeToMem<W>(ddr_obj /*obj_strm*/, schema, ddr_buff, s_buff, n_buff, l_buff, linkTable, f_buff,
                            page_size, page_vnum);
    internal::localRamToMem<W>(s_buff, n_buff, l_buff, linkTable, f_buff, page_size, page_vnum, ddr_buff);
}

} // end of dataframe namespace
} // end of data_analytics namespace
} // end of xf namespace

#endif
