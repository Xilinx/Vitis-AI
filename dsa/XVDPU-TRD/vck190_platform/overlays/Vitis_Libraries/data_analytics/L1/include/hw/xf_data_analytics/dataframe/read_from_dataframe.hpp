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
 * @file read_from_dataframe.hpp
 * @brief Read Apache Arrow format data from FPGA DDR, output as an Object sturct stream
 *
 * This file is part of Vitis Data Analytics Library.
 */

#ifndef XF_DATA_ANALYTICS_L1_DATAFRAME_READ_FROM_DATAFRAME_HPP
#define XF_DATA_ANALYTICS_L1_DATAFRAME_READ_FROM_DATAFRAME_HPP

#include "xf_data_analytics/common/obj_interface.hpp"
#include "xf_data_analytics/dataframe/df_utils.hpp"

//#define _DF_DEBUG 1

namespace xf {
namespace data_analytics {
namespace dataframe {

namespace internal {

// one time burst read from DDR
template <int W>
void bread(const ap_uint<8> nm,
           const int base_addr,
           const ap_uint<W> field_id,
           ap_uint<64>* ddr_buff,
           hls::stream<ap_uint<64> >& o_dat_strm,
           hls::stream<ap_uint<W> >& o_field_id_strm,
           hls::stream<bool>& o_e_strm) {
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
    std::cout << "id: " << field_id << ", burst read nm: " << nm << " data, addr: " << base_addr
              << ", data as follow: " << std::endl;
#endif
#endif
    for (int i = 0; i < nm; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<64> dat = ddr_buff[base_addr + i];
        o_dat_strm.write(dat);
        o_field_id_strm.write(field_id);
        o_e_strm.write(0);
    }
}
/**
 * @brief generate the reading addr of each field, except Null and Boolean
 **/
template <int W>
void genAddr64(int field_nm,
               ap_uint<4> t_buff[1 << W],
               FieldInfo f_buff[32],
               Node linkTable[1024],
               // ap_uint<12> o_str_high_addr_mem[1 << W][256],
               hls::stream<int>& o_field_id_strm,
               hls::stream<int>& o_addr_strm,
               hls::stream<int>& o_nm_strm,
               hls::stream<ap_uint<4> >& o_type_strm,
               hls::stream<bool>& o_e_strm) {
    ap_uint<1 << W> isHead;
    // flag indicates current node is the tail node. isEnd_n = 0 means currNode = tail node
    ap_uint<1 << W> isEnd_n;
    // flag indicates w32 nodes  (UFloat32, UString offset) are the end or half
    bool isHalf = false;
    for (int i = 0; i < (1 << W); i++) {
        isHead[i] = 1;
        if (i < field_nm) {
            if (t_buff[i] == TInt64 || t_buff[i] == TDouble || t_buff[i] == TDate) {
                isEnd_n[i] = 1;
            } else if (t_buff[i] == TFloat32 || t_buff[i] == TString) {
                isEnd_n[i] = 1;
            } else
                isEnd_n[i] = 0;
        } else
            isEnd_n[i] = 0;
    }

    ap_uint<12> lt_cur[1 << W];
    ap_uint<12> lt_tail[1 << W];
    ap_uint<32> lt_size;
    ap_uint<32> addr = 0;
    ap_uint<12> high_addr[1 << W];
    ap_uint<15> low_addr[1 << W];

    // pre-processing: output all string db high_addr to a FIFO
    ap_uint<12> str_lt_cur = 0;
    ap_uint<12> str_lt_tail = 0;
    ap_uint<12> str_high_addr = 0;

    bool switch_block = false;
    //
    while (isEnd_n) {
        lt_size = 0;
    LOOPF:
        for (int idx = 0; idx < field_nm; idx++) {
#pragma HLS pipeline II = 1
            if (t_buff[idx] == TInt64 || t_buff[idx] == TDouble || t_buff[idx] == TDate) {
                // get the mem block high_addr
                if (isHead[idx]) {
                    lt_cur[idx] = f_buff[idx].head;
                    lt_tail[idx] = f_buff[idx].tail;
                    isHead[idx] = false;
                } else {
                    // not the tail node, updating to next memblock
                    if (isEnd_n[idx] == 1) {
                        lt_cur[idx] = linkTable[lt_cur[idx]].next;
                    }
                }

                high_addr[idx] = linkTable[lt_cur[idx]].base_addr;
                low_addr[idx] = 0;

                // get the memblock size/nrow
                if (lt_cur[idx] != lt_tail[idx]) {
                    if (lt_size == 0) lt_size = 1 << 20;
                } else {
                    if (lt_size == 0) lt_size = f_buff[idx].size;
                    isEnd_n[idx] = 0;
                }
                // std::cout << "lt_cur[" << idx << "]: " << lt_cur[idx] << ", ";
                // std::cout << "high_addr[" << idx << "]: " << high_addr[idx] << ", ";
                // std::cout << std::endl;
            }
            // get the offset of UString
            else if (t_buff[idx] == TString) {
                if (isHead[idx]) {
                    lt_cur[idx] = f_buff[idx + 16].head;
                    lt_tail[idx] = f_buff[idx + 16].tail;
                    isHead[idx] = false;
                    high_addr[idx] = linkTable[lt_cur[idx]].base_addr;
                    low_addr[idx] = 0;
                } else {
                    // not the tail node, updating to next memblock
                    if (isEnd_n[idx] == 1 && isHalf == true) {
                        lt_cur[idx] = linkTable[lt_cur[idx]].next;
                        high_addr[idx] = linkTable[lt_cur[idx]].base_addr;
                        low_addr[idx] = 0;
                    }
                    // if isHalf == false, lt_cur, high_addr, low_addr not changing
                } // get the memblock size/nrow
                if (lt_cur[idx] != lt_tail[idx]) {
                    // if (lt_size == 0) lt_size = 1 << 20;
                } else {
                    // if (lt_size == 0) lt_size = f_buff[idx].size;
                    isEnd_n[idx] = 0;
                }
            }
        }

        // get the round of burst read, each burst 16x 64bits
        // lt_size is 32bit based counting
        int lt_size_n = lt_size / 32;
        int lt_size_ex = lt_size.range(4, 1);
        if (lt_size[0] == 1) lt_size_ex++;

#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "lt_size = " << lt_size << std::endl;
        std::cout << "lt_size_n = " << lt_size_n << std::endl;
        std::cout << "lt_size_ex = " << lt_size_ex << std::endl;
#endif
#endif

        for (int n = 0; n < lt_size_n; n++) {
        // gen addr for each field
        LOOPG:
            for (int idx = 0; idx < field_nm; idx++) {
#pragma HLS pipeline II = 1
                ap_uint<15> low_addr_tmp = low_addr[idx];
                addr.range(30, 19) = high_addr[idx];
                int nm = 0;
                if (t_buff[idx] == TInt64 || t_buff[idx] == TDouble || t_buff[idx] == TDate) {
                    addr.range(18, 4) = low_addr_tmp;
                    addr.range(3, 0) = 0;
                    nm = 16;
                } else if (t_buff[idx] == TFloat32 || t_buff[idx] == TString) {
                    addr[18] = switch_block;
                    addr.range(17, 3) = low_addr_tmp;
                    addr.range(2, 0) = 0;
                    nm = 8;
                }
                o_field_id_strm.write(idx);
                o_addr_strm.write(addr);
                o_nm_strm.write(nm);
                o_type_strm.write(t_buff[idx]);
                o_e_strm.write(0);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                std::cout << "id: " << idx << ", high_addr: " << high_addr[idx] << ", low_addr: " << low_addr[idx]
                          << ", addr: " << std::hex << addr << std::dec << ", nm: " << nm << std::endl;
#endif
#endif

                low_addr[idx]++;
            }
        }
        // last round, output the remaining data
        if (lt_size_ex != 0) {
            for (int idx = 0; idx < field_nm; idx++) {
#pragma HLS pipeline II = 1
                if (t_buff[idx] == TInt64 || t_buff[idx] == TDouble || t_buff[idx] == TDate) {
                    addr.range(30, 19) = high_addr[idx];
                    addr.range(18, 4) = low_addr[idx];
                    addr.range(3, 0) = 0;
                    o_field_id_strm.write(idx);
                    o_addr_strm.write(addr);
                    o_nm_strm.write(lt_size_ex);
                    o_type_strm.write(t_buff[idx]);
                    o_e_strm.write(0);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                    std::cout << "id: " << idx << ", high_addr: " << high_addr[idx] << ", low_addr: " << low_addr[idx]
                              << ", addr: " << std::hex << addr << std::dec << ", nm: " << lt_size_ex << std::endl;
#endif
#endif
                } else if (t_buff[idx] == TFloat32 || t_buff[idx] == TString) {
                    addr.range(30, 19) = high_addr[idx];
                    addr[18] = switch_block;
                    addr.range(17, 3) = low_addr[idx];
                    addr.range(2, 0) = 0;
                    o_field_id_strm.write(idx);
                    o_addr_strm.write(addr);
                    o_nm_strm.write(lt_size_ex);
                    o_type_strm.write(t_buff[idx]);
                    o_e_strm.write(0);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
                    std::cout << "id: " << idx << ", high_addr: " << high_addr[idx] << ", low_addr: " << low_addr[idx]
                              << ", addr: " << std::hex << addr << std::dec << ", nm: " << lt_size_ex << std::endl;
#endif
#endif
                }
            }
        }
        switch_block = ~switch_block;
    }

    o_e_strm.write(1);
}

/**
 * @brief burst read data field by field
 **/
template <int W>
void breadWrapper(int field_nm,
                  ap_uint<12> str_high_addr_mem[1 << W][256],
                  hls::stream<int>& i_field_id_strm,
                  hls::stream<int>& i_addr_strm,
                  hls::stream<int>& i_nm_strm,
                  hls::stream<ap_uint<4> >& i_type_strm,
                  hls::stream<bool>& i_e_strm,
                  ap_uint<64>* ddr_buff,
                  hls::stream<ap_uint<64> > o_dat_strm[1 << W],
                  hls::stream<ap_uint<4> > o_valid_strm[1 << W],
                  hls::stream<bool> o_e_strm[1 << W]) {
#pragma HLS inline off

    ap_uint<32> offset[32];
    int offset_base = 0;
    int last_burst_offset = 0;
    bool e = i_e_strm.read();
    int last_offset_base = 0;
    // initialize the used str_addr
    ap_uint<32> str_addr[1 << W];
    int blk_cnt[1 << W];
    for (int i = 0; i < (1 << W); i++) {
        str_addr[i].range(31, 19) = str_high_addr_mem[i][0];
        str_addr[i].range(18, 0) = 0;
        blk_cnt[i] = 0;
    }
    while (!e) {
        int addr = i_addr_strm.read();
        int nm = i_nm_strm.read();
        ap_uint<W> id = i_field_id_strm.read();

        int type = i_type_strm.read();
        int total_nm = 0;
        ap_uint<64> dat = 0;
        ap_uint<19> nrow = 0;
        // w64 data type
        if (type == TInt64 || type == TDouble || type == TDate) {
        LOOP_UFF:
            for (int i = 0; i < nm; i++) {
#pragma HLS PIPELINE II = 1
                ap_uint<64> dat = ddr_buff[addr + i];
                o_dat_strm[id].write(dat);
                o_valid_strm[id].write(8);
                o_e_strm[id].write(0);
            }
        } else if (type == TString) {
        // burst read the strlen first
        // ap_uint<4> nm_i = 0;
        LOOP_UF:
            for (int i = 0; i < nm * 2; i++) {
#pragma HLS PIPELINE II = 1
                int len = 0;
                if (i % 2 == 0) {
                    dat = ddr_buff[addr + i / 2];
                    len = dat.range(31, 0);
                    // nm_i++;
                } else {
                    len = dat.range(63, 32);
                }
                nrow += (len + 7) / 8;
                offset[i] = len;
            }
#ifdef _DF_DEBUG
#ifndef __SYNTHESISI__
            std::cout << "string lines to read: nrow = " << std::dec << nrow << std::endl;
#endif
#endif

            // read the string based on strlen
            // check whether needs to change the mem block in this burst string read.
            ap_uint<19> new_low_addr = str_addr[id].range(18, 0);
            ap_uint<20> tmp = new_low_addr + nrow;
            ap_uint<12> tmp_high_addr_exceed;
            // overflow, change the memblock
            if (tmp[19] == 1) {
                blk_cnt[id]++;
                tmp_high_addr_exceed = str_high_addr_mem[id][blk_cnt[id]];
                std::cout << "tmp_high_addr_exceed = " << tmp_high_addr_exceed << std::endl;
            }

            int o_idx = 0;
        LOOPS:
            for (int i = 0; i < nrow; i++) {
#pragma HLS PIPELINE II = 1
                // std::cout << "read addr: " << std::hex << str_addr[id] + i << std::endl;
                ap_uint<64> str_dat = ddr_buff[str_addr[id] + i];
                o_dat_strm[id].write(str_dat);
                o_e_strm[id].write(0);
                ap_uint<4> valid = 0;
                int tmp = offset[o_idx];
                offset[o_idx] -= 8;
                if (tmp > 8) {
                    valid = 8;
                } else {
                    valid = tmp;
                    o_idx++;
                }

                o_valid_strm[id].write(valid);

                if (new_low_addr + i == (1 << 19) - 1) {
                    str_addr[id].range(30, 19) = tmp_high_addr_exceed;
                }
            }
            new_low_addr += nrow;
            str_addr[id].range(18, 0) = new_low_addr;
        }
        e = i_e_strm.read();
    }

    for (int i = 0; i < field_nm; i++) {
        o_e_strm[i].write(1);
    }
}

/**
 * @brief write out all fields of each line
 **/
template <int W>
void writeObjOut(int field_nm,
                 ap_uint<4> t_buff[1 << W],
                 hls::stream<bool> i_null_strm[1 << W],
                 hls::stream<bool> i_bool_strm[1 << W],
                 hls::stream<ap_uint<64> > i_dat_strm[1 << W],
                 hls::stream<ap_uint<4> > i_valid_strm[1 << W],
                 hls::stream<bool> i_e_strm[1 << W],
                 hls::stream<Object>& obj_strm) {
    // get the data type of each field
    ap_uint<4> type[1 << W];
    bool e = false;
    for (int id = 0; id < field_nm; id++) {
        type[id] = t_buff[id];
        e &= i_e_strm[id].read();
    }
    // write out each data as an object stream
    Object obj_dat;
    ap_uint<64> dat;
    ap_uint<4> valid;
    while (!e) {
    WFOR:
        for (int id = 0; id < field_nm; id++) {
            // check if data is null
            bool isNull = i_null_strm[id].read();
            bool isBool = i_bool_strm[id].read();

            if (type[id] == TBoolean) {
                valid = 1;
                dat = isBool;
            } else if (type[id] == TInt64 || type[id] == TDouble || type[id] == TDate || type[id] == TString) {
                valid = i_valid_strm[id].read();
                dat = i_dat_strm[id].read();
                e = i_e_strm[id].read();
            }

            if (isNull) {
                valid = 0;
                dat = 0;
            }
            obj_dat.set_data(dat);
            obj_dat.set_id(id);
            obj_dat.set_valid(valid);
            obj_dat.set_type(type[id]);
            obj_strm.write(obj_dat);
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            if (isNull) std::cout << "null, ";
            std::cout << obj_dat.get_type() << ",";
            std::cout << obj_dat.get_valid() << ",";
            std::cout << obj_dat.get_id() << ",";
            std::cout << obj_dat.get_data() << "; ";
#endif
#endif
        }
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << std::endl;
#endif
#endif
    }
}

// read the null and bool info from on-chip mem
template <int W>
void readNullBool(int field_nm,
                  ap_uint<32> nrow,
                  ap_uint<64> bit_map[4096][16],
                  ap_uint<64> bool_buff[4096][16],
                  hls::stream<bool> o_null_strm[1 << W],
                  hls::stream<bool> o_bool_strm[1 << W]) {
    for (ap_uint<32> n = 0; n < nrow; n++) {
        ap_uint<6> low_addr = n.range(5, 0);
        ap_uint<12> high_addr = n.range(17, 6) + 1;

        // read one data per field
        for (int id = 0; id < field_nm; id++) {
#pragma HLS pipeline II = 1
            ap_uint<64> elem = bit_map[high_addr][id];
            bool isNull = elem[low_addr];
            if (!isNull) {
                o_null_strm[id].write(1);
            } else {
                o_null_strm[id].write(0);
            }
            // boolean
            ap_uint<64> bool_elem = bool_buff[high_addr][id];
            bool bval = bool_elem[low_addr];
            o_bool_strm[id].write(bval);
        }
    }
}
// read all data of each field as object stream
template <int W>
void readToObjStrm(int field_nm,
                   ap_uint<32> nrow,
                   ap_uint<4> t_buff[1 << W],
                   ap_uint<4> t_buff_w[1 << W],
                   FieldInfo f_buff[32],
                   ap_uint<64> bit_map[4096][16],
                   ap_uint<64> bool_buff[4096][16],
                   Node linkTable[1024],
                   ap_uint<12> str_high_addr_mem[1 << W][256],
                   ap_uint<64>* ddr_buff,
                   hls::stream<Object>& obj_strm) {
    hls::stream<bool> null_strm[1 << W];
#pragma HLS STREAM variable = null_strm depth = 64
#pragma HLS BIND_STORAGE variable = null_strm type = fifo impl

    hls::stream<bool> bool_strm[1 << W];
#pragma HLS stream variable = bool_strm depth = 64
#pragma HLS BIND_STORAGE variable = bool_strm type = fifo impl

    hls::stream<int> field_id_strm;
#pragma HLS stream variable = field_id_strm depth = 16
    hls::stream<int> addr_strm;
#pragma HLS stream variable = addr_strm depth = 16
    hls::stream<int> nm_strm;
#pragma HLS stream variable = nm_strm depth = 16
    hls::stream<ap_uint<4> > type_strm;
#pragma HLS stream variable = type_strm depth = 16
    hls::stream<bool> e_strm;
#pragma HLS stream variable = e_strm depth = 16

    hls::stream<ap_uint<64> > w_data_strm[1 << W];
#pragma HLS stream variable = w_data_strm depth = 1024
#pragma HLS BIND_STORAGE variable = w_data_strm type = fifo impl = bram
    hls::stream<ap_uint<4> > w_valid_strm[1 << W];
#pragma HLS stream variable = w_valid_strm depth = 1024
#pragma HLS BIND_STORAGE variable = w_valid_strm type = fifo impl = bram
    hls::stream<bool> w_e_strm[1 << W];
#pragma HLS stream variable = w_e_strm depth = 1024
#pragma HLS BIND_STORAGE variable = w_e_strm type = fifo impl = bram

#pragma HLS dataflow
    readNullBool<W>(field_nm, nrow, bit_map, bool_buff, null_strm, bool_strm);

    genAddr64<W>(field_nm, t_buff, f_buff, linkTable, field_id_strm, addr_strm, nm_strm, type_strm, e_strm);
    breadWrapper<W>(field_nm, str_high_addr_mem, field_id_strm, addr_strm, nm_strm, type_strm, e_strm, ddr_buff,
                    w_data_strm, w_valid_strm, w_e_strm);

    writeObjOut<W>(field_nm, t_buff_w, null_strm, bool_strm, w_data_strm, w_valid_strm, w_e_strm, obj_strm);
}

/**
 * @brief load the header info which saves length, null, bool info into on-chip mem
 **/
template <int W>
void memToLocalRam(int field_nm,
                   ap_uint<W> t_buff[1 << W],
                   ap_uint<32>* s_buff,
                   ap_uint<32>* n_buff,
                   ap_uint<32>* l_buff,
                   ap_uint<64> bit_map[4096][1 << W],
                   ap_uint<64> bool_buff[4096][1 << W],
                   Node linkTable[1024],
                   FieldInfo f_buff[2 * (1 << W)],
                   ap_uint<12> str_high_addr_mem[1 << W][256],
                   ap_uint<64>* ddr_in) {
    int addr = 1;
    // write size buff
    for (int i = 0; i < 2 * (1 << W); ++i) {
#pragma HLS pipeline II = 1
        s_buff[i] = ddr_in[addr + i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        if (i < (1 << W)) {
            std::cout << "field_id: " << i << ", size: " << s_buff[i] << std::endl;
        } else
            std::cout << "field_id: " << i << ", offset size: " << s_buff[i] << std::endl;
#endif
#endif
    }
    addr += 2 * (1 << W);
    // write length buff
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        l_buff[i] = ddr_in[addr + i];
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
        n_buff[i] = ddr_in[addr + i];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "n_buff[" << i << "] = " << n_buff[i] << std::endl;
#endif
#endif
    }
    addr += (1 << W);
    // skip bitmap address
    addr += (1 << W);
    // skip bool_buff address
    addr += (1 << W);
    // skip value address and offset address
    addr += 2 * (1 << W);
    // write bitmap buffer
    for (int i = 0; i < (1 << W); ++i) {
        ap_uint<32> nm = ddr_in[addr];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        std::cout << "bit_map " << i << ", nm: " << nm << ", data";
#endif
#endif
        for (int j = 0; j < nm; ++j) {
#pragma HLS pipeline II = 1
            bit_map[j][i] = ddr_in[addr + j];
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            if (j > 0) std::cout << "[" << j << "][" << i << "]=" << bit_map[j][i] << ",";
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
        int nm = ddr_in[addr];
        for (int j = 0; j < nm; ++j) {
#pragma HLS pipeline II = 1
            bool_buff[j][i] = ddr_in[addr + j];
        }
        addr += nm;
    }

    // write field info buffer
    for (int i = 0; i < 2 * (1 << W); ++i) {
#pragma HLS pipeline II = 1
        ap_uint<64> in = ddr_in[addr + i];
        FieldInfo fld_info;
        fld_info.head = in.range(11, 0);
        fld_info.tail = in.range(23, 12);
        fld_info.size = in.range(43, 24);
        f_buff[i] = fld_info;
    }
    addr += 2 * (1 << W);
    // write link table
    for (int i = 0; i < 1024; ++i) {
#pragma HLS pipeline II = 1
        ap_uint<64> in = ddr_in[addr + i];
        Node nd;
        nd.next = in.range(11, 0);
        nd.base_addr = in.range(23, 12);
        linkTable[i] = nd;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
        if (i < 20) {
            std::cout << "linkTable[" << i << "].base_addr= " << linkTable[i].base_addr;
            std::cout << ", linkTable[" << i << "].next = " << linkTable[i].next << std::endl;
        }
#endif
#endif
    }

    // pre-processing: output all string db high_addr to a FIFO
    ap_uint<12> str_lt_cur = 0;
    ap_uint<12> str_lt_tail = 0;
    ap_uint<12> str_high_addr = 0;

    int cnt[1 << W] = {0};
    // pre-store the string data used node info into str_high_addr_mem
    for (int idx = 0; idx < field_nm; idx++) {
        if (t_buff[idx] == TString) {
            str_lt_cur = f_buff[idx].head;
            str_lt_tail = f_buff[idx].tail;

        // no the last node
        STRW:
            while (str_lt_cur != str_lt_tail) {
#pragma HLS pipeline II = 1
                str_high_addr = linkTable[str_lt_cur].base_addr;
                str_high_addr_mem[idx][cnt[idx]++] = str_high_addr;
                str_lt_cur = linkTable[str_lt_cur].next;
            }
            // last node
            str_high_addr = linkTable[str_lt_cur].base_addr;
            str_high_addr_mem[idx][cnt[idx]++] = str_high_addr;
#ifdef _DF_DEBUG
#ifndef __SYNTHESIS__
            for (int t = 0; t < cnt[idx]; t++) {
                std::cout << "str_high_addr_mem[" << idx << "][" << t << "] = " << str_high_addr_mem[idx][t]
                          << std::endl;
            }
#endif
#endif
        }
    }
}

} // end of internal namespace

/**
 *
 * @brief read the data frame format data from DDR and pack into object streams
 *
 * @tparam field_type the data type of each field id (may obtained from schema)
 * @tparam ddr_buff the ddr that saves data
 * @tparam obj_strm output Object stream
 *
 **/
void readFromDataFrame(int field_type[17], ap_uint<64>* ddr_buff, hls::stream<Object>& obj_strm) {
    // define the filed_id number is 1 << W
    const int W = 4;

    // read in valid field_id number
    int field_nm = field_type[0];
#ifdef _DF_DEBUG
    std::cout << "valid field_nm = " << field_nm << std::endl;
#endif
    // read data type from each field_id, save to t_buff
    ap_uint<W> t_buff[1 << W];
    ap_uint<W> t_buff_w[1 << W];
    for (int i = 0; i < (1 << W); ++i) {
#pragma HLS pipeline II = 1
        t_buff[i] = field_type[i + 1];
        t_buff_w[i] = field_type[i + 1];
#ifdef _DF_DEBUG
        if (i < field_nm) std::cout << "valid field types: t_buff[" << i << "] = " << t_buff[i] << std::endl;
#endif
    }
#ifdef _DF_DEBUG
    std::cout << std::endl;
#endif

    // the total data size and offset size info of each field_id
    // w64 data size of each field_id:		s_buff[0]  s_buff[1]  ... s_buff[15]
    // string offset size of each field_id: s_buff[16] s_buff[17] ... s_buff[31]
    ap_uint<32> s_buff[32];
#pragma HLS BIND_STORAGE variable = s_buff type = ram_1p impl = lutram

    ap_uint<12> str_high_addr_mem[1 << W][256];
#pragma HLS BIND_STORAGE variable = str_high_addr_mem type = ram_2p impl = bram
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
    // the detailed link info of each mem block node
    Node linkTable[1024];
    // the link info between nodes
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
    internal::memToLocalRam<W>(field_nm, t_buff, s_buff, n_buff, l_buff, bit_map, bool_buff, linkTable, f_buff,
                               str_high_addr_mem, ddr_buff);

    ap_uint<32> nrow = l_buff[0];

    // read data field by field
    internal::readToObjStrm<4>(field_nm, nrow, t_buff, t_buff_w, f_buff, bit_map, bool_buff, linkTable,
                               str_high_addr_mem, ddr_buff, obj_strm);
}

} // end of dataframe namespace
} // end of data_analytics namespace
} // end of xf namespace

#endif
