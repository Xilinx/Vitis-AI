/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef GQE_AGGR_COMMAND_HPP
#define GQE_AGGR_COMMAND_HPP

#include "ap_int.h"
// L1
#include "xf_database/enums.hpp"
#include "xf_database/filter_config.hpp"
#include "xf_database/dynamic_alu_host.hpp"
#include "xf_database/dynamic_eval_v2_config.hpp"
// L2
#include "xf_database/gqe_utils.hpp"

namespace xf {
namespace database {

namespace gqe {

enum DIV_SCALE { nosf = 0, sf10 = 4, sf100 = 5, sf1k = 6, sf10k = 7 };

/**
 * @class AggrCommand aggr_command.hpp "xf_database/aggr_command.hpp"
 * @brief Class for generating configuration bits of ``gqeAggr`` kernel.
 */
class AggrCommand {
   private:
    enum { W = 32, size = 128 };
    ap_uint<W>* hbuf;
    ap_uint<W>* hbuf_out;

    //
    int setIndElems(int s_row, std::vector<int8_t> index, size_t max_num) {
        int validnum = 0;
        int offset = 0;
        int row_id = s_row;
        for (unsigned c = 0; c < index.size(); ++c) {
            validnum++;
            hbuf[row_id].range(offset + 7, offset) = *(index.begin() + c);
            offset += 8;
            if (offset == W) {
                offset = 0;
                row_id++;
            }
        }
        for (size_t c = index.size(); c < max_num; c++) {
            hbuf[row_id].range(offset + 7, offset) = -1;
            offset += 8;
            if (offset == W) {
                offset = 0;
                row_id++;
            }
        }
        return validnum;
    }

   public:
    /**
     * @brief construct of AggrCommand.
     *
     * The aggregate command used buffer is allocated on host.
     * Some default settings are provided.
     *
     */

    AggrCommand() {
        hbuf = gqe::utils::aligned_alloc<ap_uint<W> >(size);
        hbuf_out = gqe::utils::aligned_alloc<ap_uint<W> >(size);

        memset(hbuf, 0, sizeof(ap_uint<W>) * size);
        memset(hbuf_out, 0, sizeof(ap_uint<W>) * size);
        // filter defalut
        hbuf[66] = (1 << 31);
        // demux mux direct_aggr
        hbuf[81] = 0;
        // init aggr num, return value
        hbuf[78] = 0;
    };

    // hbuf[0x00~0x01]
    /**
     * @brief scan valid cols and shuffle for evaluation 1.
     *
     * @param index input valid column ids
     *
     */
    void Scan(std::vector<int8_t> index) {
        if (index.size() > 8) {
            std::cerr << "Most support 8 cols" << std::endl;
            exit(1);
        }
        setIndElems(0, index, 8);
    }

    /*
     *
     * 2. hbuf[0x02~0x11]
     *    hbuf[0x12~0x21]
     *
     * */

    /**
     * @brief set Evaluation string.
     *
     * @param eval_id most support two evaluations
     * @param s0 evaluation string for special strm format (not support '/')
     * @param index evaluation constants for each strm, default 0
     * @param post_div post division alu operation, nosf for no division, sf10
     * for "a/10", sf100 for "a/100",
     * sf1k for "a/1000", sf10k for "a/10000" (a is result of evaluation
     * expression)
     *
     */
    void setEvaluation(int eval_id, std::string s0, std::vector<int32_t> index, int post_div = nosf) {
        if (eval_id > 1) {
            std::cerr << "most support two evaluations" << std::endl;
            exit(1);
        }
        ap_uint<289> op = 0;
        std::vector<int32_t> conts(4);
        size_t c = 0;
        for (c = 0; c < index.size(); c++) {
            conts[c] = *(index.begin() + c);
        }
        while (c < 4) {
            conts[c] = 0;
            c++;
        }
        xf::database::dynamicALUOPCompiler<uint32_t, uint32_t, uint32_t, uint32_t>(s0.c_str(), conts[0], conts[1],
                                                                                   conts[2], conts[3], op);

        for (int i = 0; i < 9; i++) {
            hbuf[i + 2 + eval_id * 10] = op(32 * (i + 1) - 1, 32 * i);
        }
        hbuf[11 + eval_id * 10][0] = op[288];
        hbuf[11 + eval_id * 10](3, 1) = post_div;
    }

    // TODO update
    /**
     * @brief set Evaluation string.
     *
     * @param eval_id most two evaluations
     * @param eval_str evaluation expression string
     *
     */
    void setEvaluation(int eval_id, std::string eval_str) {
        if (eval_id > 1) {
            std::cerr << "eval_id can only be set 0/1" << std::endl;
            exit(1);
        }
        auto eval_bits = DynamicEvalV2Config(eval_str).getConfigBits();
        memcpy(&hbuf[eval_id * 10 + 2], eval_bits.get(), sizeof(uint32_t) * 8);
    }

    /*
     *
     * 3. hbuf[0x22~0x66]
     *
     * */
    /**
     * @brief set Filter string.
     *
     * The string uses ``a``, ``b``, ``c``, ``d`` to refer to first to the fourth
     * column,
     * and supports comparison and logical operator like C++.
     * Parentheses can be used to group logic, but notice that the underlying
     * hardware module
     * does not support comparing one column with multiple constant boundaries in
     * OR relationship.
     * Integral constants will be extracted from expression.
     *
     * For example, an expression could be ``(a < 10 && b < 20) || (c >= d)``.
     *
     * @param filter_string filter expression string
     *
     */
    void setFilter(std::string filter_string) {
        FilterConfig<32> fp(filter_string);
        auto cfg = fp.getConfigBits();
        memcpy(&hbuf[22], cfg.get(), sizeof(uint32_t) * 45);
    }

    // TODO deprecate
    void setFilter(void (*get_filter_func)(uint32_t cfg[])) {
        uint32_t cfg[45];
        get_filter_func(cfg);
        memcpy(&hbuf[22], cfg, sizeof(uint32_t) * 45);
    }

    // hbuf 67~68, for eval2 keys
    /**
     * @brief set shuffle0 column ids.
     *
     *
     * @param index shuffle0 key list,move evaluation keys to first 4 locations
     *
     */
    void setShuffle0(std::vector<int8_t> index) { setIndElems(67, index, 8); }

    // hbuf 69~70, for filter keys
    /**
     * @brief set shuffle1 column ids.
     *
     * @param index shuffle1 key list,move filter keys to first 4 locations
     *
     */
    void setShuffle1(std::vector<int8_t> index) { setIndElems(69, index, 8); }

    // hbuf 71~72, output keys
    // 76 -- key num, set in shuffle2
    /**
     * @brief set shuffle2 column ids.
     *
     * @param index set column indexes of keys
     *
     */
    void setShuffle2(std::vector<int8_t> index) {
        int validnum = setIndElems(71, index, 8);
        hbuf[76] = validnum;
    }

    // hbuf 73~74, output pld
    // 77 -- pld num, set in shuffle3
    /**
     * @brief set shuffle3 column ids.
     *
     * @param index set column indexes of payloads
     *
     */
    void setShuffle3(std::vector<int8_t> index) {
        int validnum = setIndElems(73, index, 8);
        hbuf[77] = validnum;
    }

    // 75-78
    // 75 -- aggr type of each pld
    // 76 -- key num, set in shuffle2
    // 77 -- pld num, set in shuffle3
    // 78 -- default 0 -- aggr num

    /**
     * @brief set Group Aggregation type of setting index
     *
     * @param ind set index of column
     * @param aggr_op Aggregation type of setting index
     *
     */
    void setGroupAggr(int ind, ap_uint<4> aggr_op) {
        int aggr_op_id = ind * 4;
        hbuf[75].range(aggr_op_id + 3, aggr_op_id) = aggr_op;
    }

    /**
     * @brief set Group Aggregation types
     *
     * @param index list of Aggregation types, starting from the first column,
     * align with columns in shuffle3
     *
     */
    void setGroupAggrs(std::vector<ap_uint<4> > index) {
        ap_uint<4> aggr_op[8];
        for (size_t i = 0; i < index.size(); i++) {
            aggr_op[i] = *(index.begin() + i); // xf::database::enums::AOP_MEAN;
        }
        for (int i = index.size(); i < 8; i++) {
            aggr_op[i] = xf::database::enums::AOP_COUNT; // xf::database::enums::AOP_COUNT;
        }
        hbuf[75] = (aggr_op[7], aggr_op[6], aggr_op[5], aggr_op[4], aggr_op[3], aggr_op[2], aggr_op[1], aggr_op[0]);
    }

    // 79-81
    /**
     * @brief set Merge column index in setting round of merge
     *
     * merge flow :
     * four list : key, count, aggr_l and aggr_h, each one has size 8
     * key is aligned with shuffle2, count, aggr_l and aggr_h are aligned with
     * shuffle3 and setGroupAggr
     * merge 0,1,2 merge key into the other three
     * merge 3,4   merge count pld into other two
     *
     * @param m_id round id of column merge
     * @param index merge colum list in current round
     *
     */
    void setMerge(int m_id, std::vector<uint8_t> index) {
        if (m_id > 4) {
            std::cout << "Most 5 merges" << std::endl;
            return;
        }
        ap_uint<8> merge = 0;
        for (size_t i = 0; i < index.size(); i++) {
            int bit_ind = *(index.begin() + i);
            merge.range(bit_ind, bit_ind) = 1;
        }
        int r = m_id / 3;
        int c = (m_id % 3) * 8;

        hbuf[r + 79].range(c + 7, c) = merge;
    }

    /**
     * @brief set Merge codes in each round of merge
     *
     * merge flow :
     * four list : key, count, aggr_h, aggr_l (each one has size 8)
     * key is aligned with shuffle2, count, aggr_h and aggr_l are aligned with
     * shuffle3 and setGroupAggr
     * merge 0,1,2 merge key into the other three
     * merge 3,4   merge count pld into other two
     *
     * @param index coded merging indexes
     *
     */
    void columnMerge(std::vector<ap_uint<4> > index) {
        ap_uint<4> merge[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < index.size(); i++) {
            merge[i] = *(index.begin() + i);
        }
        hbuf[79] = (0, merge[2], merge[1], merge[0]);
        hbuf[80] = (0, merge[4], merge[3]);
    }

    // 81
    /**
     * @brief set demux direct aggregations
     *
     * @param index columns of using direct aggregations
     *
     */
    void setDirectAggrs(std::vector<uint8_t> index) {
        int32_t value = 0;
        if (index.size() > 16) {
            std::cerr << "Error,max supporting col number is 16" << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < index.size(); i++) {
            value += (1 << *(index.begin() + i));
        }
        hbuf[81] = value;
    }

    // 82
    /**
     * @brief set output column ids.
     *
     * @param index output column ids
     *
     */
    void setWriteCol(std::vector<int8_t> index) {
        int32_t value = 0;
        if (index.size() > 16) {
            std::cerr << "Error,max supporting col number is 16" << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < index.size(); i++) {
            value += (1 << *(index.begin() + i));
        }
        hbuf[82] = value;
    }

    /**
     * @brief return config bit array.
     *
     */
    ap_uint<W>* getConfigBits() const { return hbuf; }

    /**
     * @brief return config out bit array.
     *
     */
    ap_uint<W>* getConfigOutBits() { return hbuf_out; };

#ifdef USER_DEBUG
    // to test
    void cmpConfig(ap_uint<32>* cmpbuf, int len = 83) {
        int difnum = 0;
        for (int i = 0; i < len; i++) {
            ap_uint<32> l = hbuf[i];
            ap_uint<32> r = cmpbuf[i];
            ap_uint<32> l_xor_r = l ^ r;
            if (l_xor_r != 0) {
                for (int k = 0; k < 32; k++) {
                    difnum += ((l_xor_r >> k) & 0x01);
                    if (((l_xor_r >> k) & 0x01)) {
                        std::cout << k << " ";
                    }
                }
                std::cout << "hbufs differ in " << i << " row" << std::endl;
            }
        }
        if (!difnum)
            std::cout << "Test Right!" << std::endl;
        else {
            std::cout << "Total " << difnum << " elems" << std::endl;
        }
    }
#endif
};

} // namespace gqe

} // namespace database
} // namespace xf

#endif // GQE_AGGR_COMMAND_HPP
