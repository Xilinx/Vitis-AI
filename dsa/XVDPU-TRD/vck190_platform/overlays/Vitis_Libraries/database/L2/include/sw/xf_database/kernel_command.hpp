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

#ifndef GQE_KERNEL_COMMAND_HPP
#define GQE_KERNEL_COMMAND_HPP

#include "ap_int.h"
// L1
#include "xf_database/enums.hpp"
#include "xf_database/filter_config.hpp"
#include "xf_database/dynamic_alu_host.hpp"
#include "xf_database/dynamic_eval_v2_config.hpp"
// L2
#include "xf_database/gqe_utils.hpp"

//#define USER_DEBUG 1

namespace xf {
namespace database {

enum JoinType { INNER_JOIN = 0, SEMMI_JOIN = 1, ANTI_JOIN = 2 };

namespace gqe {

/**
 * @class KernelCommand kernel_command.hpp "xf_database/kernel_command.hpp"
 * @brief Class for generating configuration bits of ``gqePart``, ``gqeJoin`` and  ``gqeFilter`` kernels.
 */
class KernelCommand {
   private:
    ap_uint<512>* hbuf;
    int joinType;
    bool isBypass;
    bool isAggrOn;
    bool isDualKeyOn;
    bool isBloomfilterOn;

   public:
    /**
    * @brief constructor of KernelCommand class. The kernel command used buffer is allocated on host. Some default
    * settings are provided.
    */
    KernelCommand() {
        hbuf = gqe::utils::aligned_alloc<ap_uint<512> >(14);
        memset(hbuf, 0, sizeof(ap_uint<512>) * 14);
        hbuf[0] = 1;
        hbuf[9].range(159, 159) = 1;
        hbuf[13].range(159, 159) = 1;
        isBypass = false;
        isAggrOn = false;
        isDualKeyOn = false;
    }
    ~KernelCommand() { free(hbuf); }

    /**
     * @brief scan valid cols.
     *
     * @param krn_id 0 for gqeJoin, 1 for gqePart, 2 for gqeFilter.
     * @param table_id 0 for left table, 1 for right table.
     * @param index valid input column ids.
     *
     */
    void setScanColEnable(int krn_id, int table_id, std::vector<int8_t> index) {
        if (table_id > 1) {
            std::cerr << "table_id can only be set 0/1" << std::endl;
            exit(1);
        }
        if (table_id == 0) {
            for (size_t d = 0; d < index.size(); d++) {
                // if (index[d] != -1) hbuf[krn_id].set_bit((6 + index[d]), 1);
                if (index[d] != -1) hbuf[krn_id].set_bit((6 + d), 1);
            }
        } else {
            for (size_t d = 0; d < index.size(); d++) {
                // if (index[d] != -1) hbuf[krn_id].set_bit((9 + index[d]), 1);
                if (index[d] != -1) hbuf[krn_id].set_bit((9 + d), 1);
            }
        }
#ifdef USER_DEBUG
        if (table_id == 0)
            std::cout << "table A enable col / hbuf[krn_id].range(8, 6): " << hbuf[krn_id].range(8, 6) << std::endl;
        else
            std::cout << "table B enable col / hbuf[krn_id].range(11, 9): " << hbuf[krn_id].range(11, 9) << std::endl;
#endif
    }

    /**
     * @brief set gen_rowIDEnable and validEnable flag.
     *
     * @param krn_id 0 for gqeJoin, 1 for gqePart, 2 for gqeFilter.
     * @param table_id 0 for left table, 1 for right table.
     * @param gen_rowID_en enable flag for using GQE to generate row IDs internally. 1 for enable, 0 for disable.
     * @param valid_en enable flag for getting valid bits from off-chip memory or enabing every row internally. 1 for
     * valid bits from off-chip memory, 0 for enabling every row.
     *
     */
    void setRowIDValidEnable(int krn_id, int table_id, bool gen_rowID_en, bool valid_en) {
        if (krn_id == 0 || krn_id == 2) {
            // gen-rowid for tab
            hbuf[krn_id].set_bit(16 + 2 * table_id, gen_rowID_en);
            // valid_en for tab
            hbuf[krn_id].set_bit(17 + 2 * table_id, valid_en);
        } else if (krn_id == 1) {
            hbuf[krn_id].set_bit(19 + 2 * table_id, gen_rowID_en);
            hbuf[krn_id].set_bit(20 + 2 * table_id, valid_en);
        }
    }
    /**
     * @brief set bypass on.
     */
    void setBypassOn() {
        isBypass = true;
        hbuf[0].set_bit(0, 0);
        hbuf[1].set_bit(0, 0);
        hbuf[2].set_bit(0, 0);
    }

    /**
     * @brief set bloom-filter on.
     *
     * @param bf_size size of bloom-filter in bits, we need 35 bits to represent a range between 1 to 16 Gbits.
     *
     */
    void setBloomfilterOn(ap_uint<35> bf_size) {
        isBloomfilterOn = true;
        // since bloom-filter is merged into gqeJoin
        // should use Join config bit instead of Filter
        hbuf[0].set_bit(1, 1);
        hbuf[2].range(54, 20) = bf_size;
    }

    /**
     * @brief set dual key on.
     *
     */
    void setDualKeyOn() {
        isDualKeyOn = true;
        hbuf[0].set_bit(2, 1); // set dual key for gqeJoin
        hbuf[1].set_bit(2, 1); // set dual key for gqePart
        hbuf[2].set_bit(2, 1); // set dual key for gqeFilter
    }

    /**
     * @brief set join type.
     *
     * @param jointype Join type, default is INNER_JOIN.
     */
    void setJoinType(int jointype) {
        joinType = jointype;
        hbuf[0].range(4, 3) = jointype;
    }

    /**
     * @brief enables output columns for gqeJoin/Filter.
     *
     * @param krn_id 0 for gqeJoin, 1 for gqePart, 2 for gqeFilter.
     * @param table_id 0 for left table, 1 for right table.
     * @param index output column ids.
     *
     */
    void setJoinWriteColEnable(int krn_id, int table_id, std::vector<int8_t> index) {
        int8_t value = 0;
        // gqeJoin / gqeFilter setup
        if (krn_id == 0 || krn_id == 2) {
            if (index.size() > 4) {
                std::cerr << "Error,max supporting col number is 4" << std::endl;
                exit(1);
            }
            for (size_t i = 0; i < index.size(); i++) {
                if (index[i] >= 0) value += (1 << i);
            }
            hbuf[krn_id].range(15, 12) = value;
            // gqePart setup
        } else {
            if (index.size() > 3) {
                std::cerr << "Error,max supporting col number is 3" << std::endl;
                exit(1);
            }
            for (size_t i = 0; i < index.size(); i++) {
                if (index[i] >= 0) value += (1 << i);
            }
            if (table_id) {
                // L table
                hbuf[krn_id].range(18, 16) = value;
            } else {
                // O table
                hbuf[krn_id].range(14, 12) = value;
            }
        }
    }

    /**
     * @brief enables output columns for gqePart.
     *
     * @param krn_id 0 for gqeJoin, 1 for gqePart, 2 for gqeFilter
     * @param table_id 0 for left table, 1 for right table
     * @param index output column ids
     *
     */
    void setPartWriteColEnable(int krn_id, int table_id, std::vector<int8_t> index) {
        int8_t value = 0;
        // gqeJoin / gqeFilter setup
        if (krn_id == 0 || krn_id == 2) {
            if (index.size() > 4) {
                std::cerr << "Error,max supporting col number is 4" << std::endl;
                exit(1);
            }
            for (size_t i = 0; i < index.size(); i++) {
                if (index[i] >= 0) value += (1 << index[i]);
            }
            hbuf[krn_id].range(15, 12) = value;
            // gqePart setup
        } else {
            if (index.size() > 3) {
                std::cerr << "Error,max supporting col number is 3" << std::endl;
                exit(1);
            }
            for (size_t i = 0; i < index.size(); i++) {
                if (index[i] >= 0) value += (1 << index[i]);
            }
            if (table_id) {
                // L table
                hbuf[krn_id].range(18, 16) = value;
            } else {
                // O table
                hbuf[krn_id].range(14, 12) = value;
            }
        }
    }

    /*
     *
     * 3. hbuf[0x06~0x13]
     *
     * */
    /**
     * @brief set Filter string.
     *
     * The string uses ``a``, ``b``, ``c``, ``d`` to refer to first to the fourth column,
     * and supports comparison and logical operator like C++.
     * Parentheses can be used to group logic, but notice that the underlying hardware module
     * does not support comparing one column with multiple constant boundaries in OR relationship.
     * Integral constants will be extracted from expression.
     *
     * For example, an expression could be ``(a < 10 && b < 20) || (c >= d)``.
     *
     * @param table_id 0 for left table, 1 for right table
     * @param filter_string filter expression string
     *
     */
    void setFilter(int table_id, std::string filter_string) {
        FilterConfig<64> fp(filter_string);
        auto cfg = fp.getConfigBits();
        memcpy(&hbuf[table_id * 4 + 6], cfg.get(), sizeof(uint32_t) * 53);
#ifdef USER_DEBUG
        std::cout << "setFilter cfg---------" << std::endl;
        for (int i = 0; i < 53; i++) {
            std::cout << cfg[i] << std::endl;
        }
        std::cout << "setFilter cfg end ---------" << std::endl;
#endif
    }

    /**
     * @brief return config bit array.
     *
     */
    ap_uint<512>* getConfigBits() const { return hbuf; }

#ifdef USER_DEBUG
    /*
        std::shared_ptr<ap_uint<512> > getConfigBits() const {
            std::shared_ptr<ap_uint<512> > cfgs(hbuf);
            return cfgs;
        }
    */
    /**
     * @brief for comparing the current configurations with the original ones.
     *
     * @param cmpbuf input configurations to be compared.
     * @param len length of the configurations, currently we have 14 * 512-bits size of kernel configurations.
    */
    void cmpConfig(ap_uint<512>* cmpbuf, int len = 14) {
        int difnum = 0;
        for (int i = 0; i < len; i++) {
            ap_uint<512> l = hbuf[i];
            ap_uint<512> r = cmpbuf[i];
            ap_uint<512> l_xor_r = l ^ r;
            if (l_xor_r != 0) {
                for (int k = 0; k < 512; k++) {
                    difnum += ((l_xor_r >> k) & 0x01);
                    if (((l_xor_r >> k) & 0x01)) {
                        std::cout << k << " ";
                    }
                }
                std::cout << "hbufs differ in " << i << " row" << std::endl;
            }
        }
        if (!difnum) {
            std::cout << "Test Right!" << std::endl;
        } else {
            std::cout << "Total " << difnum << " elems" << std::endl;
        }
    }
#endif
};

} // namespace gqe
} // namespace database
} // namespace xf

#endif // GQE_JOIN_COMMAND_HPP
