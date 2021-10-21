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
#ifndef PREPARE_HPP
#define PREPARE_HPP
#include "tpch_read_2.hpp"
#include "x_utils.hpp"

#include <cstdio>
#include <fstream>
// C++11 thread
#include <thread>
#include <cstring>

// ------------------------------------------------------------
//
void generate_tbl(const std::string& dbgen_dir, const std::string& tbl_dir, const int sf) {
    std::string cmd = "git clone https://github.com/electrum/ssb-dbgen.git " + dbgen_dir;
    system(cmd.c_str());
    cmd = "sed -i 's/MACHINE =MAC/MACHINE =LINUX/' " + dbgen_dir + "/makefile";
    system(cmd.c_str());
    cmd = "sed -i 's/WORKLOAD =SSBM/WORKLOAD =TPCH/' " + dbgen_dir + "/makefile";
    system(cmd.c_str());
    cmd = "make -C " + dbgen_dir + " 1>" + dbgen_dir + "/mk.log 2>&1";
    system(cmd.c_str());
    cmd = "cd " + dbgen_dir + " && ./dbgen -f -s " + std::to_string(sf);
    system(cmd.c_str());
    cmd = "mkdir -p " + tbl_dir;
    system(cmd.c_str());
    cmd = "mv " + dbgen_dir + "/*.tbl " + tbl_dir;
    system(cmd.c_str());
    cmd = "rm -rf " + dbgen_dir;
    system(cmd.c_str());
}

template <typename T>
void read_tbl(std::string _path, std::vector<T>& _vec) {
    std::ifstream ifs;
    ifs.open(_path, std::ios_base::in);
    if (!ifs) {
        printf("ERROR: %s cannot ben opened for read.\n", _path.c_str());
        return;
    }
    // read data line
    while (ifs) {
        T t;
        ifs >> t;
        if (ifs) {
            _vec.push_back(t);
        }
    }
    ifs.close();
    printf("INFO: Loaded %s from disk.\n", _path.c_str());
}

// ------------------------------------------------------------

void columnize_r(region_t* buf_r,
                 size_t nrow, //
                 TPCH_INT* col_r_regionkey,
                 std::vector<std::array<char, TPCH_READ_REGION_LEN + 1> >& col_r_name,
                 std::vector<std::array<char, TPCH_READ_R_CMNT_MAX + 1> >& col_r_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_r_regionkey[i] = buf_r[i].regionkey;
        memcpy(col_r_name[i].data(), buf_r[i].name.data, TPCH_READ_REGION_LEN + 1);
        memcpy(col_r_comment[i].data(), buf_r[i].comment.data, TPCH_READ_R_CMNT_MAX + 1);
    }
}

void columnize_n(nation_t* buf_n,
                 size_t nrow, //
                 TPCH_INT* col_n_nationkey,
                 TPCH_INT* col_n_regionkey,
                 std::vector<std::array<char, TPCH_READ_NATION_LEN + 1> >& col_n_name,
                 std::vector<std::array<char, TPCH_READ_N_CMNT_MAX + 1> >& col_n_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_n_nationkey[i] = buf_n[i].nationkey;
        col_n_regionkey[i] = buf_n[i].regionkey;
        memcpy(col_n_name[i].data(), buf_n[i].name.data, TPCH_READ_NATION_LEN + 1);
        memcpy(col_n_comment[i].data(), buf_n[i].name.data, TPCH_READ_N_CMNT_MAX + 1);
    }
}

void columnize_c(customer_t* buf_c,
                 size_t nrow, //
                 TPCH_INT* col_c_custkey,
                 std::vector<std::array<char, TPCH_READ_C_NAME_LEN + 1> >& col_c_name,
                 std::vector<std::array<char, TPCH_READ_S_ADDR_MAX + 1> >& col_c_address,
                 TPCH_INT* col_c_nationkey,
                 std::vector<std::array<char, TPCH_READ_PHONE_LEN + 1> >& col_c_phone,
                 TPCH_INT* col_c_acctbal,
                 std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> >& col_c_mktsegment,
                 std::vector<std::array<char, TPCH_READ_S_CMNT_MAX + 1> >& col_c_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_c_custkey[i] = buf_c[i].custkey;
        col_c_nationkey[i] = buf_c[i].nationkey;
        col_c_acctbal[i] = buf_c[i].acctbal;
        memcpy(col_c_mktsegment[i].data(), buf_c[i].mktsegment.data, TPCH_READ_MAXAGG_LEN + 1);
        memcpy(col_c_name[i].data(), buf_c[i].name.data, TPCH_READ_C_NAME_LEN + 1);
        memcpy(col_c_address[i].data(), buf_c[i].address.data, TPCH_READ_S_ADDR_MAX + 1);
        memcpy(col_c_phone[i].data(), buf_c[i].phone.data, TPCH_READ_PHONE_LEN + 1);
        memcpy(col_c_comment[i].data(), buf_c[i].comment.data, TPCH_READ_S_CMNT_MAX + 1);
    }
}

void columnize_o(orders_t* buf_o,
                 size_t nrow, //
                 TPCH_INT* col_o_orderkey,
                 TPCH_INT* col_o_custkey,
                 TPCH_INT* col_o_orderstatus,
                 TPCH_INT* col_o_totalprice,
                 TPCH_INT* col_o_orderdate,
                 std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> >& col_o_orderpriority,
                 std::vector<std::array<char, TPCH_READ_O_CLRK_LEN + 1> >& col_o_clerk,
                 TPCH_INT* col_o_shippriority,
                 std::vector<std::array<char, TPCH_READ_O_CMNT_MAX + 1> >& col_o_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_o_orderkey[i] = buf_o[i].orderkey;
        col_o_custkey[i] = buf_o[i].custkey;
        col_o_orderstatus[i] = buf_o[i].orderstatus.data[0];
        col_o_totalprice[i] = buf_o[i].totalprice;
        col_o_orderdate[i] = buf_o[i].orderdate;
        memcpy(col_o_orderpriority[i].data(), buf_o[i].orderpriority.data, TPCH_READ_MAXAGG_LEN + 1);
        memcpy(col_o_clerk[i].data(), buf_o[i].clerk.data, TPCH_READ_O_CLRK_LEN + 1);
        col_o_shippriority[i] = buf_o[i].shippriority;
        memcpy(col_o_comment[i].data(), buf_o[i].comment.data, TPCH_READ_O_CMNT_MAX + 1);
    }
}

void columnize_l(lineitem_t* buf_l,
                 size_t nrow, //
                 TPCH_INT* col_l_orderkey,
                 TPCH_INT* col_l_partkey,
                 TPCH_INT* col_l_suppkey,
                 TPCH_INT* col_l_linenumber,
                 TPCH_INT* col_l_quantity,
                 TPCH_INT* col_l_extendedprice,
                 TPCH_INT* col_l_discount,
                 TPCH_INT* col_l_tax,
                 TPCH_INT* col_l_returnflag,
                 TPCH_INT* col_l_linestatus,
                 TPCH_INT* col_l_shipdate,
                 TPCH_INT* col_l_commitdate,
                 TPCH_INT* col_l_receiptdate,
                 std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> >& col_l_shipinstruct,
                 std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> >& col_l_shipmode,
                 std::vector<std::array<char, TPCH_READ_L_CMNT_MAX + 1> >& col_l_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_l_orderkey[i] = buf_l[i].orderkey;
        col_l_partkey[i] = buf_l[i].partkey;
        col_l_suppkey[i] = buf_l[i].suppkey;
        col_l_linenumber[i] = buf_l[i].linenumber;
        col_l_quantity[i] = buf_l[i].quantity;
        col_l_extendedprice[i] = buf_l[i].extendedprice;
        col_l_discount[i] = buf_l[i].discount;
        col_l_tax[i] = buf_l[i].tax;
        col_l_linestatus[i] = (int)buf_l[i].linestatus.data[0];
        col_l_returnflag[i] = (int)buf_l[i].returnflag.data[0];
        col_l_shipdate[i] = buf_l[i].shipdate;
        col_l_commitdate[i] = buf_l[i].commitdate;
        col_l_receiptdate[i] = buf_l[i].receiptdate;
        memcpy(col_l_shipinstruct[i].data(), buf_l[i].shipinstruct.data, TPCH_READ_MAXAGG_LEN + 1);
        memcpy(col_l_shipmode[i].data(), buf_l[i].shipmode.data, TPCH_READ_MAXAGG_LEN + 1);
        memcpy(col_l_comment[i].data(), buf_l[i].comment.data, TPCH_READ_L_CMNT_MAX + 1);
    }
}

void columnize_s(supplier_t* buf_s,
                 size_t nrow, //
                 TPCH_INT* col_s_suppkey,
                 std::vector<std::array<char, TPCH_READ_S_NAME_LEN + 1> >& col_s_name,
                 std::vector<std::array<char, TPCH_READ_S_ADDR_MAX + 1> >& col_s_address,
                 TPCH_INT* col_s_nationkey,
                 std::vector<std::array<char, TPCH_READ_PHONE_LEN + 1> >& col_s_phone,
                 TPCH_INT* col_s_acctbal,
                 std::vector<std::array<char, TPCH_READ_S_CMNT_MAX + 1> >& col_s_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_s_suppkey[i] = buf_s[i].suppkey;
        memcpy(col_s_name[i].data(), buf_s[i].name.data, TPCH_READ_S_NAME_LEN + 1);
        memcpy(col_s_address[i].data(), buf_s[i].address.data, TPCH_READ_S_ADDR_MAX + 1);
        col_s_nationkey[i] = buf_s[i].nationkey;
        memcpy(col_s_phone[i].data(), buf_s[i].phone.data, TPCH_READ_PHONE_LEN + 1);
        col_s_acctbal[i] = buf_s[i].acctbal;
        memcpy(col_s_comment[i].data(), buf_s[i].comment.data, TPCH_READ_S_CMNT_MAX + 1);
    }
}

void columnize_p(part_t* buf_p,
                 size_t nrow, //
                 TPCH_INT* col_p_partkey,
                 std::vector<std::array<char, TPCH_READ_P_NAME_LEN + 1> >& col_p_name,
                 std::vector<std::array<char, TPCH_READ_P_MFG_LEN + 1> >& col_p_mfgr,
                 std::vector<std::array<char, TPCH_READ_P_BRND_LEN + 1> >& col_p_brand,
                 std::vector<std::array<char, TPCH_READ_P_TYPE_LEN + 1> >& col_p_type,
                 TPCH_INT* col_p_size,
                 std::vector<std::array<char, TPCH_READ_P_CNTR_LEN + 1> >& col_p_container,
                 TPCH_INT* col_p_retailprice,
                 std::vector<std::array<char, TPCH_READ_P_CMNT_MAX + 1> >& col_p_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_p_partkey[i] = buf_p[i].partkey;
        memcpy(col_p_name[i].data(), buf_p[i].name.data, TPCH_READ_P_NAME_LEN + 1);
        memcpy(col_p_mfgr[i].data(), buf_p[i].mfgr.data, TPCH_READ_P_MFG_LEN + 1);
        memcpy(col_p_brand[i].data(), buf_p[i].brand.data, TPCH_READ_P_BRND_LEN + 1);
        memcpy(col_p_type[i].data(), buf_p[i].type.data, TPCH_READ_P_TYPE_LEN + 1);
        col_p_size[i] = buf_p[i].size;
        memcpy(col_p_container[i].data(), buf_p[i].container.data, TPCH_READ_P_CNTR_LEN + 1);

        col_p_retailprice[i] = buf_p[i].retailprice;
        memcpy(col_p_comment[i].data(), buf_p[i].comment.data, TPCH_READ_P_CMNT_MAX + 1);
    }
}

void columnize_ps(partsupp_t* buf_ps,
                  size_t nrow, //
                  TPCH_INT* col_ps_partkey,
                  TPCH_INT* col_ps_suppkey,
                  TPCH_INT* col_ps_availqty,
                  TPCH_INT* col_ps_supplycost,
                  std::vector<std::array<char, TPCH_READ_PS_CMNT_MAX + 1> >& col_ps_comment) {
    for (size_t i = 0; i < nrow; ++i) {
        col_ps_partkey[i] = buf_ps[i].partkey;
        col_ps_suppkey[i] = buf_ps[i].suppkey;
        col_ps_availqty[i] = buf_ps[i].availqty;
        col_ps_supplycost[i] = buf_ps[i].supplycost;
        memcpy(col_ps_comment[i].data(), buf_ps[i].comment.data, TPCH_READ_PS_CMNT_MAX + 1);
    }
}

// ------------------------------------------------------------

// template <typename T, std::enable_if<std::is_integral<T>::value, int>::type = 0>
template <typename T>
int write_to_file(const std::string& fn, const std::vector<T>& d) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) {
        printf("ERROR: %s cannot be opened for write.\n", fn.c_str());
        return 1;
    }
    int n = fwrite(d.data(), sizeof(T), d.size(), f);
    fclose(f);
    return n;
}

template <int N>
int write_to_file(const std::string& fn, const std::vector<std::array<char, N> >& d) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) {
        printf("ERROR: %s cannot be opened for write.\n", fn.c_str());
        return 1;
    }
    int n = 0;
    for (int i = 0; i < d.size(); ++i) {
        n += fwrite(d[i].data(), N, 1, f);
    }
    fclose(f);
    return n;
}

int convert_dat(const std::string& in_dir, const std::string& out_dir) {
    // cmd arg parser.
    std::string cmd = "mkdir -p " + out_dir;
    system(cmd.c_str());

    int err = 0;

    if (!x_utils::is_dir(in_dir)) {
        printf("ERROR: \"%s\" is not a directory!\n", in_dir.c_str());
        ++err;
    }

    std::string r_path = in_dir + "/region.tbl";
    if (!x_utils::is_file(r_path)) {
        printf("ERROR: \"%s\" is not a file!\n", r_path.c_str());
        ++err;
    }
    std::string n_path = in_dir + "/nation.tbl";
    if (!x_utils::is_file(n_path)) {
        printf("ERROR: \"%s\" is not a file!\n", n_path.c_str());
        ++err;
    }
    std::string c_path = in_dir + "/customer.tbl";
    if (!x_utils::is_file(c_path)) {
        printf("ERROR: \"%s\" is not a file!\n", c_path.c_str());
        ++err;
    }
    std::string o_path = in_dir + "/orders.tbl";
    if (!x_utils::is_file(o_path)) {
        printf("ERROR: \"%s\" is not a file!\n", o_path.c_str());
        ++err;
    }
    std::string l_path = in_dir + "/lineitem.tbl";
    if (!x_utils::is_file(l_path)) {
        printf("ERROR: \"%s\" is not a file!\n", l_path.c_str());
        ++err;
    }
    std::string s_path = in_dir + "/supplier.tbl";
    if (!x_utils::is_file(s_path)) {
        printf("ERROR: \"%s\" is not a file!\n", s_path.c_str());
        ++err;
    }

    std::string p_path = in_dir + "/part.tbl";
    if (!x_utils::is_file(p_path)) {
        printf("ERROR: \"%s\" is not a file!\n", p_path.c_str());
        ++err;
    }

    std::string ps_path = in_dir + "/partsupp.tbl";
    if (!x_utils::is_file(ps_path)) {
        printf("ERROR: \"%s\" is not a file!\n", ps_path.c_str());
        ++err;
    }

    if (err) return err;

    // set up input data and buffers

    std::vector<region_t> r_vec;
    std::vector<nation_t> n_vec;
    std::vector<customer_t> c_vec;
    std::vector<orders_t> o_vec;
    std::vector<lineitem_t> l_vec;
    std::vector<supplier_t> s_vec;
    std::vector<part_t> p_vec;
    std::vector<partsupp_t> ps_vec;

    std::thread r_thread;
    std::thread n_thread;
    std::thread c_thread;
    std::thread o_thread;
    std::thread l_thread;
    std::thread s_thread;
    std::thread p_thread;
    std::thread ps_thread;

    struct timeval tv0, tv1;
    int usec;

    gettimeofday(&tv0, 0);

    r_thread = std::thread(read_tbl<region_t>, r_path, std::ref(r_vec));
    n_thread = std::thread(read_tbl<nation_t>, n_path, std::ref(n_vec));
    c_thread = std::thread(read_tbl<customer_t>, c_path, std::ref(c_vec));
    o_thread = std::thread(read_tbl<orders_t>, o_path, std::ref(o_vec));
    l_thread = std::thread(read_tbl<lineitem_t>, l_path, std::ref(l_vec));
    s_thread = std::thread(read_tbl<supplier_t>, s_path, std::ref(s_vec));
    p_thread = std::thread(read_tbl<part_t>, p_path, std::ref(p_vec));
    ps_thread = std::thread(read_tbl<partsupp_t>, ps_path, std::ref(ps_vec));

    r_thread.join();
    n_thread.join();
    c_thread.join();
    o_thread.join();
    l_thread.join();
    s_thread.join();
    p_thread.join();
    ps_thread.join();

    gettimeofday(&tv1, 0);
    usec = x_utils::tvdiff(tv0, tv1);
    printf("Time to load table: %d usec.\n", usec);

    size_t r_nrow = r_vec.size();
    std::vector<TPCH_INT> col_r_regionkey(r_nrow);
    std::vector<std::array<char, TPCH_READ_REGION_LEN + 1> > col_r_name(r_nrow);
    std::vector<std::array<char, TPCH_READ_N_CMNT_MAX + 1> > col_r_comment(r_nrow);

    size_t n_nrow = n_vec.size();
    std::vector<TPCH_INT> col_n_nationkey(n_nrow);
    std::vector<TPCH_INT> col_n_regionkey(n_nrow);
    std::vector<std::array<char, TPCH_READ_NATION_LEN + 1> > col_n_name(n_nrow);
    std::vector<std::array<char, TPCH_READ_N_CMNT_MAX + 1> > col_n_comment(n_nrow);

    size_t c_nrow = c_vec.size();
    std::vector<TPCH_INT> col_c_custkey(c_nrow);
    std::vector<TPCH_INT> col_c_nationkey(c_nrow);
    std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> > col_c_mktsegment(c_nrow);
    std::vector<std::array<char, TPCH_READ_C_NAME_LEN + 1> > col_c_name(c_nrow);
    std::vector<TPCH_INT> col_c_acctbal(c_nrow);
    std::vector<std::array<char, TPCH_READ_S_ADDR_MAX + 1> > col_c_address(c_nrow);
    std::vector<std::array<char, TPCH_READ_PHONE_LEN + 1> > col_c_phone(c_nrow);
    std::vector<std::array<char, TPCH_READ_S_CMNT_MAX + 1> > col_c_comment(c_nrow);

    size_t o_nrow = o_vec.size();
    std::vector<TPCH_INT> col_o_orderkey(o_nrow);
    std::vector<TPCH_INT> col_o_custkey(o_nrow);
    std::vector<TPCH_INT> col_o_orderstatus(o_nrow);
    std::vector<TPCH_INT> col_o_totalprice(o_nrow);
    std::vector<TPCH_INT> col_o_orderdate(o_nrow);
    std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> > col_o_orderpriority(o_nrow);
    std::vector<std::array<char, TPCH_READ_O_CLRK_LEN + 1> > col_o_clerk(o_nrow);
    std::vector<TPCH_INT> col_o_shippriority(o_nrow);
    std::vector<std::array<char, TPCH_READ_O_CMNT_MAX + 1> > col_o_comment(o_nrow);

    size_t l_nrow = l_vec.size();
    std::vector<TPCH_INT> col_l_orderkey(l_nrow);
    std::vector<TPCH_INT> col_l_partkey(l_nrow);
    std::vector<TPCH_INT> col_l_suppkey(l_nrow);
    std::vector<TPCH_INT> col_l_linenumber(l_nrow);
    std::vector<TPCH_INT> col_l_quantity(l_nrow);
    std::vector<TPCH_INT> col_l_extendedprice(l_nrow);
    std::vector<TPCH_INT> col_l_discount(l_nrow);
    std::vector<TPCH_INT> col_l_tax(l_nrow);
    std::vector<TPCH_INT> col_l_returnflag(l_nrow);
    std::vector<TPCH_INT> col_l_linestatus(l_nrow);
    std::vector<TPCH_INT> col_l_shipdate(l_nrow);
    std::vector<TPCH_INT> col_l_commitdate(l_nrow);
    std::vector<TPCH_INT> col_l_receiptdate(l_nrow);
    std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> > col_l_shipinstruct(l_nrow);
    std::vector<std::array<char, TPCH_READ_MAXAGG_LEN + 1> > col_l_shipmode(l_nrow);
    std::vector<std::array<char, TPCH_READ_L_CMNT_MAX + 1> > col_l_comment(l_nrow);

    size_t s_nrow = s_vec.size();
    std::vector<TPCH_INT> col_s_suppkey(s_nrow);
    std::vector<std::array<char, TPCH_READ_S_NAME_LEN + 1> > col_s_name(s_nrow);
    std::vector<std::array<char, TPCH_READ_S_ADDR_MAX + 1> > col_s_address(s_nrow);
    std::vector<TPCH_INT> col_s_nationkey(s_nrow);
    std::vector<std::array<char, TPCH_READ_PHONE_LEN + 1> > col_s_phone(s_nrow);
    std::vector<TPCH_INT> col_s_acctbal(s_nrow);
    std::vector<std::array<char, TPCH_READ_S_CMNT_MAX + 1> > col_s_comment(s_nrow);

    size_t p_nrow = p_vec.size();
    std::vector<TPCH_INT> col_p_partkey(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_NAME_LEN + 1> > col_p_name(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_MFG_LEN + 1> > col_p_mfgr(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_BRND_LEN + 1> > col_p_brand(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_TYPE_LEN + 1> > col_p_type(p_nrow);
    std::vector<TPCH_INT> col_p_size(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_CNTR_LEN + 1> > col_p_container(p_nrow);
    std::vector<TPCH_INT> col_p_retailprice(p_nrow);
    std::vector<std::array<char, TPCH_READ_P_CMNT_MAX + 1> > col_p_comment(p_nrow);

    size_t ps_nrow = ps_vec.size();
    std::vector<TPCH_INT> col_ps_partkey(ps_nrow);
    std::vector<TPCH_INT> col_ps_suppkey(ps_nrow);
    std::vector<TPCH_INT> col_ps_availqty(ps_nrow);
    std::vector<TPCH_INT> col_ps_supplycost(ps_nrow);
    std::vector<std::array<char, TPCH_READ_PS_CMNT_MAX + 1> > col_ps_comment(ps_nrow);

    // start timing and launch threads.
    // ************************************************************

    gettimeofday(&tv0, 0);

    r_thread = std::thread(columnize_r, r_vec.data(), r_nrow, col_r_regionkey.data(), std::ref(col_r_name),
                           std::ref(col_r_comment));

    n_thread = std::thread(columnize_n, n_vec.data(), n_nrow, col_n_nationkey.data(), col_n_regionkey.data(),
                           std::ref(col_n_name), std::ref(col_n_comment));

    c_thread = std::thread(columnize_c, c_vec.data(), c_nrow, col_c_custkey.data(), std::ref(col_c_name),
                           std::ref(col_c_address), col_c_nationkey.data(), std::ref(col_c_phone), col_c_acctbal.data(),
                           std::ref(col_c_mktsegment), std::ref(col_c_comment));

    o_thread = std::thread(columnize_o, o_vec.data(), o_nrow, col_o_orderkey.data(), col_o_custkey.data(),
                           col_o_orderstatus.data(), col_o_totalprice.data(), col_o_orderdate.data(),
                           std::ref(col_o_orderpriority), std::ref(col_o_clerk), col_o_shippriority.data(),
                           std::ref(col_o_comment));

    l_thread =
        std::thread(columnize_l, l_vec.data(), l_nrow, col_l_orderkey.data(), col_l_partkey.data(),
                    col_l_suppkey.data(), col_l_linenumber.data(), col_l_quantity.data(), col_l_extendedprice.data(),
                    col_l_discount.data(), col_l_tax.data(), col_l_returnflag.data(), col_l_linestatus.data(),
                    col_l_shipdate.data(), col_l_commitdate.data(), col_l_receiptdate.data(),
                    std::ref(col_l_shipinstruct), std::ref(col_l_shipmode), std::ref(col_l_comment));

    s_thread = std::thread(columnize_s, s_vec.data(), s_nrow, col_s_suppkey.data(), std::ref(col_s_name),
                           std::ref(col_s_address), col_s_nationkey.data(), std::ref(col_s_phone), col_s_acctbal.data(),
                           std::ref(col_s_comment));

    p_thread = std::thread(columnize_p, p_vec.data(), p_nrow, col_p_partkey.data(), std::ref(col_p_name),
                           std::ref(col_p_mfgr), std::ref(col_p_brand), std::ref(col_p_type), col_p_size.data(),
                           std::ref(col_p_container), col_p_retailprice.data(), std::ref(col_p_comment));

    ps_thread = std::thread(columnize_ps, ps_vec.data(), ps_nrow, col_ps_partkey.data(), col_ps_suppkey.data(),
                            col_ps_availqty.data(), col_ps_supplycost.data(), std::ref(col_ps_comment));

    r_thread.join();
    n_thread.join();
    c_thread.join();
    o_thread.join();
    l_thread.join();
    s_thread.join();
    p_thread.join();
    ps_thread.join();

    gettimeofday(&tv1, 0);
    usec = x_utils::tvdiff(tv0, tv1);
    printf("Time to columnize tables: %d usec.\n", usec);

    // ************************************************************
    // done and write data to file.

    write_to_file(out_dir + "/r_regionkey.dat", col_r_regionkey);
    write_to_file(out_dir + "/r_name.dat", col_r_name);
    write_to_file(out_dir + "/r_comment.dat", col_r_comment);

    write_to_file(out_dir + "/n_nationkey.dat", col_n_nationkey);
    write_to_file(out_dir + "/n_regionkey.dat", col_n_regionkey);
    write_to_file(out_dir + "/n_name.dat", col_n_name);
    write_to_file(out_dir + "/n_comment.dat", col_n_comment);

    write_to_file(out_dir + "/c_custkey.dat", col_c_custkey);
    write_to_file(out_dir + "/c_name.dat", col_c_name);
    write_to_file(out_dir + "/c_address.dat", col_c_address);
    write_to_file(out_dir + "/c_nationkey.dat", col_c_nationkey);
    write_to_file(out_dir + "/c_phone.dat", col_c_phone);
    write_to_file(out_dir + "/c_acctbal.dat", col_c_acctbal);
    write_to_file(out_dir + "/c_mktsegment.dat", col_c_mktsegment);
    write_to_file(out_dir + "/c_commet.dat", col_c_comment);

    write_to_file(out_dir + "/o_orderkey.dat", col_o_orderkey);
    write_to_file(out_dir + "/o_custkey.dat", col_o_custkey);
    write_to_file(out_dir + "/o_orderstatus.dat", col_o_orderstatus);
    write_to_file(out_dir + "/o_totalprice.dat", col_o_totalprice);
    write_to_file(out_dir + "/o_orderdate.dat", col_o_orderdate);
    write_to_file(out_dir + "/o_orderpriority.dat", col_o_orderpriority);
    write_to_file(out_dir + "/o_clerk.dat", col_o_clerk);
    write_to_file(out_dir + "/o_shippriority.dat", col_o_shippriority);
    write_to_file(out_dir + "/o_comment.dat", col_o_comment);

    write_to_file(out_dir + "/l_orderkey.dat", col_l_orderkey);
    write_to_file(out_dir + "/l_partkey.dat", col_l_partkey);
    write_to_file(out_dir + "/l_suppkey.dat", col_l_suppkey);
    write_to_file(out_dir + "/l_linenumber.dat", col_l_linenumber);
    write_to_file(out_dir + "/l_quantity.dat", col_l_quantity);
    write_to_file(out_dir + "/l_extendedprice.dat", col_l_extendedprice);
    write_to_file(out_dir + "/l_discount.dat", col_l_discount);
    write_to_file(out_dir + "/l_tax.dat", col_l_tax);
    write_to_file(out_dir + "/l_returnflag.dat", col_l_returnflag);
    write_to_file(out_dir + "/l_linestatus.dat", col_l_linestatus);
    write_to_file(out_dir + "/l_shipdate.dat", col_l_shipdate);
    write_to_file(out_dir + "/l_commitdate.dat", col_l_commitdate);
    write_to_file(out_dir + "/l_receiptdate.dat", col_l_receiptdate);
    write_to_file(out_dir + "/l_shipinstruct.dat", col_l_shipinstruct);
    write_to_file(out_dir + "/l_shipmode.dat", col_l_shipmode);
    write_to_file(out_dir + "/l_comment.dat", col_l_comment);

    write_to_file(out_dir + "/s_suppkey.dat", col_s_suppkey);
    write_to_file(out_dir + "/s_name.dat", col_s_name);
    write_to_file(out_dir + "/s_address.dat", col_s_address);
    write_to_file(out_dir + "/s_nationkey.dat", col_s_nationkey);
    write_to_file(out_dir + "/s_phone.dat", col_s_phone);
    write_to_file(out_dir + "/s_acctbal.dat", col_s_acctbal);
    write_to_file(out_dir + "/s_comment.dat", col_s_comment);

    write_to_file(out_dir + "/p_partkey.dat", col_p_partkey);
    write_to_file(out_dir + "/p_name.dat", col_p_name);
    write_to_file(out_dir + "/p_mfgr.dat", col_p_mfgr);
    write_to_file(out_dir + "/p_brand.dat", col_p_brand);
    write_to_file(out_dir + "/p_type.dat", col_p_type);
    write_to_file(out_dir + "/p_size.dat", col_p_size);
    write_to_file(out_dir + "/p_container.dat", col_p_container);
    write_to_file(out_dir + "/p_retailprice.dat", col_p_retailprice);
    write_to_file(out_dir + "/p_comment.dat", col_p_comment);

    write_to_file(out_dir + "/ps_partkey.dat", col_ps_partkey);
    write_to_file(out_dir + "/ps_suppkey.dat", col_ps_suppkey);
    write_to_file(out_dir + "/ps_availqty.dat", col_ps_availqty);
    write_to_file(out_dir + "/ps_supplycost.dat", col_ps_supplycost);
    write_to_file(out_dir + "/ps_comment.dat", col_ps_comment);

    return 0;
}

std::string prepare(std::string& wrk_dir, const int sf, std::vector<std::string> col_names) {
    const bool debug = true;
    std::string dbgen_dir = wrk_dir;
    dbgen_dir += "/dbgen";
    std::string tbl_dir = wrk_dir;
    tbl_dir += "/tbl";
    std::string dat_dir = wrk_dir;
    dat_dir += "/dat" + std::to_string(sf);
    if (debug) std::cout << "dat dir is: " << dat_dir << std::endl;
    bool must_generate = false;
    for (std::string col_name : col_names) {
        if (x_utils::is_file(dat_dir + "/" + col_name + ".dat")) {
            if (debug) std::cout << col_name << ".dat exists." << std::endl;
        } else {
            if (debug) std::cout << col_name << ".dat needs to be generated." << std::endl;
            must_generate = true;
        }
    }
    if (must_generate) {
        if (debug) std::cout << "calling data generator..." << std::endl;
        if (!x_utils::is_dir(wrk_dir)) {
            std::string cmd = "mkdir -p " + wrk_dir;
            system(cmd.c_str());
        }
        generate_tbl(dbgen_dir, tbl_dir, sf);
        convert_dat(tbl_dir, dat_dir);
        if (debug) std::cout << ".dat file genrated" << std::endl;
    }
    return dat_dir;
}

inline void generate_valid(const std::string& in_dir,
                           const int sf,
                           const std::string& v_name,
                           const std::string& col_name,
                           const unsigned elem_bits,
                           const uint64_t ge_v,
                           const uint64_t lt_v) {
    assert((elem_bits == 32 || elem_bits == 64) && "unknown element width");
    std::string dir = in_dir + "/dat" + std::to_string(sf) + "/";
    if (x_utils::is_file(dir + v_name + ".dat")) {
        std::cout << v_name << ".dat exists, skip generation." << std::endl;
        return;
    }
    std::ifstream fin(dir + col_name + ".dat", std::ios::ate | std::ios::binary);
    size_t n = fin.tellg() / (elem_bits / 8);
    std::cout << "Filtering " << n << " rows into validaty bits..." << std::flush;
    fin.seekg(std::ios::beg);
    std::ofstream fout(dir + v_name + ".dat", std::ios::binary);
    char t;
    unsigned bit_cnt = 0;
    size_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        char pass = 1;
        if (elem_bits == 64) {
            uint64_t d;
            fin.read(reinterpret_cast<char*>(&d), 8);
            pass = (d >= ge_v) && (d < lt_v);
        } else {
            uint32_t d;
            fin.read(reinterpret_cast<char*>(&d), 4);
            pass = (d >= ge_v) && (d < lt_v);
        }
        v += pass;
        if (bit_cnt == 7) {
            t |= (pass << bit_cnt);
            fout.write(&t, 1);
            t = 0;
            bit_cnt = 0;
        } else {
            t |= (pass << bit_cnt);
            ++bit_cnt;
        }
    }
    if (bit_cnt != 0) {
        fout.write(&t, 1);
    }
    std::cout << "Done. " << v << " rows valid." << std::endl;
    fout.close();
    fin.close();
}

#endif
