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
#ifndef _TPCH_READ_H
#define _TPCH_READ_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <stdint.h>
typedef int32_t TPCH_INT;

#define TPCH_READ_PHONE_LEN 15
#define TPCH_READ_C_NAME_LEN 18
#define TPCH_READ_C_ADDR_MAX 40
#define TPCH_READ_MAXAGG_LEN 20
#define TPCH_READ_C_CMNT_MAX 117
#define TPCH_READ_DATE_LEN 13
#define TPCH_READ_O_CLRK_LEN 15
#define TPCH_READ_O_CMNT_MAX 79
#define TPCH_READ_L_CMNT_MAX 44
#define TPCH_READ_NATION_LEN 25
#define TPCH_READ_N_CMNT_MAX 152
#define TPCH_READ_REGION_LEN 25
#define TPCH_READ_R_CMNT_MAX 152
#define TPCH_READ_PS_CMNT_MAX 199
#define TPCH_READ_S_NAME_LEN 25
#define TPCH_READ_S_ADDR_MAX 40
#define TPCH_READ_S_CMNT_MAX 101
#define TPCH_READ_P_NAME_LEN 55
#define TPCH_READ_P_MFG_LEN 25
#define TPCH_READ_P_BRND_LEN 10
#define TPCH_READ_P_TYPE_LEN 25
#define TPCH_READ_P_CNTR_LEN 10
#define TPCH_READ_P_CMNT_MAX 23

/* tpch_read.h provide read interface to access .tbl of TPC-H.
 * There're 8 kinds of .tbl files and tpch_read.h offers Read Row and Read Column.
 *
 * class d_long is wrapper of "long". class template d_string is wrapper of "char[]".
 * They both have overload operator << and >> to facilitate input and output.
 * d_money is wrapper of "long", too. it storage money in cent.
 * d_string's member "char data[num]" contains effective chars(num-1 chars,at most), ending with '\0'.
 * d_string's member "int len" indicate effective number of chars, including spaces.
 *
 * xxx_t is "Row" of the tbl file. and xxx_t_col is "Column" of the tbl file.
 * xxx_t_col stores each column in tbl in an vector, along with "long volumn" and "long count".
 * "long volumn" indicates the size of vector, "long count" indicates effective number of record in vector.
 * member function change_volumn is to resize each vector.
 * operator << will output effective data, keep the remaining data.
 * operator >> will overwrite its data from the begining and stop when it's fullfilled or runout of data.
 * "long count" will equals to "long volumn" or num of records.
 * */

class d_long {
   public:
    TPCH_INT data;

    d_long() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, const d_long& source) {
        output << (source.data);
        return output;
    }

    friend std::istream& operator>>(std::istream& input, d_long& target) {
        input >> (target.data);
        input.get();
        return input;
    }

    void read(std::ifstream& is) { is.read((char*)&data, sizeof(TPCH_INT)); }
    void write(std::ofstream& os) const { os.write((char*)&data, sizeof(TPCH_INT)); }
};

class d_date {
   public:
    TPCH_INT data;

    d_date() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, const d_date& source) {
        int year, month, day;
        year = source.data / 10000;
        month = (source.data % 10000) / 100;
        day = source.data % 100;
        output << year << '-';
        if (month < 10) {
            output << '0';
        }
        output << month << '-';
        if (day < 10) {
            output << '0';
        }
        output << day;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, d_date& target) {
        int year, month, day;
        input >> year;
        input.get();
        input >> month;
        input.get();
        input >> day;
        input.get();
        target.data = year * 10000 + month * 100 + day;
        return input;
    }

    void read(std::ifstream& is) { is.read((char*)&data, sizeof(TPCH_INT)); }
    void write(std::ofstream& os) const { os.write((char*)&data, sizeof(TPCH_INT)); }
};

class d_money {
   public:
    TPCH_INT data;

    d_money() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, const d_money& source) {
        TPCH_INT cents, dollars;
        cents = source.data;
        if (cents < 0) {
            output << '.';
            cents = -cents;
        }
        dollars = cents / 100;
        cents %= 100;
        output << dollars << '.';
        if (cents < 10) {
            output << '0';
        }
        output << cents;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, d_money& target) {
        TPCH_INT cents, dollars;
        input >> dollars;
        input.get();
        input >> cents;
        input.get();
        if (dollars < 0) {
            target.data = -((-dollars) * 100 + cents);
        } else {
            target.data = dollars * 100 + cents;
        }
        return input;
    }

    void read(std::ifstream& is) { is.read((char*)&data, sizeof(TPCH_INT)); }
    void write(std::ofstream& os) const { os.write((char*)&data, sizeof(TPCH_INT)); }
};

template <long num>
class d_string {
   public:
    char data[num];
    int len;

    d_string() : len(0) {}

    friend std::ostream& operator<<(std::ostream& output, const d_string<num>& source) {
        for (long i = 0; i < source.len; i++) {
            output.put(source.data[i]);
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, d_string<num>& target) {
        char tmp;
        int i;
        for (i = 0; i < num; i++) {
            tmp = input.get();
            if (tmp != '|') {
                target.data[i] = tmp;
            } else {
                target.data[i] = '\0';
                break;
            }
        }
        target.len = i;
        if (i == num) {
            target.data[num - 1] = '\0';
        }
        for (; i < num; i++) {
            target.data[i] = '\0';
        }
        return input;
    }

    void read(std::ifstream& is) { is.read(data, num); }
    void write(std::ofstream& os) const { os.write(data, num); }
};

template <class... Ts>
struct table {
    void set(std::istream& input) {}
};

template <class T, class... Ts>
struct table<T, Ts...> : public table<Ts...> {
    table() {}
    table(T t, Ts... ts) : v(t), table<Ts...>(ts...) {}
    T v;

    // find and return
    const T& get(const T&) const { return v; }
    // try recurse down
    template <typename Tc>
    const Tc& get(const Tc& c) const {
        const table<Ts...>& base = *this;
        return base.get(c);
    }

    void set(std::istream& input) {
        input >> v;
        table<Ts...>& base = *this;
        base.set(input);
    }
    friend std::istream& operator>>(std::istream& input, table<T, Ts...>& target) {
        target.set(input);
        return input;
    };
};

#define COL_STR_(x) #x
#define COL_STR(x) COL_STR_(x)
#define DEFINE_COL_STRUCT(n, t)                             \
    struct n : public t {                                   \
        const char* get_name() const { return COL_STR(n); } \
    }

DEFINE_COL_STRUCT(c_custkey, d_long);
DEFINE_COL_STRUCT(c_name, d_string<TPCH_READ_C_NAME_LEN + 1>);
DEFINE_COL_STRUCT(c_address, d_string<TPCH_READ_C_ADDR_MAX + 1>);
DEFINE_COL_STRUCT(c_nationkey, d_long);
DEFINE_COL_STRUCT(c_phone, d_string<TPCH_READ_PHONE_LEN + 1>);
DEFINE_COL_STRUCT(c_acctbal, d_money);
DEFINE_COL_STRUCT(c_mktsegment, d_string<TPCH_READ_MAXAGG_LEN + 1>);
DEFINE_COL_STRUCT(c_comment, d_string<TPCH_READ_C_CMNT_MAX + 1>);

typedef table<c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_mktsegment, c_comment> customer_t;

DEFINE_COL_STRUCT(o_orderkey, d_long);
DEFINE_COL_STRUCT(o_custkey, d_long);
DEFINE_COL_STRUCT(o_orderstatus, d_string<2>);
DEFINE_COL_STRUCT(o_totalprice, d_money);
DEFINE_COL_STRUCT(o_orderdate, d_date);
DEFINE_COL_STRUCT(o_orderpriority, d_string<TPCH_READ_MAXAGG_LEN + 1>);
DEFINE_COL_STRUCT(o_clerk, d_string<TPCH_READ_O_CLRK_LEN + 1>);
DEFINE_COL_STRUCT(o_shippriority, d_long);
DEFINE_COL_STRUCT(o_comment, d_string<TPCH_READ_O_CMNT_MAX + 1>);

typedef table<o_orderkey,
              o_custkey,
              o_orderstatus,
              o_totalprice,
              o_orderdate,
              o_orderpriority,
              o_clerk,
              o_shippriority,
              o_comment>
    orders_t;

DEFINE_COL_STRUCT(l_orderkey, d_long);
DEFINE_COL_STRUCT(l_partkey, d_long);
DEFINE_COL_STRUCT(l_suppkey, d_long);
DEFINE_COL_STRUCT(l_linenumber, d_long);
DEFINE_COL_STRUCT(l_quantity, d_long);
DEFINE_COL_STRUCT(l_extendedprice, d_money);
DEFINE_COL_STRUCT(l_discount, d_money);
DEFINE_COL_STRUCT(l_tax, d_money);
DEFINE_COL_STRUCT(l_returnflag, d_string<2>);
DEFINE_COL_STRUCT(l_linestatus, d_string<2>);
DEFINE_COL_STRUCT(l_shipdate, d_date);
DEFINE_COL_STRUCT(l_commitdate, d_date);
DEFINE_COL_STRUCT(l_receiptdate, d_date);
DEFINE_COL_STRUCT(l_shipinstruct, d_string<TPCH_READ_MAXAGG_LEN + 1>);
DEFINE_COL_STRUCT(l_shipmode, d_string<TPCH_READ_MAXAGG_LEN + 1>);
DEFINE_COL_STRUCT(l_comment, d_string<TPCH_READ_L_CMNT_MAX + 1>);
typedef table<l_orderkey,
              l_partkey,
              l_suppkey,
              l_linenumber,
              l_quantity,
              l_extendedprice,
              l_discount,
              l_tax,
              l_returnflag,
              l_linestatus,
              l_shipdate,
              l_commitdate,
              l_receiptdate,
              l_shipinstruct,
              l_shipmode,
              l_comment>
    lineitem_t;

DEFINE_COL_STRUCT(n_nationkey, d_long);
DEFINE_COL_STRUCT(n_name, d_string<TPCH_READ_NATION_LEN + 1>);
DEFINE_COL_STRUCT(n_regionkey, d_long);
DEFINE_COL_STRUCT(n_comment, d_string<TPCH_READ_N_CMNT_MAX + 1>);

typedef table<n_nationkey, n_name, n_regionkey, n_comment> nation_t;

DEFINE_COL_STRUCT(r_regionkey, d_long);
DEFINE_COL_STRUCT(r_name, d_string<TPCH_READ_REGION_LEN + 1>);
DEFINE_COL_STRUCT(r_comment, d_string<TPCH_READ_R_CMNT_MAX + 1>);

typedef table<r_regionkey, r_name, r_comment> region_t;

DEFINE_COL_STRUCT(ps_partkey, d_long);
DEFINE_COL_STRUCT(ps_suppkey, d_long);
DEFINE_COL_STRUCT(ps_availqty, d_long);
DEFINE_COL_STRUCT(ps_supplycost, d_money);
DEFINE_COL_STRUCT(ps_comment, d_string<TPCH_READ_PS_CMNT_MAX + 1>);

typedef table<ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment> partsupp_t;

DEFINE_COL_STRUCT(s_suppkey, d_long);
DEFINE_COL_STRUCT(s_name, d_string<TPCH_READ_S_NAME_LEN + 1>);
DEFINE_COL_STRUCT(s_address, d_string<TPCH_READ_S_ADDR_MAX + 1>);
DEFINE_COL_STRUCT(s_nationkey, d_long);
DEFINE_COL_STRUCT(s_phone, d_string<TPCH_READ_PHONE_LEN + 1>);
DEFINE_COL_STRUCT(s_acctbal, d_money);
DEFINE_COL_STRUCT(s_comment, d_string<TPCH_READ_S_CMNT_MAX + 1>);

typedef table<s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment> supplier_t;

DEFINE_COL_STRUCT(p_partkey, d_long);
DEFINE_COL_STRUCT(p_name, d_string<TPCH_READ_P_NAME_LEN + 1>);
DEFINE_COL_STRUCT(p_mfgr, d_string<TPCH_READ_P_MFG_LEN + 1>);
DEFINE_COL_STRUCT(p_brand, d_string<TPCH_READ_P_BRND_LEN + 1>);
DEFINE_COL_STRUCT(p_type, d_string<TPCH_READ_P_TYPE_LEN + 1>);
DEFINE_COL_STRUCT(p_size, d_long);
DEFINE_COL_STRUCT(p_container, d_string<TPCH_READ_P_CNTR_LEN + 1>);
DEFINE_COL_STRUCT(p_retailprice, d_money);
DEFINE_COL_STRUCT(p_comment, d_string<TPCH_READ_P_CMNT_MAX + 1>);

typedef table<p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment> part_t;

#undef COL_STR_
#undef COL_STR
#undef DEFINE_COL_STRUCT

template <typename Table, typename T>
void write_file(std::unordered_map<const char*, std::ofstream*>& m,
                const std::string& dout_name,
                const Table& tab,
                const T& col) {
    const char* cnp = col.get_name();
    std::ofstream* ofsp;
    auto it = m.find(cnp);
    if (it == m.end()) {
        std::string fn = dout_name + "/" + cnp + ".dat";
        ofsp = new std::ofstream(fn, std::ofstream::binary);
        if (ofsp->fail()) {
            std::cerr << "Fail to open " << fn << std::endl;
            exit(1);
        }
        m[cnp] = ofsp;
    } else {
        ofsp = it->second;
    }
    // write to its file
    tab.get(col).write(*ofsp);
}

template <typename Table, typename T, typename... Args>
void write_file(std::unordered_map<const char*, std::ofstream*>& m,
                const std::string& dout_name,
                const Table& tab,
                const T& col,
                Args... args) {
    write_file(m, dout_name, tab, col);
    write_file(m, dout_name, tab, args...);
}

template <typename Table, typename... Args>
void extract_col(const std::string& fin_name, const std::string& dout_name, Args... args) {
    std::unordered_map<const char*, std::ofstream*> m;
    std::ifstream ifs(fin_name, std::ifstream::in);
    if (ifs) {
        std::cout << "INFO: Extracting columns from " << fin_name << "..." << std::endl;
        while (ifs) {
            Table row;
            ifs >> row;
            if (ifs) {
                write_file(m, dout_name, row, args...);
            }
        }
    } else {
        std::cerr << "WARNING: " << fin_name << " does not exist, and is skipped!" << std::endl;
    }
    ifs.close();
    // close output files
    for (auto& it : m) {
        it.second->close();
        delete it.second;
    }
}

#endif
