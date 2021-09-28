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
#include <istream>
#include <ostream>
#include <vector>

#include "table_dt.hpp"

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
 * There're 8 kinds of .tbl files and tpch_read.h offers Read Row and Read
 * Column.
 *
 * class d_long is wrapper of "long". class template d_string is wrapper of
 * "char[]".
 * They both have overload operator << and >> to facilitate input and output.
 * d_money is wrapper of "long", too. it storage money in cent.
 * d_string's member "char data[num]" contains effective chars(num-1 chars,at
 * most), ending with '\0'.
 * d_string's member "int len" indicate effective number of chars, including
 * spaces.
 *
 * xxx_t is "Row" of the tbl file. and xxx_t_col is "Column" of the tbl file.
 * xxx_t_col stores each column in tbl in an vector, along with "long volumn"
 * and "long count".
 * "long volumn" indicates the size of vector, "long count" indicates effective
 * number of record in vector.
 * member function change_volumn is to resize each vector.
 * operator << will output effective data, keep the remaining data.
 * operator >> will overwrite its data from the begining and stop when it's
 * fullfilled or runout of data.
 * "long count" will equals to "long volumn" or num of records.
 * */

class d_long {
   public:
    TPCH_INT data;

    d_long() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, d_long& source) {
        output << (source.data) << '|';
        return output;
    }

    friend std::istream& operator>>(std::istream& input, d_long& target) {
        input >> (target.data);
        input.get();
        return input;
    }
};

class d_date {
   public:
    TPCH_INT data;

    d_date() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, d_date& source) {
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
        output << day << '|';
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
};

class d_money {
   public:
    TPCH_INT data;

    d_money() : data(0) {}

    operator TPCH_INT() const { return data; }

    friend std::ostream& operator<<(std::ostream& output, d_money& source) {
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
        output << cents << '|';
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
};

template <long num>
class d_string {
   public:
    char data[num];
    int len;

    d_string() : len(0) {}

    friend std::ostream& operator<<(std::ostream& output, d_string<num>& source) {
        long i;
        for (i = 0; i < source.len; i++) {
            output.put(source.data[i]);
        }
        output.put('|');
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
};

class customer_t {
   public:
    d_long custkey;
    d_string<TPCH_READ_C_NAME_LEN + 1> name;
    d_string<TPCH_READ_C_ADDR_MAX + 1> address;
    d_long nationkey;
    d_string<TPCH_READ_PHONE_LEN + 1> phone;
    d_money acctbal;
    d_string<TPCH_READ_MAXAGG_LEN + 1> mktsegment;
    d_string<TPCH_READ_C_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, customer_t& source) {
        output << source.custkey << source.name << source.address << source.nationkey << source.phone << source.acctbal
               << source.mktsegment << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, customer_t& target) {
        input >> target.custkey >> target.name >> target.address >> target.nationkey >> target.phone >>
            target.acctbal >> target.mktsegment >> target.comment;
        input.get();
        return input;
    }
};

class customer_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> custkey;
    std::vector<d_string<TPCH_READ_C_NAME_LEN + 1> > name;
    std::vector<d_string<TPCH_READ_C_ADDR_MAX + 1> > address;
    std::vector<d_long> nationkey;
    std::vector<d_string<TPCH_READ_PHONE_LEN + 1> > phone;
    std::vector<d_money> acctbal;
    std::vector<d_string<TPCH_READ_MAXAGG_LEN + 1> > mktsegment;
    std::vector<d_string<TPCH_READ_C_CMNT_MAX + 1> > comment;

    customer_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        custkey.resize(ss);
        name.resize(ss);
        address.resize(ss);
        nationkey.resize(ss);
        phone.resize(ss);
        acctbal.resize(ss);
        mktsegment.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, customer_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.custkey[i] << source.name[i] << source.address[i] << source.nationkey[i] << source.phone[i]
                   << source.acctbal[i] << source.mktsegment[i] << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, customer_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.custkey[i] >> target.name[i] >> target.address[i] >> target.nationkey[i] >>
                target.phone[i] >> target.acctbal[i] >> target.mktsegment[i] >> target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class orders_t {
   public:
    d_long orderkey;
    d_long custkey;
    d_string<2> orderstatus;
    d_money totalprice;
    d_date orderdate;
    d_string<TPCH_READ_MAXAGG_LEN + 1> orderpriority;
    d_string<TPCH_READ_O_CLRK_LEN + 1> clerk;
    d_long shippriority;
    d_string<TPCH_READ_O_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, orders_t& source) {
        output << source.orderkey << source.custkey << source.orderstatus << source.totalprice << source.orderdate
               << source.orderpriority << source.clerk << source.shippriority << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, orders_t& target) {
        input >> target.orderkey >> target.custkey >> target.orderstatus >> target.totalprice >> target.orderdate >>
            target.orderpriority >> target.clerk >> target.shippriority >> target.comment;
        input.get();
        return input;
    }
};

class orders_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> orderkey;
    std::vector<d_long> custkey;
    std::vector<d_string<2> > orderstatus;
    std::vector<d_money> totalprice;
    std::vector<d_date> orderdate;
    std::vector<d_string<TPCH_READ_MAXAGG_LEN + 1> > orderpriority;
    std::vector<d_string<TPCH_READ_O_CLRK_LEN + 1> > clerk;
    std::vector<d_long> shippriority;
    std::vector<d_string<TPCH_READ_O_CMNT_MAX + 1> > comment;

    orders_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        orderkey.resize(ss);
        custkey.resize(ss);
        orderstatus.resize(ss);
        totalprice.resize(ss);
        orderdate.resize(ss);
        orderpriority.resize(ss);
        clerk.resize(ss);
        shippriority.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, orders_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.orderkey[i] << source.custkey[i] << source.orderstatus[i] << source.totalprice[i]
                   << source.orderdate[i] << source.orderpriority[i] << source.clerk[i] << source.shippriority[i]
                   << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, orders_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.orderkey[i] >> target.custkey[i] >> target.orderstatus[i] >> target.totalprice[i] >>
                target.orderdate[i] >> target.orderpriority[i] >> target.clerk[i] >> target.shippriority[i] >>
                target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class lineitem_t {
   public:
    d_long orderkey;
    d_long partkey;
    d_long suppkey;
    d_long linenumber;
    d_long quantity;
    d_money extendedprice;
    d_money discount;
    d_money tax;
    d_string<2> returnflag;
    d_string<2> linestatus;
    d_date shipdate;
    d_date commitdate;
    d_date receiptdate;
    d_string<TPCH_READ_MAXAGG_LEN + 1> shipinstruct;
    d_string<TPCH_READ_MAXAGG_LEN + 1> shipmode;
    d_string<TPCH_READ_L_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, lineitem_t& source) {
        output << source.orderkey << source.partkey << source.suppkey << source.linenumber << source.quantity
               << source.extendedprice << source.discount << source.tax << source.returnflag << source.linestatus
               << source.shipdate << source.commitdate << source.receiptdate << source.shipinstruct << source.shipmode
               << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, lineitem_t& target) {
        input >> target.orderkey >> target.partkey >> target.suppkey >> target.linenumber >> target.quantity >>
            target.extendedprice >> target.discount >> target.tax >> target.returnflag >> target.linestatus >>
            target.shipdate >> target.commitdate >> target.receiptdate >> target.shipinstruct >> target.shipmode >>
            target.comment;
        input.get();
        return input;
    }
};

class lineitem_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> orderkey;
    std::vector<d_long> partkey;
    std::vector<d_long> suppkey;
    std::vector<d_long> linenumber;
    std::vector<d_long> quantity;
    std::vector<d_money> extendedprice;
    std::vector<d_money> discount;
    std::vector<d_money> tax;
    std::vector<d_string<2> > returnflag;
    std::vector<d_string<2> > linestatus;
    std::vector<d_date> commitdate;
    std::vector<d_date> shipdate;
    std::vector<d_date> receiptdate;
    std::vector<d_string<TPCH_READ_MAXAGG_LEN + 1> > shipinstruct;
    std::vector<d_string<TPCH_READ_MAXAGG_LEN + 1> > shipmode;
    std::vector<d_string<TPCH_READ_L_CMNT_MAX + 1> > comment;

    lineitem_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        orderkey.resize(ss);
        partkey.resize(ss);
        suppkey.resize(ss);
        linenumber.resize(ss);
        quantity.resize(ss);
        extendedprice.resize(ss);
        discount.resize(ss);
        tax.resize(ss);
        returnflag.resize(ss);
        linestatus.resize(ss);
        commitdate.resize(ss);
        shipdate.resize(ss);
        receiptdate.resize(ss);
        shipinstruct.resize(ss);
        shipmode.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, lineitem_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.orderkey[i] << source.partkey[i] << source.suppkey[i] << source.linenumber[i]
                   << source.quantity[i] << source.extendedprice[i] << source.discount[i] << source.tax[i]
                   << source.returnflag[i] << source.linestatus[i] << source.shipdate[i] << source.commitdate[i]
                   << source.receiptdate[i] << source.shipinstruct[i] << source.shipmode[i] << source.comment[i]
                   << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, lineitem_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.orderkey[i] >> target.partkey[i] >> target.suppkey[i] >> target.linenumber[i] >>
                target.quantity[i] >> target.extendedprice[i] >> target.discount[i] >> target.tax[i] >>
                target.returnflag[i] >> target.linestatus[i] >> target.shipdate[i] >> target.commitdate[i] >>
                target.receiptdate[i] >> target.shipinstruct[i] >> target.shipmode[i] >> target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class nation_t {
   public:
    d_long nationkey;
    d_string<TPCH_READ_NATION_LEN + 1> name;
    d_long regionkey;
    d_string<TPCH_READ_N_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, nation_t& source) {
        output << source.nationkey << source.name << source.regionkey << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, nation_t& target) {
        input >> target.nationkey >> target.name >> target.regionkey >> target.comment;
        input.get();
        return input;
    }
};

class nation_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> nationkey;
    std::vector<d_string<TPCH_READ_NATION_LEN + 1> > name;
    std::vector<d_long> regionkey;
    std::vector<d_string<TPCH_READ_N_CMNT_MAX + 1> > comment;

    nation_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        nationkey.resize(ss);
        name.resize(ss);
        regionkey.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, nation_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.nationkey[i] << source.name[i] << source.regionkey[i] << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, nation_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.nationkey[i] >> target.name[i] >> target.regionkey[i] >> target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class region_t {
   public:
    d_long regionkey;
    d_string<TPCH_READ_REGION_LEN + 1> name;
    d_string<TPCH_READ_R_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, region_t& source) {
        output << source.regionkey << source.name << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, region_t& target) {
        input >> target.regionkey >> target.name >> target.comment;
        input.get();
        return input;
    }
};

class region_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> regionkey;
    std::vector<d_string<TPCH_READ_REGION_LEN + 1> > name;
    std::vector<d_string<TPCH_READ_R_CMNT_MAX + 1> > comment;

    region_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        regionkey.resize(ss);
        name.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, region_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.regionkey[i] << source.name[i] << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, region_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.regionkey[i] >> target.name[i] >> target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class partsupp_t {
   public:
    d_long partkey;
    d_long suppkey;
    d_long availqty;
    d_money supplycost;
    d_string<TPCH_READ_PS_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, partsupp_t& source) {
        output << source.partkey << source.suppkey << source.availqty << source.supplycost << source.comment
               << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, partsupp_t& target) {
        input >> target.partkey >> target.suppkey >> target.availqty >> target.supplycost >> target.comment;
        input.get();
        return input;
    }
};

class partsupp_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> partkey;
    std::vector<d_long> suppkey;
    std::vector<d_long> availqty;
    std::vector<d_money> supplycost;
    std::vector<d_string<TPCH_READ_PS_CMNT_MAX + 1> > comment;

    partsupp_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        partkey.resize(ss);
        suppkey.resize(ss);
        availqty.resize(ss);
        supplycost.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, partsupp_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.partkey[i] << source.suppkey[i] << source.availqty[i] << source.supplycost[i]
                   << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, partsupp_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.partkey[i] >> target.suppkey[i] >> target.availqty[i] >> target.supplycost[i] >>
                target.comment[i];
            input.get();
        }
        target.count = target.volumn;
        return input;
    }
};

class supplier_t {
   public:
    d_long suppkey;
    d_string<TPCH_READ_S_NAME_LEN + 1> name;
    d_string<TPCH_READ_S_ADDR_MAX + 1> address;
    d_long nationkey;
    d_string<TPCH_READ_PHONE_LEN + 1> phone;
    d_money acctbal;
    d_string<TPCH_READ_S_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, supplier_t& source) {
        output << source.suppkey << source.name << source.address << source.nationkey << source.phone << source.acctbal
               << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, supplier_t& target) {
        input >> target.suppkey >> target.name >> target.address >> target.nationkey >> target.phone >>
            target.acctbal >> target.comment;
        input.get();
        return input;
    }
};

class supplier_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> suppkey;
    std::vector<d_string<TPCH_READ_S_NAME_LEN + 1> > name;
    std::vector<d_string<TPCH_READ_S_ADDR_MAX + 1> > address;
    std::vector<d_long> nationkey;
    std::vector<d_string<TPCH_READ_PHONE_LEN + 1> > phone;
    std::vector<d_money> acctbal;
    std::vector<d_string<TPCH_READ_S_CMNT_MAX + 1> > comment;

    supplier_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        suppkey.resize(ss);
        name.resize(ss);
        address.resize(ss);
        nationkey.resize(ss);
        phone.resize(ss);
        acctbal.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, supplier_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.suppkey[i] << source.name[i] << source.address[i] << source.nationkey[i] << source.phone[i]
                   << source.acctbal[i] << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, supplier_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.suppkey[i] >> target.name[i] >> target.address[i] >> target.nationkey[i] >>
                target.phone[i] >> target.acctbal[i] >> target.comment[i];
        }
        target.count = target.volumn;
        return input;
    }
};

class part_t {
   public:
    d_long partkey;
    d_string<TPCH_READ_P_NAME_LEN + 1> name;
    d_string<TPCH_READ_P_MFG_LEN + 1> mfgr;
    d_string<TPCH_READ_P_BRND_LEN + 1> brand;
    d_string<TPCH_READ_P_TYPE_LEN + 1> type;
    d_long size;
    d_string<TPCH_READ_P_CNTR_LEN + 1> container;
    d_money retailprice;
    d_string<TPCH_READ_P_CMNT_MAX + 1> comment;

    friend std::ostream& operator<<(std::ostream& output, part_t& source) {
        output << source.partkey << source.name << source.mfgr << source.brand << source.type << source.size
               << source.container << source.retailprice << source.comment << std::endl;
        return output;
    }

    friend std::istream& operator>>(std::istream& input, part_t& target) {
        input >> target.partkey >> target.name >> target.mfgr >> target.brand >> target.type >> target.size >>
            target.container >> target.retailprice >> target.comment;
        input.get();
        return input;
    }
};

class part_t_col {
   public:
    long volumn;
    long count;
    std::vector<d_long> partkey;
    std::vector<d_string<TPCH_READ_P_NAME_LEN + 1> > name;
    std::vector<d_string<TPCH_READ_P_MFG_LEN + 1> > mfgr;
    std::vector<d_string<TPCH_READ_P_BRND_LEN + 1> > brand;
    std::vector<d_string<TPCH_READ_P_TYPE_LEN + 1> > type;
    std::vector<d_long> size;
    std::vector<d_string<TPCH_READ_P_CNTR_LEN + 1> > container;
    std::vector<d_money> retailprice;
    std::vector<d_string<TPCH_READ_P_CMNT_MAX + 1> > comment;

    part_t_col() : volumn(0), count(0) {}
    void change_volumn(long ss) {
        volumn = ss;
        count = 0;
        partkey.resize(ss);
        name.resize(ss);
        mfgr.resize(ss);
        brand.resize(ss);
        type.resize(ss);
        size.resize(ss);
        container.resize(ss);
        retailprice.resize(ss);
        comment.resize(ss);
    }

    friend std::ostream& operator<<(std::ostream& output, part_t_col& source) {
        for (int i = 0; i < source.count; i++) {
            output << source.partkey[i] << source.name[i] << source.mfgr[i] << source.brand[i] << source.type[i]
                   << source.size[i] << source.container[i] << source.retailprice[i] << source.comment[i] << std::endl;
        }
        return output;
    }

    friend std::istream& operator>>(std::istream& input, part_t_col& target) {
        for (int i = 0; i < target.volumn; i++) {
            if (input.peek() == EOF) {
                target.count = i;
                return input;
            }
            input >> target.partkey[i] >> target.name[i] >> target.mfgr[i] >> target.brand[i] >> target.type[i] >>
                target.size[i] >> target.container[i] >> target.retailprice[i] >> target.comment[i];
        }
        target.count = target.volumn;
        return input;
    }
};

#endif
