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
#ifndef Q1_CPU_HPP
#define Q1_CPU_HPP
#include <unordered_map>
struct Q1GroupKey {
    int32_t k1; // l_returnflag
    int32_t k2; // l_linestatus

    bool operator==(const Q1GroupKey& other) const { return (k1 == other.k1 && k2 == other.k2); }
};
namespace std {

template <>
struct hash<Q1GroupKey> {
    std::size_t operator()(const Q1GroupKey& k) const {
        using std::size_t;
        using std::hash;
        // using std::string;
        return (hash<int>()(k.k1)) + (hash<int>()(k.k2));
    }
};
}
struct Q1GroupValue {
    int64_t sum_qty;        // sum(l_quantity)
    int64_t sum_price;      // sum(l_extendedprice)
    int64_t sum_disc_price; // sum(l_extendedprice * (1 - l_discount))
    int64_t sum_charge;     // sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))
    // avg(l_quantity)      sum_qty/sum_count
    // avg(l_extendedprice) sum_price/sum_count
    int64_t sum_disc;  // avg(l_discount) sum_disc/sum_count
    int64_t sum_count; // count(*)
};
void cpuQ1(int* table_in_col0,
           int* table_in_col1,
           int* table_in_col2,
           int* table_in_col3,
           int* table_in_col4,
           int* table_in_col5,
           int* table_in_col6,
           const int lineitem_n,
           int64_t* golden[10],
           int32_t& golden_n) {
    // int r = 0;
    std::unordered_map<Q1GroupKey, Q1GroupValue> m;
    for (int i = 0; i < lineitem_n; ++i) {
        int32_t l_shipdate = table_in_col6[i];
        // if(i<10) std::cout<<std::dec<<l_shipdate<<std::endl;
        if (l_shipdate <= 19980902) {
            int32_t l_returnflag = table_in_col0[i];
            int32_t l_linestatus = table_in_col1[i];
            int32_t l_quantity = table_in_col2[i];
            int32_t l_extendedprice = table_in_col3[i];
            int32_t l_discount = table_in_col4[i];
            int32_t l_tax = table_in_col5[i];
            int32_t eval0 = l_extendedprice * (100 - l_discount) / 100;
            int64_t eval1 = (int64_t)l_extendedprice * (100 - l_discount) * (100 + l_tax) / 10000;
            //    int64_t eval1 = eval0 / 100 * (100 + l_tax);
            Q1GroupKey k{l_returnflag, l_linestatus};
            auto it = m.find(k);
            if (it != m.end()) {
                int64_t sum_qty = it->second.sum_qty + l_quantity;
                int64_t sum_price = it->second.sum_price + l_extendedprice;
                int64_t sum_disc_price = it->second.sum_disc_price + eval0;
                int64_t sum_charge = it->second.sum_charge + eval1;
                int64_t sum_disc = it->second.sum_disc + l_discount;
                int64_t sum_count = it->second.sum_count + 1;
                Q1GroupValue s{sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, sum_count};
                m[k] = s;
            } else {
                Q1GroupValue s{l_quantity, l_extendedprice, eval0, eval1, l_discount, 1};
                m.insert(std::make_pair(k, s));
            }
            // if (i < 10) std::cout << l_returnflag << "," << l_linestatus << std::endl;
        }
    }
    struct Q1SortRow {
        int64_t l_returnflag;
        int64_t l_linestatus;
        int64_t sum_qty;
        int64_t sum_price;
        int64_t sum_disc_price;
        int64_t sum_charge;
        int64_t avg_qty;
        int64_t avg_price;
        int64_t avg_disc;
        int64_t sum_count;
    };

    struct {
        // operator <
        bool operator()(const Q1SortRow& a, const Q1SortRow& b) const {
            return a.l_returnflag < b.l_returnflag ||
                   (a.l_returnflag == b.l_returnflag && a.l_linestatus < b.l_linestatus);
        }
    } Q1SortLess;
    std::vector<Q1SortRow> rows;
    for (auto& it : m) {
        int64_t sum_qty = it.second.sum_qty;
        int64_t sum_price = it.second.sum_price;
        int64_t sum_disc = it.second.sum_disc;
        int64_t sum_count = it.second.sum_count;
        int64_t avg_qty = sum_qty / sum_count;
        int64_t avg_price = sum_price / sum_count;
        int64_t avg_disc = sum_disc / sum_count;
        Q1SortRow t = {it.first.k1,          it.first.k2, sum_qty,   sum_price, it.second.sum_disc_price,
                       it.second.sum_charge, avg_qty,     avg_price, avg_disc,  sum_count};
        rows.push_back(t);
    }
    // tout.setNumRow(r);

    std::sort(rows.begin(), rows.end(), Q1SortLess);

    int r = 0;
    for (auto& it : rows) {
        golden[0][r] = it.l_returnflag;
        golden[1][r] = it.l_linestatus;
        golden[2][r] = it.sum_qty;
        golden[3][r] = it.sum_price;
        golden[4][r] = it.sum_disc_price;
        golden[5][r] = it.sum_charge;
        golden[6][r] = it.avg_qty;
        golden[7][r] = it.avg_price;
        golden[8][r] = it.avg_disc;
        golden[9][r] = it.sum_count;
        ++r;
    }
    golden_n = r;
}
int check_result(xf::database::gqe::Table tout, int64_t* golden[10], int32_t golden_n) {
    int nrow = tout.getRowNum();
    int ncol = tout.getColNum();
    if (ncol < 10) {
        std::cout << "Error in ncol!" << std::endl;
        exit(1);
    }
    // std::vector<bool> _w_tags = tout.getTagWCol();
    char** _col_ptr = new char*[ncol];
    int* tab_col_size = new int[ncol];
    for (int i = 0; i < ncol; i++) {
        _col_ptr[i] = tout.getColPointer(i);
        tab_col_size[i] = tout.getColTypeSize(i);
    }
    // int _w_ncol = _w_tags.size();
    if (nrow != golden_n) {
        printf("kernel result row:%d\n", nrow);
        printf("ref result row:%d\n", golden_n);
        std::cout << "Output row number is wrong!" << std::endl;
        return 1;
    }
    std::cout << "Output row number check passed,start checking each value..." << std::endl;
    struct Q1SortRow {
        int64_t* onerow;
    };

    struct {
        // operator <
        bool operator()(const Q1SortRow& a, const Q1SortRow& b) const {
            return a.onerow[0] < b.onerow[0] || (a.onerow[0] == b.onerow[0] && a.onerow[1] < b.onerow[1]);
        }
    } Q1SortLess;
    std::vector<Q1SortRow> rows;
    for (int i = 0; i < nrow; ++i) {
        int64_t* tmps = new int64_t[ncol];
        for (int j = 0; j < ncol; j++) {
            memcpy(&tmps[j], _col_ptr[j] + i * tab_col_size[j], tab_col_size[j]);
        }
        Q1SortRow t = {tmps};
        rows.push_back(t);
    }
    std::sort(rows.begin(), rows.end(), Q1SortLess);
    int nerror = 0;

    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; j++) {
            if (rows[i].onerow[j] != golden[j][i]) {
                nerror++;
                std::cout << "(" << i << "," << j << "): " << rows[i].onerow[j] << "!=" << golden[j][i] << std::endl;
            }
        }
    }
    std::cout << nerror << " Errors found." << std::endl;
    if (nerror == 0)
        std::cout << "Validate Pass" << std::endl;
    else
        std::cout << "Validate Failed" << std::endl;
    return nerror;
}

#endif
