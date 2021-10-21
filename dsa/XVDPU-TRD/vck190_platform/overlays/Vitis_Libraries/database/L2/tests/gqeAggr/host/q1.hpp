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
#ifndef Q1_HPP
#define Q1_HPP

#include "ap_int.h"
#include "xf_database/meta_table.hpp"

// helpers for OpenCL C API
#include "xclhost.hpp"

class Table {
   public:
    std::string name;
    size_t nrow;
    size_t ncol;
    int npart;
    std::vector<std::string> colsname;
    std::vector<size_t> colswidth;
    std::vector<size_t> isrowid;
    std::vector<size_t> iskdata;

    std::string dir;

    std::vector<size_t> size512;

    ap_uint<512>* data;
    ap_uint<512>* datak;

    Table(){};

    Table(std::string name_, size_t nrow_, size_t ncol_, std::string dir_) {
        name = name_;
        nrow = nrow_;
        ncol = ncol_;
        npart = 1;
        dir = dir_;
        size512.push_back(0);
        mode = 2;
    };

    // no header assigned to table data[0]
    Table(std::string name_, size_t nrow_, size_t ncol_, size_t mode_, std::string dir_) {
        name = name_;
        nrow = nrow_;
        ncol = ncol_;
        npart = 1;
        dir = dir_;
        size512.push_back(0);
        mode = 4; // only supports mode 4
    };

    Table(size_t size) {
        size512.push_back(size / 64);
        mode = 3;
    };

    //! Add column
    void addCol(std::string columname, size_t width, int isrid = 0, int iskda = 1) {
        size_t depth = nrow + VEC_LEN * 2 - 1;
        size_t sizeonecol = size_t((width * depth + 64 - 1) / 64);
        size512.push_back(size512.back() + sizeonecol);
        colsname.push_back(columname);
        isrowid.push_back(isrid);
        iskdata.push_back(iskda);
        colswidth.push_back(width);
        mode = 1;
    };

    //! CPU memory allocation
    void allocateHost() { // col added manually
        if (mode == 1) {
            data = xclhost::aligned_alloc<ap_uint<512> >(size512.back());
            data[0] = get_table_header(size512[1], nrow); // TO CHECK
            for (size_t j = 1; j < ncol; j++) {
                data[size512[j]] = 0;
            };
        } else if (mode == 2) { // col added automatically
            size_t depth = nrow + VEC_LEN;
            size_t sizeonecol = size_t((4 * depth + 64 - 1) / 64);
            for (size_t i = 0; i < ncol; i++) {
                size512.push_back(size512.back() + sizeonecol);
                iskdata.push_back(1);
            };
            data = xclhost::aligned_alloc<ap_uint<512> >(size512.back());
            data[0] = get_table_header(size512[1], 0); // TO CHECK
        } else if (mode == 3) {
            data = xclhost::aligned_alloc<ap_uint<512> >(size512.back());
        } else if (mode == 4) {
            size_t depth = nrow + VEC_LEN;
            size_t sizeonecol = size_t((4 * depth + 64 - 1) / 64);
            for (size_t i = 0; i < ncol; i++) {
                size512.push_back(sizeonecol);
                iskdata.push_back(1);
            }
            data = xclhost::aligned_alloc<ap_uint<512> >(sizeonecol);
        } else {
            std::cout << "ERROR: Table mode not supported" << std::endl;
        }
    };

    void allocateHost(float f, int p_num) { // col added manually
        if ((f == 0) || (p_num == 0))
            std::cout << "ERROR: p_num (" << p_num << ")should be bigger than 1,"
                      << "f(" << f << ")should be bigger than 1" << std::endl;
        if (mode == 2) { // col added automatically
            npart = p_num;
            size_t depth = nrow + VEC_LEN;
            size_t sizeonecol = size_t((4 * depth + 64 - 1) / 64);
            size_t alignedSizeOneCol = (f * sizeonecol + p_num - 1) / p_num;
            for (int j = 0; j < p_num; j++) {
                for (size_t i = 0; i < ncol; i++) {
                    size512.push_back(size512.back() + alignedSizeOneCol);
                    iskdata.push_back(1);
                };
            }
            data = xclhost::aligned_alloc<ap_uint<512> >(size512.back());
            data[0] = get_table_header(size512[ncol], alignedSizeOneCol, 0); // TO CHECK
            for (int j = 1; j < p_num; j++) data[size512[ncol] * j] = get_table_header(alignedSizeOneCol, 0);
        } else {
            std::cout << "ERROR: Table mode not supported" << std::endl;
        }
    };

    int getNumRow() {
        int nm = *(data);
        return nm;
    };

    int getNumRow(int BLOCK_SIZE) {
        int nm = *(data + BLOCK_SIZE * 16 - 1);
        return nm;
    };

    void setNumRow(int n) {
        data[0].range(31, 0) = n; // TO CHECK
    };

    void setNumRow(int BLOCK_SIZE, int n) {
        data[BLOCK_SIZE * 16 - 1].range(31, 0) = n; // TO CHECK
    };

    template <class T, int N>
    void setcharN(int r, int l, std::array<T, N> array_) {
        long offset = (long)r * N * sizeof(T);
        memcpy((char*)(data + size512[l]) + offset, array_.data(), N * sizeof(T));
    }
    void setInt32(int r, int l, int d) { memcpy((char*)(data + size512[l]) + r * sizeof(int), &d, sizeof(int)); };
    void setInt64(int r, int l, int64_t d) {
        long offset = (long)r * sizeof(int64_t);
        memcpy((char*)(data + size512[l]) + offset, &d, sizeof(int64_t));
    };
    void setInt64_l(int r, int l, int64_t d) { setInt32(r, l, d & 0x000000007fffffff); };
    void setInt64_h(int r, int l, int64_t d) { setInt32(r, l, (int32_t)(d >> 31)); };
    template <class T, int N>
    std::array<T, N> getcharN(int r, int l) {
        std::array<T, N> arr;
        long offset = (long)r * N * sizeof(T);
        memcpy(arr.data(), (char*)(data + size512[l]) + offset, N * sizeof(T));
        return arr;
    };
    int getInt32K(int r, int l) {
        int d;
        memcpy(&d, (char*)(datak + size512[l]) + r * sizeof(int), sizeof(int));
        return d;
    };
    int getInt32(int r, int l) {
        int d;
        memcpy(&d, (char*)(data + size512[l]) + r * sizeof(int), sizeof(int));
        return d;
    };
    int64_t getInt64(int r, int l) {
        int64_t d;
        long offset = (long)r * sizeof(int64_t);
        memcpy(&d, (char*)(data + size512[l]) + offset, sizeof(int64_t));
        return d;
    };
    uint32_t getUint32(int r, int l) {
        uint32_t d;
        memcpy(&d, (char*)(data + size512[l]) + r * sizeof(uint32_t), sizeof(uint32_t));
        return d;
    };
    int64_t combineInt64(int r, int l0, int l1) {
        ap_uint<32> h; // h
        ap_uint<32> l; // l
        int64_t d;

        memcpy(&h, (char*)(data + size512[l0]) + r * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&l, (char*)(data + size512[l1]) + r * sizeof(uint32_t), sizeof(uint32_t));
        d = (h, l);

        return d;
    };

    int64_t mergeInt64(int32_t l, int32_t h) {
        int64_t h_ = (int64_t)h;
        int64_t h_l = (h_ << 31) + l;
        return h_l;
    }

    bool getKdata() {
        bool datacp = 0;
        int cpNum = 0;
        for (size_t i = 0; i < ncol; i++) {
            if (iskdata[i] == 0) {
                datacp = 1;
            } else {
                cpNum++;
            }
        }
        if (datacp) {
            size_t depth = nrow + VEC_LEN * 2 - 1;
            size_t sizeonecol = size_t((4 * depth + 64 - 1) / 64);
            datak = xclhost::aligned_alloc<ap_uint<512> >(sizeonecol * cpNum);
            int ki = 0;
            for (size_t i = 0; i < ncol; i++) {
                if (iskdata[i] == 1) {
                    memcpy(datak + ki, data + size512[i], sizeonecol * 64);
                    ki += sizeonecol;
                }
            }
            datak[0] = get_table_header(sizeonecol, nrow); // TO CHECK
        }

        return datacp;
    }

   private:
    ap_uint<512> get_table_header(int n512b, int nrow) {
        ap_uint<512> th = 0;
        th.range(31, 0) = nrow;
        th.range(63, 32) = n512b;
        return th;
    };

    ap_uint<512> get_table_header(int hp_size, int blk_size, int nrow) {
        ap_uint<512> th = 0;
        th.range(31, 0) = nrow;
        th.range(63, 32) = blk_size;
        th.range(95, 64) = hp_size;
        return th;
    }

    int mode;
};

void gatherTable_col(Table& tin1, Table& tin2, Table& tout) {
    int nrow = tin1.getNumRow();
    if (nrow != tin2.getNumRow()) {
        std::cout << "colums gathering must tin1.row == tin2.row" << std::endl;
        return;
    }
    for (int r = 0; r < nrow; r++) {
        int tin1_0 = tin1.getInt32(r, 0);
        int tin1_1 = tin1.getInt32(r, 1);
        int tin2_0 = tin2.getInt32(r, 2);
        int tin2_1 = tin2.getInt32(r, 3);
        tout.setInt32(r, 0, tin1_0);
        tout.setInt32(r, 1, tin1_1);
        tout.setInt32(r, 2, tin2_0);
        tout.setInt32(r, 3, tin2_1);
    }
    tout.setNumRow(nrow);
}

void gatherTable_row(Table& tin1, Table& tin2, Table& tout) {
    int nrow1 = tin1.getNumRow();
    int nrow2 = tin2.getNumRow();
    tout.setNumRow(nrow1 + nrow2);
}

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
           int* table_in_col7,
           int lineitem_n,
           Table& golden) {
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
        golden.setInt64(r, 0, it.l_returnflag);
        golden.setInt64(r, 1, it.l_linestatus);
        golden.setInt64(r, 2, it.sum_qty);
        golden.setInt64(r, 3, it.sum_price);
        golden.setInt64(r, 4, it.sum_disc_price);
        golden.setInt64(r, 5, it.sum_charge);
        golden.setInt64(r, 6, it.avg_qty);
        golden.setInt64(r, 7, it.avg_price);
        golden.setInt64(r, 8, it.avg_disc);
        golden.setInt64(r, 9, it.sum_count);

        ++r;
        /*
        if (r < 10)
            std::cout << it.l_returnflag << "," << it.l_linestatus << "," << it.sum_qty << "," << it.sum_price << ","
                      << it.sum_disc_price << "," << it.sum_charge << "," << it.avg_qty << "," << it.avg_price << ","
                      << it.avg_disc << "," << it.sum_count << std::endl;
        */
    }
    golden.setNumRow(64, r);
    // golden.setNumRow(r);

    // std::cout << "Golden value: " << std::dec << r << "," << golden.getNumRow(64) << std::endl;
    std::cout << "Golden value: " << std::dec << r << std::endl;
}

void q1Sort(Table& tin, Table& tout, xf::database::gqe::MetaTable& tout_meta) {
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

    // int nrow = tin.getNumRow(64);
    int nrow = tout_meta.getColLen();
    std::vector<Q1SortRow> rows;
    for (int i = 0; i < nrow; i++) {
        Q1SortRow t = {tin.getInt32(i, 8),         tin.getInt32(i, 9),         tin.combineInt64(i, 13, 5),
                       tin.combineInt64(i, 14, 6), tin.combineInt64(i, 11, 3), tin.combineInt64(i, 12, 4),
                       tin.getUint32(i, 0),        tin.getUint32(i, 1),        tin.combineInt64(i, 10, 2),
                       tin.getUint32(i, 15)};
        rows.push_back(t);
    }

    std::sort(rows.begin(), rows.end(), Q1SortLess);

    int r = 0;
    for (auto& it : rows) {
        tout.setInt64(r, 0, it.l_returnflag);
        tout.setInt64(r, 1, it.l_linestatus);
        tout.setInt64(r, 2, it.sum_qty);
        tout.setInt64(r, 3, it.sum_price);
        tout.setInt64(r, 4, it.sum_disc_price);
        tout.setInt64(r, 5, it.sum_charge);
        tout.setInt64(r, 6, it.avg_qty);
        tout.setInt64(r, 7, it.avg_price);
        tout.setInt64(r, 8, it.avg_disc);
        tout.setInt64(r, 9, it.sum_count);

        ++r;
    }
    tout.setNumRow(64, r);
}
void q1Print(Table& tout) {
    unsigned nrow = tout.getNumRow(64);
    printf("after sort:%d\n", nrow);
    int r = nrow > 32 ? 32 : nrow;
    std::cout << "Query result:" << nrow << std::endl;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < 10; j++) std::cout << std::dec << tout.getInt64(i, j) << " ";
        std::cout << std::endl;
    }
}

int check_result(Table& tout, Table& golden) {
    int nerror = 0;
    unsigned nrow = tout.getNumRow(64);
    unsigned nrow_g = golden.getNumRow(64);
    printf("kernel result row:%d\n", nrow);
    printf("ref result row:%d\n", nrow_g);
    if (nrow == nrow_g) {
        std::cout << "Checking each value..." << std::endl;
        for (unsigned i = 0; i < nrow; ++i) {
            /*
            if (tout.getInt64(i, 0) != golden.getInt64(i, 0)) {
                std::cout << i << ": " << tout.getInt64(i, 0) << "," << golden.getInt64(i, 0) << std::endl;
                nerror++;
            } else if (tout.getInt64(i, 1) != golden.getInt64(i, 1)) {
                std::cout << i << ": " << tout.getInt64(i, 1) << "," << golden.getInt64(i, 1) << std::endl;
                nerror++;
            }
            */
            for (int j = 0; j < 10; j++) {
                // std::cout << tout.getInt64(i, j) << "," << golden.getInt64(i, j) << std::endl;
                if (tout.getInt64(i, j) != golden.getInt64(i, j)) {
                    std::cout << i << ": " << tout.getInt64(i, j) << "," << golden.getInt64(i, j) << std::endl;
                    nerror++;
                    break;
                }
            }
        }
        std::cout << nerror << " Errors found." << std::endl;
        if (nerror == 0)
            std::cout << "Validate Pass" << std::endl;
        else
            std::cout << "Validate Failed" << std::endl;
        return nerror;
    } else {
        std::cout << "FPGA aggr rows: " << nrow_g << ", Golden value:" << nrow << std::endl;
        return 1;
    }
}

#endif
