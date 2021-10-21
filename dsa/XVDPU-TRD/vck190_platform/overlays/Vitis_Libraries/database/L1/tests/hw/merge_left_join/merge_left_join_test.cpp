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

#include <vector>
#include <algorithm>
#include <hls_stream.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>

#include "xf_database/merge_left_join.hpp"

#define FIELD_RAND_MIN 'a'
#define FIELD_RAND_MAX 'z'
#define KEY_RAND_MIN 0
#define KEY_RAND_MAX 100

#define ISASCEND true

#define DATA_TYPE char
#define KEY_TYPE int

template <typename FIELD_T, typename KEY_T>
struct row_msg {
    FIELD_T field;
    KEY_T key;
};

template <typename FIELD_T, typename KEY_T>
struct right_row_msg {
    FIELD_T field;
    KEY_T key;
    bool isnull;
};

int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}

template <typename FIELD_T, typename KEY_T>
bool compare_key(row_msg<FIELD_T, KEY_T> a, row_msg<FIELD_T, KEY_T> b) {
    return ISASCEND ? a.key < b.key : a.key > b.key;
}

template <typename FIELD_T, typename KEY_T>
bool same_key(row_msg<FIELD_T, KEY_T> a, row_msg<FIELD_T, KEY_T> b) {
    return a.key == b.key;
}

// generate a sorted table with duplicated keys
template <typename FIELD_T, typename KEY_T>
void generate_sort_dupl_data(std::vector<row_msg<FIELD_T, KEY_T> >& testvector) {
    int randnum = rand_int(1, 100);

    for (int i = 0; i < randnum; i++) {
        row_msg<FIELD_T, KEY_T> row;
        row.field = rand_int(FIELD_RAND_MIN, FIELD_RAND_MAX);
        row.key = rand_int(KEY_RAND_MIN, KEY_RAND_MAX);
        testvector.push_back(row);
    }

    if (ISASCEND) {
        row_msg<FIELD_T, KEY_T> row;
        row.field = 'a';
        row.key = 101;
        testvector.push_back(row);
    } else {
        row_msg<FIELD_T, KEY_T> row;
        row.field = 'a';
        row.key = -1;
        testvector.push_back(row);
    }

    std::sort(testvector.begin(), testvector.end(), compare_key<FIELD_T, KEY_T>);
    std::cout << " random test data generated! " << std::endl;
}

// generate a sorted table without duplicated keys
template <typename FIELD_T, typename KEY_T>
void generate_sort_nodupl_data(std::vector<row_msg<FIELD_T, KEY_T> >& testvector) {
    int randnum = rand_int(1, 100);
    typename std::vector<row_msg<FIELD_T, KEY_T> >::iterator it;

    for (int i = 0; i < randnum; i++) {
        row_msg<FIELD_T, KEY_T> row;
        row.field = rand_int(FIELD_RAND_MIN, FIELD_RAND_MAX);
        row.key = rand_int(KEY_RAND_MIN, KEY_RAND_MAX);
        testvector.push_back(row);
    }

    if (ISASCEND) {
        row_msg<FIELD_T, KEY_T> row;
        row.field = 'a';
        row.key = 101;
        testvector.push_back(row);
    } else {
        row_msg<FIELD_T, KEY_T> row;
        row.field = 'a';
        row.key = -1;
        testvector.push_back(row);
    }

    std::sort(testvector.begin(), testvector.end(), compare_key<FIELD_T, KEY_T>);
    it = std::unique(testvector.begin(), testvector.end(), same_key<FIELD_T, KEY_T>);
    testvector.resize(std::distance(testvector.begin(), it));

    std::cout << " random test data generated! " << std::endl;
}

// nested loop join as a reference
void org_merge_left_join(std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& left_in_Vector,
                         std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& right_in_Vector,
                         std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& left_out_Vector,
                         std::vector<right_row_msg<DATA_TYPE, KEY_TYPE> >& right_out_Vector) {
    int i, j;
    bool righthas = false;

    for (i = 0; i < left_in_Vector.size(); i++) {
        righthas = false;
        for (j = 0; j < right_in_Vector.size(); j++) {
            if (left_in_Vector[i].key == right_in_Vector[j].key) {
                row_msg<DATA_TYPE, KEY_TYPE> row;
                row.key = left_in_Vector[i].key;
                row.field = left_in_Vector[i].field;
                left_out_Vector.push_back(row);
                right_row_msg<DATA_TYPE, KEY_TYPE> right_row;
                right_row.key = right_in_Vector[j].key;
                right_row.field = right_in_Vector[j].field;
                right_row.isnull = false;
                right_out_Vector.push_back(right_row);
                righthas = true;
            }
        }
        if (!righthas) {
            row_msg<DATA_TYPE, KEY_TYPE> row;
            row.key = left_in_Vector[i].key;
            row.field = left_in_Vector[i].field;
            left_out_Vector.push_back(row);
            right_row_msg<DATA_TYPE, KEY_TYPE> right_row;
            right_row.key = right_in_Vector[j].key;
            right_row.field = right_in_Vector[j].field;
            right_row.isnull = true;
            right_out_Vector.push_back(right_row);
        }
    }
}

// for Cosim
void syn_merge_left_join(hls::stream<KEY_TYPE>& left_strm_in_key,
                         hls::stream<DATA_TYPE>& left_strm_in_field,
                         hls::stream<bool>& left_e_strm,

                         hls::stream<KEY_TYPE>& right_strm_in_key,
                         hls::stream<DATA_TYPE>& right_strm_in_field,
                         hls::stream<bool>& right_e_strm,

                         hls::stream<KEY_TYPE>& left_strm_out_key,
                         hls::stream<DATA_TYPE>& left_strm_out_field,

                         hls::stream<KEY_TYPE>& right_strm_out_key,
                         hls::stream<DATA_TYPE>& right_strm_out_field,

                         hls::stream<bool>& out_e_strm,
                         hls::stream<bool>& isnull_strm) {
    xf::database::mergeLeftJoin<KEY_TYPE, DATA_TYPE, DATA_TYPE>(
        ISASCEND, left_strm_in_key, left_strm_in_field, left_e_strm, right_strm_in_key, right_strm_in_field,
        right_e_strm, left_strm_out_key, left_strm_out_field, right_strm_out_key, right_strm_out_field, out_e_strm,
        isnull_strm);
}

int main() {
    int i;
    int nerror = 0;

    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > left_in_Vector;
    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > right_in_Vector;

    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > left_out_Vector;
    std::vector<right_row_msg<DATA_TYPE, KEY_TYPE> > right_out_Vector;

    hls::stream<KEY_TYPE> left_strm_key("left_strm_key");
    hls::stream<DATA_TYPE> left_strm_field("left_strm_field");
    hls::stream<bool> left_e_strm("left_e_strm");

    hls::stream<KEY_TYPE> right_strm_key("right_strm_key");
    hls::stream<DATA_TYPE> right_strm_field("right_strm_field");
    hls::stream<bool> right_e_strm("right_e_strm");

    hls::stream<KEY_TYPE> left_out_key_strm("left_out_key_strm");
    hls::stream<DATA_TYPE> left_out_field_strm("left_out_field_strm");
    hls::stream<KEY_TYPE> right_out_key_strm("right_out_key_strm");
    hls::stream<DATA_TYPE> right_out_field_strm("right_out_field_strm");

    hls::stream<bool> out_e_strm("out_e_strm");
    hls::stream<bool> isnull_strm("isnull_strm");

    bool out_e;
    bool isnull;

    //----------------------test empty input tables-------------------//
    left_e_strm.write(true); // left table is empty

    right_strm_key.write(1);
    right_strm_field.write('a');
    right_e_strm.write(false);
    right_e_strm.write(true); // right table is not empty

    syn_merge_left_join(left_strm_key, left_strm_field, left_e_strm, right_strm_key, right_strm_field, right_e_strm,
                        left_out_key_strm, left_out_field_strm, right_out_key_strm, right_out_field_strm, out_e_strm,
                        isnull_strm);
    out_e = out_e_strm.read();
    if (!out_e) {
        nerror++;
    }

    right_e_strm.write(true); // right table is empty

    left_strm_key.write(1);
    left_strm_field.write('a');
    left_e_strm.write(false);
    left_e_strm.write(true); // left table is not empty

    syn_merge_left_join(left_strm_key, left_strm_field, left_e_strm, right_strm_key, right_strm_field, right_e_strm,
                        left_out_key_strm, left_out_field_strm, right_out_key_strm, right_out_field_strm, out_e_strm,
                        isnull_strm);

    if (left_out_key_strm.read() != 1 || left_out_field_strm.read() != 'a' || out_e_strm.read() != false ||
        isnull_strm.read() != true) {
        nerror++;
    }
    right_out_key_strm.read();
    right_out_field_strm.read();

    out_e = out_e_strm.read();
    if (!out_e) {
        nerror++;
    }

    right_e_strm.write(true);
    left_e_strm.write(true); // both tables are empty

    syn_merge_left_join(left_strm_key, left_strm_field, left_e_strm, right_strm_key, right_strm_field, right_e_strm,
                        left_out_key_strm, left_out_field_strm, right_out_key_strm, right_out_field_strm, out_e_strm,
                        isnull_strm);
    out_e = out_e_strm.read();
    if (!out_e) {
        nerror++;
    }

    //---------------------------test empty table finish----------------//

    // generate left table
    generate_sort_nodupl_data<DATA_TYPE, KEY_TYPE>(left_in_Vector);
    // generate right table
    generate_sort_dupl_data<DATA_TYPE, KEY_TYPE>(right_in_Vector);

    // print left table
    std::cout << "left table:" << std::endl;
    for (i = 0; i < left_in_Vector.size(); i++) {
        std::cout << std::setw(5) << left_in_Vector[i].key << " " << std::setw(5) << left_in_Vector[i].field
                  << std::endl;
    }

    // print right table
    std::cout << "right table:" << std::endl;
    for (i = 0; i < right_in_Vector.size(); i++) {
        std::cout << std::setw(5) << right_in_Vector[i].key << " " << std::setw(5) << right_in_Vector[i].field
                  << std::endl;
    }

    // push the left table to the stream
    for (i = 0; i < left_in_Vector.size(); i++) {
        left_strm_key.write(left_in_Vector[i].key);
        left_strm_field.write(left_in_Vector[i].field);
        left_e_strm.write(false);
    }
    left_e_strm.write(true);

    // push the right table to the stream
    for (i = 0; i < right_in_Vector.size(); i++) {
        right_strm_key.write(right_in_Vector[i].key);
        right_strm_field.write(right_in_Vector[i].field);
        right_e_strm.write(false);
    }
    right_e_strm.write(true);

    // nested loop join as a reference
    org_merge_left_join(left_in_Vector, right_in_Vector, left_out_Vector, right_out_Vector);

    // for CoSim
    syn_merge_left_join(left_strm_key, left_strm_field, left_e_strm, right_strm_key, right_strm_field, right_e_strm,
                        left_out_key_strm, left_out_field_strm, right_out_key_strm, right_out_field_strm, out_e_strm,
                        isnull_strm);

    // start comparison
    out_e = out_e_strm.read();
    i = 0;

    std::cout << "" << std::endl;

    while (!out_e) {
        if (i >= left_out_Vector.size()) {
            nerror++;
            break;
        }

        if (left_out_Vector[i].key != left_out_key_strm.read()) {
            nerror++;
        }

        if (left_out_Vector[i].field != left_out_field_strm.read()) {
            nerror++;
        }

        if (right_out_Vector[i].isnull != isnull_strm.read()) {
            nerror++;
        }

        if (!right_out_Vector[i].isnull) {
            if (right_out_Vector[i].key != right_out_key_strm.read()) {
                nerror++;
            }

            if (right_out_Vector[i].field != right_out_field_strm.read()) {
                nerror++;
            }
        } else {
            right_out_key_strm.read();
            right_out_field_strm.read();
        }

        std::cout << std::setw(5) << left_out_Vector[i].key << " " << std::setw(5) << left_out_Vector[i].field << " "
                  << std::setw(5) << right_out_Vector[i].key << " " << std::setw(5) << right_out_Vector[i].field << " "
                  << std::setw(5) << right_out_Vector[i].isnull << std::endl;

        i++;
        out_e = out_e_strm.read();
    }

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }

    return nerror;
}
