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

#include "xf_database/nested_loop_join.hpp"
#include <vector>
#include <hls_stream.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <algorithm>

#define FIELD_RAND_MIN 'a'
#define FIELD_RAND_MAX 'z'
#define KEY_RAND_MIN 0
#define KEY_RAND_MAX 100

#define CMP_NUM 20
#define DATA_TYPE int
#define KEY_TYPE int

template <typename FIELD_T, typename KEY_T>
struct row_msg {
    FIELD_T field;
    KEY_T key;
};

int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}

bool same(row_msg<DATA_TYPE, KEY_TYPE> a, row_msg<DATA_TYPE, KEY_TYPE> b) {
    return (a.key == b.key) && (a.field == b.field);
}

bool compare(row_msg<DATA_TYPE, KEY_TYPE> a, row_msg<DATA_TYPE, KEY_TYPE> b) {
    if (a.key != b.key)
        return a.key < b.key;
    else
        return a.field < b.field;
}

// Generate random table
void generate_data(std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& testvector, int num) {
    row_msg<DATA_TYPE, KEY_TYPE> init_row;
    init_row.field = 'a';
    init_row.key = -1;
    testvector.push_back(init_row);

    for (int i = 0; i < num - 2; i++) {
        row_msg<DATA_TYPE, KEY_TYPE> row;
        row.field = rand_int(FIELD_RAND_MIN, FIELD_RAND_MAX);
        row.key = rand_int(KEY_RAND_MIN, KEY_RAND_MAX);
        //		row.field = rand_int(FIELD_RAND_MIN, FIELD_RAND_MAX);
        //		row.key = 1;
        testvector.push_back(row);
    }

    row_msg<DATA_TYPE, KEY_TYPE> fin_row;
    fin_row.field = 'b';
    fin_row.key = -2;
    testvector.push_back(fin_row);

    std::cout << " random test data generated! " << std::endl;
}

void strm_to_vector(std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& left_syn_Vector,
                    std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& right_syn_Vector,

                    hls::stream<KEY_TYPE> strm_in_left_key[CMP_NUM],
                    hls::stream<DATA_TYPE> strm_in_left_field[CMP_NUM],

                    hls::stream<KEY_TYPE> strm_in_right_key[CMP_NUM],
                    hls::stream<DATA_TYPE> strm_in_right_field[CMP_NUM],
                    hls::stream<bool> strm_in_e[CMP_NUM]) {
    int i;
    int out_left_key;
    char out_left_field;
    int out_right_key;
    char out_right_field;
    bool end;

    for (i = 0; i < CMP_NUM; i++) {
        end = strm_in_e[i].read();
        while (!end) {
            out_left_key = strm_in_left_key[i].read();
            out_left_field = strm_in_left_field[i].read();
            out_right_key = strm_in_right_key[i].read();
            out_right_field = strm_in_right_field[i].read();

            row_msg<DATA_TYPE, KEY_TYPE> left_row;
            left_row.key = out_left_key;
            left_row.field = out_left_field;
            left_syn_Vector.push_back(left_row);

            row_msg<DATA_TYPE, KEY_TYPE> right_row;
            right_row.key = out_right_key;
            right_row.field = out_right_field;
            right_syn_Vector.push_back(right_row);

            end = strm_in_e[i].read();
        }
    }
}

// software nested loop join as a reference
void org_nested_loop_join(std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& left_in_Vector,
                          std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& right_in_Vector,
                          std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& left_out_Vector,
                          std::vector<row_msg<DATA_TYPE, KEY_TYPE> >& right_out_Vector) {
    int i, j;

    for (i = 0; i < left_in_Vector.size(); i++) {
        for (j = 0; j < right_in_Vector.size(); j++) {
            if (left_in_Vector[i].key == right_in_Vector[j].key) {
                row_msg<DATA_TYPE, KEY_TYPE> row;
                row.key = left_in_Vector[i].key;
                row.field = left_in_Vector[i].field;
                left_out_Vector.push_back(row);
                row_msg<DATA_TYPE, KEY_TYPE> right_row;
                right_row.key = right_in_Vector[j].key;
                right_row.field = right_in_Vector[j].field;
                right_out_Vector.push_back(right_row);
            }
        }
    }
}

// nested loop join for CoSim
void syn_nested_loop_join(hls::stream<KEY_TYPE>& strm_in_left_key,
                          hls::stream<DATA_TYPE>& strm_in_left_field,
                          hls::stream<bool>& strm_in_left_e,

                          hls::stream<KEY_TYPE>& strm_in_right_key,
                          hls::stream<DATA_TYPE>& strm_in_right_field,
                          hls::stream<bool>& strm_in_right_e,

                          hls::stream<KEY_TYPE> strm_out_left_key[CMP_NUM],
                          hls::stream<DATA_TYPE> strm_out_left_field[CMP_NUM],

                          hls::stream<KEY_TYPE> strm_out_right_key[CMP_NUM],
                          hls::stream<DATA_TYPE> strm_out_right_field[CMP_NUM],

                          hls::stream<bool> strm_out_e[CMP_NUM]) {
    xf::database::nestedLoopJoin<CMP_NUM, KEY_TYPE, DATA_TYPE, DATA_TYPE>(
        strm_in_left_key, strm_in_left_field, strm_in_left_e, strm_in_right_key, strm_in_right_field, strm_in_right_e,
        strm_out_left_key, strm_out_left_field, strm_out_right_key, strm_out_right_field, strm_out_e);
}

int main() {
    int i, j;
    int nerror = 0;

    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > left_in_Vector;
    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > right_in_Vector;

    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > left_ref_Vector;
    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > right_ref_Vector;

    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > left_syn_Vector;
    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > right_syn_Vector;

    hls::stream<KEY_TYPE> strm_in_left_key("strm_in_left_key");
    hls::stream<DATA_TYPE> strm_in_left_field("strm_in_left_field");
    hls::stream<bool> strm_in_left_e("strm_in_left_e");

    hls::stream<KEY_TYPE> strm_in_right_key("strm_in_right_key");
    hls::stream<DATA_TYPE> strm_in_right_field("strm_in_right_field");
    hls::stream<bool> strm_in_right_e("strm_in_right_e");

    hls::stream<KEY_TYPE> strm_out_left_key[CMP_NUM];
    hls::stream<DATA_TYPE> strm_out_left_field[CMP_NUM];

    hls::stream<KEY_TYPE> strm_out_right_key[CMP_NUM];
    hls::stream<DATA_TYPE> strm_out_right_field[CMP_NUM];

    hls::stream<bool> strm_out_e[CMP_NUM];

    int out_left_key;
    char out_left_field;
    int out_right_key;
    char out_right_field;
    bool end;

    //----------------------test empty input tables-------------------//

    strm_in_left_e.write(true); // left table is empty

    strm_in_right_key.write(1);
    strm_in_right_field.write('a');
    strm_in_right_e.write(false);
    strm_in_right_e.write(true); // right table is not empty

    syn_nested_loop_join(strm_in_left_key, strm_in_left_field, strm_in_left_e, strm_in_right_key, strm_in_right_field,
                         strm_in_right_e, strm_out_left_key, strm_out_left_field, strm_out_right_key,
                         strm_out_right_field, strm_out_e);
    for (i = 0; i < CMP_NUM; i++) {
        end = strm_out_e[i].read();
        if (!end) {
            nerror++;
        }
    }

    strm_in_left_key.write(1);
    strm_in_left_field.write('a');
    strm_in_left_e.write(false);
    strm_in_left_e.write(true); // left table is not empty

    strm_in_right_e.write(true); // right table is empty

    syn_nested_loop_join(strm_in_left_key, strm_in_left_field, strm_in_left_e, strm_in_right_key, strm_in_right_field,
                         strm_in_right_e, strm_out_left_key, strm_out_left_field, strm_out_right_key,
                         strm_out_right_field, strm_out_e);
    for (i = 0; i < CMP_NUM; i++) {
        end = strm_out_e[i].read();
        if (!end) {
            nerror++;
        }
    }

    strm_in_right_e.write(true);
    strm_in_left_e.write(true); // both tables are empty

    syn_nested_loop_join(strm_in_left_key, strm_in_left_field, strm_in_left_e, strm_in_right_key, strm_in_right_field,
                         strm_in_right_e, strm_out_left_key, strm_out_left_field, strm_out_right_key,
                         strm_out_right_field, strm_out_e);
    for (i = 0; i < CMP_NUM; i++) {
        end = strm_out_e[i].read();
        if (!end) {
            nerror++;
        }
    }

    //---------------------------test empty table finish----------------//

    // generate left and right tables
    generate_data(left_in_Vector, CMP_NUM);
    generate_data(right_in_Vector, 100);

    // push the left table into the stream
    for (i = 0; i < left_in_Vector.size(); i++) {
        strm_in_left_key.write(left_in_Vector[i].key);
        strm_in_left_field.write(left_in_Vector[i].field);
        strm_in_left_e.write(false);
    }
    strm_in_left_e.write(true);

    // push the right table into the stream
    for (j = 0; j < right_in_Vector.size(); j++) {
        strm_in_right_key.write(right_in_Vector[j].key);
        strm_in_right_field.write(right_in_Vector[j].field);
        strm_in_right_e.write(false);
    }
    strm_in_right_e.write(true);

    // run the software nested loop join as a reference
    org_nested_loop_join(left_in_Vector, right_in_Vector, left_ref_Vector, right_ref_Vector);

    // synthesis the implemented nested loop join
    syn_nested_loop_join(strm_in_left_key, strm_in_left_field, strm_in_left_e, strm_in_right_key, strm_in_right_field,
                         strm_in_right_e, strm_out_left_key, strm_out_left_field, strm_out_right_key,
                         strm_out_right_field, strm_out_e);

    strm_to_vector(left_syn_Vector, right_syn_Vector, strm_out_left_key, strm_out_left_field, strm_out_right_key,
                   strm_out_right_field, strm_out_e);

    // print the left table
    std::cout << "left table:" << std::endl;
    for (i = 0; i < left_in_Vector.size(); i++) {
        std::cout << std::setw(5) << left_in_Vector[i].key << " " << std::setw(5) << left_in_Vector[i].field
                  << std::endl;
    }

    // print the right table
    std::cout << "right table:" << std::endl;
    for (i = 0; i < right_in_Vector.size(); i++) {
        std::cout << std::setw(5) << right_in_Vector[i].key << " " << std::setw(5) << right_in_Vector[i].field
                  << std::endl;
    }

    // print the ref result table
    std::cout << "ref result table:" << std::endl;
    for (i = 0; i < left_ref_Vector.size(); i++) {
        std::cout << std::setw(5) << left_ref_Vector[i].key << " " << std::setw(5) << left_ref_Vector[i].field << " "
                  << right_ref_Vector[i].key << " " << std::setw(5) << right_ref_Vector[i].field << std::endl;
    }

    // print the syn result table
    std::cout << "syn result table:" << std::endl;
    for (i = 0; i < left_syn_Vector.size(); i++) {
        std::cout << std::setw(5) << left_syn_Vector[i].key << " " << std::setw(5) << left_syn_Vector[i].field << " "
                  << right_syn_Vector[i].key << " " << std::setw(5) << right_syn_Vector[i].field << std::endl;
    }

    std::sort(left_ref_Vector.begin(), left_ref_Vector.end(), compare);
    std::sort(right_ref_Vector.begin(), right_ref_Vector.end(), compare);

    std::sort(left_syn_Vector.begin(), left_syn_Vector.end(), compare);
    std::sort(right_syn_Vector.begin(), right_syn_Vector.end(), compare);

    for (i = 0; i < left_ref_Vector.size(); i++) {
        if (!(same(left_ref_Vector[i], left_syn_Vector[i]) && same(right_ref_Vector[i], right_syn_Vector[i]))) {
            nerror++;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }

    return nerror;
}
