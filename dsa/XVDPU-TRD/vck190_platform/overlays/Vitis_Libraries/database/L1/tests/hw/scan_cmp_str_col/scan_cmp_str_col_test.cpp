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

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "xf_database/scan_cmp_str_col.hpp"

#define ITERA_NUM 100
#define COL_NUM 100
#define STR_NUM 128
#define MAX_LEN 64

static const char word_file[] = "dictionary.dat";

// top-function
void dut(ap_uint<512>* ddr_ptr,
         hls::stream<int>& size,
         hls::stream<int>& num_str,
         hls::stream<ap_uint<512> >& cnst_stream,
         hls::stream<bool>& out_stream,
         hls::stream<bool>& e_str_o) {
#pragma HLS INTERFACE s_axilite port = ddr_ptr
#pragma HLS INTERFACE m_axi depth = 32 port = ddr_ptr offset = slave
    xf::database::scanCmpStrCol(ddr_ptr, size, num_str, //
                                cnst_stream, out_stream, e_str_o);
}

// golden reference function
void ref_str_equal(const ap_uint<512>* ddr_ptr,
                   const int size,
                   const int num_str,
                   hls::stream<ap_uint<512> >& cnst_stream,
                   hls::stream<bool>& out_stream,
                   hls::stream<bool>& e_str_o) {
    std::string str_i = "";
    std::string str_ref = "";
    ap_uint<512> ddr_copy;
    bool is_equal, is_cross = 0;
    int rest = 0;
    int counter = 0;

    ap_uint<512> config = cnst_stream.read();
    for (int i = MAX_LEN - 1; i > 0; i--) {
        char c = config.range(8 * i - 1, 8 * i - 8);
        if (c != '\0') str_ref.push_back(c);
    }
    // std::cout << "str_ref:" << str_ref << std::endl;

    for (int i = 0; i < size; i++) {
        ddr_copy = ddr_ptr[i];
        int loc = MAX_LEN - 1;
        if (is_cross) { // cross 512bit
            for (int j = 0; j < rest; j++) str_i.push_back(ddr_copy.range(511 - 8 * j, 504 - 8 * j));
            loc -= rest;
            is_cross = 0;

            if (!str_i.compare(str_ref))
                is_equal = true;
            else
                is_equal = false;
            out_stream << is_equal;
            e_str_o << false;
            // std::cout << "str-i:" << str_i << std::endl;
            str_i.clear();

            if (++counter == num_str) break;
        }

        while (loc > 0) {
            int len = (int)ddr_copy.range(8 * loc + 7, 8 * loc);
            if (len == 0) {
                loc--;
                continue;
            }
            if (loc < len) {
                is_cross = 1;
                for (int j = loc - 1; j >= 0; j--) {
                    str_i.push_back(ddr_copy.range(8 * j + 7, 8 * j));
                }
                rest = len - loc;
                loc = 0;
                continue;
            }
            for (int j = 0; j < len; j++) str_i.push_back(ddr_copy.range(8 * (loc - j) - 1, 8 * (loc - j - 1)));

            if (!str_i.compare(str_ref))
                is_equal = true;
            else
                is_equal = false;
            out_stream << is_equal;
            e_str_o << false;
            // std::cout << "str_i:" << str_i << std::endl;
            str_i.clear();
            loc -= (len + 1);

            if (++counter == num_str) break;
        }

        if (counter == num_str) break;
    }

    // end of transfer
    e_str_o << true;
}

// generate multiple column 512b test data
int generate_test_data(
    int div_set[8], ap_uint<512>* ddr_mem, std::string& const_str, const char* file_name, const int num_str) {
    char word[MAX_LEN], part_of_word[MAX_LEN];
    unsigned char len = 0, rest_len = 64;
    int counter = 0;
    static std::streamoff offset_ptr = 0;
    bool is_cross = 0;

    // srand(time(0));
    int rnd_index = rand() % num_str;
    // int rnd_index = 19;

    // open word file
    std::ifstream infile(file_name, std::ios::in); // Attention: the results in window and linux are different
    if (!infile.is_open()) {
        std::cout << "Error! File " << file_name << " cannot be open!" << std::endl;
        return true;
    }

    // read words and generate columns of string
    infile.seekg(offset_ptr, std::ios::beg);
    for (int i = 0; i < num_str;) {
        for (int j = 0; j < MAX_LEN; j++) word[j] = '\0'; // clear the buffer
        if (!infile.eof() && !is_cross) {
            infile >> word;
            std::string str_t(word);
            div_set[str_t.length() / 8]++;
            // std::cout << str_t << " " << str_t.length() << std::endl;
            if (i == rnd_index) const_str = word;
            i++;
        } else
            for (int j = 0; j < MAX_LEN; j++) word[j] = part_of_word[j];

        for (int j = 0; j < MAX_LEN; j++) { // length of heading and char
            if (word[j] == '\0') {
                if (!is_cross)
                    len = j + 1;
                else
                    len = j;
                break;
            }
        }

        if (rest_len >= len) { // normal concat
            if (!is_cross) {
                for (int j = 0; j < len; j++) {
                    if (j == 0)
                        ddr_mem[counter].range(7, 0) = len - 1;
                    else
                        ddr_mem[counter].range(7, 0) = word[j - 1];

                    if ((j == len - 1) && ((rest_len - len) / 8 == 0)) {
                    } else
                        ddr_mem[counter] <<= 8;
                }
            } else {
                for (int j = 0; j < len; j++) {
                    ddr_mem[counter].range(7, 0) = word[j];
                    ddr_mem[counter] <<= 8;
                }
            }

            is_cross = 0;
            rest_len -= len;

            if (len % 8 != 0) {
                ddr_mem[counter] <<= 8 * (8 - len % 8); // align to 64bit
                rest_len -= (8 - len % 8);
            }
        } else { // cross two 512bit
            is_cross = 1;
            ddr_mem[counter] <<= 8 * (rest_len - 1);
            for (int j = rest_len - 1; j >= 0; j--) { // load upper part
                if (j == rest_len - 1)
                    ddr_mem[counter].range(8 * rest_len - 1, 8 * rest_len - 8) = len - 1;
                else
                    ddr_mem[counter].range(8 * j + 7, 8 * j) = word[rest_len - j - 2];
            }

            for (int j = 0; j < MAX_LEN; j++) { // backup the rest part
                if (rest_len + j >= len)
                    part_of_word[j] = '\0';
                else
                    part_of_word[j] = word[rest_len + j - 1];
            }

            rest_len = 0;
        }

        if (rest_len == 0) {
            counter++;
            rest_len = 64;
        }

        if (i == num_str) {
            if (!is_cross) {
                ddr_mem[counter] <<= 8 * (rest_len - 1); // shift to leftmost for last 512bit
            } else {
                for (int j = MAX_LEN - 1; j >= 0; j--) { // extend to next 512bit for last long string
                    ddr_mem[counter].range(8 * j + 7, 8 * j) = part_of_word[63 - j];
                }
            }
            counter++;
        }
    }

    offset_ptr = (std::streamoff)infile.tellg();
    infile.close();
    return counter;
}

int main() {
    int nerror = 0;

    hls::stream<bool> out_stream("out_stream");
    hls::stream<bool> e_str_o("e_str_o");
    hls::stream<ap_uint<512> > cnst_stream("cnst_stream");
    hls::stream<bool> ref_out_stream("ref_out_stream");
    hls::stream<bool> ref_e_str_o("ref_e_str_o");
    hls::stream<ap_uint<512> > ref_cnst_stream("ref_cnst_stream");

    int real_col = 0;
    std::string const_str;
    ap_uint<512> ddr_content[COL_NUM];
    ap_uint<512> padding_const_str;

    std::vector<bool> hw_result;
    std::vector<bool> sw_result;

    for (int n = 0; n < ITERA_NUM; n++) {
        // genrate test data for ddr memory
        int div_set[8] = {0};
        memset(ddr_content, 0, COL_NUM);
        real_col = generate_test_data(div_set, ddr_content, const_str, word_file, STR_NUM);
        std::cout << "Number of column:" << real_col << std::endl;
        for (int m = 0; m < 8; m++) {
            double ratio = 100 * div_set[m] / double(STR_NUM);
            std::cout << "Length-" << m << ":" << ratio << "% ";
        }
        std::cout << std::endl << "Total string number:" << STR_NUM << std::endl;

        // generate constant string stream
        for (int i = 64; i > 0; i--) {
            if (i == 64)
                padding_const_str.range(8 * i - 1, 8 * i - 8) = const_str.length();
            else if (i > (63 - const_str.length()))
                padding_const_str.range(8 * i - 1, 8 * i - 8) = const_str.at(63 - i);
            else
                padding_const_str.range(8 * i - 1, 8 * i - 8) = 0;
        }
        cnst_stream << padding_const_str;
        ref_cnst_stream << padding_const_str;

        hls::stream<int> size, num_str;
        size.write(real_col);
        num_str.write(STR_NUM);

        // call filter string equal function
        dut(ddr_content, size, num_str, cnst_stream, out_stream, e_str_o);

        while (!e_str_o.read()) {
            hw_result.push_back(out_stream.read());
        }

        // call golden function and record result
        ref_str_equal(ddr_content, real_col, STR_NUM, ref_cnst_stream, ref_out_stream, ref_e_str_o);
        while (!ref_e_str_o.read()) {
            sw_result.push_back(ref_out_stream.read());
        }

        // compare gloden and hardware result
        // for(std::vector<bool>::iterator it1 = sw_result.begin(); it1 !=
        // sw_result.end(); ++it1) {
        //   std::cout << "sw:" << *it1 << " ";
        // }

        // std::cout << std::endl;
        // for(std::vector<bool>::iterator it1 = hw_result.begin(); it1 !=
        // hw_result.end(); ++it1) {
        //   std::cout << "hw:" << *it1 << " ";
        // }

        if (hw_result.size() == sw_result.size()) {
            std::vector<bool>::iterator it2 = hw_result.begin();
            for (std::vector<bool>::iterator it1 = sw_result.begin(); it1 != sw_result.end(); ++it1) {
                // std::cout << "hw:" << *it2 << "-sw:" << *it1 << std::endl;
                if (*it2 != *it1) nerror++;
                it2++;
            }
        } else {
            std::cout << "The length of hw & sw sequence is not equal!" << std::endl
                      << "sw: " << sw_result.size() << ", hw: " << hw_result.size() << std::endl;
            nerror++;
        }

        hw_result.clear();
        sw_result.clear();
    }

    return nerror;
}
