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

#include <vector> // std::vector
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "dut.hpp"

const static int num_of_class = 10;
const static int num_of_terms = (1 << 17) / num_of_class;

const std::string d_name = "train.dat";

template <typename MType>
union f_cast {
    MType f;
    MType i;
};

template <>
union f_cast<unsigned int> {
    unsigned int f;
    unsigned int i;
};

template <>
union f_cast<unsigned long long> {
    unsigned long long f;
    unsigned long long i;
};

template <>
union f_cast<double> {
    double f;
    unsigned long long i;
};

template <>
union f_cast<float> {
    float f;
    unsigned int i;
};

inline void splitStr(const std::string& s, std::vector<std::string>& v, const std::string& c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) v.push_back(s.substr(pos1));
}

int load_dat(std::vector<ap_uint<64> >& dataset, const std::string& dir) {
    std::string fn = dir + "/" + d_name;
    std::ifstream ifs(fn, std::ifstream::in);

    if (ifs) {
        while (ifs.good()) {
            std::string str;
            std::vector<std::string> std_vec;

            std::getline(ifs, str);
            if (ifs.good()) {
                splitStr(str, std_vec, " ");

                ap_uint<12> type = std::stoi(std_vec[0]);
                for (int i = 1; i < std_vec.size(); i++) {
                    std::vector<std::string> vec_t;
                    splitStr(std_vec[i], vec_t, ":");
                    if (vec_t.size() != 2)
                        return -1;
                    else {
                        ap_uint<20> term = std::stoi(vec_t[0]);
                        ap_uint<32> tf = std::stoi(vec_t[1]);
                        dataset.push_back((type, term, tf));
                        // std::cout << "Input:" << type << " " << term << " " << tf << std::endl;
                    }
                }

                ap_uint<64> end;
                end(31, 0) = 0;
                end(51, 32) = -1;
                end(63, 52) = type;
                dataset.push_back(end); // end of this sample
            }
        }
    } else {
        std::cerr << "ERROR: "
                  << "Failed to open dat file!\n";
        return -1;
    }

    ifs.close();

    return 0;
}

int main() {
    int err = 0;

    std::vector<ap_uint<64> > dataset;

    // load_dat(dataset, "/wrk/xsjhdnobkup5/xingw/xf_data_analytics/L1/benchmarks/naive_bayes/dat");
    load_dat(dataset, "../../../../");
    int fnum = dataset.size();
    if (fnum % 8 != 0) {
        for (int i = 0; i < (8 - fnum % 8); i++) {
            dataset.push_back(0);
        }
    }

    std::cout << "length:" << dataset.size() / 8 << std::endl;

    hls::stream<ap_uint<64> > i_data_array[PU];
    hls::stream<bool> i_e_array[PU];

    for (int i = 0; i < dataset.size() / 8; i++) {
        for (int j = 0; j < 8; j++) {
            i_data_array[j].write(dataset[i * 8 + j]);
            i_e_array[j].write(false);
        }
    }
    for (int j = 0; j < 8; j++) {
        i_e_array[j].write(true);
    }

    hls::stream<int> o_terms_strm;
    hls::stream<ap_uint<64> > mnnb_d0_strm[PU];
    hls::stream<ap_uint<64> > mnnb_d1_strm[PU];

    multinomialNB(num_of_class, num_of_terms, i_data_array, i_e_array, o_terms_strm, mnnb_d0_strm, mnnb_d1_strm);

    const int num_terms = o_terms_strm.read();
    const int n0 = ((num_terms + PU - 1) / PU) * num_of_class;
    const int n1 = (num_of_class + PU - 1) / PU;

    int cnt = 0;
    for (int i = 0; i < n0; i++) {
        for (int p = 0; p < PU; p++) {
            f_cast<double> cc0;
            cc0.i = mnnb_d0_strm[p].read();
            if (cnt < num_terms) {
                std::cout << cc0.f << " ";
                cnt++;
            } else if (cnt >= num_terms && p > 0 && p != PU - 1) {
            } else {
                std::cout << std::endl;
                cnt = 0;
            }
        }
    }

    // std::cout << std::endl;
    for (int i = 0; i < n1; i++) {
        for (int p = 0; p < PU; p++) {
            f_cast<double> cc0;
            cc0.i = mnnb_d1_strm[p].read();
            if (cnt < num_of_class) {
                std::cout << cc0.f << " ";
                cnt++;
            } else if (cnt >= num_of_class && p > 0 && p != PU - 1) {
            } else {
                std::cout << std::endl;
                cnt = 0;
            }
        }
    }

    return err;
}
