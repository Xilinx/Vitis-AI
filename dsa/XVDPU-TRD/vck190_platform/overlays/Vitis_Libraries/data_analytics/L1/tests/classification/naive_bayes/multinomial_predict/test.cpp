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
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "dut.hpp"

const static int num_of_class = 10;
const static int num_of_term = 10;
const std::string d_name = "test.dat";
const std::string m_name = "test.model";

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

int load_dat(std::vector<ap_uint<32> >& dataset, std::vector<ap_uint<64> >& dataset2, const std::string& dir) {
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
                        dataset.push_back(tf);
                        dataset2.push_back((type, term, tf));
                    }
                }

                if (num_of_term % CH_NM != 0) {
                    for (int i = 0; i < (CH_NM - num_of_term % CH_NM); i++) { // padding to 512b
                        dataset.push_back(0);
                    }
                }

                ap_uint<64> end;
                end(31, 0) = 0;
                end(51, 32) = -1;
                end(63, 52) = type;
                dataset2.push_back(end); // end of this sample

                if (std_vec.size() % 8 != 0) {
                    for (int i = 0; i < (8 - std_vec.size() % 8); i++) { // padding to 512b
                        dataset2.push_back(0);
                    }
                }
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

int load_model(const std::string& dir,
               std::vector<std::vector<double> >& theta,
               std::vector<double>& pi,
               hls::stream<ap_uint<64> >& theta_strm,
               hls::stream<ap_uint<64> >& prior_strm) {
    std::string fn = dir + "/" + m_name;
    std::ifstream ifs(fn, std::ifstream::in);
    int cls_cnt = 0;

    if (ifs) {
        while (ifs.good()) {
            std::string str;
            std::vector<std::string> std_vec;
            std::getline(ifs, str);
            if (ifs.good()) {
                splitStr(str, std_vec, " ");

                for (int i = 0; i < std_vec.size(); i++) {
                    if (cls_cnt < num_of_class) {
                        if (std_vec.size() != num_of_term)
                            return -1;
                        else {
                            f_cast<double> cc0;
                            cc0.f = std::stod(std_vec[i], NULL);
                            theta_strm.write(cc0.i);
                            theta[cls_cnt][i] = cc0.f;
                        }
                    } else {
                        if (std_vec.size() != num_of_class)
                            return -1;
                        else {
                            f_cast<double> cc0;
                            cc0.f = std::stod(std_vec[i], NULL);
                            prior_strm.write(cc0.i);
                            pi[i] = cc0.f;
                        }
                    }
                }

                cls_cnt++;
            }
        }
    } else {
        std::cerr << "ERROR: "
                  << "Failed to open dat file!\n";
        return -1;
    }

    ifs.close();

    if (cls_cnt != num_of_class + 1) return -1;

    return 0;
}

static bool abs_compare(double a, double b) {
    return (b - a > 1e-8);
}

void generate_golden(const std::vector<std::vector<double> > theta,
                     const std::vector<double> pi,
                     const std::vector<ap_uint<64> > dataset,
                     std::vector<int>& result) {
    std::vector<double> sum(num_of_class, 0.0);

    int cnt = 0;
    for (auto it = dataset.begin(); it != dataset.end(); it++) {
        ap_uint<64> t = *it;
        if (t(51, 32) == (ap_uint<20>)-1) {
            cnt++;
            std::vector<double> sum_tmp;
            for (int i = 0; i < num_of_class; i++) {
                double tmp = pi[i] + sum[i];
                sum_tmp.push_back(tmp);
                sum[i] = 0.0;
            }
            std::vector<double>::iterator tmp = std::max_element(sum_tmp.begin(), sum_tmp.end(), abs_compare);
            result.push_back(std::distance(sum_tmp.begin(), tmp));
            sum_tmp.clear();
        }

        if (t == 0 || t(51, 32) == (ap_uint<20>)-1) continue;

        int x = t(51, 32).to_int();
        int y = t(31, 0).to_int();
        for (int i = 0; i < num_of_class; i++) {
            sum[i] += theta[i][x - 1] * y;
        }
    }
}

int main() {
    int err = 0;

    std::vector<ap_uint<32> > dataset;
    std::vector<ap_uint<64> > dataset2;

    std::string in_dir = "../../../../";
    err = load_dat(dataset, dataset2, in_dir);
    if (err) {
        std::cout << "Error found during load_dat\n";
        return 1;
    }

    hls::stream<ap_uint<32> > i_data_array[CH_NM];
    hls::stream<bool> i_e_array;

    for (int i = 0; i < dataset.size() / CH_NM; i++) {
        for (int j = 0; j < CH_NM; j++) {
            i_data_array[j].write(dataset[i * CH_NM + j]);
        }
        i_e_array.write(false);
    }
    i_e_array.write(true);

    std::vector<std::vector<double> > model_theta(num_of_class, std::vector<double>(num_of_term, 0.0));
    std::vector<double> model_pi(num_of_class, 0.0);

    hls::stream<ap_uint<64> > i_model_theta_strm;
    hls::stream<ap_uint<64> > i_model_prior_strm;

    err = load_model(in_dir, model_theta, model_pi, i_model_theta_strm, i_model_prior_strm);
    if (err) {
        std::cout << "Error found during load_model\n";
        return 1;
    }

    hls::stream<ap_uint<10> > o_class_strm;
    hls::stream<bool> o_e_strm;

    multinomialNB(num_of_class, num_of_term, i_model_theta_strm, i_model_prior_strm, i_data_array, i_e_array,
                  o_class_strm, o_e_strm);

    std::vector<int> sw_result;
    generate_golden(model_theta, model_pi, dataset2, sw_result);

    int cnt = 0;
    bool last = o_e_strm.read();
    while (!last) {
        last = o_e_strm.read();
        ap_uint<10> t = o_class_strm.read();
        if (t != sw_result[cnt]) {
            std::cout << "The " << (cnt + 1) << "th sample failed to predict!!! sw:" << sw_result[cnt]
                      << " --> hw:" << t << std::endl;
            err++;
        }

        cnt++;
    }

    return err;
}
