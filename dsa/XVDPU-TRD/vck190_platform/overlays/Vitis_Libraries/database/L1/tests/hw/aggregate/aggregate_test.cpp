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

#include <time.h> //time
#include <cmath>
#include <cstdlib> // std::rand, std::srand
#include <iomanip>
#include <iostream> // std::cout
#include <numeric>
#include <sstream>
#include <string>
#include <valarray> // std::valarray
#include <vector>   // std::vector

#define AP_INT_MAX_W 4096
#include <ap_int.h>

#include "xf_database/aggregate.hpp"
#include "hls_stream.h"

#define AOP xf::database::AOP_MAX
#define FUN max

#define INPUT_LENGTH 100
#define DATA_TYPE double // int||double
#define RANDOM_TYPE 2    // enum typeName{Int,Float,Double};
#define IMIN -100
#define IMAX 100
#define FMIN -10.00
#define FMAX 10.00

void hls_db_aggregate_function(hls::stream<DATA_TYPE>& in_strm,
                               hls::stream<bool>& e_in_strm,
                               hls::stream<DATA_TYPE>& out_strm,
                               hls::stream<bool>& e_out_strm) {
    xf::database::aggregate<AOP>(in_strm, e_in_strm, out_strm, e_out_strm);
}

namespace ref_aggregate {
template <typename T>
void max(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    T ret = in_strm.read();
    for (int i = 1; i < len; i++) {
        ret = std::max(ret, (in_strm.read()));
    }
    out_strm.write(ret);
}

template <typename T>
void min(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    T ret = in_strm.read();
    for (int i = 1; i < len; i++) {
        ret = std::min(ret, (in_strm.read()));
    }
    out_strm.write(ret);
}

template <typename T>
void sum(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    std::vector<T> in_vector(len);
    T ret;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
    }
    ret = std::accumulate(in_vector.begin(), in_vector.end(), 0.0);
    std::cout << "return" << ret << std::endl;
    out_strm.write(ret);
}

template <typename T>
void mean(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    std::vector<T> in_vector(len);
    double ret;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
    }
    ret = std::accumulate(in_vector.begin(), in_vector.end(), 0.0) / len;
    std::cout << "mean org" << ret << std::endl;
    out_strm.write(ret);
    std::cout << "mean fininal" << ret << std::endl;
}

template <typename T>
void numNonZeros(hls::stream<T>& in_strm, uint64_t len, hls::stream<int>& out_strm) {
    std::vector<T> in_vector(len);
    uint64_t countVal = 0;
    uint64_t ret = 0;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
    }
    countVal = std::count(in_vector.begin(), in_vector.end(), 0);
    ret = len - countVal;
    out_strm.write(ret);
    std::cout << "numNonZeros:" << ret << std::endl;
}

template <typename T>
void variance(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    std::vector<T> in_vector(len);
    double mean = 0;
    double mean_temp = 0;
    double ret = 0;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
    }
    mean = std::accumulate(in_vector.begin(), in_vector.end(), 0.0) / len;
    std::cout << "mean:" << mean << std::endl;
    for (uint64_t i = 0; i < len; i++) {
        ret += (in_vector[i] - mean) * (in_vector[i] - mean);
    }
    ret = ret / (len);
    std::cout << "variance:" << ret << std::endl;
    out_strm.write(ret);
}

template <typename T>
void normL1(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    std::vector<T> in_vector(len);
    double ret = 0;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
        in_vector[i] = std::abs(in_vector[i]);
    }

    ret = std::accumulate(in_vector.begin(), in_vector.end(), 0.0);
    std::cout << "normL1:" << ret << std::endl;
    out_strm.write(ret);
}

template <typename T>
void normL2(hls::stream<T>& in_strm, uint64_t len, hls::stream<T>& out_strm) {
    std::vector<T> in_vector(len);
    double ret = 0;
    for (uint64_t i = 0; i < len; i++) {
        in_vector[i] = in_strm.read();
        in_vector[i] *= in_vector[i];
    }
    ret = std::accumulate(in_vector.begin(), in_vector.end(), 0.0);
    ret = sqrt(ret);
    std::cout << "normL2:" << ret << std::endl;
    out_strm.write(ret);
}
}

// generate a random integer sequence between speified limits a and b (a<b);
int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}

// generate a random double sequence between speified limits a and b (a<b);
double rand_double(double a, double b) {
    return (double)rand() / RAND_MAX * (b - a) + a;
}

template <typename T>
void generate_test_data(uint64_t len, std::vector<T>& testVector) {
    typename std::vector<T>::iterator it;
    enum typeName { Int, Float, Double };
    char random_type;
    random_type = RANDOM_TYPE;
    if (random_type == Int) {
        // srand(time(NULL));
        srand(1);
        for (int i = 0; i < len; i++) {
            testVector.push_back(rand_int(IMIN, IMAX)); // generate a random value
        }
    } else if (random_type == Double) {
        // srand(time(NULL));
        for (int i = 0; i < len; i++) {
            testVector.push_back(rand_double(FMIN, FMAX));
        }
    }
    // print testdata
    std::cout << "TestData contains:";
    for (it = testVector.begin(); it != testVector.end(); it++) std::cout << ' ' << *it;
    std::cout << '\n';
}

int main() {
    std::vector<DATA_TYPE> testVector;
    hls::stream<DATA_TYPE> in_strm("in_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<DATA_TYPE> out_strm("out_strm");
    hls::stream<bool> e_out_strm("e_out_strm");
    hls::stream<DATA_TYPE> ref_in_strm("ref_in_strm");
    hls::stream<DATA_TYPE> ref_out_strm("ref_out_strm");
    int nerror = 0;
    uint64_t len = INPUT_LENGTH;
    DATA_TYPE retstr;
    DATA_TYPE goldenVal;

    // generate test data
    generate_test_data<DATA_TYPE>(len, testVector);

    // prepare input data
    std::cout << "testVector List:" << std::endl;
    for (std::string::size_type i = 0; i < len; i++) {
        // print vector value
        std::cout << "Index=" << i << ' ';
        std::cout << "Vector=" << testVector[i] << std::endl;

        in_strm.write(testVector[i]); // write data to in_strm
        e_in_strm.write(false);
        ref_in_strm.write(testVector[i]); // write data to ref_in_strm
    }
    e_in_strm.write(true);

    // call aggregate function
    hls_db_aggregate_function(in_strm, e_in_strm, out_strm, e_out_strm);

    bool e0 = e_out_strm.read();
    bool e1 = e_out_strm.read();
    if (e0 != false) ++nerror;
    if (e1 != true) ++nerror;

    retstr = out_strm.read();
    std::cout << "test_data=" << retstr << std::endl;

    // check output
    ref_aggregate::FUN<DATA_TYPE>(ref_in_strm, len, ref_out_strm);
    goldenVal = ref_out_strm.read();
    std::cout << "golden_data=" << goldenVal << std::endl;

    // compare golden and test data
    if (retstr != goldenVal) ++nerror;

    // print result
    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}
