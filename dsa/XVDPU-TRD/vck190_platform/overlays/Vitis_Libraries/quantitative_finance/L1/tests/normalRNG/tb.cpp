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

#include <ap_int.h>
#include <math.h>
#include <iostream>
#include "dut.hpp"
extern "C" {
#include "dc.h"
}

#define SAMPLE_NUM (1 << 10)

extern "C" void dut(const int num,
                    const int preRun,
                    ap_uint<32> st[4],
                    double outputMT19937ICN[SAMPLE_NUM],
                    double outputMT2203ICN[SAMPLE_NUM],
                    double outputMT19937BoxMuller[SAMPLE_NUM]);

int main() {
    const int sampleNum = SAMPLE_NUM;
    const int preRun = 0;

    // Get rng init parameters
    mt_struct* mts[1];
    init_dc(4172);
    mts[0] = get_mt_parameter_id(32, 2203, 0);
    sgenrand_mt(1234, mts[0]);
    ap_uint<32> st[4];
    st[0] = 1234;
    st[1] = mts[0]->aaa;
    st[2] = mts[0]->maskB;
    st[3] = mts[0]->maskC;
    //
    double resultMT19937ICN[sampleNum];
    double resultMT2203ICN[sampleNum];
    double resultMT19937BoxMuller[sampleNum];

    double avgMT19937ICN = 0;
    double avgMT2203ICN = 0;
    double avgMT19937BoxMuller = 0;
    double sdMT19937ICN = 0;
    double sdMT2203ICN = 0;
    double sdMT19937BoxMuller = 0;

    dut(sampleNum, preRun, st, resultMT19937ICN, resultMT2203ICN, resultMT19937BoxMuller);

    for (int i = 0; i < sampleNum; i++) {
        avgMT19937ICN += resultMT19937ICN[i];
        avgMT2203ICN += resultMT2203ICN[i];
        avgMT19937BoxMuller += resultMT19937BoxMuller[i];
        // std::cout << i << " : " << resultMT19937BoxMuller[i] << " ,sum: " << avgMT19937BoxMuller << std::endl;
    }
    avgMT19937ICN /= sampleNum;
    avgMT2203ICN /= sampleNum;
    avgMT19937BoxMuller /= sampleNum;

    for (int i = 0; i < sampleNum; i++) {
        sdMT19937ICN += (resultMT19937ICN[i] - avgMT19937ICN) * (resultMT19937ICN[i] - avgMT19937ICN);
        sdMT2203ICN += (resultMT2203ICN[i] - avgMT2203ICN) * (resultMT2203ICN[i] - avgMT2203ICN);
        sdMT19937BoxMuller +=
            (resultMT19937BoxMuller[i] - avgMT19937BoxMuller) * (resultMT19937BoxMuller[i] - avgMT19937BoxMuller);
        // std::cout << "resultboxmuller: " << resultMT19937BoxMuller[i] << std::endl;
    }
    sdMT19937ICN = sqrt(sdMT19937ICN / (sampleNum - 1));
    sdMT2203ICN = sqrt(sdMT2203ICN / (sampleNum - 1));
    sdMT19937BoxMuller = sqrt(sdMT19937BoxMuller / (sampleNum - 1));

    std::cout << "SampleNumber: " << sampleNum << "   Prerun: " << preRun << std::endl;

    std::cout << "Average of " << sampleNum << " MT19937IcnRng samples: " << avgMT19937ICN << std::endl;
    std::cout << "Average of " << sampleNum << " MT2203IcnRng samples: " << avgMT2203ICN << std::endl;
    std::cout << "Average of " << sampleNum << " MT19937BoxMullerNormalRng samples: " << avgMT19937BoxMuller
              << std::endl;

    std::cout << "Standard Deviation of " << sampleNum << " MT19937IcnRng samples: " << sdMT19937ICN << std::endl;
    std::cout << "Standard Deviation of " << sampleNum << " MT2203IcnRng samples: " << sdMT2203ICN << std::endl;
    std::cout << "Standard Deviation of " << sampleNum << " MT19937BoxMullerNormalRng samples: " << sdMT19937BoxMuller
              << std::endl;

    /*
        double tMT19937ICN = (avgMT19937ICN - 0.0) / (sdMT19937ICN / sqrt(sampleNum));
        double tMT2203ICN = (avgMT2203ICN - 0.0) / (sdMT2203ICN / sqrt(sampleNum));
        double tMT19937BoxMuller = (avgMT19937BoxMuller - 0.0) / (sdMT19937BoxMuller / sqrt(sampleNum));

        std::cout << "Student's t-test on MT19937ICN : " << tMT19937ICN << std::endl;
        std::cout << "Student's t-test on MT2203ICN : " << tMT2203ICN << std::endl;
        std::cout << "Student's t-test on MT19937BoxMuller : " << tMT19937BoxMuller << std::endl;
    */

    if (abs(avgMT19937ICN) > sqrt(1.0 / sampleNum)) {
        std::cout << "MT19937 sample average is out of one sigma" << std::endl;
        return 1;
    }
    if (abs(avgMT2203ICN) > sqrt(1.0 / sampleNum)) {
        std::cout << "MT19937 sample average is out of one sigma" << std::endl;
        return 1;
    }
    if (abs(avgMT19937BoxMuller) > sqrt(1.0 / sampleNum)) {
        std::cout << "MT19937 sample average is out of one sigma" << std::endl;
        return 1;
    }
    return 0;
}
