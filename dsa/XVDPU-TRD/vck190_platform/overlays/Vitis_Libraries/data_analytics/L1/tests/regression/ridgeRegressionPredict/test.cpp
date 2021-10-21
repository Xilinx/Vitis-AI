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

#include "xf_data_analytics/regression/linearRegression.hpp"
#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/common/enums.hpp"

extern void dut(ap_uint<32> cols,
                double weight[8][25],
                double intercept,
                hls::stream<double> opStrm[8],
                hls::stream<bool>& eOpStrm,
                hls::stream<double> retStrm[1],
                hls::stream<bool>& eRetStrm) {
    using namespace xf::data_analytics::regression;
    using namespace xf::data_analytics::regression::internal;
    using namespace xf::data_analytics::common::internal;
    ridgeRegressionPredict<double, 8, 25, URAM, URAM> processor;
    processor.setWeight(weight, cols);
    processor.setIntercept(intercept);
    processor.predict(opStrm, eOpStrm, retStrm, eRetStrm, cols);
}

#ifndef __SYNTHESIS__
bool diff(double a, double b) {
    double tmp = a - b;
    if (tmp > 0.00001 || tmp < -0.00001) {
        std::cout << "a: " << a << "   b: " << b << std::endl;
        return false;
    } else {
        return true;
    }
}

int main() {
    xf::data_analytics::common::internal::MT19937 rng;
    rng.seedInitialization(42);
    const int rows = 200;
    const int cols = 23;
    double weight[8][25];
    for (int i = 0; i < cols; i++) {
        weight[i % 8][i / 8] = rng.next();
    }
    double intercept = rng.next();

    hls::stream<double> opStrm[8];
    hls::stream<bool> eOpStrm;
    hls::stream<double> retStrm[1];
    hls::stream<bool> eRetStrm;
    double golden[rows];

    for (int i = 0; i < rows; i++) {
        double tmp = intercept;
        for (int j = 0; j < cols; j += 8) {
            for (int k = 0; k < 8; k++) {
                double r = rng.next();
                opStrm[k].write(r);
                if ((j + k) < cols) {
                    tmp += (r * weight[k][j / 8]);
                }
            }
            eOpStrm.write(false);
        }
        golden[i] = tmp;
    }
    eOpStrm.write(true);

    dut(cols, weight, intercept, opStrm, eOpStrm, retStrm, eRetStrm);

    bool res = true;
    for (int i = 0; i < rows; i++) {
        eRetStrm.read();
        res = diff(retStrm[0].read(), golden[i]);
    }
    eRetStrm.read();

    if (res) {
        return 0;
    } else {
        return 1;
    }
}
#endif
