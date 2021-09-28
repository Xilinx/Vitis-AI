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

#include "xf_fintech/bs_model.hpp"
void dut(double riskFreeRate,
         double dividendYield,
         double timeLength,
         double volatility,
         double underlying,
         unsigned int timeSteps,
         unsigned int paths,
         double rand_in[10240],
         double out[10240]) {
    xf::fintech::BSModel<double> BSInst;

    double dt = timeLength / timeSteps;

    BSInst.riskFreeRate = riskFreeRate;
    BSInst.dividendYield = dividendYield;
    BSInst.volatility = volatility;

    BSInst.variance(dt);
    BSInst.stdDeviation();
    BSInst.updateDrift(dt);

    double xBuff[1024];
#ifndef __SYNTHESIS__
    assert(paths <= 1024);
#endif
    for (int i = 0; i < timeSteps; ++i) {
        for (int j = 0; j < 1024; ++j) {
#pragma HLS pipeline
            double x0;
            if (i == 0)
                x0 = underlying;
            else
                x0 = xBuff[j];
            double dw = rand_in[i * paths + j];
#ifndef __SYNTHESIS__
            if (j < 10) std::cout << "i=" << i << " ,j=" << j << " ,dw=" << dw << std::endl;
#endif
            double s = BSInst.evolve(x0, dt, dw);
#ifndef __SYNTHESIS__
            if (j < 10) std::cout << "i=" << i << " ,j=" << j << " ,s=" << s << std::endl;
#endif
            out[i * paths + j] = s;
            xBuff[j] = s;
        }
    }
}
