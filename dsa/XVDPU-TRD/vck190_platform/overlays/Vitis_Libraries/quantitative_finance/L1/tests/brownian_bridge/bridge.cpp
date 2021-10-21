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
#include "xf_fintech/brownian_bridge.hpp"

void read_data(int steps, double* inputVal, hls::stream<double>& strm) {
    for (int i = 0; i < steps; ++i) {
#pragma HLS pipeline II = 1
        strm.write(inputVal[i]);
    }
}
void write_data(int steps, double* outputVal, hls::stream<double>& strm) {
    for (int i = 0; i < steps; ++i) {
#pragma HLS pipeline II = 1
        outputVal[i] = strm.read();
        ;
    }
}
void brownian_trans(int steps,
                    double* inputVal,
                    double* outputVal,
                    xf::fintech::BrownianBridge<double, 1024>& bridge_inst) {
#pragma HLS dataflow
    hls::stream<double> in_strm;
    hls::stream<double> out_strm;

    read_data(steps, inputVal, in_strm);
    bridge_inst.transform(in_strm, out_strm);
    write_data(steps, outputVal, out_strm);
}
void brownian_bridge_top(int steps, double inputVal[100], double outputVal[100]) {
    // void brownian_bridge_top(int steps, hls::stream<double>& in_strm,
    // hls::stream<double>& out_strm) {

    xf::fintech::BrownianBridge<double, 1024> bridge;

    bridge.initialize(steps);

    // bridge.transform(in_strm, out_strm);

    brownian_trans(steps, inputVal, outputVal, bridge);
}
