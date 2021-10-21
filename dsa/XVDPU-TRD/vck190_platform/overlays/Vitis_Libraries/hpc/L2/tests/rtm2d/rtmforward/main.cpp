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

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "rtm2d.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "assert.hpp"
using namespace std;

typedef WideData<RTM_dataType, RTM_nPE> RTM_wideType;
typedef WideData<RTM_wideType, 2> RTM_pairType;

int main(int argc, char** argv) {
    int err4 = 0, err5 = 0, err6 = 0;
    bool pass4, pass5, pass6;

    const int requiredArg = 6;
    if (argc < requiredArg) return EXIT_FAILURE;

    unsigned int argId = 0;

    string l_xclbinFile(argv[++argId]);
    string filePath = argv[++argId];
    int l_height = atoi(argv[++argId]);
    int l_width = atoi(argv[++argId]);
    int l_time = atoi(argv[++argId]);
    int l_shot = 1;

    int l_imgX = l_width - 2 * RTM_NXB;
    int l_imgZ = l_height - 2 * RTM_NZB;
    int l_area = l_width * l_height / RTM_nPE;

    bool l_verify = false;
    if (argc > 1 + argId) l_verify = atoi(argv[++argId]) == 0 ? false : true;

    unsigned int l_deviceId = 0;
    if (argc > 1 + argId) l_deviceId = atoi(argv[++argId]);

    string msg;

    msg = "l_height must be less than " + to_string(RTM_maxDim);
    myAssert(l_height <= RTM_maxDim, msg);

    msg = "l_time must be multiple of " + to_string(RTM_numFSMs);
    myAssert(l_time % RTM_numFSMs == 0, msg);

    msg = "Image size must be multiple of " + to_string(RTM_parEntries);
    myAssert(l_width * l_height % RTM_parEntries == 0, msg);

    host_buffer_t<RTM_pairType> p_snap;
    host_buffer_t<RTM_dataType> p_upb;
    vector<RTM_dataType> p, pp, snap0, snap1, ref;

    p.resize(l_width * l_height);
    pp.resize(l_width * l_height);

    FPGA fpga(l_deviceId);
    fpga.xclbin(l_xclbinFile);

    ForwardKernel<RTM_dataType, RTM_order, RTM_nPE> fwd(&fpga, l_height, l_width, RTM_NZB, RTM_NXB, l_time, l_shot);

    fwd.loadData(filePath);
    double elapsedF = 0;

    auto start = chrono::high_resolution_clock::now();
    for (int s = 0; s < l_shot; s++) {
        elapsedF += fwd.run(s, l_width / 2, p_snap, p_upb);

        if (l_verify) {
            for (int i = 0; i < l_area; i++) {
                for (int pe = 0; pe < RTM_nPE; pe++) {
                    pp[i * RTM_nPE + pe] = p_snap[i][0][pe];
                    p[i * RTM_nPE + pe] = p_snap[i][1][pe];
                }
            }

            readBin(filePath + "upb.bin", sizeof(float) * l_width * RTM_order * l_time / 2, ref);
            readBin(filePath + "snap0.bin", sizeof(float) * l_width * l_height, snap0);
            readBin(filePath + "snap1.bin", sizeof(float) * l_width * l_height, snap1);

            pass4 = compare<RTM_dataType>(l_width * RTM_order * l_time / 2, p_upb.data(), ref.data(), err4);
            cout << "There are in total " << err4 << " errors in upb v.s. ref" << endl;

            pass5 = compare<RTM_dataType>(l_width * l_height, pp.data(), snap0.data(), err5);
            cout << "There are in total " << err5 << " errors in pp v.s. snap0" << endl;

            pass6 = compare<RTM_dataType>(l_width * l_height, p.data(), snap1.data(), err6);
            cout << "There are in total " << err6 << " errors in p v.s. snap1" << endl;

            if (pass4 && pass5 && pass6) {
                cout << "Test passed!" << endl;
            } else {
                cout << "Test failed, there are in total " << err4 + err5 + err6 << " errors!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = finish - start;

    cout << "Execution completed" << endl;
    cout << "Average forward execution time " << elapsedF / l_shot << "s." << endl;
    cout << "Average total execution time " << elapsed.count() / l_shot << "s." << endl;

    return EXIT_SUCCESS;
}
