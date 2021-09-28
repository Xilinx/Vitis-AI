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

#include <iostream>
#include <fstream>
#include <hls_stream.h>
#include "xf_data_analytics/common/obj_interface.hpp"
#include "xf_data_analytics/dataframe/df_utils.hpp"
#include "xf_data_analytics/dataframe/read_from_dataframe.hpp"

using namespace xf::data_analytics::dataframe;

void writeToDataFrameWrapper(hls::stream<Object>& obj_strm, ap_uint<64> buff0[26214400]);

int main() {
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "------------------ in main ------------------" << std::endl;

    std::cout << "--- generate data in object stream struct ---" << std::endl;

    hls::stream<Object> obj_strm("in_obj_strm");
    Object obj_data;

    // 3 rounds to finish 1 file
    int round = 1;
    for (int r = 0; r < round; r++) {
        // each col has 10 json lines
        for (int l = 0; l < 1000; l++) {
            // each json line has 4 data
            for (int n = 0; n < 5; n++) {
                ap_uint<64> dat = ((n + l) * 8) << (16 + r) + (n * 8); // 64bit double data
                ap_uint<16> field_id = n;                              // 4 fields
                ap_uint<4> valid = 8;                                  // 64-bit valid

                ap_uint<4> type = TInt64; // int64
                if (n == 2) {
                    ap_uint<4> type = TDouble; // int64
                }
                if (n == 3) {
                    type = TBoolean;
                }
                if (n == 4) {
                    type = TString;
                    if (l == 0) valid = 8;
                    if (l == 1) valid = 7;
                    if (l == 2) valid = 8;
                    if (l == 3) valid = 2;
                    if (l == 4) valid = 5;
                    if (l == 5) valid = 8;
                    if (l == 6) valid = 7;
                    if (l == 7) valid = 8;
                    if (l == 8) valid = 8;
                    if (l == 9) valid = 4;
                }

                if (n == 0 && l == 4) {
                    valid = 0;
                }
                if (n == 1 && l == 6) {
                    valid = 0;
                }

                obj_data.set_data(dat);
                obj_data.set_id(field_id);
                obj_data.set_valid(valid);
                obj_data.set_type(type);
                obj_strm.write(obj_data);

                std::cout << obj_data.get_type() << ",";
                if (obj_data.get_valid() == 0) {
                    std::cout << "null;        ";
                } else {
                    std::cout << obj_data.get_valid() << ",";
                    std::cout << obj_data.get_id() << ",";
                    std::cout << obj_data.get_data() << "; ";
                }
            }

            ap_uint<4> type = 13; // end of json line
            obj_data.set_type(type);
            obj_strm.write(obj_data);

            ap_uint<4> tf = obj_data.get_type();
            std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
            std::cout << tf_str << ",";
            // std::cout << obj_data.get_valid() << ",";
            // std::cout << obj_data.get_id() << ",";
            // std::cout << obj_data.get_data() << std::endl;
            std::cout << std::endl;
        }

        ap_uint<4> type = 14; // end of col
        obj_data.set_type(type);
        obj_strm.write(obj_data);

        ap_uint<4> tf = obj_data.get_type();
        std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
        std::cout << tf_str << ",";
        // std::cout << obj_data.get_valid() << ",";
        // std::cout << obj_data.get_id() << ",";
        // std::cout << obj_data.get_data();
        std::cout << std::endl << std::endl;
    }
    ap_uint<4> type = 15; // end of file
    obj_data.set_type(type);
    obj_strm.write(obj_data);

    ap_uint<4> tf = obj_data.get_type();
    std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
    std::cout << tf_str << ",";
    // std::cout << obj_data.get_valid() << ",";
    // std::cout << obj_data.get_id() << ",";
    // std::cout << obj_data.get_data();
    std::cout << std::endl << std::endl;

    ap_uint<64>* buff0;
    buff0 = (ap_uint<64>*)malloc(26214400 * sizeof(ap_uint<64>));
    memset(buff0, 0, sizeof(ap_uint<64>) * 26214400);

    writeToDataFrameWrapper(obj_strm, buff0);

    return 0;
}
