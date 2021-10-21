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
#include <hls_stream.h>
#include "xf_data_analytics/dataframe/write_to_dataframe.hpp"

using namespace xf::data_analytics::dataframe;

void writeToDataFrameWrapper(hls::stream<Object>& obj_strm, ap_uint<64> buff0[26214400]) {
    xf::data_analytics::dataframe::writeToDataFrame(obj_strm, buff0);
}
