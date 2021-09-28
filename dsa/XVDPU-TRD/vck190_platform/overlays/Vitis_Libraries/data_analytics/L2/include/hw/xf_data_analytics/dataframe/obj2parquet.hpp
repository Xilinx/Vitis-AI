
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
#ifndef _XF_DATA_ANALYTICS_ETL_HPP_
#define _XF_DATA_ANALYTICS_ETL_HPP_
/**
 * @brief From DataFrame to Parquet
 * \rst
 * \endrst
 *
 * @param
 * TBD
 *
 */
extern "C" void ObjToParquet(ap_uint<88> ddr_obj[1 << 25], ap_uint<8> schema[16], ap_uint<64> ddr_parquet[1 << 25]);
#endif
