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
#include <cstring>
#include <iostream>
#include <stdlib.h>

#include "code.hpp"

// random data
ap_uint<W_DATA> rand_t() {
    int min = W_DATA >= 16 ? 16 : W_DATA;
    ap_uint<W_DATA> c = rand() % (1 << min);
    return c;
}

/**
 * @brief generate test data
 * @param data_buf stored data
 * @param num  the number of generated data
 */
void gen_data(t_data* data_buf, const int num) {
    for (int i = 0; i < num; ++i) data_buf[i] = rand_t();
}
/**
 * @brief check the result
 * @param data_buf  original data
 * @param res_buf   updated data
 * @param num  the number of checking data
 * @return 0-pass, 1-fail
 */
int check_data(t_data* data_buf, t_data* res_buf, const int num) {
    // check the result by their sums because distribution on load balance is NOT order-preserving.
    int sum_gld = 0;
    int sum_res = 0;
    for (int i = 0; i < num; ++i) {
        t_data new_data = update_data(data_buf[i]);
        sum_gld += calculate(new_data);
        sum_res += calculate(res_buf[i]);
    }
    return (sum_gld != sum_res);
}

int main() {
    t_data* data_ddr = (t_data*)malloc(DATA_LEN * sizeof(t_data));
    t_data* res_ddr = (t_data*)malloc(DATA_LEN * sizeof(t_data));

    // reshape:  total bits = DATA_LEN*sizeof(t_data)*8 = W_STRM * num
    int num = DATA_LEN * sizeof(t_data) * 8 / (W_STRM);
    const int len = DATA_LEN;
    // generate data
    gen_data(data_ddr, len);
    std::cout << " W_AXI  = " << W_AXI << "  num = " << num << std::endl;
    std::cout << " W_DATA = " << W_DATA << "  len = " << len << std::endl;
    // core
    top_core((ap_uint<W_AXI>*)data_ddr, (ap_uint<W_AXI>*)res_ddr, num);

    //  check result
    int err = check_data(data_ddr, res_ddr, len);
    if (err == 0)
        std::cout << "********PASS*********" << std::endl;
    else
        std::cout << "********FAIL*********" << std::endl;
    free(data_ddr);
    free(res_ddr);
    return err;
}
