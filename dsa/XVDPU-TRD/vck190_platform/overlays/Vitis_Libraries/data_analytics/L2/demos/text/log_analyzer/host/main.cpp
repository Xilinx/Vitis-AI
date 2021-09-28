/*
 * Copyright 2020 Xilinx, Inc.
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
#include "log_analyzer_config.hpp"
#include "log_analyzer.hpp"
#include "ref_result.hpp"
#include <fstream>
#include "oniguruma.h"
enum {
    MAX_MSG_DEPTH = 250000000,
    MAX_LNM = 6000000,
    GEO_DB_DEPTH = 2147483648,
    GEO_DB_LNM = 5000000,
    JSON_OUT_DEPTH = 4294967295
};

int store_dt(std::string out_path, const uint8_t* out_buff) {
    uint64_t sz = 0;
    memcpy(&sz, out_buff, 8);

    printf("total write to disk %ld Byte data\n", sz);
    std::ofstream out_file(out_path);
    if (!out_file.is_open()) {
        std::cerr << "ERROR: " << out_path << "cannot be opened for read.\n";
        return -1;
    }
    out_file.write((char*)out_buff + 8, sz);
    out_file.close();
    return 0;
}
int load_msg(std::string log_path, uint64_t* msg_buff, uint16_t* len_buff, uint32_t& lnm, int limit_ln) {
    typedef union {
        char c_a[8];
        uint64_t d;
    } uint64_un;

    std::ifstream log_file(log_path);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: " << log_path << "cannot be opened for read.\n";
        return -1;
    }

    lnm = 0;
    uint32_t offt = 0;
    std::string line;
    // for(int l = 0; l < 20; ++l) {
    //      std::ifstream log_file(log_path);
    //      if (!log_file.is_open()) {
    //          std::cerr << "ERROR: " << log_path << "cannot be opened for read.\n";
    //          return -1;
    //      }
    while (getline(log_file, line)) {
        size_t sz = line.size();
        if (sz > 4090) {
            std::cout << "WARNNING: length of log exceeds 4090.\n";
            // return 1;
            // ingore the empty line
        } else if (sz > 0 && offt < MAX_MSG_DEPTH && lnm < MAX_LNM && (limit_ln == -1 || lnm < limit_ln)) {
            len_buff[lnm] = sz;
            for (int i = 0; i < (sz + 7) / 8; ++i) {
                uint64_un out;
                for (unsigned int j = 0; j < 8; ++j) {
                    if (i * 8 + j < sz) {
                        out.c_a[j] = line[i * 8 + j];
                    } else {
                        out.c_a[j] = ' ';
                    }
                }
                msg_buff[offt++] = out.d;
            }
            lnm++;
        }
    }
    log_file.close();
    //}
    // one more
    return 0;
}
int load_geoIP_dt(std::string geo_path, uint8_t* geo_buff, uint32_t* geo_oft_buff, uint32_t& lnm) {
    std::ifstream geo_file(geo_path);
    if (!geo_file.is_open()) {
        std::cerr << "ERROR: " << geo_path << "cannot be opened for read.\n";
        return -1;
    }
    uint32_t offt = 0;
    std::string line;
    while (getline(geo_file, line)) {
        size_t sz = line.size();
        memcpy(geo_buff + offt, line.c_str(), sz);
        geo_oft_buff[lnm] = offt;
        offt += sz;
        lnm++;
    }
    geo_oft_buff[lnm] = offt;
    geo_file.close();
    return 0;
}

int refComp(std::string ref_path, uint8_t* out_value) {
    std::ifstream inFile(ref_path, std::ios::in);
    std::string lineStr;
    int err = 0;
    uint64_t cnt = 8;
    while (getline(inFile, lineStr)) {
        uint8_t* tmp_dec = (uint8_t*)lineStr.data();
        for (int i = 0; i < lineStr.size(); i++) {
            if (tmp_dec[i] != out_value[cnt++]) {
                err++;
                std::cout << "tmp_dec[" << i << "]=" << tmp_dec[i] << ",out_value[" << cnt - 1 << "]=" << out_value[cnt]
                          << std::endl;
            }
        }
        cnt++;
    }
    return err;
}

int main(int argc, const char* argv[]) {
    std::cout << "----------------------log analyzer----------------" << std::endl;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // command argument parser
    x_utils::ArgParser parser(argc, argv);

    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
    std::string log_path;
    if (!parser.getCmdOption("-log", log_path)) {
        std::cout << "ERROR:  input log path is not specified.\n";
        return 1;
    }
    std::string geoip_path;
    if (!parser.getCmdOption("-dat", geoip_path)) {
        std::cout << "ERROR:  input geo path is not specified.\n";
        return 1;
    }

    std::string golden_path;
    bool ref_golden = 1;
    if (!parser.getCmdOption("-ref", golden_path)) {
        ref_golden = 0;
    }

    std::string mmdb_path;
    bool ref_mmdb = 1;
    if (!parser.getCmdOption("-mmdb", mmdb_path)) {
        ref_mmdb = 0;
    }

    std::string out_path;
    bool out_flag = 1;
    if (!parser.getCmdOption("-out", out_path)) {
        out_flag = 0;
    }

    std::string ln_nm;
    int limit_ln = -1;
    if (parser.getCmdOption("-lnm", ln_nm)) {
        try {
            limit_ln = std::stoi(ln_nm);
        } catch (...) {
            limit_ln = -1;
        }
    }
    std::string pattern =
        "^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \\[(?<time>[^\\]]*)\\] \"(?<method>\\S+)(?: "
        "+(?<path>[^\\\"]*?)(?: +\\S*)?)?\" (?<code>[^ ]*) (?<size>[^ ]*)(?: \"(?<referer>[^\\\"]*)\" "
        "\"(?<agent>[^\\\"]*)\"(?:\\s+(?<http_x_forwarded_for>[^ ]+))?)?$";

    // allocate the in-memory buffer
    x_utils::MM mm;
    uint64_t* msg_buff = mm.aligned_alloc<uint64_t>(MAX_MSG_DEPTH);
    uint16_t* msg_len_buff = mm.aligned_alloc<uint16_t>(MAX_LNM);
    uint8_t* geo_buff = mm.aligned_alloc<uint8_t>(GEO_DB_DEPTH);
    uint32_t* geo_oft_buff = mm.aligned_alloc<uint32_t>(GEO_DB_LNM);

    uint8_t* out_buff = mm.aligned_alloc<uint8_t>(JSON_OUT_DEPTH);
    // make the buffer really is allocated.
    for (int i = 0; i < JSON_OUT_DEPTH / (1024 * 4); ++i) {
        out_buff[i * 1024 * 4] = 0;
    }
    // memset(out_buff, 0, JSON_OUT_DEPTH);
    // constructor of reEngine
    xf::search::logAnalyzer logAnalyzerInst(xclbin_path);
    xf::search::ErrCode err_code;
    // compile pattern
    err_code = logAnalyzerInst.compile(pattern);

    if (err_code != 0) return -1;

    uint32_t msg_lnm = 0;
    // load message data from disk to in-memory buffer
    printf("load log from disk to in-memory buffer\n");
    load_msg(log_path, msg_buff, msg_len_buff, msg_lnm, limit_ln);

    // load GeoIP data from disk to in-memory buffer
    uint32_t geo_lnm = 0;
    printf("load geoip database disk to in-memory buffer\n");
    load_geoIP_dt(geoip_path, geo_buff, geo_oft_buff, geo_lnm);

    // call logAnalyzerInst to do analyzer
    printf("execute log analyzer\n");
    err_code = logAnalyzerInst.analyze(msg_buff, msg_len_buff, msg_lnm, geo_buff, geo_oft_buff, geo_lnm, out_buff);
    if (err_code != 0) return -1;
    // check the result
    int err = 0;
    if (ref_golden)
        err = refComp(golden_path, out_buff);
    else if (ref_mmdb)
        err = refCalcu(log_path, pattern, mmdb_path, out_buff);

    if (out_flag) {
        // store the result from in-memory buffer to disk in JSON format
        printf("write result from in-memory buffer to disk\n");
        store_dt(out_path, out_buff);
    }
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
