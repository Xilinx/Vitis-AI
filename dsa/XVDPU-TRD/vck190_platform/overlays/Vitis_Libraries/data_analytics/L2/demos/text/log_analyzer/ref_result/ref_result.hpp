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
#include "grok.hpp"
#include "geoip2.hpp"
#include <fstream>
#include <vector>
#include "x_utils.hpp"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
using namespace x_utils;
const std::string geoip2FN[] = {
    "postal_code",    "latitude",         "longitude",    "accuracy_radius",        "continent_code",
    "continent_name", "country_iso_code", "country_name", "subdivision_1_iso_code", "subdivision_1_name",
    "city_name",      "metro_code",       "time_zone"};

const int idFN[] = {0x7, 0x12, 0x13, 0x11, 0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x6, 0x10, 0x8};

int refCalcu(std::string msg_path,
             std::string pattern,
             std::string geoip_path,
             uint8_t* test_result = NULL,
             std::string json_path = "") {
    int nerr = 0;
    // init
    const int subNum = 13;

    std::string fieldName[] = {
        "remote", "host", "user", "time", "method", "path", "code", "size", "referer", "agent", "http_x_forwarded_for"};

    regex_t* reg;
    OnigRegion* region;
    region = onig_region_new();
    OnigErrorInfo einfo;
    OnigEncoding use_encs[1];
    use_encs[0] = ONIG_ENCODING_ASCII;
    onig_initialize(use_encs, sizeof(use_encs) / sizeof(use_encs[0]));
    UChar* pattern_c = (UChar*)pattern.c_str();
    int r = onig_new(&reg, pattern_c, pattern_c + strlen((char*)pattern_c), ONIG_OPTION_DEFAULT, ONIG_ENCODING_ASCII,
                     ONIG_SYNTAX_DEFAULT, &einfo);

    if (r != ONIG_NORMAL) {
        char s[ONIG_MAX_ERROR_MESSAGE_LEN];
        onig_error_code_to_str((UChar*)s, r, &einfo);
        fprintf(stderr, "ERROR: %s\n", s);
        return -1;
    }

    std::string ip_address = "192.168.2.1";
    MMDB_s mmdb;
    int status = MMDB_open(geoip_path.c_str(), MMDB_MODE_MMAP, &mmdb);
    if (MMDB_SUCCESS != status) {
        fprintf(stderr, "\n  Can't open %s - %s\n", geoip_path.c_str(), MMDB_strerror(status));
        if (MMDB_IO_ERROR == status) {
            fprintf(stderr, "    IO error: %s\n", strerror(errno));
        }
        exit(1);
    }

    int exit_code = 0;
    MMDB_entry_data_s entry_data;
    MMDB_entry_data_list_s* entry_data_list = NULL;
    char buffer[20];
    int gai_error, mmdb_error;
    uint16_t resInt;
    double resDlb;

    uint32_t* offsets = (uint32_t*)malloc(sizeof(uint32_t) * subNum);

    // core
    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);

    // read file
    // std::cout << "read file: " << msg_path << "\n";
    std::string lineStr;
    int msg_num = 0;
    std::vector<std::string> msgs;
    std::ifstream inFile(msg_path, std::ios::in);
    while (getline(inFile, lineStr)) {
        msgs.push_back(lineStr);
        msg_num++;
    }
    // std::cout << "msg_num = " << msg_num << std::endl;

    uint64_t char_cnt = 8;
    uint64_t char_cnt_last = 8;
    int mismatch_flag = 0, mismatch_cnt = 0;
    int geoip_err_flag = 0;
    std::string outjsons = "";
    uint64_t input_size = 0;
    std::cout << "msg num=" << msg_num << std::endl;
    for (int m = 0; m < msg_num; m++) {
        std::string msg = msgs[m];
        if (msg.size() == 0 || msg.size() > 4090) continue;
        input_size += msg.size();
        mismatch_flag = 0;
        geoip_err_flag = 0;
        char onejson[10240];
        rapidjson::StringBuffer s;
        rapidjson::Writer<rapidjson::StringBuffer> writer(s);
        writer.StartObject();
        offsets[0] = 0;
        regexTop(subNum, reg, region, (UChar*)msg.c_str(), offsets);
        if (offsets[0] > 0 && offsets[0] <= subNum) {
            for (int j = 2; j < subNum; j++) {
                uint32_t tmp = offsets[j];
                uint32_t begin = tmp % N16;
                uint32_t end = tmp / N16;
                if (j == 2) ip_address = msg.substr(begin, end - begin);
                writer.Key(fieldName[j - 2].c_str());
                writer.String(msg.substr(begin, end - begin).c_str());
            }
            MMDB_lookup_result_s result = MMDB_lookup_string(&mmdb, ip_address.c_str(), &gai_error, &mmdb_error);
            if ((0 != gai_error) || (MMDB_SUCCESS != mmdb_error) || (!result.found_entry)) {
                // std::cout << "ip_address=" << ip_address << ",\n msg=" << msg << std::endl;
                writer.Key("tag");
                writer.String("geoip_failure");
                geoip_err_flag = 1;
            }

            if (geoip_err_flag == 0 && result.found_entry) {
                writer.Key("geoip");
                writer.StartObject();
                for (int i = 0; i < 13; i++) {
                    int id_fn = idFN[i];
                    if (id_fn >> 4) {
                        if (0 == getDecValue(entry_data, result, id_fn & 0xF, resInt, resDlb)) {
                            writer.Key(geoip2FN[i].c_str());
                            if ((id_fn & 0xF) < 2) {
                                writer.Uint(resInt);
                            } else {
                                writer.Double(resDlb);
                            }
                        }
                    } else {
                        if (0 == getStrValue(entry_data, result, id_fn & 0xF, buffer)) {
                            writer.Key(geoip2FN[i].c_str());
                            writer.String(buffer);
                        }
                    }
                }
                writer.EndObject();
            }
        } else {
            writer.Key("tag");
            writer.String("grok_failure");
        }
        writer.EndObject();
#ifndef BS_TEST
        std::string tmp = s.GetString();
        tmp += "\n";
        uint8_t* tmp_dec = (uint8_t*)tmp.data();
        for (int i = 0; i < tmp.length(); i++) {
            if (test_result[char_cnt] != tmp_dec[i]) {
                while (tmp_dec[i] == '\\') i++;
                if (test_result[char_cnt] != tmp_dec[i]) {
                    mismatch_flag = 1;
                    mismatch_cnt++;
                    std::cout << "\n\nm = " << m << ", mismatch_cnt=" << mismatch_cnt
                              << ", ratio=" << mismatch_cnt * 100.0 / m << std::endl;
                    std::cout << "ERROR: test[" << char_cnt << "]=" << test_result[char_cnt] << ", golden[" << m << "]["
                              << i << "]=" << tmp_dec[i] << std::endl;
                    nerr++;
                    break;
                }
            }
            char_cnt++;
        }
        if (mismatch_flag) {
            for (int i = 0; i < 10239; i++) {
                if (test_result[char_cnt_last + i] != '\n') {
                    onejson[i] = test_result[char_cnt_last + i];
                } else {
                    onejson[i] = test_result[char_cnt_last + i];
                    onejson[i + 1] = '\0';
                    char_cnt_last += i + 1;
                    break;
                }
            }
            char_cnt = char_cnt_last;
            std::cout << "out json line is \n" << onejson << std::endl;
            std::cout << "ref json line is \n" << s.GetString() << std::endl;
        } else {
            char_cnt_last = char_cnt;
        }
#else
        // std::cout << s.GetString() << std::endl;
        outjsons += s.GetString();
        outjsons += "\n";
#endif
    }
#ifdef BS_TEST
    gettimeofday(&end_time, 0);
    std::cout << "Execution time " << tvdiff(start_time, end_time) / 1000.0 << " ms" << std::endl;
    std::cout << "Input Size " << input_size / 1000000.0 << "MB, Throughput "
              << input_size / (tvdiff(start_time, end_time) / 1.0) << " MB/s\n";
    std::fstream f(json_path.c_str(), std::ios::out);
    f << outjsons;
    // std::cout << outjsons;
    f.close();

    MMDB_free_entry_data_list(entry_data_list);
    MMDB_close(&mmdb);
    exit(exit_code);

    onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
    onig_free(reg);
    onig_end();
#endif
    return nerr;
}
