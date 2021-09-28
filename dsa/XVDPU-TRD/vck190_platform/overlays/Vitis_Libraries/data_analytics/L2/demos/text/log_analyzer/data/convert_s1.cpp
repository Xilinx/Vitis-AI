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
#include "convert_s1.hpp"
#include <fstream>

int main(int argc, char** argv) {
    char* filename = argv[1];
    char* outfile = argv[2];

    MMDB_s mmdb;
    int status = MMDB_open(filename, MMDB_MODE_MMAP, &mmdb);

    if (MMDB_SUCCESS != status) {
        fprintf(stderr, "\n  Can't open %s - %s\n", filename, MMDB_strerror(status));

        if (MMDB_IO_ERROR == status) {
            fprintf(stderr, "    IO error: %s\n", strerror(errno));
        }
        exit(1);
    }

    std::fstream f(outfile, std::ios::out);

    int exit_code = 0;
    uint32_t geoname_id = 0, geoname_id_last = 0;
    std::string ip_address_base;
    const uint64_t TH = 16797696;
    // const uint64_t TH = 4294967296;
    std::string geo_result, geo_result_last = "", write_str;
    for (uint64_t i = 0; i < TH; i++) {
        // generate ip
        uint8_t A1 = (i & 0xFF000000) >> 24;
        uint8_t A2 = (i & 0xFF0000) >> 16;
        uint8_t A3 = (i & 0xFF00) >> 8;
        uint8_t A4 = i & 0xFF;
        std::string ip_address =
            std::to_string(A1) + "." + std::to_string(A2) + "." + std::to_string(A3) + "." + std::to_string(A4);

        if (i % 1000000 == 0) std::cout << "i=" << i << ",ip=" << ip_address << ", ratio=" << i * 100.0 / TH << "%\n";
        // get geo value of ip
        int gai_error, mmdb_error;
        MMDB_lookup_result_s result = MMDB_lookup_string(&mmdb, ip_address.c_str(), &gai_error, &mmdb_error);

        if (0 != gai_error) {
            fprintf(stderr, "\n  Error from getaddrinfo for %s - %s\n\n", ip_address.c_str(), gai_strerror(gai_error));
            exit(2);
        }

        if (MMDB_SUCCESS != mmdb_error) {
            fprintf(stderr, "\n  Got an error from libmaxminddb: %s\n\n", MMDB_strerror(mmdb_error));
            exit(3);
        }

        MMDB_entry_data_list_s* entry_data_list = NULL;

        // parser result and write file
        int status = 0;
        if (result.found_entry) {
            // status = getCityGeoNameId(result, geoname_id);
            if (1) { //(status == MMDB_SUCCESS) {
                getJsonPayload(result, geo_result);
                if (geo_result != geo_result_last) {
                    // std::cout << "1: " << ip_address << " " << geo_result << std::endl;
                    // std::cout << "2: " << ip_address << " " << geo_result_last << std::endl;
                    write_str = std::to_string(i) + " " + ip_address + " X1 " + "#\",\"geoip\":" + geo_result + "}";
                    // write_str = ip_address + " 1 " + std::to_string(geoname_id) + " " + geo_result;
                    // std::cout << write_str << std::endl << std::endl << std::endl;
                    f << write_str << std::endl;
                }
            } else {
                geoname_id = 0xFFFFFFFF;
                if (geoname_id != geoname_id_last) {
                    write_str = std::to_string(i) + " " + ip_address + " X0 " + "#\"tag\":\"geoip_failure\"}}";
                    // write_str = ip_address + " 0 " + std::to_string(geoname_id) + " 0";
                    // std::cout << write_str << std::endl;
                    f << write_str << std::endl;
                }
            }
        } else {
            geoname_id = 0xFFFFFFFF;
            geo_result = "#\"tag\":\"geoip_failure\"}}";
            if (geo_result != geo_result_last) {
                // if (geoname_id != geoname_id_last) {
                write_str = std::to_string(i) + " " + ip_address + " X0 " + geo_result;
                // write_str = ip_address + " 0 " + std::to_string(geoname_id) + " 0";
                // std::cout << write_str << std::endl;
                f << write_str << std::endl;
            }
        }
        geo_result_last = geo_result;
        geoname_id_last = geoname_id;
    end:
        MMDB_free_entry_data_list(entry_data_list);
    }

    MMDB_close(&mmdb);
    exit(exit_code);
}
