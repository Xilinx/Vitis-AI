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
#include <errno.h>
#include <maxminddb.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "geoip2.hpp"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

const std::string geoip2FN[] = {
    "postal_code",    "latitude",         "longitude",    "accuracy_radius",        "continent_code",
    "continent_name", "country_iso_code", "country_name", "subdivision_1_iso_code", "subdivision_1_name",
    "city_name",      "metro_code",       "time_zone"};

const int idFN[] = {0x7, 0x12, 0x13, 0x11, 0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x6, 0x10, 0x8};

const int sizeFN = 13;

int getCityGeoNameId(MMDB_lookup_result_s result, uint32_t& geoname_id) {
    MMDB_entry_data_s entry_data;
    int status = MMDB_get_value(&result.entry, &entry_data, "city", "geoname_id", NULL);
    if (MMDB_SUCCESS != status) status = MMDB_get_value(&result.entry, &entry_data, "country", "geoname_id", NULL);
    if (MMDB_SUCCESS == status) {
        if (entry_data.has_data) {
            geoname_id = entry_data.uint32;
        }
    }
    return status;
}

int getJsonPayload(MMDB_lookup_result_s result, std::string& geo_result) {
    MMDB_entry_data_s entry_data;
    char buffer[20];
    uint16_t resInt;
    double resDlb;
    rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
    // writer.Key("geoip");
    writer.StartObject();
    for (int i = 0; i < sizeFN; i++) {
        int id_fn = idFN[i];
        if (id_fn >> 4) {
            if (0 == getDecValue(entry_data, result, id_fn & 0xF, resInt, resDlb)) {
                writer.Key(geoip2FN[i].c_str());
                if ((id_fn & 0xF) < 2) {
                    // std::cout << resInt << std::endl;
                    writer.Uint(resInt);
                } else {
                    // std::cout << resDlb << std::endl;
                    writer.Double(resDlb);
                }
            }
        } else {
            if (0 == getStrValue(entry_data, result, id_fn & 0xF, buffer)) {
                // std::cout << "buffer=" << buffer << std::endl;
                writer.Key(geoip2FN[i].c_str());
                writer.String(buffer);
            }
        }
    }
    writer.EndObject();
    geo_result = s.GetString();
    // std::cout << s.GetString() << std::endl;
}

uint32_t calcuDelta(uint32_t i) {
    uint32_t b = 0;
    if (i == 0) return 4294967295;
    while (1) {
        uint32_t delta = 1 << b++;
        if (i / delta * delta != i) {
            return delta;
        }
    }
}
