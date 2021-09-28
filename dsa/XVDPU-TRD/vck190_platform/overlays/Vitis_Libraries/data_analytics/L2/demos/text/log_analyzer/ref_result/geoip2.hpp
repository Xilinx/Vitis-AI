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
#include <maxminddb.h>

const std::string Str1[] = {"continent",    "continent", "country", "country", "subdivisions",
                            "subdivisions", "city",      "postal",  "location"};
const std::string Str2[] = {"metro_code", "accuracy_radius", "latitude", "longitude", "time_zone"};

int getDecValue(MMDB_entry_data_s entry_data, MMDB_lookup_result_s result, int i, uint16_t& resInt, double& resDlb) {
    int status = MMDB_get_value(&result.entry, &entry_data, "location", Str2[i].c_str(), NULL);
    if (MMDB_SUCCESS == status) {
        if (entry_data.has_data) {
            if (entry_data.type == MMDB_DATA_TYPE_UINT16) {
                resInt = entry_data.uint16;
            } else if (entry_data.type == MMDB_DATA_TYPE_DOUBLE) {
                resDlb = entry_data.double_value;
            } else {
                // printf("data_type = %d\n", entry_data.type);
                return 1;
            }
        } else {
            // std::cout << "entry_data.has_data fail!\n";
            return 2;
        }
    } else {
        // std::cout << "MMDB_SUCCESS fail!\n";
        return 3;
    }
    return 0;
}

int getStrValue(MMDB_entry_data_s entry_data, MMDB_lookup_result_s result, int i, char* buffer) {
    int status;
    if (i == 1)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "code", NULL);
    else if (i == 3)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "iso_code", NULL);
    else if (i == 4)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "0", "names", "en", NULL);
    else if (i == 5)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "0", "iso_code", NULL);
    else if (i == 7)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "code", NULL);
    else if (i == 8)
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "time_zone", NULL);
    else
        status = MMDB_get_value(&result.entry, &entry_data, Str1[i].c_str(), "names", "en", NULL);
    if (MMDB_SUCCESS == status) {
        if (entry_data.has_data) {
            if (entry_data.type == MMDB_DATA_TYPE_UTF8_STRING) {
                strncpy(buffer, entry_data.utf8_string, entry_data.data_size);
                buffer[entry_data.data_size] = '\0';
            } else {
                // printf("data_type = %d\n", entry_data.type);
                return 1;
            }
        } else {
            // std::cout << "entry_data.has_data fail!\n";
            return 2;
        }
    } else {
        // std::cout << "MMDB_SUCCESS fail!\n";
        return 3;
    }
    return 0;
}
