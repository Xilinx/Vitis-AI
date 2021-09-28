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
#include "ref_result.hpp"

int main(int argc, const char** argv) {
    std::cout << "-------- Integration (Grok + GeoIP + WriteJson) Test --------\n";

    // parser
    x_utils::ArgParser parser(argc, argv);
    std::string msg_path;
    std::string geoip_path;
    std::string json_path;
    parser.getCmdOption("-msg", msg_path);
    parser.getCmdOption("-geoip", geoip_path);
    parser.getCmdOption("-o", json_path);
    std::string pattern =
        "^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \\[(?<time>[^\\]]*)\\] \"(?<method>\\S+)(?: "
        "+(?<path>[^\\\"]*?)(?: +\\S*)?)?\" (?<code>[^ ]*) (?<size>[^ ]*)(?: \"(?<referer>[^\\\"]*)\" "
        "\"(?<agent>[^\\\"]*)\"(?:\\s+(?<http_x_forwarded_for>[^ ]+))?)?$";

    return refCalcu(msg_path, pattern, geoip_path, NULL, json_path);
}
