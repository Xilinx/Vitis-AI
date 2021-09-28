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
#include <iostream>
#include <fstream>

//#define DEBUG

void ipConvert(uint32_t ip_num, std::string& ip_str) {
    uint8_t A1 = (ip_num & 0xFF000000) >> 24;
    uint8_t A2 = (ip_num & 0xFF0000) >> 16;
    uint8_t A3 = (ip_num & 0xFF00) >> 8;
    uint8_t A4 = ip_num & 0xFF;
    ip_str = std::to_string(A1) + "." + std::to_string(A2) + "." + std::to_string(A3) + "." + std::to_string(A4);
}

int main(int argc, const char** argv) {
    std::cout << "Convert Flow\n";
    const char* in_file_path = argv[1];
    const char* out_file_path = argv[2];

    std::ifstream in_file(in_file_path, std::ios::in);
    std::ofstream out_file(out_file_path, std::ios::out);
    uint64_t cnt = 0, last_cnt = 0, delta = 0;

    std::string in_line, out_line;
    getline(in_file, in_line);
    int flag = std::stoi(in_line.substr(in_line.find('X') + 1, in_line.find('X') + 2));
    uint32_t first_space = in_line.find(' ');
    uint32_t second_space = in_line.find(' ', first_space + 1);
    std::string str1 = in_line.substr(0, first_space); // string substr (size_t pos = 0, size_t len = npos) const;
    std::string substr = in_line.substr(second_space, in_line.size() - second_space);
    last_cnt = std::stol(str1);
#ifdef DEBUG
    std::cout << in_line << std::endl;
#endif
    int i = 0;
    while (getline(in_file, in_line)) {
#ifdef DEBUG
        std::cout << in_line << std::endl;
#endif
        cnt = std::stol(in_line.substr(0, in_line.find(' ')));
        if (i % 1000 == 0) std::cout << "i=" << i << ", cnt=" << cnt << "\n";
        if (last_cnt % 65536 > 0 && (cnt / 65536 - last_cnt / 65536 > 0)) {
            delta = 65536 - last_cnt % 65536;
            uint8_t b = 0;
            while (flag) {
                uint32_t t = 1 << b;
                if (t > delta)
                    break;
                else if ((0x01 & (delta >> b)) == 0) {
                } else {
                    std::string ip_str;
                    ipConvert(last_cnt, ip_str);
                    std::string w_line =
                        std::to_string(last_cnt) + "/" + std::to_string(32 - b) + " " + ip_str + substr;
                    out_file << w_line << std::endl;
#ifdef DEBUG
                    std::cout << "out: " << w_line << std::endl;
#endif
                    delta -= t;
                    last_cnt += t;
                }
                b++;
            }
        }
        if (cnt / 65536 - last_cnt / 65536 > 0) {
            delta = (cnt / 65536 - last_cnt / 65536) * 65536;
            uint8_t b = 0;
            while (flag) {
                uint32_t t = 1 << b;
                if (t > delta)
                    break;
                else if ((0x01 & (delta >> b)) == 0) {
                } else {
                    std::string ip_str;
                    ipConvert(last_cnt, ip_str);
                    std::string w_line =
                        std::to_string(last_cnt) + "/" + std::to_string(32 - b) + " " + ip_str + substr;
                    out_file << w_line << std::endl;
#ifdef DEBUG
                    std::cout << "out: " << w_line << std::endl;
#endif
                    delta -= t;
                    last_cnt += t;
                }
                b++;
            }
        }

        delta = cnt - last_cnt;
        uint8_t b = 0;
        while (flag && delta) {
            uint32_t t = 1 << b;
#ifdef DEBUG
            std::cout << "b=" << b - 1 << ",t=" << t << ",delta=" << delta << std::endl;
#endif
            if (t > delta)
                break;
            else if ((0x01 & (delta >> b)) == 0) {
            } else {
                std::string ip_str;
                ipConvert(last_cnt, ip_str);
                std::string w_line = std::to_string(last_cnt) + "/" + std::to_string(32 - b) + " " + ip_str + substr;
                out_file << w_line << std::endl;
#ifdef DEBUG
                std::cout << "out: " << w_line << std::endl;
#endif
                delta -= t;
                last_cnt += t;
            }
            b++;
        }
        flag = std::stol(in_line.substr(in_line.find('X') + 1, in_line.find('X') + 2));
        first_space = in_line.find(' ');
        second_space = in_line.find(' ', first_space + 1);
        substr = in_line.substr(second_space, in_line.size() - second_space);
        last_cnt = cnt;
        i++;
    }
    in_file.close();
    out_file.close();
    return 0;
}
