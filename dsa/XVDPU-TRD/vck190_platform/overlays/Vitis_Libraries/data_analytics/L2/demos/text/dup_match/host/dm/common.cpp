/*
 * Copyright 2021 Xilinx, Inc.
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

#include "common.hpp"

namespace dup_match {
namespace internal {

std::string strip(std::string str, char ch = ' ') {
    int i = 0;
    while (str[i] == ch) i++; // start n ch
    int j = str.size() - 1;
    while (str[j] == ch) j--;
    return str.substr(i, j + 1 - i);
}

std::string to_preprocess(std::string str) {
    std::replace(str.begin(), str.end(), '\n', ' ');
    str = strip(strip(strip(strip(str), '"'), '\''));
    char last_ch = ' ';
    std::string out;
    for (int i = 0; i < str.size(); i++) {
        char ch = str[i];
        if (!(ch == last_ch && ch == ' ')) {
            out += std::tolower(static_cast<unsigned char>(str[i]));
        }
        last_ch = ch;
    }
    return out;
}

int checkAlphaNum(char in) {
    if (in >= 48 && in <= 57) // 0~9
        return 1;
    else if (in >= 97 && in <= 122) // a-z
        return 1;
    else if (in >= 65 && in <= 90) // A-Z
        return 1;
    else if (in == ' ') //
        return 1;
    else
        return 0;
}

int checkWordChar(char in) {
    if (in >= 48 && in <= 57) // 0~9
        return 1;
    else if (in >= 97 && in <= 122) // a-z
        return 1;
    else if (in >= 65 && in <= 90) // A-Z
        return 1;
    else if (in == 95 || in == '\'') // _ '
        return 1;
    else
        return 0;
}

// split string into words
void splitWord(std::string& inStr, std::vector<std::string>& terms, std::string& outStr) {
    int begin = 0;
    for (int i = 0; i < inStr.size(); i++) {
        if (!checkWordChar(inStr[i])) {
            if (i > begin) {
                terms.push_back(inStr.substr(begin, i - begin));
                outStr += " " + inStr.substr(begin, i - begin);
            }
            begin = i + 1;
        }
    }
    if (inStr.size() > begin) {
        terms.push_back(inStr.substr(begin, inStr.size() - begin));
        outStr += " " + inStr.substr(begin, inStr.size() - begin);
    }
}

void addMerge(std::vector<udPT>& in1, std::vector<udPT>& in2, std::vector<udPT>& out) {
    int i1 = 0, i2 = 0;
    while (i1 < in1.size() || i2 < in2.size()) {
        if (i1 < in1.size() && i2 < in2.size()) {
            if (in1[i1].first == in2[i2].first) {
                out.push_back(udPT(in1[i1].first, in1[i1].second + in2[i2].second));
                i1++;
                i2++;
            } else if (in1[i1].first < in2[i2].first)
                out.push_back(in1[i1++]);
            else
                out.push_back(in2[i2++]);
        } else if (i1 < in1.size())
            out.push_back(in1[i1++]);
        else
            out.push_back(in2[i2++]);
    }
}

} // internal
} // dup_match
