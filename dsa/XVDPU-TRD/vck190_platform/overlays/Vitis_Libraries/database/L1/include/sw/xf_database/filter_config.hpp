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
#ifndef XF_DB_FILTER_CONFIG_H
#define XF_DB_FILTER_CONFIG_H
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <string>
#include <regex>
#include <memory>
// Op code, etc.
#include "xf_database/enums.hpp"
// HW header for config details
#include "xf_database/dynamic_filter.hpp"

//#define DEBUGTT 1

namespace xf {
namespace database {
namespace internals {
namespace filter_config {

static const std::map<std::string, int> opMap = {{"<", FOP_LTU},  {"<=", FOP_LEU}, {">", FOP_GTU},
                                                 {">=", FOP_GEU}, {"==", FOP_EQ},  {"!=", FOP_NE}};
static const std::map<std::string, int> revs_opMap = {{"<", FOP_GTU},  {"<=", FOP_GEU}, {">", FOP_LTU},
                                                      {">=", FOP_LEU}, {"==", FOP_EQ},  {"!=", FOP_NE}};

static const std::map<int, int> bitmapIndMap = {{FOP_LTU, 1}, {FOP_LEU, 1}, {FOP_GTU, 0}, {FOP_GEU, 0}};

static const std::map<std::string, int> opMap_sign = {{"<", FOP_LT},  {"<=", FOP_LE}, {">", FOP_GT},
                                                      {">=", FOP_GE}, {"==", FOP_EQ}, {"!=", FOP_NE}};
static const std::map<std::string, int> revs_opMap_sign = {{"<", FOP_GT},  {"<=", FOP_GE}, {">", FOP_LT},
                                                           {">=", FOP_LE}, {"==", FOP_EQ}, {"!=", FOP_NE}};

static const std::map<int, int> bitmapIndMap_sign = {{FOP_LT, 1}, {FOP_LE, 1}, {FOP_GT, 0}, {FOP_GE, 0}};

static const std::map<std::string, int> opPriorty = {{"&&", 1}, {"||", 1}, {"!", 3}, {"(", 4},  {">=", 2},
                                                     {">", 2},  {"<=", 2}, {"<", 2}, {"==", 2}, {"!=", 2}};

static const std::map<std::string, int> mapKey = {{"ra", 0}, {"rb", 1}, {"rc", 2}, {"rd", 3}, {"ab", 4},
                                                  {"ac", 5}, {"ad", 6}, {"bc", 7}, {"bd", 8}, {"cd", 9}};

/* filter (true) */
// eliminate all space in the string
inline void trim(std::string& s) {
    size_t index = 0;
    if (!s.empty()) {
        while ((index = s.find(' ', index)) != std::string::npos) {
            s.erase(index, 1);
        }
    }
}

template <typename T, unsigned NCOL>
class TTParser {
    std::string midExpr;
    std::vector<std::string> preExpr;
    std::vector<std::string> sufExpr;
    std::vector<bool> sufExprTag;
    std::vector<std::vector<int> > trueTable;
    bool isDigit(char a) { return (a >= 0x30 && a <= 0x39); }

    bool isData(char a) { return (a >= 0x61 && a <= 0x64) || (a == 'r'); }

    bool isOp(char a) {
        if (a == '|' || a == '!' || a == '&' || a == '(' || a == ')')
            return true;
        else
            return false;
    }

   public:
    TTParser(std::string s, int n = NCOL) {
        midExpr = s;

        trueTable.resize(n + 1);
        for (int i = 0; i < n + 1; i++) trueTable[i].resize(1 << n);
        unsigned num_to_fill = 1U << (n - 1);
        // 0 cols is the low
        for (int col = 0; col < n; col++, num_to_fill >>= 1) {
            for (unsigned row = num_to_fill; row < (1U << n); row += (num_to_fill * 2)) {
                std::fill_n(&trueTable[n - 1 - col][row], num_to_fill, 1);
            }
        }
    }

    void calcu(std::string op, std::stack<T>& data) {
        if (op == "!") {
            T a = !data.top();
            data.pop();
            data.push(a);
        } else if (op == "||") {
            T a = data.top();
            data.pop();
            T b = data.top();
            data.pop();
            data.push(a | b);
        } else if (op == "&&") {
            T a = data.top();
            data.pop();
            T b = data.top();
            data.pop();
            data.push(a && b);
#ifdef DEBUGTT
// std::cout << "in calcu " << a << " " << b << " " << data.top() << std::endl;
#endif
        }
    }

    T computeSufExpr(int row) {
        std::stack<std::string> op;
        std::stack<T> data;
        for (unsigned i = 0; i < sufExpr.size(); i++) {
            bool ifdata = sufExprTag[i];
            std::string tmp = sufExpr[i];
            if (ifdata) {
                int index = stoi(tmp);
                data.push(trueTable[index][row]);
#ifdef DEBUGTT
// std::cout << "push data " << data.top() << std::endl;
#endif
            } else {
                calcu(tmp, data);
#ifdef DEBUGTT
// std::cout << "calculate" << data.top() << std::endl;
#endif
            }
        }
        return data.top();
    }

    void trueTableCfg(int n = NCOL) {
        for (int x = 0; x < (1 << n); x++) {
#ifdef DEBUGTT
            for (int y = n - 1; y >= 0; y--) {
                std::cout << trueTable[y][x] << " ";
            }
#endif
            T c = computeSufExpr(x);
            trueTable[n][x] = c;
#ifdef DEBUGTT
            std::cout << "  " << c << std::endl;
#endif
        }
    }
    void storeTrueTableCfg(uint32_t* tCfg, int n = NCOL) {
        for (int i = 0; i < (1 << n); i += 32) {
            uint32_t x = 0;
            for (int j = 0; j < 32; j++) {
                x ^= (trueTable[n][i + j] << j);
#ifdef DEBUGTT
                std::cout << "  " << trueTable[n][i + j];
#endif
            }
            tCfg[i / 32] = x;
#ifdef DEBUGTT
            std::cout << std::endl;
#endif
        }
    }

    void toSufExpr() {
        std::stack<std::string> st1;
        for (unsigned i = 0; i < midExpr.length();) {
            if (isOp(midExpr[i])) { // when the oprator
                if ('(' == midExpr[i]) {
                    st1.push("(");
                    i++;
                } else if (')' == midExpr[i]) {
                    while (!st1.empty() && (st1.top() != "(")) {
                        std::string c = st1.top();
                        sufExpr.push_back(c);
                        sufExprTag.push_back(false);
                        st1.pop();
                    }
                    st1.pop(); // discart left brace
                    i++;
                } else {
                    char op[10];
                    int op_len = 0;
                    for (; isOp(midExpr[i]) && '(' != midExpr[i] && i < midExpr.length(); i++) {
                        op[op_len++] = midExpr[i];
                    }
                    op[op_len] = 0;
                    std::string opstr(op, strlen(op));
                    while (1) {
                        if (st1.empty() || st1.top() == "(" || opPriorty.at(st1.top()) < opPriorty.at(opstr)) {
                            st1.push(opstr);
                            break;
                        } else {
                            std::string tmpstr = st1.top();
                            sufExpr.push_back(tmpstr);
                            sufExprTag.push_back(false);
                            st1.pop();
                        }
                    }
                }
            } else if (isData(midExpr[i])) { // when oprator data, push stack S2 (final stack, save the index instead of
                                             // the string)
                char data[10];
                int data_len = 0;
                for (; isData(midExpr[i]) && i < midExpr.length(); i++) {
                    data[data_len++] = midExpr[i];
                }
                data[data_len] = 0;
                std::string datastr(data, strlen(data)); // strlen(data) = data_len+1
                int mapk = mapKey.at(datastr);
                sufExpr.push_back(std::to_string(mapk));
                sufExprTag.push_back(true);
            } else {
                std::cerr << "not supported ops" << std::endl;
                exit(0);
            }
        }

        while (!st1.empty()) {
            std::string tmp = st1.top();
            sufExpr.push_back(tmp);
            sufExprTag.push_back(false);
            st1.pop();
        }

#ifdef DEBUGTT
        std::cout << "debug trans to suffix expr!" << std::endl;
        int i = 0;
        for (std::string s : sufExpr) {
            std::cout << s << "   isop?:" << sufExprTag[i++] << " " << std::endl;
        }
#endif
    }
    void doParser(uint32_t* cfg) {
        trim(midExpr);
        toSufExpr();
        trueTableCfg();
        storeTrueTableCfg(cfg);
    }
};

class FilterParser {
    const int bits;
    const int dw_per_imm;
    std::string midExpr;
    std::vector<std::string> sufExpr;
    std::vector<bool> sufExprTag;

    // 0~9
    bool isDigit(char a) { return (a >= 0x30 && a <= 0x39); }

    // a || b|| c || d
    bool isData(char a) { return (a >= 0x61 && a <= 0x64) || (a >= 0x30 && a <= 0x39); }

    // supported operation
    bool isOp(char a) {
        if (a == '|' || a == '!' || a == '&' || a == '(' || a == ')' || a == '>' || a == '<' || a == '=')
            return true;
        else
            return false;
    }

    std::string doCmp(std::string lstr, std::string rstr, std::string op, std::unique_ptr<uint32_t[]>& cfg) {
#ifdef DEBUGFP
        std::cout << "doCmp: " << lstr << std::endl;
        std::cout << "doCmp: " << rstr << std::endl;
#endif
        std::string restr;
        uint64_t tmps[2];
        int opcode = FOP_DC;

        int type = -1;
        const char* l_array = lstr.c_str();
        const char* r_array = rstr.c_str();
        char lchar = l_array[0];
        char rchar = r_array[0];
        // left is the a|b|c|d
        if (lchar >= 0x61 && lchar <= 0x64) {
            type = 0;
            // default-0: left is col, right is number
        }
        if (rchar >= 0x61 && rchar <= 0x64) {
            if (type == 0) {
                type = 2; // both left and right are cols
            } else {
                type = 1; // rigit is col, left is number
            }
        }
#ifdef DEBUGFP
        std::cout << "debug l/r char:" << lchar << " " << rchar << std::endl;
#endif
        //
        try {
            if (type == 0) {
                tmps[0] = lchar - 0x61;
                tmps[1] = std::stoull(rstr);
                opcode = opMap.at(op);
                restr = "r";
                restr.append(1, lchar);
            } else if (type == 1) {
                tmps[0] = rchar - 0x61;
                tmps[1] = std::stoull(lstr);
                opcode = revs_opMap.at(op);
                lchar = rchar;
                type = 0;
                restr = "r";
                restr.append(1, lchar);
            } else if (type == 2) {
                opcode = opMap.at(op);
                if (lchar > rchar) {
                    int tmp = lchar;
                    lchar = rchar;
                    rchar = tmp;
                    opcode = revs_opMap.at(op);
                }
                tmps[0] = lchar - 0x61;
                tmps[1] = rchar - 0x61;
                restr = lchar;
                restr.append(1, rchar);
            } else {
                std::cerr << "wrong input type!\n";
                exit(0);
            }
        } catch (std::invalid_argument& e) {
            std::cerr << "catched invalid_arguments!\n";
            exit(0);
        }

#ifdef DEBUGFP
        std::cout << "debug bitmap:" << opcode << " " << tmps[0] << " " << tmps[1] << std::endl;
#endif
        // add for bitmap
        int off = tmps[0] * (2 * dw_per_imm + 1); // 2 immediate + two op in one dword.
        int ind = bitmapIndMap.at(opcode);
        if (ind != 0 && ind != 1) {
            std::cerr << "wrong input pattern!\n";
        }
        if (type == 0) {
            cfg[off + ind * dw_per_imm] = (uint32_t)(tmps[1] & 0xffffffff);
            if (dw_per_imm > 1) {
                cfg[off + ind * dw_per_imm + 1] = (uint32_t)(tmps[1] >> 32);
            }
            cfg[off + 2 * dw_per_imm] |= (opcode << ((1 - ind) * FilterOpWidth));
#ifdef DEBUGFP
            std::cout << "off: " << off << ", ind: " << ind << std::endl;
#endif
        }
        if (type == 2) {
            int base = 0;
            if (tmps[0] == 1)
                base = 3;
            else if (tmps[0] == 2)
                base = 5;
            int sh = (tmps[1] - tmps[0] + base - 1) * FilterOpWidth;
#if __cplusplus >= 201103L
            static_assert(FilterOpWidth == 4); // otherwise may need more than one dword.
#endif
            cfg[(2 * dw_per_imm + 1) * 4] |= (uint32_t)(opcode << sh);
        }
#ifdef DEBUGFP
        std::cout << "in doCmp:" << restr << std::endl;
#endif
        return restr;
    }

    // in the replacement , when the operator is ||, do this
    bool validate(std::string l, std::string r) {
        std::size_t l_check = l.find("ra");
        std::size_t r_check = r.find("ra");
        if (l_check != std::string::npos && r_check != std::string::npos) return false;

        l_check = l.find("rb");
        r_check = r.find("rb");
        if (l_check != std::string::npos && r_check != std::string::npos) return false;

        l_check = l.find("rc");
        r_check = r.find("rc");
        if (l_check != std::string::npos && r_check != std::string::npos) return false;

        l_check = l.find("rd");
        r_check = r.find("rd");
        if (l_check != std::string::npos && r_check != std::string::npos) return false;
        return true;
    }

    template <typename T = std::string>
    void processR(T& a, std::unique_ptr<uint32_t[]>& cfg) {
        // if a is one char, then a=(doCmp(a,'1','!='))
        // continue to process
        if (a == "a" || a == "b" || a == "c" || a == "d") {
            a = doCmp(a, "0", "!=", cfg);
        }
    }

    template <typename T = std::string>
    void calcu(std::string op, std::stack<T>& data, std::unique_ptr<uint32_t[]>& cfg) {
        if (op == ">" || op == ">=" || op == "<" || op == "<=" || op == "!=" || op == "==") {
            T l = data.top();
            data.pop();
            T r = data.top();
            data.pop();
            data.push(doCmp(r, l, op, cfg));
        } else if (op == "||") {
            T l = data.top();
            processR(l, cfg);
            data.pop();
            T r = data.top();
            processR(r, cfg);
            data.pop();
            if (!validate(l, r)) {
                std::cerr << "Err: not supported filter!" << std::endl;
                exit(0);
            }
            data.push("(" + r + "||" + l + ")");
        } else if (op == "&&") {
            T l = data.top();
            processR(l, cfg);
            data.pop();
            T r = data.top();
            processR(r, cfg);
            data.pop();
            data.push("(" + r + "&&" + l + ")");
        } else if (op == "!") {
            T l = data.top();
            processR(l, cfg);
            data.top();
            data.push("(!" + l + ")");
        }
    }

   public:
    FilterParser(std::string s, int b) : bits(b), dw_per_imm((b - 1) / 32 + 1) {
        midExpr = s;
        trim(midExpr);
    }

    std::string computeSufExpr(std::unique_ptr<uint32_t[]>& cfg) {
        std::stack<std::string> op;
        std::stack<std::string> data;
        for (unsigned i = 0; i < sufExpr.size(); i++) {
            bool ifdata = sufExprTag[i];
            std::string tmp = sufExpr[i];
            if (ifdata) {
                data.push(tmp);
            } else {
                calcu(tmp, data, cfg);
            }
        }
        return data.top();
    }

    void toSufExpr() {
        std::stack<std::string> st1;
        for (unsigned i = 0; i < midExpr.length();) {
            if (isOp(midExpr[i])) { // when the oprator
                if ('(' == midExpr[i]) {
                    st1.push("(");
                    i++;
                } else if (')' == midExpr[i]) {
                    while (!st1.empty() && (st1.top() != "(")) {
                        std::string c = st1.top();
                        sufExpr.push_back(c);
                        sufExprTag.push_back(false);
                        st1.pop();
                    }
                    st1.pop(); // discart left brace
                    i++;
                } else {
                    char op[10];
                    int op_len = 0;
                    for (; isOp(midExpr[i]) && '(' != midExpr[i] && i < midExpr.length();) {
                        op[op_len++] = midExpr[i++];
                        if ('!' == midExpr[i]) break;
                    }
                    op[op_len] = 0;
                    std::string opstr(op, strlen(op));
                    while (1) {
                        if (st1.empty() || st1.top() == "(" || opPriorty.at(st1.top()) < opPriorty.at(opstr)) {
                            st1.push(opstr);
                            break;
                        } else {
                            std::string tmpstr = st1.top();
                            sufExpr.push_back(tmpstr);
                            sufExprTag.push_back(false);
                            st1.pop();
                        }
                    }
                }
            } else if (isData(midExpr[i])) { // when oprator data, push stack S2 (final stack, save the index instead of
                                             // the string)
                char data[10];
                int data_len = 0;
                for (; isData(midExpr[i]) && i < midExpr.length(); i++) {
                    data[data_len++] = midExpr[i];
                }
                data[data_len] = 0;
                std::string datastr(data, strlen(data)); // strlen(data) = data_len+1
                sufExpr.push_back(datastr);
                sufExprTag.push_back(true);
            } else {
                std::cerr << "not supported ops" << std::endl;
                exit(0);
            }
        }

        while (!st1.empty()) {
            std::string tmp = st1.top();
            sufExpr.push_back(tmp);
            sufExprTag.push_back(false);
            st1.pop();
        }

#ifdef DEBUGFP
        int i = 0;
        for (std::string s : sufExpr) {
            std::cout << s << "   isop?:" << sufExprTag[i++] << " " << std::endl;
        }
#endif
    }

    void doParser(std::unique_ptr<uint32_t[]>& cfg) {
        toSufExpr();
        std::string result_s = computeSufExpr(cfg);
        TTParser<bool, 10> e(result_s);
        e.doParser(cfg.get() + 4 * (dw_per_imm * 2 + 1) + 1);
    }
};

} // namespace filter_config
} // namespace internals

/**
 * @class Dynamic Filter Configuration Generator.
 *
 * @tparam BITS number of bits for each element, default to 32, can also be 64.
 */
template <int BITS = 32>
class FilterConfig {
#if __cplusplus >= 201103L
    static_assert(BITS == 32 || BITS == 64);
#endif
    std::string filter_str;

   public:
    /**
     * @brief construct with filter condition string.
     *
     * The string uses ``a``, ``b``, ``c``, ``d`` to refer to first to the fourth column,
     * and supports comparison and logical operator like C++.
     * Parentheses can be used to group logic, but notice that the underlying hardware module
            // off++;
     * does not support comparing one column with multiple constant boundaries in OR relationship.
     * Integral constants will be extracted from expression.
     *
     * For example, an expression could be ``(a < 10 && b < 20) || (c >= d)``.
     *
     * @param s the filter expression string.
     */
    FilterConfig(const std::string& s) {
        filter_str = s;
        internals::filter_config::trim(filter_str);
    }

    /**
     * @brief generate a configuration bits buffer.
     * @return a unique_ptr to configuration bits buffer of ``uint32_t`` type.
     */
    std::unique_ptr<uint32_t[]> getConfigBits() const {
        const unsigned dw_num = DynamicFilterInfo<4, BITS>::dwords_num;
        std::cout << "Num of dword in filter config: " << dw_num << std::endl;
        std::unique_ptr<uint32_t[]> cfg(new uint32_t[dw_num]);
        memset(cfg.get(), 0, sizeof(uint32_t) * dw_num);

        if (filter_str.size() == 0) { // if input filter_str is empty, then no filter and set true table to all "1"
            const unsigned tt_dw_num = details::true_table_info<4>::dwords_num;
            for (unsigned int i = dw_num - 1; i >= dw_num - tt_dw_num; i--) {
                cfg[i] = 0xFFFFFFFF;
            }
        } else { // if input filter_str is not empty. Might need extra valid check.
            internals::filter_config::FilterParser fp(filter_str, BITS);
            fp.doParser(cfg);
        }
        return cfg;
    }
};

} // namespace database
} // namespace xf
#endif // XF_DB_FILTER_CONFIG_H
