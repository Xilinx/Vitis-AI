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
#ifndef XF_DB_DYN_EVAL_CONFIG_H
#define XF_DB_DYN_EVAL_CONFIG_H
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <string>
#include <regex>
#include <memory>
#include <string>
#include <cstdint>
#include <cassert>
#include <unordered_map>

// fetch hw module definitions
#include "xf_database/enums.hpp"
#include "xf_database/dynamic_eval_v2.hpp"

namespace xf {
namespace database {

namespace internals {
namespace eval_config {

inline uint16_t dynamicEvalV2Inst(const uint32_t op, const uint32_t lhs, const uint32_t rhs) {
    return (uint16_t)((op << 12) | (lhs << 6) | rhs);
}

class DynEvalCompiler {
   public:
    DynEvalCompiler(){};

    void writeOp(uint32_t* config, const std::string& expr) {
        uint16_t inst[xf::database::DYN_EVAL_NSTEP];
        emitRPNToInst(inst, parseExprToRPN(expr));
        int c = 0;
        uint32_t inst2 = 0;
        for (int i = 0; i < xf::database::DYN_EVAL_NSTEP; i++) {
            int ii = i % 0x02;
            if (ii == 0) {
                inst2 |= (inst[i] & 0xffffU);
                if (i == xf::database::DYN_EVAL_NSTEP - 1) {
                    config[c++] = inst2;
                }
            } else {
                inst2 |= (((uint32_t)inst[i] << 16) & 0xffff0000U);
                config[c++] = inst2;
                inst2 = 0;
            }
        }
    }

   private:
    void checkList(const std::string& prefix, const std::vector<std::string>& list) {
        return;
        std::cout << prefix;
        for (auto s : list) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
    }
    std::vector<std::string> parseExprToRPN(const std::string& expr) {
        using namespace std;
        vector<string> tokens;
        char t = '\0';
        for (auto c : expr) {
            if (c == ' ' || c == '\t') {
                continue;
            } else if (isOperator(c) || c == '(' || c == ')') {
                string s{c};
                tokens.push_back(s);
            } else if (c == 'v' || c == 'c') {
                t = c;
            } else if (c >= '0' && c <= '3') {
                string s{t, c};
                tokens.push_back(s);
                t = '\0';
            }
        }
        checkList("tokens:", tokens);
        vector<string> stack;
        vector<string> RPN;
        for (auto token : tokens) {
            if (isSymbol(token)) {
                RPN.push_back(token);
                checkList("stack:", stack);
                checkList("RPN:", RPN);
            } else if (isOperator(token)) {
                while (!stack.empty() && isOperator(stack.back())) {
                    if (precedence(token) <= precedence(stack.back())) {
                        RPN.push_back(stack.back());
                        stack.pop_back();
                        continue;
                    }
                    break;
                }
                stack.push_back(token);
                checkList("stack:", stack);
                checkList("RPN:", RPN);
            } else if (token == "(") {
                stack.push_back(token);
                checkList("stack:", stack);
                checkList("RPN:", RPN);
            } else if (token == ")") {
                while (!stack.empty() && stack.back() != "(") {
                    RPN.push_back(stack.back());
                    stack.pop_back();
                }
                stack.pop_back();
                checkList("stack:", stack);
                checkList("RPN:", RPN);
            }
        }
        while (stack.size()) {
            RPN.push_back(stack.back());
            stack.pop_back();
        }
        checkList("RPN:", RPN);
        return RPN;
    }

    void emitRPNToInst(uint16_t insts[], const std::vector<std::string>& RPN) {
        using namespace std;
        vector<string> t{"t0", "t1", "t2", "t3", "t4", "t5", "t6"};
        int n = 0;
        vector<string> stack;
        for (auto token : RPN) {
            if (isSymbol(token)) {
                stack.push_back(token);
            } else if (isOperator(token)) {
                string lhs = stack[stack.size() - 2];
                string rhs = stack.back();
                insts[n] = dynamicEvalV2Inst(convertSymToEnum(token), convertSymToEnum(lhs), convertSymToEnum(rhs));
                // string s = t[n] + "=" + lhs + token + rhs;
                // cout << s << endl;
                stack.pop_back();
                stack.pop_back();
                stack.push_back(t[n]);
                ++n;
            }
            assert(n <= 7 && "Too many steps!");
        }
        while (n < 7) {
            insts[n] = dynamicEvalV2Inst(xf::database::DYN_EVAL_OP_NOP, convertSymToEnum(t[n - 1]),
                                         convertSymToEnum(t[n - 1]));
            // string s = t[n] + "=" + t[n - 1];
            // cout << s << endl;
            ++n;
        }
    }

    uint32_t convertSymToEnum(const std::string& sym) {
        std::unordered_map<std::string, uint32_t> dict{// var
                                                       {"v0", xf::database::DYN_EVAL_REG_V0},
                                                       {"v1", xf::database::DYN_EVAL_REG_V1},
                                                       {"v2", xf::database::DYN_EVAL_REG_V2},
                                                       {"v3", xf::database::DYN_EVAL_REG_V3},
                                                       // const
                                                       {"c0", xf::database::DYN_EVAL_REG_I0},
                                                       {"c1", xf::database::DYN_EVAL_REG_I1},
                                                       {"c2", xf::database::DYN_EVAL_REG_I2},
                                                       {"c3", xf::database::DYN_EVAL_REG_I3},
                                                       // tmp
                                                       {"t0", xf::database::DYN_EVAL_REG_T0},
                                                       {"t1", xf::database::DYN_EVAL_REG_T1},
                                                       {"t2", xf::database::DYN_EVAL_REG_T2},
                                                       {"t3", xf::database::DYN_EVAL_REG_T3},
                                                       {"t4", xf::database::DYN_EVAL_REG_T4},
                                                       {"t5", xf::database::DYN_EVAL_REG_T5},
                                                       {"t6", xf::database::DYN_EVAL_REG_T6},
                                                       // op
                                                       {"+", xf::database::DYN_EVAL_OP_ADD},
                                                       {"-", xf::database::DYN_EVAL_OP_SUB},
                                                       {"*", xf::database::DYN_EVAL_OP_MUL},
                                                       {"/", xf::database::DYN_EVAL_OP_DIV},
                                                       // not used
                                                       {" ", 0UL}};
        auto p = dict.find(sym);
        if (p != dict.end()) {
            return p->second;
        } else {
            assert(0 && "Unknown sym!");
        }
        return 0;
    }

    bool isOperator(const char& c) { return (c == '+' || c == '-' || c == '*' || c == '/'); }
    bool isOperator(const std::string& s) { return (s == "+" || s == "-" || s == "*" || s == "/"); }
    bool isSymbol(const std::string& s) {
        return (s.size() == 2 && (s[0] == 'v' || s[0] == 'c') && (s[1] >= '0' && s[1] <= '3'));
    }
    int precedence(const std::string& op) {
        if (op == "+" || op == "-") {
            return 1;
        } else if (op == "*" || op == "/") {
            return 2;
        }
        return 0;
    }
}; // DynEvalCompiler

static const std::map<std::string, int> opPriorty = {{"NULL", 0}, {"&&", 1}, {"||", 1}, {"!", 3}, {"(", 4},
                                                     {">=", 2},   {">", 2},  {"<=", 2}, {"<", 2}, {"==", 2},
                                                     {"!=", 2},   {"*", 3},  {"+", 2},  {"-", 2}};

/* filter (true) */
// eliminate all space in the string
//
inline void trim(std::string& s) {
    size_t index = 0;
    if (!s.empty()) {
        while ((index = s.find(' ', index)) != std::string::npos) {
            s.erase(index, 1);
        }
    }
}

class EvalParser {
   private:
    // record opratores between the four, a b c d
    std::vector<std::string> operators_1;
    // init with 0
    std::vector<int> constants;
    // init with false
    std::vector<bool> strm;
    // init with false, when true, strm with a '-' sign
    std::vector<bool> strm_notation;

    int ch_num;

    std::string midExpr;
    std::vector<std::string> sufExpr;
    std::vector<bool> sufExprTag;

    bool if_strm_str = false;

    // 0~9
    bool isDigit(char a) { return (a >= 0x30 && a <= 0x39); }

    bool isCapital(char a) { return (a >= 0x61 && a <= 0x64); }

    // a || b|| c || d
    bool isData(char a) { return (a >= 0x61 && a <= 0x64) || (a >= 0x30 && a <= 0x39); }

    // supported operation
    bool isOp(char op) {
        return (op == ':' || op == '?' || op == '+' || op == '*' || op == '=' || op == '!' || op == '>' || op == '<' ||
                op == '|' || op == '&' || op == '^' || op == '~' || op == '(' || op == ')' || op == '-');
    }
    void set_if_strm_str(bool tag) { if_strm_str = tag; }
    void use_old_compiler() { if_strm_str = true; }

    void toSufExpr() {
        std::stack<std::string> st1;
        for (unsigned i = 0; i < midExpr.length();) {
            // if the (-a, (-b, (-c, (-d
            if (midExpr[i] == '-' && i != 0 && i < midExpr.length() - 1 && midExpr[i - 1] == '(' &&
                isCapital(midExpr[i + 1])) {
                if (isCapital(midExpr[i + 2])) {
                    std::cerr << "error: invalid input !" << std::endl;
                    exit(1);
                }
                int ind = midExpr[i + 1] - 0x61;
                // - notation not push into stack, only tag the strm_notatin
                strm_notation[ind] = true;
                i++;
            } else if ((isOp(midExpr[i]) && midExpr[i] != '-') ||
                       (midExpr[i] == '-' && i != 0 && isCapital(midExpr[i - 1])) ||
                       (midExpr[i] == '-' && i < midExpr.length() - 1 &&
                        isCapital(midExpr[i + 1]))) { // when the oprator, and not a sign
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
                        if ('!' == midExpr[i] || '~' == midExpr[i]) break;
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
            } else if (isData(midExpr[i]) || (midExpr[i] == '-' && i < midExpr.length() - 1 &&
                                              isDigit(midExpr[i + 1]))) { // when oprator signed/unsigned data
                                                                          //
                char data[10];
                int data_len = 0;
                if (midExpr[i] == '-') {
                    data[0] = '-';
                    data_len = 1;
                    i++;
                }
                for (; isData(midExpr[i]) && i < midExpr.length(); i++) {
                    data[data_len++] = midExpr[i];
                }
                data[data_len] = 0;
                std::string datastr(data, strlen(data)); // strlen(data) = data_len+1
                sufExpr.push_back(datastr);
                sufExprTag.push_back(true);
            } else {
                std::cerr << "not supported ops:" << midExpr[i] << std::endl;
                exit(1);
            }
        }

        while (!st1.empty()) {
            std::string tmp = st1.top();
            sufExpr.push_back(tmp);
            sufExprTag.push_back(false);
            st1.pop();
        }

#ifdef DEBUGEVAL
        int i = 0;
        for (std::string s : sufExpr) {
            std::cout << s << "   isop?:" << sufExprTag[i++] << " " << std::endl;
        }
#endif
    }
    std::string process_pair(std::string l, std::string r, std::string op) {
        std::string result;
        // rule for first walk;
        const char* l_array = l.c_str();
        const char* r_array = r.c_str();
        bool if_c_l = true;
        bool if_c_r = true;
        bool if_tmp_l = false;
        bool if_tmp_r = false;
        for (size_t i = 0; i < strlen(l_array); i++) {
            char a = l_array[i];
            if (i == 0) {
                if (!isDigit(a) && a != '-') {
                    if_c_l = false;
                    break;
                }
            } else if (!isDigit(a)) {
                if_c_l = false;
                break;
            }
        }
        for (size_t i = 0; i < strlen(r_array); i++) {
            char a = r_array[i];
            if (i == 0) {
                if (!isDigit(a) && a != '-') {
                    if_c_r = false;
                    break;
                }
            } else if (!isDigit(a)) {
                if_c_r = false;
                break;
            }
        }
        if (!if_c_l) {
            if (l.length() == 1 && l_array[0] >= 'a' && l_array[0] <= 'd') {
                if_tmp_l = false;
            } else {
                if_tmp_l = true;
            }
        }

        if (!if_c_r) {
            if (r.length() == 1 && r_array[0] >= 'a' && r_array[0] <= 'd') {
                if_tmp_r = false;
            } else {
                if_tmp_r = true;
            }
        }
        std::string lstr;
        std::string rstr;
        std::string opstr;
        if (if_c_r) {
            int cont = std::stoi(r);
            if (if_c_l || if_tmp_l) {
                std::cerr << "please put the constant operand behind a variable (a/b/c/d)" << std::endl;
                exit(1);
            } else { // then l is a/b/c/d
                int ind = l_array[0] - 'a';
                strm[ind] = true;
                constants[ind] = cont;
                opstr = op;
                if (if_strm_str) {
                    lstr = "strm" + std::to_string(ind + 1);
                    rstr = "c" + std::to_string(ind + 1);
                } else {
                    lstr = "v" + std::to_string(ind);
                    rstr = "c" + std::to_string(ind);
                }
#ifdef DEBUGEVAL
                std::cout << "l_var and r_con:strm" << ind << ", op:" << op << ",con:" << cont << std::endl;
#endif
                if (strm_notation[ind]) {
                    std::cerr << "not support signed strm, please transform into minus operation!" << std::endl;
                } else {
                    return "(" + lstr + opstr + rstr + ")";
                }
            }
        }

        if (if_c_l) {
            int cont = std::stoi(l);
            if (if_c_r || if_tmp_r) {
                std::cerr << "please put the constant operand behind a variable (a/b/c/d)" << std::endl;
            } else {
                int ind = r_array[0] - 0x61;
                strm[ind] = true;
                constants[ind] = cont;
                opstr = op;
                if (if_strm_str) {
                    lstr = "strm" + std::to_string(ind + 1);
                    rstr = "c" + std::to_string(ind + 1);
                } else {
                    lstr = "v" + std::to_string(ind);
                    rstr = "c" + std::to_string(ind);
                }
                if (strm_notation[ind]) {
                    std::cerr << "not support signed strm, please transform into minus operation!" << std::endl;
                } else {
                    return "(" + lstr + opstr + rstr + ")";
                }
#ifdef DEBUGEVAL
                std::cout << "l_var and r_var:" << lstr << "," << rstr << ", op:" << op << std::endl;
#endif
            }
        }
        if (!if_c_l && !if_c_r) {
            if (if_tmp_l) {
                lstr = l;
            } else {
                int ind_l = l_array[0] - 0x61;
                strm[ind_l] = true;
                if (strm_notation[ind_l]) {
                    std::cerr << "not support signed strm, please transform into minus operation!" << std::endl;
                } else {
                    if (if_strm_str) {
                        lstr = "strm" + std::to_string(ind_l + 1);
                    } else {
                        lstr = "v" + std::to_string(ind_l);
                    }
                }
            }
            if (if_tmp_r) {
                rstr = r;
            } else {
                int ind_r = r_array[0] - 0x61;
                strm[ind_r] = true;
                if (strm_notation[ind_r]) {
                    std::cerr << "not support signed strm, please transform into minus operation!" << std::endl;
                } else {
                    if (if_strm_str) {
                        rstr = "strm" + std::to_string(ind_r + 1);
                    } else {
                        rstr = "v" + std::to_string(ind_r);
                    }
                }
            }
#ifdef DEBUGEVAL
            std::cout << "l_var and r_var:" << lstr << "," << rstr << ", op:" << op << std::endl;
#endif
            return "(" + lstr + op + rstr + ")";
        }
        return lstr + op + rstr;
    }

    std::string process_single(std::string l, std::string op) {
        bool if_var_l = (l == "a" || l == "b" || l == "c" || l == "d");
        if (if_var_l) {
            std::cerr << "invalid input!" << std::endl;
            exit(1);
        }
        const char* l_array = l.c_str();
        int ind_l = l_array[0] - 0x61;
        strm[ind_l] = true;
        operators_1[ind_l] = op;
        return (l);
    }
    void process(std::string op, std::stack<std::string>& data) {
        if (op == "!" || op == "~") {
            std::string r = data.top();
            data.pop();
            std::string rel = process_single(r, op);
            data.push(rel);
        } else {
            std::string l = data.top();
            data.pop();
            std::string r = data.top();
            data.pop();
            std::string rel = process_pair(r, l, op);
#ifdef DEBUGEVAL
            std::cout << "walk tree, process op: " << op << ", operands::" << r << "," << l << ",results:" << rel
                      << std::endl;
#endif
            data.push(rel);
        }
    }

    std::string walkTree() {
        std::stack<std::string> op;
        std::stack<std::string> data;
        for (unsigned i = 0; i < sufExpr.size(); i++) {
            bool ifdata = sufExprTag[i];
            std::string tmp = sufExpr[i];
            if (ifdata) {
                data.push(tmp);
#ifdef DEBUGEVAL
                std::cout << "walk tree, push data: " << tmp << std::endl;
#endif
            } else {
                process(tmp, data);
            }
        }
        return data.top();
    }

   public:
    EvalParser(std::string s, int _ch_num = 4) {
        midExpr = s;
        ch_num = _ch_num;
        trim(midExpr);
        for (int i = 0; i < ch_num; i++) {
            operators_1.push_back("NULL");
            constants.push_back(0);
            strm.push_back(false);
            strm_notation.push_back(false);
        }
    }

    std::string doParser() {
        toSufExpr();
        std::string s = walkTree();
        return s;
    }
    std::vector<int> getConts() const { return constants; }

    void printConts() {
        for (int cont : constants) {
            std::cout << cont << " " << std::endl;
        }
    }
};

} // namespace eval_config
} // namespace internals

/**
 * @class Dynamic Evaluation Configuration Generator.
 */
class DynamicEvalV2Config {
    std::string eval_str;
    enum { DYN_EVAL_CFG_LENGTH = (DYN_EVAL_NSTEP + 1) / 2 + DYN_EVAL_NCOL };

   public:
    /**
     * @brief construct with evaluation expression string.
     *
     * The string uses ``a``, ``b``, ``c``, ``d`` to refer to first to the fourth column,
     * and supports four interger operators including ``+``, ``-``, ``*`` and  ``/`` , like C++
     * Note that it does not support signed varibles, like ``-a``, ``-b``, ``-c``, ``-d``
     * so please try to convert them into a binary ``-`` operation.
     *
     * For example, an expression could be ``(-100 - b) * (10 - a) + (d + 3) * (c)``.
     *
     * @param s the evaluation expression string.
     */

    DynamicEvalV2Config(std::string s) { eval_str = s; }

    /**
     * @brief get the length of config bits buffer.
     */
    size_t getConfigLen() const { return DYN_EVAL_CFG_LENGTH; }

    /**
     * @brief generate a configuration bits buffer.
     * @return a unique_ptr to configuration bits buffer of ``uint32_t`` type.
     */

    std::unique_ptr<uint32_t[]> getConfigBits() const {
        const int inst_len = (DYN_EVAL_NSTEP + 1) / 2;
        std::unique_ptr<uint32_t[]> cfgs(new uint32_t[DYN_EVAL_CFG_LENGTH]);
        internals::eval_config::EvalParser ep(eval_str);
        std::string str = ep.doParser();
        std::vector<int> conts = ep.getConts();

#ifdef DEBUGEVAL
        std::cout << "final str: " << str << std::endl;
        std::cout << "conts: " << conts[0] << "," << conts[1] << "," << conts[2] << "," << conts[3] << std::endl;
        ep.printConts();
#endif
        uint32_t inst[inst_len];
        internals::eval_config::DynEvalCompiler ec;
        ec.writeOp(inst, str);
        int imm[DYN_EVAL_NCOL] = {conts[0], conts[1], conts[2], conts[3]};
        for (int i = 0; i < inst_len; ++i) {
            cfgs[i] = inst[i];
        }
        for (int i = 0; i < DYN_EVAL_NCOL; ++i) {
            cfgs[inst_len + i] = imm[i];
        }
        return cfgs;
    }
};

} // namespace database
} // namespace xf
#endif // XF_DB_DYN_EVAL_CONFIG_H
