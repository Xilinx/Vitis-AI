/*
 * Copyright 2019 Xilinx, Inc.
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

#ifdef ONIG_ESCAPE_UCHAR_COLLISION
#undef ONIG_ESCAPE_UCHAR_COLLISION
#endif

extern "C" {
#include "oniguruma.h"
#include "xf_data_analytics/text/xf_re_compile.h"
}

#include <iostream>
#include <stdio.h>
#include <cstring>
#include <string.h>

#include "ap_int.h"
#include "xf_data_analytics/text/regexVM.hpp"

#define SLEN(s) strlen(s)

#define BIT_SET_SIZE (1024)     // max supported size of the bit map for op cclass
#define INSTRUC_SIZE (32768)    // max supported size of the instruction table
#define MESSAGE_SIZE (2048 / 8) // max supported length of the input string
#define CAP_GRP_SIZE (256)      // max supported size of the capturing groups
#define STACK_SIZE (16384)      // max supported internal stack for backtracking

void dut(unsigned int bitset[BIT_SET_SIZE],
         ap_uint<64> instr_buff[INSTRUC_SIZE],
         ap_uint<32> msg_buff[MESSAGE_SIZE],
         unsigned int str_len,
         ap_uint<2> out[1],
         ap_uint<16> offset_buff[CAP_GRP_SIZE * 2]) {
#pragma HLS bind_storage variable = bitset type = ram_t2p impl = bram
//#pragma HLS resource variable = instr_buff core = RAM_2P_URAM uram
#pragma HLS bind_storage variable = instr_buff type = ram_t2p impl = bram
#pragma HLS bind_storage variable = msg_buff type = ram_t2p impl = bram
#pragma HLS bind_storage variable = offset_buff type = ram_t2p impl = bram
    // initialize buffer
    for (int i = 0; i < 2 * CAP_GRP_SIZE; ++i) {
        offset_buff[i] = -1;
    }

    ap_uint<2> match = false;
    // xf::data_analytics::text::regexVM<STACK_SIZE>((ap_uint<32>*)bitset, instr_buff, msg_buff, str_len, match,
    // offset_buff);
    xf::data_analytics::text::regexVM_opt<STACK_SIZE>((ap_uint<32>*)bitset, instr_buff, msg_buff, str_len, match,
                                                      offset_buff);
    // out_buff[0] = match;
    out[0] = match;
    // for(int i = 0; i < CAP_GRP_SIZE*2; ++i) {
    //    out_buff[i+1] = offset_buff[i];
    //}
}

static int nsucc = 0;
static int nfail = 0;
static int nerror = 0;
static int nsupport = 0;
static int curr_test = 0;

static OnigRegion* region;

int compile(const char* pattern,
            int error_no,
            int* nsucc,
            int* nfail,
            int* nerror,
            unsigned int* bitset,
            uint64_t* instr_buff,
            unsigned int* instr_num,
            unsigned int* cclass_num,
            unsigned int* cpgp_num) {
    int r = xf_re_compile(pattern, bitset, instr_buff, instr_num, cclass_num, cpgp_num, NULL, NULL);
    if (r != ONIG_NORMAL && r != XF_UNSUPPORTED_OPCODE) {
        char s[ONIG_MAX_ERROR_MESSAGE_LEN];

        if (error_no == 0) {
            onig_error_code_to_str((UChar*)s, r);
            fprintf(stdout, "ERROR: %s  /%s/\n", s, pattern);
            (*nerror)++;
        } else {
            if (r == error_no) {
                fprintf(stdout, "OK(ERROR): /%s/ %d\n", pattern, r);
                (*nsucc)++;
            } else {
                fprintf(stdout, "FAIL(ERROR): /%s/ , %d, %d\n", pattern, error_no, r);
                (*nfail)++;
            }
        }
    }
    return r;
}
static void single_test(const char* pattern, const char* str, int from, int to, int mem, int negative, int error_no) {
    printf("Test case No.%d\n", curr_test);
    int r;
    unsigned int instr_num = 0;
    unsigned int cclass_num = 0;
    unsigned int cpgp_num = 0;
    bool support = false;
    // bit set map for op cclass
    unsigned int bitset[BIT_SET_SIZE];
    // instruction list
    uint64_t instr_buff[INSTRUC_SIZE];
    // message buffer
    ap_uint<32> msg_buff[MESSAGE_SIZE];
    for (int i = 0; i < MESSAGE_SIZE; i++) {
        msg_buff[i] = 0;
    }
    // captured group offset addresses
    ap_uint<16> offset_buff[2 * CAP_GRP_SIZE];
    for (int i = 0; i < 2 * CAP_GRP_SIZE; i++) {
        offset_buff[i] = 0;
    }

    r = compile(pattern, error_no, &nsucc, &nfail, &nerror, bitset, instr_buff, &instr_num, &cclass_num, &cpgp_num);
    // printf("instr_num = %d, cclass_num = %d, cpgp_num = %d\n", instr_num, cclass_num, cpgp_num);

    if (r == XF_UNSUPPORTED_OPCODE) {
        fprintf(stdout, "UNSUPPORT: /%s/ '%s'\n", pattern, str);
        nsupport++;
    }
    if (r == ONIG_NORMAL) {
        regex_t* reg;
        OnigErrorInfo einfo;
        r = onig_new(&reg, (UChar*)pattern, (UChar*)(pattern + SLEN(pattern)), ONIG_OPTION_DEFAULT, ONIG_ENCODING_ASCII,
                     ONIG_SYNTAX_DEFAULT, &einfo);
        r = onig_search(reg, (UChar*)str, (UChar*)(str + SLEN(str)), (UChar*)str, (UChar*)(str + SLEN(str)), region,
                        ONIG_OPTION_NONE);

        if (!region->beg[0]) {
            unsigned int str_len = strlen((const char*)str);
            // prepare input message buffer for FPGA
            for (int i = 0; i < (str_len + 3) / 4; i++) {
                for (int k = 0; k < 4; ++k) {
                    if (i * 4 + k < str_len) {
                        msg_buff[i].range((k + 1) * 8 - 1, k * 8) = str[i * 4 + k];
                    } else {
                        msg_buff[i].range((k + 1) * 8 - 1, k * 8) = ' ';
                    }
                    // printf("%c", (unsigned char)msg_buff[i].range((k+1)*8-1,k*8));
                }
            }
            printf("\n");

            // call FPGA
            ap_uint<2> out[1];
            dut(bitset, (ap_uint<64>*)instr_buff, msg_buff, str_len, out, offset_buff);

            if (r == ONIG_MISMATCH) {
                if (negative && (out[0] == 0)) {
                    fprintf(stdout, "OK(N): /%s/ '%s'\n", pattern, str);
                    nsucc++;
                } else {
                    fprintf(stdout, "FAIL: /%s/ '%s'\n", pattern, str);
                    nfail++;
                }
            } else {
                if (!negative && out[0] == 1) {
                    if (region->beg[mem] == (int)offset_buff[2 * mem] &&
                        region->end[mem] == (int)offset_buff[2 * mem + 1]) {
                        fprintf(stdout, "OK: /%s/ '%s'\n", pattern, str);
                        nsucc++;
                    } else {
                        fprintf(stdout, "FAIL: /%s/ '%s' %d-%d : %d-%d\n", pattern, str, region->beg[mem],
                                region->end[mem], offset_buff[2 * mem], offset_buff[2 * mem + 1]);
                        nfail++;
                    }
                } else {
                    fprintf(stdout, "FAIL(N): /%s/ '%s'\n", pattern, str);
                    nfail++;
                }
            }
        } else {
            fprintf(stdout, "UNSUPPORT: /%s/ '%s'\n", pattern, str);
            nsupport++;
        }
        onig_free(reg);
    }
    curr_test++;
}

static void x2(const std::string& pattern, const std::string& str, int from, int to) {
    single_test(pattern.c_str(), str.c_str(), from, to, 0, 0, 0);
}

static void x3(const std::string& pattern, const std::string& str, int from, int to, int mem) {
    single_test(pattern.c_str(), str.c_str(), from, to, mem, 0, 0);
}

static void n(const std::string& pattern, const std::string& str) {
    single_test(pattern.c_str(), str.c_str(), 0, 0, 0, 1, 0);
}

static void e(const std::string& pattern, const std::string& str, int error_no) {
    single_test(pattern.c_str(), str.c_str(), 0, 0, 0, 0, error_no);
}

extern int main(int argc, char* argv[]) {
    OnigEncoding use_encs[1];
    use_encs[0] = ONIG_ENCODING_ASCII;
    onig_initialize(use_encs, sizeof(use_encs) / sizeof(use_encs[0]));

    region = onig_region_new();

    // test case 0
    x2("", "", 0, 0);
    // test case 1
    x2("^", "", 0, 0);
    // test case 2
    x2("^a", "\na", 1, 2);
    // test case 3
    x2("$", "", 0, 0);
    // test case 4
    x2("$\\O", "bb\n", 2, 3);
    // test case 5
    x2("\\G", "", 0, 0);
    // test case 6
    x2("\\A", "", 0, 0);
    // test case 7
    x2("\\Z", "", 0, 0);
    // test case 8
    x2("\\z", "", 0, 0);
    // test case 9
    x2("^$", "", 0, 0);
    // test case 10
    x2("\\ca", "\001", 0, 1);
    // test case 11
    x2("\\C-b", "\002", 0, 1);
    // test case 12
    x2("\\c\\\\", "\034", 0, 1);
    // test case 13
    x2("q[\\c\\\\]", "q\034", 0, 2);
    // test case 14
    x2("", "a", 0, 0);
    // test case 15
    x2("a", "a", 0, 1);
    // test case 16
    x2("\\x61", "a", 0, 1);
    // test case 17
    x2("aa", "aa", 0, 2);
    // test case 18
    x2("aaa", "aaa", 0, 3);
    // test case 19
    x2("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0, 35);
    // test case 20
    x2("ab", "ab", 0, 2);
    // test case 21
    x2("b", "ab", 1, 2);
    // test case 22
    x2("bc", "abc", 1, 3);
    // test case 23
    x2("(?i:#RET#)", "#INS##RET#", 5, 10);
    // test case 24
    x2("\\17", "\017", 0, 1);
    // test case 25
    x2("\\x1f", "\x1f", 0, 1);
    // test case 26
    x2("a(?#....\\\\JJJJ)b", "ab", 0, 2);
    // test case 27
    x2("(?x)  G (o O(?-x)oO) g L", "GoOoOgLe", 0, 7);
    // test case 28
    x2(".", "a", 0, 1);
    // test case 29
    n(".", "");
    // test case 30
    x2("..", "ab", 0, 2);
    // test case 31
    x2("\\w", "e", 0, 1);
    // test case 32
    n("\\W", "e");
    // test case 33
    x2("\\s", " ", 0, 1);
    // test case 34
    x2("\\S", "b", 0, 1);
    // test case 35
    x2("\\d", "4", 0, 1);
    // test case 36
    n("\\D", "4");
    // test case 37
    x2("\\b", "z ", 0, 0);
    // test case 38
    x2("\\b", " z", 1, 1);
    // test case 39
    x2("\\b", "  z ", 2, 2);
    // test case 40
    x2("\\B", "zz ", 1, 1);
    // test case 41
    x2("\\B", "z ", 2, 2);
    // test case 42
    x2("\\B", " z", 0, 0);
    // test case 43
    x2("[ab]", "b", 0, 1);
    // test case 44
    n("[ab]", "c");
    // test case 45
    x2("[a-z]", "t", 0, 1);
    // test case 46
    n("[^a]", "a");
    // test case 47
    x2("[^a]", "\n", 0, 1);
    // test case 48
    x2("[]]", "]", 0, 1);
    // test case 49
    n("[^]]", "]");
    // test case 50
    x2("[\\^]+", "0^^1", 1, 3);
    // test case 51
    x2("[b-]", "b", 0, 1);
    // test case 52
    x2("[b-]", "-", 0, 1);
    // test case 53
    x2("[\\w]", "z", 0, 1);
    // test case 54
    n("[\\w]", " ");
    // test case 55
    x2("[\\W]", "b$", 1, 2);
    // test case 56
    x2("[\\d]", "5", 0, 1);
    // test case 57
    n("[\\d]", "e");
    // test case 58
    x2("[\\D]", "t", 0, 1);
    // test case 59
    n("[\\D]", "3");
    // test case 60
    x2("[\\s]", " ", 0, 1);
    // test case 61
    n("[\\s]", "a");
    // test case 62
    x2("[\\S]", "b", 0, 1);
    // test case 63
    n("[\\S]", " ");
    // test case 64
    x2("[\\w\\d]", "2", 0, 1);
    // test case 65
    n("[\\w\\d]", " ");
    // test case 66
    x2("[[:upper:]]", "B", 0, 1);
    // test case 67
    x2("[*[:xdigit:]+]", "+", 0, 1);
    // test case 68
    x2("[*[:xdigit:]+]", "GHIKK-9+*", 6, 7);
    // test case 69
    x2("[*[:xdigit:]+]", "-@^+", 3, 4);
    // test case 70
    n("[[:upper]]", "A");
    // test case 71
    x2("[[:upper]]", ":", 0, 1);
    // test case 72
    x2("[\\044-\\047]", "\046", 0, 1);
    // test case 73
    x2("[\\x5a-\\x5c]", "\x5b", 0, 1);
    // test case 74
    x2("[\\x6A-\\x6D]", "\x6c", 0, 1);
    // test case 75
    n("[\\x6A-\\x6D]", "\x6E");
    // test case 76
    n("^[0-9A-F]+ 0+ UNDEF ", "75F 00000000 SECT14A notype ()    External    | _rb_apply");
    // test case 77
    x2("[\\[]", "[", 0, 1);
    // test case 78
    x2("[\\]]", "]", 0, 1);
    // test case 79
    x2("[&]", "&", 0, 1);
    // test case 80
    x2("[[ab]]", "b", 0, 1);
    // test case 81
    x2("[[ab]c]", "c", 0, 1);
    // test case 82
    n("[[^a]]", "a");
    // test case 83
    n("[^[a]]", "a");
    // test case 84
    x2("[[ab]&&bc]", "b", 0, 1);
    // test case 85
    n("[[ab]&&bc]", "a");
    // test case 86
    n("[[ab]&&bc]", "c");
    // test case 87
    x2("[a-z&&b-y&&c-x]", "w", 0, 1);
    // test case 88
    n("[^a-z&&b-y&&c-x]", "w");
    // test case 89
    x2("[[^a&&a]&&a-z]", "b", 0, 1);
    // test case 90
    n("[[^a&&a]&&a-z]", "a");
    // test case 91
    x2("[[^a-z&&bcdef]&&[^c-g]]", "h", 0, 1);
    // test case 92
    n("[[^a-z&&bcdef]&&[^c-g]]", "c");
    // test case 93
    x2("[^[^abc]&&[^cde]]", "c", 0, 1);
    // test case 94
    x2("[^[^abc]&&[^cde]]", "e", 0, 1);
    // test case 95
    n("[^[^abc]&&[^cde]]", "f");
    // test case 96
    x2("[a-&&-a]", "-", 0, 1);
    // test case 97
    n("[a\\-&&\\-a]", "&");
    // test case 98
    n("\\wabc", " abc");
    // test case 99
    x2("a\\Wbc", "a bc", 0, 4);
    // test case 100
    x2("a.b.c", "aabbc", 0, 5);
    // test case 101
    x2(".\\wb\\W..c", "abb bcc", 0, 7);
    // test case 102
    x2("\\s\\wzzz", " zzzz", 0, 5);
    // test case 103
    x2("aa.b", "aabb", 0, 4);
    // test case 104
    n(".a", "ab");
    // test case 105
    x2(".a", "aa", 0, 2);
    // test case 106
    x2("^a", "a", 0, 1);
    // test case 107
    x2("^a$", "a", 0, 1);
    // test case 108
    x2("^\\w$", "a", 0, 1);
    // test case 109
    n("^\\w$", " ");
    // test case 110
    x2("^\\wab$", "zab", 0, 3);
    // test case 111
    x2("^\\wabcdef$", "zabcdef", 0, 7);
    // test case 112
    x2("^\\w...def$", "zabcdef", 0, 7);
    // test case 113
    x2("\\w\\w\\s\\Waaa\\d", "aa  aaa4", 0, 8);
    // test case 114
    x2("\\A\\Z", "", 0, 0);
    // test case 115
    x2("\\Axyz", "xyz", 0, 3);
    // test case 116
    x2("xyz\\Z", "xyz", 0, 3);
    // test case 117
    x2("xyz\\z", "xyz", 0, 3);
    // test case 118
    x2("a\\Z", "a", 0, 1);
    // test case 119
    x2("\\Gaz", "az", 0, 2);
    // test case 120
    n("\\Gz", "bza");
    // test case 121
    n("az\\G", "az");
    // test case 122
    n("az\\A", "az");
    // test case 123
    n("a\\Az", "az");
    // test case 124
    x2("\\^\\$", "^$", 0, 2);
    // test case 125
    x2("^x?y", "xy", 0, 2);
    // test case 126
    x2("^(x?y)", "xy", 0, 2);
    // test case 127
    x2("\\w", "_", 0, 1);
    // test case 128
    n("\\W", "_");
    // test case 129
    x2("(?=z)z", "z", 0, 1);
    // test case 130
    n("(?=z).", "a");
    // test case 131
    x2("(?!z)a", "a", 0, 1);
    // test case 132
    n("(?!z)a", "z");
    // test case 133
    x2("(?i:a)", "a", 0, 1);
    // test case 134
    x2("(?i:a)", "A", 0, 1);
    // test case 135
    x2("(?i:A)", "a", 0, 1);
    // test case 136
    x2("(?i:i)", "I", 0, 1);
    // test case 137
    x2("(?i:I)", "i", 0, 1);
    // test case 138
    x2("(?i:[A-Z])", "i", 0, 1);
    // test case 139
    x2("(?i:[a-z])", "I", 0, 1);
    // test case 140
    n("(?i:A)", "b");
    // test case 141
    x2("(?i:ss)", "ss", 0, 2);
    // test case 142
    x2("(?i:ss)", "Ss", 0, 2);
    // test case 143
    x2("(?i:ss)", "SS", 0, 2);
    // x2("(?i:ss)", "\xc5\xbfS", 0, 3);
    // x2("(?i:ss)", "s\xc5\xbf", 0, 3);
    // x2("(?i:ss)", "\xc3\x9f", 0, 2);
    // x2("(?i:ss)", "\xe1\xba\x9e", 0, 3);
    // test case 144
    x2("(?i:xssy)", "xssy", 0, 4);
    // test case 145
    x2("(?i:xssy)", "xSsy", 0, 4);
    // test case 146
    x2("(?i:xssy)", "xSSy", 0, 4);
    // x2("(?i:xssy)", "x\xc5\xbfSy", 0, 5);
    // x2("(?i:xssy)", "xs\xc5\xbfy", 0, 5);
    // x2("(?i:xssy)", "x\xc3\x9fy", 0, 4);
    // x2("(?i:xssy)", "x\xe1\xba\x9ey", 0, 5);
    // x2("(?i:x\xc3\x9fy)", "xssy", 0, 4);
    // x2("(?i:x\xc3\x9fy)", "xSSy", 0, 4);
    // x2("(?i:\xc3\x9f)", "ss", 0, 2);
    // x2("(?i:\xc3\x9f)", "SS", 0, 2);
    // x2("(?i:[\xc3\x9f])", "ss", 0, 2);
    // x2("(?i:[\xc3\x9f])", "SS", 0, 2);
    // test case 147
    x2("(?i)(?<!ss)z", "qqz", 2, 3);
    // test case 148
    x2("(?i:[A-Z])", "a", 0, 1);
    // test case 149
    x2("(?i:[f-m])", "H", 0, 1);
    // test case 150
    x2("(?i:[f-m])", "h", 0, 1);
    // test case 151
    n("(?i:[f-m])", "e");
    // test case 152
    x2("(?i:[A-c])", "D", 0, 1);
    // test case 153
    n("(?i:[^a-z])", "A");
    // test case 154
    n("(?i:[^a-z])", "a");
    // test case 155
    x2("(?i:[!-k])", "Z", 0, 1);
    // test case 156
    x2("(?i:[!-k])", "7", 0, 1);
    // test case 157
    x2("(?i:[T-}])", "b", 0, 1);
    // test case 158
    x2("(?i:[T-}])", "{", 0, 1);
    // test case 159
    x2("(?i:\\?a)", "?A", 0, 2);
    // test case 160
    x2("(?i:\\*A)", "*a", 0, 2);
    // test case 161
    n(".", "\n");
    // test case 162
    x2("(?m:.)", "\n", 0, 1);
    // test case 163
    x2("(?m:a.)", "a\n", 0, 2);
    // test case 164
    x2("(?m:.b)", "a\nb", 1, 3);
    // test case 165
    x2(".*abc", "dddabdd\nddabc", 8, 13);
    // test case 166
    x2(".+abc", "dddabdd\nddabcaa\naaaabc", 8, 13);
    // test case 167
    x2("(?m:.*abc)", "dddabddabc", 0, 10);
    // test case 168
    n("(?i)(?-i)a", "A");
    // test case 169
    n("(?i)(?-i:a)", "A");
    // test case 170
    x2("a?", "", 0, 0);
    // test case 171
    x2("a?", "b", 0, 0);
    // test case 172
    x2("a?", "a", 0, 1);
    // test case 173
    x2("a*", "", 0, 0);
    // test case 174
    x2("a*", "a", 0, 1);
    // test case 175
    x2("a*", "aaa", 0, 3);
    // test case 176
    x2("a*", "baaaa", 0, 0);
    // test case 177
    n("a+", "");
    // test case 178
    x2("a+", "a", 0, 1);
    // test case 179
    x2("a+", "aaaa", 0, 4);
    // test case 180
    x2("a+", "aabbb", 0, 2);
    // test case 181
    x2("a+", "baaaa", 1, 5);
    // test case 182
    x2(".?", "", 0, 0);
    // test case 183
    x2(".?", "f", 0, 1);
    // test case 184
    x2(".?", "f", 0, 1);
    // test case 185
    x2(".?", "\n", 0, 0);
    // test case 186
    x2(".*", "", 0, 0);
    // test case 187
    x2(".*", "abcde", 0, 5);
    // test case 188
    x2(".+", "z", 0, 1);
    // test case 189
    x2(".+", "zdswer\n", 0, 6);
    // test case 190
    x2("(.*)a\\1f", "babfbac", 0, 4);
    // test case 191
    x2("(.*)a\\1f", "bacbabf", 3, 7);
    // test case 192
    x2("((.*)a\\2f)", "bacbabf", 3, 7);
    // test case 193
    x2("(.*)a\\1f", "baczzzzzz\nbazz\nzzzzbabf", 19, 23);
    // test case 194
    x2("a|b", "a", 0, 1);
    // test case 195
    x2("a|b", "b", 0, 1);
    // test case 196
    x2("|a", "a", 0, 0);
    // test case 197
    x2("(|a)", "a", 0, 0);
    // test case 198
    x2("ab|bc", "ab", 0, 2);
    // test case 199
    x2("ab|bc", "bc", 0, 2);
    // test case 200
    x2("z(?:ab|bc)", "zbc", 0, 3);
    // test case 201
    x2("a(?:ab|bc)c", "aabc", 0, 4);
    // test case 202
    x2("ab|(?:ac|az)", "az", 0, 2);
    // test case 203
    x2("a|b|c", "dc", 1, 2);
    // test case 204
    x2("a|b|cd|efg|h|ijk|lmn|o|pq|rstuvwx|yz", "pqr", 0, 2);
    // test case 205
    n("a|b|cd|efg|h|ijk|lmn|o|pq|rstuvwx|yz", "mn");
    // test case 206
    x2("a|^z", "ba", 1, 2);
    // test case 207
    x2("a|^z", "za", 0, 1);
    // test case 208
    x2("a|\\Gz", "bza", 2, 3);
    // test case 209
    x2("a|\\Gz", "za", 0, 1);
    // test case 210
    x2("a|\\Az", "bza", 2, 3);
    // test case 211
    x2("a|\\Az", "za", 0, 1);
    // test case 212
    x2("a|b\\Z", "ba", 1, 2);
    // test case 213
    x2("a|b\\Z", "b", 0, 1);
    // test case 214
    x2("a|b\\z", "ba", 1, 2);
    // test case 215
    x2("a|b\\z", "b", 0, 1);
    // test case 216
    x2("\\w|\\s", " ", 0, 1);
    // test case 217
    n("\\w|\\w", " ");
    // test case 218
    x2("\\w|%", "%", 0, 1);
    // test case 219
    x2("\\w|[&$]", "&", 0, 1);
    // test case 220
    x2("[b-d]|[^e-z]", "a", 0, 1);
    // test case 221
    x2("(?:a|[c-f])|bz", "dz", 0, 1);
    // test case 222
    x2("(?:a|[c-f])|bz", "bz", 0, 2);
    // test case 223
    x2("abc|(?=zz)..f", "zzf", 0, 3);
    // test case 224
    x2("abc|(?!zz)..f", "abf", 0, 3);
    // test case 225
    x2("(?=za)..a|(?=zz)..a", "zza", 0, 3);
    // test case 226
    n("(?>a|abd)c", "abdc");
    // test case 227
    x2("(?>abd|a)c", "abdc", 0, 4);
    // test case 228
    x2("a?|b", "a", 0, 1);
    // test case 229
    x2("a?|b", "b", 0, 0);
    // test case 230
    x2("a?|b", "", 0, 0);
    // test case 231
    x2("a*|b", "aa", 0, 2);
    // test case 232
    x2("a*|b*", "ba", 0, 0);
    // test case 233
    x2("a*|b*", "ab", 0, 1);
    // test case 234
    x2("a+|b*", "", 0, 0);
    // test case 235
    x2("a+|b*", "bbb", 0, 3);
    // test case 236
    x2("a+|b*", "abbb", 0, 1);
    // test case 237
    n("a+|b+", "");
    // test case 238
    x2("(a|b)?", "b", 0, 1);
    // test case 239
    x2("(a|b)*", "ba", 0, 2);
    // test case 240
    x2("(a|b)+", "bab", 0, 3);
    // test case 241
    x2("(ab|ca)+", "caabbc", 0, 4);
    // test case 242
    x2("(ab|ca)+", "aabca", 1, 5);
    // test case 243
    x2("(ab|ca)+", "abzca", 0, 2);
    // test case 244
    x2("(a|bab)+", "ababa", 0, 5);
    // test case 245
    x2("(a|bab)+", "ba", 1, 2);
    // test case 246
    x2("(a|bab)+", "baaaba", 1, 4);
    // test case 247
    x2("(?:a|b)(?:a|b)", "ab", 0, 2);
    // test case 248
    x2("(?:a*|b*)(?:a*|b*)", "aaabbb", 0, 3);
    // test case 249
    x2("(?:a*|b*)(?:a+|b+)", "aaabbb", 0, 6);
    // test case 250
    x2("(?:a+|b+){2}", "aaabbb", 0, 6);
    // test case 251
    x2("h{0,}", "hhhh", 0, 4);
    // test case 252
    x2("(?:a+|b+){1,2}", "aaabbb", 0, 6);
    // test case 253
    n("ax{2}*a", "0axxxa1");
    // test case 254
    n("a.{0,2}a", "0aXXXa0");
    // test case 255
    n("a.{0,2}?a", "0aXXXa0");
    // test case 256
    n("a.{0,2}?a", "0aXXXXa0");
    // test case 257
    x2("^a{2,}?a$", "aaa", 0, 3);
    // test case 258
    x2("^[a-z]{2,}?$", "aaa", 0, 3);
    // test case 259
    x2("(?:a+|\\Ab*)cc", "cc", 0, 2);
    // test case 260
    n("(?:a+|\\Ab*)cc", "abcc");
    // test case 261
    x2("(?:^a+|b+)*c", "aabbbabc", 6, 8);
    // test case 262
    x2("(?:^a+|b+)*c", "aabbbbc", 0, 7);
    // test case 263
    x2("a|(?i)c", "C", 0, 1);
    // test case 264
    x2("(?i)c|a", "C", 0, 1);
    // test case 265
    x2("(?i)c|a", "A", 0, 1);
    // test case 266
    x2("a(?i)b|c", "aB", 0, 2);
    // test case 267
    x2("a(?i)b|c", "aC", 0, 2);
    // test case 268
    n("a(?i)b|c", "AC");
    // test case 269
    n("a(?:(?i)b)|c", "aC");
    // test case 270
    x2("(?i:c)|a", "C", 0, 1);
    // test case 271
    n("(?i:c)|a", "A");
    // test case 272
    x2("[abc]?", "abc", 0, 1);
    // test case 273
    x2("[abc]*", "abc", 0, 3);
    // test case 274
    x2("[^abc]*", "abc", 0, 0);
    // test case 275
    n("[^abc]+", "abc");
    // test case 276
    x2("a?\?", "aaa", 0, 0);
    // test case 277
    x2("ba?\?b", "bab", 0, 3);
    // test case 278
    x2("a*?", "aaa", 0, 0);
    // test case 279
    x2("ba*?", "baa", 0, 1);
    // test case 280
    x2("ba*?b", "baab", 0, 4);
    // test case 281
    x2("a+?", "aaa", 0, 1);
    // test case 282
    x2("ba+?", "baa", 0, 2);
    // test case 283
    x2("ba+?b", "baab", 0, 4);
    // test case 284
    x2("(?:a?)?\?", "a", 0, 0);
    // test case 285
    x2("(?:a?\?)?", "a", 0, 0);
    // test case 286
    x2("(?:a?)+?", "aaa", 0, 1);
    // test case 287
    x2("(?:a+)?\?", "aaa", 0, 0);
    // test case 288
    x2("(?:a+)?\?b", "aaab", 0, 4);
    // test case 289
    x2("(?:ab)?{2}", "", 0, 0);
    // test case 290
    x2("(?:ab)?{2}", "ababa", 0, 4);
    // test case 291
    x2("(?:ab)*{0}", "ababa", 0, 0);
    // test case 292
    x2("(?:ab){3,}", "abababab", 0, 8);
    // test case 293
    n("(?:ab){3,}", "abab");
    // test case 294
    x2("(?:ab){2,4}", "ababab", 0, 6);
    // test case 295
    x2("(?:ab){2,4}", "ababababab", 0, 8);
    // test case 296
    x2("(?:ab){2,4}?", "ababababab", 0, 4);
    // test case 297
    x2("(?:ab){,}", "ab{,}", 0, 5);
    // test case 298
    x2("(?:abc)+?{2}", "abcabcabc", 0, 6);
    // test case 299
    x2("(?:X*)(?i:xa)", "XXXa", 0, 4);
    // test case 300
    x2("(d+)([^abc]z)", "dddz", 0, 4);
    // test case 301
    x2("([^abc]*)([^abc]z)", "dddz", 0, 4);
    // test case 302
    x2("(\\w+)(\\wz)", "dddz", 0, 4);
    // test case 303
    x3("(a)", "a", 0, 1, 1);
    // test case 304
    x3("(ab)", "ab", 0, 2, 1);
    // test case 305
    x2("((ab))", "ab", 0, 2);
    // test case 306
    x3("((ab))", "ab", 0, 2, 1);
    // test case 307
    x3("((ab))", "ab", 0, 2, 2);
    // test case 308
    x3("((((((((((((((((((((ab))))))))))))))))))))", "ab", 0, 2, 20);
    // test case 309
    x3("(ab)(cd)", "abcd", 0, 2, 1);
    // test case 310
    x3("(ab)(cd)", "abcd", 2, 4, 2);
    // test case 311
    x3("()(a)bc(def)ghijk", "abcdefghijk", 3, 6, 3);
    // test case 312
    x3("(()(a)bc(def)ghijk)", "abcdefghijk", 3, 6, 4);
    // test case 313
    x2("(^a)", "a", 0, 1);
    // test case 314
    x3("(a)|(a)", "ba", 1, 2, 1);
    // test case 315
    x3("(^a)|(a)", "ba", 1, 2, 2);
    // test case 316
    x3("(a?)", "aaa", 0, 1, 1);
    // test case 317
    x3("(a*)", "aaa", 0, 3, 1);
    // test case 318
    x3("(a*)", "", 0, 0, 1);
    // test case 319
    x3("(a+)", "aaaaaaa", 0, 7, 1);
    // test case 320
    x3("(a+|b*)", "bbbaa", 0, 3, 1);
    // test case 321
    x3("(a+|b?)", "bbbaa", 0, 1, 1);
    // test case 322
    x3("(abc)?", "abc", 0, 3, 1);
    // test case 323
    x3("(abc)*", "abc", 0, 3, 1);
    // test case 324
    x3("(abc)+", "abc", 0, 3, 1);
    // test case 325
    x3("(xyz|abc)+", "abc", 0, 3, 1);
    // test case 326
    x3("([xyz][abc]|abc)+", "abc", 0, 3, 1);
    // test case 327
    x3("((?i:abc))", "AbC", 0, 3, 1);
    // test case 328
    x2("(abc)(?i:\\1)", "abcABC", 0, 6);
    // test case 329
    x3("((?m:a.c))", "a\nc", 0, 3, 1);
    // test case 330
    x3("((?=az)a)", "azb", 0, 1, 1);
    // test case 331
    x3("abc|(.abd)", "zabd", 0, 4, 1);
    // test case 332
    x2("(?:abc)|(ABC)", "abc", 0, 3);
    // test case 333
    x3("(?i:(abc))|(zzz)", "ABC", 0, 3, 1);
    // test case 334
    x3("a*(.)", "aaaaz", 4, 5, 1);
    // test case 335
    x3("a*?(.)", "aaaaz", 0, 1, 1);
    // test case 336
    x3("a*?(c)", "aaaac", 4, 5, 1);
    // test case 337
    x3("[bcd]a*(.)", "caaaaz", 5, 6, 1);
    // test case 338
    x3("(\\Abb)cc", "bbcc", 0, 2, 1);
    // test case 339
    n("(\\Abb)cc", "zbbcc");
    // test case 340
    x3("(^bb)cc", "bbcc", 0, 2, 1);
    // test case 341
    n("(^bb)cc", "zbbcc");
    // test case 342
    x3("cc(bb$)", "ccbb", 2, 4, 1);
    // test case 343
    n("cc(bb$)", "ccbbb");
    // test case 344
    n("(\\1)", "");
    // test case 345
    n("\\1(a)", "aa");
    // test case 346
    n("(a(b)\\1)\\2+", "ababb");
    // test case 347
    n("(?:(?:\\1|z)(a))+$", "zaa");
    // test case 348
    x2("(?:(?:\\1|z)(a))+$", "zaaa", 0, 4);
    // test case 349
    x2("(a)(?=\\1)", "aa", 0, 1);
    // test case 350
    n("(a)$|\\1", "az");
    // test case 351
    x2("(a)\\1", "aa", 0, 2);
    // test case 352
    n("(a)\\1", "ab");
    // test case 353
    x2("(a?)\\1", "aa", 0, 2);
    // test case 354
    x2("(a?\?)\\1", "aa", 0, 0);
    // test case 355
    x2("(a*)\\1", "aaaaa", 0, 4);
    // test case 356
    x3("(a*)\\1", "aaaaa", 0, 2, 1);
    // test case 357
    x2("a(b*)\\1", "abbbb", 0, 5);
    // test case 358
    x2("a(b*)\\1", "ab", 0, 1);
    // test case 359
    x2("(a*)(b*)\\1\\2", "aaabbaaabb", 0, 10);
    // test case 360
    x2("(a*)(b*)\\2", "aaabbbb", 0, 7);
    // test case 361
    x2("(((((((a*)b))))))c\\7", "aaabcaaa", 0, 8);
    // test case 362
    x3("(((((((a*)b))))))c\\7", "aaabcaaa", 0, 3, 7);
    // test case 363
    x2("(a)(b)(c)\\2\\1\\3", "abcbac", 0, 6);
    // test case 364
    x2("([a-d])\\1", "cc", 0, 2);
    // test case 365
    x2("(\\w\\d\\s)\\1", "f5 f5 ", 0, 6);
    // test case 366
    n("(\\w\\d\\s)\\1", "f5 f5");
    // test case 367
    x2("(who|[a-c]{3})\\1", "whowho", 0, 6);
    // test case 368
    x2("...(who|[a-c]{3})\\1", "abcwhowho", 0, 9);
    // test case 369
    x2("(who|[a-c]{3})\\1", "cbccbc", 0, 6);
    // test case 370
    x2("(^a)\\1", "aa", 0, 2);
    // test case 371
    n("(^a)\\1", "baa");
    // test case 372
    n("(a$)\\1", "aa");
    // test case 373
    n("(ab\\Z)\\1", "ab");
    // test case 374
    x2("(a*\\Z)\\1", "a", 1, 1);
    // test case 375
    x2(".(a*\\Z)\\1", "ba", 1, 2);
    // test case 376
    x3("(.(abc)\\2)", "zabcabc", 0, 7, 1);
    // test case 377
    x3("(.(..\\d.)\\2)", "z12341234", 0, 9, 1);
    // test case 378
    x2("((?i:az))\\1", "AzAz", 0, 4);
    // test case 379
    n("((?i:az))\\1", "Azaz");
    // test case 380
    x2("(?<=a)b", "ab", 1, 2);
    // test case 381
    n("(?<=a)b", "bb");
    // test case 382
    x2("(?<=a|b)b", "bb", 1, 2);
    // test case 383
    x2("(?<=a|bc)b", "bcb", 2, 3);
    // test case 384
    x2("(?<=a|bc)b", "ab", 1, 2);
    // test case 385
    x2("(?<=a|bc||defghij|klmnopq|r)z", "rz", 1, 2);
    // test case 386
    x3("(?<=(abc))d", "abcd", 0, 3, 1);
    // test case 387
    x2("(?<=(?i:abc))d", "ABCd", 3, 4);
    // test case 388
    x2("(?<=^|b)c", " cbc", 3, 4);
    // test case 389
    x2("(?<=a|^|b)c", " cbc", 3, 4);
    // test case 390
    x2("(?<=a|(^)|b)c", " cbc", 3, 4);
    // test case 391
    x2("(?<=a|(^)|b)c", "cbc", 0, 1);
    // test case 392
    n("(Q)|(?<=a|(?(1))|b)c", "czc");
    // test case 393
    x2("(Q)(?<=a|(?(1))|b)c", "cQc", 1, 3);
    // test case 394
    x2("(?<=a|(?~END)|b)c", "ENDc", 3, 4);
    // test case 395
    n("(?<!^|b)c", "cbc");
    // test case 396
    n("(?<!a|^|b)c", "cbc");
    // test case 397
    n("(?<!a|(?:^)|b)c", "cbc");
    // test case 398
    x2("(?<!a|(?:^)|b)c", " cbc", 1, 2);
    // test case 399
    x2("(a)\\g<1>", "aa", 0, 2);
    // test case 400
    x2("(?<!a)b", "cb", 1, 2);
    // test case 401
    n("(?<!a)b", "ab");
    // test case 402
    x2("(?<!a|bc)b", "bbb", 0, 1);
    // test case 403
    n("(?<!a|bc)z", "bcz");
    // test case 404
    x2("(?<name1>a)", "a", 0, 1);
    // test case 405
    x2("(?<name_2>ab)\\g<name_2>", "abab", 0, 4);
    // test case 406
    x2("(?<name_3>.zv.)\\k<name_3>", "azvbazvb", 0, 8);
    // test case 407
    x2("(?<=\\g<ab>)|-\\zEND (?<ab>XyZ)", "XyZ", 3, 3);
    // test case 408
    x2("(?<n>|a\\g<n>)+", "", 0, 0);
    // test case 409
    x2("(?<n>|\\(\\g<n>\\))+$", "()(())", 0, 6);
    // test case 410
    x3("\\g<n>(?<n>.){0}", "X", 0, 1, 1);
    // test case 411
    x2("\\g<n>(abc|df(?<n>.YZ){2,8}){0}", "XYZ", 0, 3);
    // test case 412
    x2("\\A(?<n>(a\\g<n>)|)\\z", "aaaa", 0, 4);
    // test case 413
    x2("(?<n>|\\g<m>\\g<n>)\\z|\\zEND (?<m>a|(b)\\g<m>)", "bbbbabba", 0, 8);
    // test case 414
    x2("(?<name1240>\\w+\\sx)a+\\k<name1240>", "  fg xaaaaaaaafg x", 2, 18);
    // test case 415
    x3("(z)()()(?<_9>a)\\g<_9>", "zaa", 2, 3, 1);
    // test case 416
    x2("(.)(((?<_>a)))\\k<_>", "zaa", 0, 3);
    // test case 417
    x2("((?<name1>\\d)|(?<name2>\\w))(\\k<name1>|\\k<name2>)", "ff", 0, 2);
    // test case 418
    x2("(?:(?<x>)|(?<x>efg))\\k<x>", "", 0, 0);
    // test case 419
    x2("(?:(?<x>abc)|(?<x>efg))\\k<x>", "abcefgefg", 3, 9);
    // test case 420
    n("(?:(?<x>abc)|(?<x>efg))\\k<x>", "abcefg");
    // test case 421
    x2("(?:(?<n1>.)|(?<n1>..)|(?<n1>...)|(?<n1>....)|(?<n1>.....)|(?<n1>......)|(?<n1>.......)|(?<n1>........)|(?<n1>.."
       ".......)|(?<n1>..........)|(?<n1>...........)|(?<n1>............)|(?<n1>.............)|(?<n1>..............))"
       "\\k<n1>$",
       "a-pyumpyum", 2, 10);
    // test case 422
    x3("(?:(?<n1>.)|(?<n1>..)|(?<n1>...)|(?<n1>....)|(?<n1>.....)|(?<n1>......)|(?<n1>.......)|(?<n1>........)|(?<n1>.."
       ".......)|(?<n1>..........)|(?<n1>...........)|(?<n1>............)|(?<n1>.............)|(?<n1>..............))"
       "\\k<n1>$",
       "xxxxabcdefghijklmnabcdefghijklmn", 4, 18, 14);
    // test case 423
    x3("(?<name1>)(?<name2>)(?<name3>)(?<name4>)(?<name5>)(?<name6>)(?<name7>)(?<name8>)(?<name9>)(?<name10>)(?<name11>"
       ")(?<name12>)(?<name13>)(?<name14>)(?<name15>)(?<name16>aaa)(?<name17>)$",
       "aaa", 0, 3, 16);
    // test case 424
    x2("(?<foo>a|\\(\\g<foo>\\))", "a", 0, 1);
    // test case 425
    x2("(?<foo>a|\\(\\g<foo>\\))", "((((((a))))))", 0, 13);
    // test case 426
    x3("(?<foo>a|\\(\\g<foo>\\))", "((((((((a))))))))", 0, 17, 1);
    // test case 427
    x2("\\g<bar>|\\zEND(?<bar>.*abc$)", "abcxxxabc", 0, 9);
    // test case 428
    x2("\\g<1>|\\zEND(.a.)", "bac", 0, 3);
    // test case 429
    x3("\\g<_A>\\g<_A>|\\zEND(.a.)(?<_A>.b.)", "xbxyby", 3, 6, 1);
    // test case 430
    x2("\\A(?:\\g<pon>|\\g<pan>|\\zEND  (?<pan>a|c\\g<pon>c)(?<pon>b|d\\g<pan>d))$", "cdcbcdc", 0, 7);
    // test case 431
    x2("\\A(?<n>|a\\g<m>)\\z|\\zEND (?<m>\\g<n>)", "aaaa", 0, 4);
    // test case 432
    x2("(?<n>(a|b\\g<n>c){3,5})", "baaaaca", 1, 5);
    // test case 433
    x2("(?<n>(a|b\\g<n>c){3,5})", "baaaacaaaaa", 0, 10);
    // test case 434
    x2("(?<pare>\\(([^\\(\\)]++|\\g<pare>)*+\\))", "((a))", 0, 5);
    // test case 435
    x2("()*\\1", "", 0, 0);
    // test case 436
    x2("(?:()|())*\\1\\2", "", 0, 0);
    // test case 437
    x2("(?:a*|b*)*c", "abadc", 4, 5);
    // test case 438
    x3("(?:\\1a|())*", "a", 0, 0, 1);
    // test case 439
    x2("x((.)*)*x", "0x1x2x3", 1, 6);
    // test case 440
    x2("x((.)*)*x(?i:\\1)\\Z", "0x1x2x1X2", 1, 9);
    // test case 441
    x2("(?:()|()|()|()|()|())*\\2\\5", "", 0, 0);
    // test case 442
    x2("(?:()|()|()|(x)|()|())*\\2b\\5", "b", 0, 1);
    // test case 443
    x2("[0-9-a]", "-", 0, 1); // PR#44
    // test case 444
    n("[0-9-a]", ":"); // PR#44
    // test case 445
    x3("(\\(((?:[^(]|\\g<1>)*)\\))", "(abc)(abc)", 1, 4, 2); // PR#43
    // test case 446
    x2("\\o{101}", "A", 0, 1);
    // test case 447
    x2("\\A(a|b\\g<1>c)\\k<1+3>\\z", "bbacca", 0, 6);
    // test case 448
    n("\\A(a|b\\g<1>c)\\k<1+3>\\z", "bbaccb");
    // test case 449
    x2("(?i)\\A(a|b\\g<1>c)\\k<1+2>\\z", "bBACcbac", 0, 8);
    // test case 450
    x2("(?i)(?<X>aa)|(?<X>bb)\\k<X>", "BBbb", 0, 4);
    // test case 451
    x2("(?:\\k'+1'B|(A)C)*", "ACAB", 0, 4); // relative backref by postitive number
    // test case 452
    x2("\\g<+2>(abc)(ABC){0}", "ABCabc", 0, 6); // relative call by positive number
    // test case 453
    x2("A\\g'0'|B()", "AAAAB", 0, 5);
    // test case 454
    x3("(A\\g'0')|B", "AAAAB", 0, 5, 1);
    // test case 455
    x2("(a*)(?(1))aa", "aaaaa", 0, 5);
    // test case 456
    x2("(a*)(?(-1))aa", "aaaaa", 0, 5);
    // test case 457
    x2("(?<name>aaa)(?('name'))aa", "aaaaa", 0, 5);
    // test case 458
    x2("(a)(?(1)aa|bb)a", "aaaaa", 0, 4);
    // test case 459
    x2("(?:aa|())(?(<1>)aa|bb)a", "aabba", 0, 5);
    // test case 460
    x2("(?:aa|())(?('1')aa|bb|cc)a", "aacca", 0, 5);
    // test case 461
    x3("(a*)(?(1)aa|a)b", "aaab", 0, 1, 1);
    // test case 462
    n("(a)(?(1)a|b)c", "abc");
    // test case 463
    x2("(a)(?(1)|)c", "ac", 0, 2);
    // test case 464
    n("(?()aaa|bbb)", "bbb");
    // test case 465
    x2("(a)(?(1+0)b|c)d", "abd", 0, 3);
    // test case 466
    x2("(?:(?'name'a)|(?'name'b))(?('name')c|d)e", "ace", 0, 3);
    // test case 467
    x2("(?:(?'name'a)|(?'name'b))(?('name')c|d)e", "bce", 0, 3);
    // test case 468
    x2("\\R", "\r\n", 0, 2);
    // test case 469
    x2("\\R", "\r", 0, 1);
    // test case 470
    x2("\\R", "\n", 0, 1);
    // test case 471
    x2("\\R", "\x0b", 0, 1);
    // test case 472
    n("\\R\\n", "\r\n");
    // x2("\\R", "\xc2\x85", 0, 2);
    // test case 473
    x2("\\N", "a", 0, 1);
    // test case 474
    n("\\N", "\n");
    // test case 475
    n("(?m:\\N)", "\n");
    // test case 476
    n("(?-m:\\N)", "\n");
    // test case 477
    x2("\\O", "a", 0, 1);
    // test case 478
    x2("\\O", "\n", 0, 1);
    // test case 479
    x2("(?m:\\O)", "\n", 0, 1);
    // test case 480
    x2("(?-m:\\O)", "\n", 0, 1);
    // test case 481
    x2("\\K", "a", 0, 0);
    // test case 482
    x2("a\\K", "a", 1, 1);
    // test case 483
    x2("a\\Kb", "ab", 1, 2);
    // test case 484
    x2("(a\\Kb|ac\\Kd)", "acd", 2, 3);
    // test case 485
    x2("(a\\Kb|\\Kac\\K)*", "acababacab", 9, 10);
    // test case 486
    x2("(?:()|())*\\1", "abc", 0, 0);
    // test case 487
    x2("(?:()|())*\\2", "abc", 0, 0);
    // test case 488
    x2("(?:()|()|())*\\3\\1", "abc", 0, 0);
    // test case 489
    x2("(|(?:a(?:\\g'1')*))b|", "abc", 0, 2);
    // test case 490
    x2("^(\"|)(.*)\\1$", "XX", 0, 2);
    // test case 491
    x2("(abc|def|ghi|jkl|mno|pqr|stu){0,10}?\\z", "admno", 2, 5);
    // test case 492
    x2("(abc|(def|ghi|jkl|mno|pqr){0,7}?){5}\\z", "adpqrpqrpqr", 2, 11); // cover OP_REPEAT_INC_NG_SG
    // test case 493
    x2("(?!abc).*\\z", "abcde", 1, 5); // cover OP_PREC_READ_NOT_END
    // test case 494
    x2("(.{2,})?", "abcde", 0, 5); // up coverage
    // test case 495
    x2("((a|b|c|d|e|f|g|h|i|j|k|l|m|n)+)?", "abcde", 0, 5); // up coverage
    // test case 496
    x2("((a|b|c|d|e|f|g|h|i|j|k|l|m|n){3,})?", "abcde", 0, 5); // up coverage
    // test case 497
    x2("((?:a(?:b|c|d|e|f|g|h|i|j|k|l|m|n))+)?", "abacadae", 0, 8); // up coverage
    // test case 498
    x2("((?:a(?:b|c|d|e|f|g|h|i|j|k|l|m|n))+?)?z", "abacadaez", 0, 9); // up coverage
    // test case 499
    x2("\\A((a|b)\?\?)?z", "bz", 0, 2); // up coverage
    // test case 500
    x2("((?<x>abc){0}a\\g<x>d)+", "aabcd", 0, 5); // up coverage
    // test case 501
    x2("((?(abc)true|false))+", "false", 0, 5); // up coverage
    // test case 502
    x2("((?i:abc)d)+", "abcdABCd", 0, 8); // up coverage
    // test case 503
    x2("((?<!abc)def)+", "bcdef", 2, 5); // up coverage
    // test case 504
    x2("(\\ba)+", "aaa", 0, 1); // up coverage
    // test case 505
    x2("()(?<x>ab)(?(<x>)a|b)", "aba", 0, 3); // up coverage
    // test case 506
    x2("(?<=a.b)c", "azbc", 3, 4); // up coverage
    // test case 507
    n("(?<=(?:abcde){30})z", "abc"); // up coverage
    // test case 508
    x2("(?<=(?(a)a|bb))z", "aaz", 2, 3); // up coverage
    // test case 509
    x2("[a]*\\W", "aa@", 0, 3); // up coverage
    // test case 510
    x2("[a]*[b]", "aab", 0, 3); // up coverage
    // test case 511
    n("a*\\W", "aaa"); // up coverage
    // test case 512
    n("(?W)a*\\W", "aaa"); // up coverage
    // test case 513
    x2("(?<=ab(?<=ab))", "ab", 2, 2); // up coverage
    // test case 514
    x2("(?<x>a)(?<x>b)(\\k<x>)+", "abbaab", 0, 6); // up coverage
    // test case 515
    x2("()(\\1)(\\2)", "abc", 0, 0); // up coverage
    // test case 516
    x2("((?(a)b|c))(\\1)", "abab", 0, 4); // up coverage
    // test case 517
    x2("(?<x>$|b\\g<x>)", "bbb", 0, 3); // up coverage
    // test case 518
    x2("(?<x>(?(a)a|b)|c\\g<x>)", "cccb", 0, 4); // up coverage
    // test case 519
    x2("(a)(?(1)a*|b*)+", "aaaa", 0, 4); // up coverage
    // test case 520
    x2("[[^abc]&&cde]*", "de", 0, 2); // up coverage
    // test case 521
    n("(a){10}{10}", "aa"); // up coverage
    // test case 522
    x2("(?:a?)+", "aa", 0, 2); // up coverage
    // test case 523
    x2("(?:a?)*?", "a", 0, 0); // up coverage
    // test case 524
    x2("(?:a*)*?", "a", 0, 0); // up coverage
    // test case 525
    x2("(?:a+?)*", "a", 0, 1); // up coverage
    // test case 526
    x2("\\h", "5", 0, 1); // up coverage
    // test case 527
    x2("\\H", "z", 0, 1); // up coverage
    // test case 528
    x2("[\\h]", "5", 0, 1); // up coverage
    // test case 529
    x2("[\\H]", "z", 0, 1); // up coverage
    // test case 530
    x2("[\\o{101}]", "A", 0, 1); // up coverage
    // test case 531
    x2("[\\u0041]", "A", 0, 1); // up coverage
    // test case 532
    x2("(?~)", "", 0, 0);
    // test case 533
    x2("(?~)", "A", 0, 0);
    // test case 534
    x2("(?~ab)", "abc", 0, 0);
    // test case 535
    x2("(?~abc)", "abc", 0, 0);
    // test case 536
    x2("(?~abc|ab)", "abc", 0, 0);
    // test case 537
    x2("(?~ab|abc)", "abc", 0, 0);
    // test case 538
    x2("(?~a.c)", "abc", 0, 0);
    // test case 539
    x2("(?~a.c|ab)", "abc", 0, 0);
    // test case 540
    x2("(?~ab|a.c)", "abc", 0, 0);
    // test case 541
    x2("aaaaa(?~)", "aaaaaaaaaa", 0, 5);
    // test case 542
    x2("(?~(?:|aaa))", "aaa", 0, 0);
    // test case 543
    x2("(?~aaa|)", "aaa", 0, 0);
    // test case 544
    x2("a(?~(?~)).", "abcdefghijklmnopqrstuvwxyz", 0, 26); // nested absent functions cause strange result
    // test case 545
    x2("/\\*(?~\\*/)\\*/", "/* */ */", 0, 5);
    // test case 546
    x2("(?~\\w+)zzzzz", "zzzzz", 0, 5);
    // test case 547
    x2("(?~\\w*)zzzzz", "zzzzz", 0, 5);
    // test case 548
    x2("(?~A.C|B)", "ABC", 0, 0);
    // test case 549
    x2("(?~XYZ|ABC)a", "ABCa", 1, 4);
    // test case 550
    x2("(?~XYZ|ABC)a", "aABCa", 0, 1);
    // test case 551
    x2("<[^>]*>(?~[<>])</[^>]*>", "<a>vvv</a>   <b>  </b>", 0, 10);
    // test case 552
    x2("(?~ab)", "ccc\ndab", 0, 5);
    // test case 553
    x2("(?m:(?~ab))", "ccc\ndab", 0, 5);
    // test case 554
    x2("(?-m:(?~ab))", "ccc\ndab", 0, 5);
    // test case 555
    x2("(?~abc)xyz", "xyz012345678901234567890123456789abc", 0, 3);

    // absent with expr
    // test case 556
    x2("(?~|78|\\d*)", "123456789", 0, 6);
    // test case 557
    x2("(?~|def|(?:abc|de|f){0,100})", "abcdedeabcfdefabc", 0, 11);
    // test case 558
    x2("(?~|ab|.*)", "ccc\nddd", 0, 3);
    // test case 559
    x2("(?~|ab|\\O*)", "ccc\ndab", 0, 5);
    // test case 560
    x2("(?~|ab|\\O{2,10})", "ccc\ndab", 0, 5);
    // test case 561
    x2("(?~|ab|\\O{1,10})", "ab", 1, 2);
    // test case 562
    n("(?~|ab|\\O{2,10})", "ab");
    // test case 563
    x2("(?~|abc|\\O{1,10})", "abc", 1, 3);
    // test case 564
    x2("(?~|ab|\\O{5,10})|abc", "abc", 0, 3);
    // test case 565
    x2("(?~|ab|\\O{1,10})", "cccccccccccab", 0, 10);
    // test case 566
    x2("(?~|aaa|)", "aaa", 0, 0);
    // test case 567
    x2("(?~||a*)", "aaaaaa", 0, 0);
    // test case 568
    x2("(?~||a*?)", "aaaaaa", 0, 0);
    // test case 569
    x2("(a)(?~|b|\\1)", "aaaaaa", 0, 2);
    // test case 570
    x2("(a)(?~|bb|(?:a\\1)*)", "aaaaaa", 0, 5);
    // test case 571
    x2("(b|c)(?~|abac|(?:a\\1)*)", "abababacabab", 1, 4);
    // test case 572
    n("(?~|c|a*+)a", "aaaaa");
    // test case 573
    x2("(?~|aaaaa|a*+)", "aaaaa", 0, 0);
    // test case 574
    x2("(?~|aaaaaa|a*+)b", "aaaaaab", 1, 7);
    // test case 575
    x2("(?~|abcd|(?>))", "zzzabcd", 0, 0);
    // test case 576
    x2("(?~|abc|a*?)", "aaaabc", 0, 0);

    // absent range cutter
    // test case 577
    x2("(?~|abc)a*", "aaaaaabc", 0, 5);
    // test case 578
    x2("(?~|abc)a*z|aaaaaabc", "aaaaaabc", 0, 8);
    // test case 579
    x2("(?~|aaaaaa)a*", "aaaaaa", 0, 0);
    // test case 580
    x2("(?~|abc)aaaa|aaaabc", "aaaabc", 0, 6);
    // test case 581
    x2("(?>(?~|abc))aaaa|aaaabc", "aaaabc", 0, 6);
    // test case 582
    x2("(?~|)a", "a", 0, 1);
    // test case 583
    n("(?~|a)a", "a");
    // test case 584
    x2("(?~|a)(?~|)a", "a", 0, 1);
    // test case 585
    x2("(?~|a).*(?~|)a", "bbbbbbbbbbbbbbbbbbbba", 0, 21);
    // test case 586
    x2("(?~|abc).*(xyz|pqr)(?~|)abc", "aaaaxyzaaapqrabc", 0, 16);
    // test case 587
    x2("(?~|abc).*(xyz|pqr)(?~|)abc", "aaaaxyzaaaabcpqrabc", 11, 19);
    // test case 588
    n("\\A(?~|abc).*(xyz|pqrabc)(?~|)abc", "aaaaxyzaaaabcpqrabcabc");
    // test case 589
    x2("", "あ", 0, 0);
    // test case 590
    x2("あ", "あ", 0, 3);
    // test case 591
    n("い", "あ");
    // test case 592
    x2("うう", "うう", 0, 6);
    // test case 593
    x2("あいう", "あいう", 0, 9);
    // test case 594
    x2("こここここここここここここここここここここここここここここここここここ",
       "こここここここここここここここここここここここここここここここここここ", 0, 105);
    // test case 595
    x2("あ", "いあ", 3, 6);
    // test case 596
    x2("いう", "あいう", 3, 9);
    // test case 597
    x2("\\xca\\xb8", "\xca\xb8", 0, 2);
    // x2(".", "あ", 0, 3);
    // x2("..", "かき", 0, 6);
    // x2("\\w", "お", 0, 3);
    // n("\\W", "あ");
    // x2("[\\W]", "う$", 3, 4);
    // x2("\\S", "そ", 0, 3);
    // x2("\\S", "漢", 0, 3);
    // x2("\\b", "気 ", 0, 0);
    // x2("\\b", " ほ", 1, 1);
    // x2("\\B", "せそ ", 3, 3);
    // x2("\\B", "う ", 4, 4);
    // test case 598
    x2("\\B", " い", 0, 0);
    // x2("[たち]", "ち", 0, 3);
    // n("[なに]", "ぬ");
    // x2("[う-お]", "え", 0, 3);
    // test case 599
    n("[^け]", "け");
    // x2("[\\w]", "ね", 0, 3);
    // test case 600
    n("[\\d]", "ふ");
    // x2("[\\D]", "は", 0, 3);
    // test case 601
    n("[\\s]", "く");
    // x2("[\\S]", "へ", 0, 3);
    // x2("[\\w\\d]", "よ", 0, 3);
    // x2("[\\w\\d]", "   よ", 3, 6);
    // test case 602
    n("\\w鬼車", " 鬼車");
    // test case 603
    x2("鬼\\W車", "鬼 車", 0, 7);
    // x2("あ.い.う", "ああいいう", 0, 15);
    // x2(".\\wう\\W..ぞ", "えうう うぞぞ", 0, 19);
    // x2("\\s\\wこここ", " ここここ", 0, 13);
    // x2("ああ.け", "ああけけ", 0, 12);
    // test case 604
    n(".い", "いえ");
    // x2(".お", "おお", 0, 6);
    // test case 605
    x2("^あ", "あ", 0, 3);
    // test case 606
    x2("^む$", "む", 0, 3);
    // x2("^\\w$", "に", 0, 3);
    // test case 607
    x2("^\\wかきくけこ$", "zかきくけこ", 0, 16);
    // x2("^\\w...うえお$", "zあいううえお", 0, 19);
    // x2("\\w\\w\\s\\Wおおお\\d", "aお  おおお4", 0, 16);
    // test case 608
    x2("\\Aたちつ", "たちつ", 0, 9);
    // test case 609
    x2("むめも\\Z", "むめも", 0, 9);
    // test case 610
    x2("かきく\\z", "かきく", 0, 9);
    // test case 611
    x2("かきく\\Z", "かきく\n", 0, 9);
    // test case 612
    x2("\\Gぽぴ", "ぽぴ", 0, 6);
    // test case 613
    n("\\Gえ", "うえお");
    // test case 614
    n("とて\\G", "とて");
    // test case 615
    n("まみ\\A", "まみ");
    // test case 616
    n("ま\\Aみ", "まみ");
    // test case 617
    x2("(?=せ)せ", "せ", 0, 3);
    // test case 618
    n("(?=う).", "い");
    // test case 619
    x2("(?!う)か", "か", 0, 3);
    // test case 620
    n("(?!と)あ", "と");
    // test case 621
    x2("(?i:あ)", "あ", 0, 3);
    // test case 622
    x2("(?i:ぶべ)", "ぶべ", 0, 6);
    // test case 623
    n("(?i:い)", "う");
    // test case 624
    x2("(?m:よ.)", "よ\n", 0, 4);
    // test case 625
    x2("(?m:.め)", "ま\nめ", 3, 7);
    // x2("あ?", "", 0, 0);
    // x2("変?", "化", 0, 0);
    // test case 626
    x2("変?", "変", 0, 3);
    // x2("量*", "", 0, 0);
    // test case 627
    x2("量*", "量", 0, 3);
    // x2("子*", "子子子", 0, 9);
    // x2("馬*", "鹿馬馬馬馬", 0, 0);
    // test case 628
    n("山+", "");
    // test case 629
    x2("河+", "河", 0, 3);
    // x2("時+", "時時時時", 0, 12);
    // x2("え+", "ええううう", 0, 6);
    // x2("う+", "おうううう", 3, 15);
    // x2(".?", "た", 0, 3);
    // test case 630
    x2(".*", "ぱぴぷぺ", 0, 12);
    // test case 631
    x2(".+", "ろ", 0, 3);
    // test case 632
    x2(".+", "いうえか\n", 0, 12);
    // test case 633
    x2("あ|い", "あ", 0, 3);
    // test case 634
    x2("あ|い", "い", 0, 3);
    // test case 635
    x2("あい|いう", "あい", 0, 6);
    // test case 636
    x2("あい|いう", "いう", 0, 6);
    // test case 637
    x2("を(?:かき|きく)", "をかき", 0, 9);
    // test case 638
    x2("を(?:かき|きく)け", "をきくけ", 0, 12);
    // test case 639
    x2("あい|(?:あう|あを)", "あを", 0, 6);
    // test case 640
    x2("あ|い|う", "えう", 3, 6);
    // test case 641
    x2("あ|い|うえ|おかき|く|けこさ|しすせ|そ|たち|つてとなに|ぬね", "しすせ", 0, 9);
    // test case 642
    n("あ|い|うえ|おかき|く|けこさ|しすせ|そ|たち|つてとなに|ぬね", "すせ");
    // test case 643
    x2("あ|^わ", "ぶあ", 3, 6);
    // test case 644
    x2("あ|^を", "をあ", 0, 3);
    // test case 645
    x2("鬼|\\G車", "け車鬼", 6, 9);
    // test case 646
    x2("鬼|\\G車", "車鬼", 0, 3);
    // test case 647
    x2("鬼|\\A車", "b車鬼", 4, 7);
    // test case 648
    x2("鬼|\\A車", "車", 0, 3);
    // test case 649
    x2("鬼|車\\Z", "車鬼", 3, 6);
    // test case 650
    x2("鬼|車\\Z", "車", 0, 3);
    // test case 651
    x2("鬼|車\\Z", "車\n", 0, 3);
    // test case 652
    x2("鬼|車\\z", "車鬼", 3, 6);
    // test case 653
    x2("鬼|車\\z", "車", 0, 3);
    // x2("\\w|\\s", "お", 0, 3);
    // test case 654
    x2("\\w|%", "%お", 0, 1);
    // x2("\\w|[&$]", "う&", 0, 3);
    // x2("[い-け]", "う", 0, 3);
    // x2("[い-け]|[^か-こ]", "あ", 0, 3);
    // x2("[い-け]|[^か-こ]", "か", 0, 3);
    // test case 655
    x2("[^あ]", "\n", 0, 1);
    // x2("(?:あ|[う-き])|いを", "うを", 0, 3);
    // x2("(?:あ|[う-き])|いを", "いを", 0, 6);
    // x2("あいう|(?=けけ)..ほ", "けけほ", 0, 9);
    // x2("あいう|(?!けけ)..ほ", "あいほ", 0, 9);
    // x2("(?=をあ)..あ|(?=をを)..あ", "ををあ", 0, 9);
    // test case 656
    x2("(?<=あ|いう)い", "いうい", 6, 9);
    // test case 657
    n("(?>あ|あいえ)う", "あいえう");
    // test case 658
    x2("(?>あいえ|あ)う", "あいえう", 0, 12);
    // test case 659
    x2("あ?|い", "あ", 0, 3);
    // x2("あ?|い", "い", 0, 0);
    // x2("あ?|い", "", 0, 0);
    // x2("あ*|い", "ああ", 0, 6);
    // x2("あ*|い*", "いあ", 0, 0);
    // test case 660
    x2("あ*|い*", "あい", 0, 3);
    // x2("[aあ]*|い*", "aあいいい", 0, 4);
    // x2("あ+|い*", "", 0, 0);
    // x2("あ+|い*", "いいい", 0, 9);
    // test case 661
    x2("あ+|い*", "あいいい", 0, 3);
    // x2("あ+|い*", "aあいいい", 0, 0);
    // test case 662
    n("あ+|い+", "");
    // test case 663
    x2("(あ|い)?", "い", 0, 3);
    // test case 664
    x2("(あ|い)*", "いあ", 0, 6);
    // test case 665
    x2("(あ|い)+", "いあい", 0, 9);
    // test case 666
    x2("(あい|うあ)+", "うああいうえ", 0, 12);
    // test case 667
    x2("(あい|うえ)+", "うああいうえ", 6, 18);
    // test case 668
    x2("(あい|うあ)+", "ああいうあ", 3, 15);
    // test case 669
    x2("(あい|うあ)+", "あいをうあ", 0, 6);
    // test case 670
    x2("(あい|うあ)+", "$$zzzzあいをうあ", 6, 12);
    // test case 671
    x2("(あ|いあい)+", "あいあいあ", 0, 15);
    // test case 672
    x2("(あ|いあい)+", "いあ", 3, 6);
    // test case 673
    x2("(あ|いあい)+", "いあああいあ", 3, 12);
    // test case 674
    x2("(?:あ|い)(?:あ|い)", "あい", 0, 6);
    // x2("(?:あ*|い*)(?:あ*|い*)", "あああいいい", 0, 9);
    // x2("(?:あ*|い*)(?:あ+|い+)", "あああいいい", 0, 18);
    // x2("(?:あ+|い+){2}", "あああいいい", 0, 18);
    // x2("(?:あ+|い+){1,2}", "あああいいい", 0, 18);
    // x2("(?:あ+|\\Aい*)うう", "うう", 0, 6);
    // test case 675
    n("(?:あ+|\\Aい*)うう", "あいうう");
    // test case 676
    x2("(?:^あ+|い+)*う", "ああいいいあいう", 18, 24);
    // x2("(?:^あ+|い+)*う", "ああいいいいう", 0, 21);
    // x2("う{0,}", "うううう", 0, 12);
    // test case 677
    x2("あ|(?i)c", "C", 0, 1);
    // test case 678
    x2("(?i)c|あ", "C", 0, 1);
    // test case 679
    x2("(?i:あ)|a", "a", 0, 1);
    // test case 680
    n("(?i:あ)|a", "A");
    // x2("[あいう]?", "あいう", 0, 3);
    // test case 681
    x2("[あいう]*", "あいう", 0, 9);
    // test case 682
    x2("[^あいう]*", "あいう", 0, 0);
    // test case 683
    n("[^あいう]+", "あいう");
    // x2("あ?\?", "あああ", 0, 0);
    // test case 684
    x2("いあ?\?い", "いあい", 0, 9);
    // x2("あ*?", "あああ", 0, 0);
    // x2("いあ*?", "いああ", 0, 3);
    // x2("いあ*?い", "いああい", 0, 12);
    // test case 685
    x2("あ+?", "あああ", 0, 3);
    // test case 686
    x2("いあ+?", "いああ", 0, 6);
    // x2("いあ+?い", "いああい", 0, 12);
    // test case 687
    x2("(?:天?)?\?", "天", 0, 0);
    // x2("(?:天?\?)?", "天", 0, 0);
    // test case 688
    x2("(?:夢?)+?", "夢夢夢", 0, 3);
    // test case 689
    x2("(?:風+)?\?", "風風風", 0, 0);
    // x2("(?:雪+)?\?霜", "雪雪雪霜", 0, 12);
    // test case 690
    x2("(?:あい)?{2}", "", 0, 0);
    // test case 691
    x2("(?:鬼車)?{2}", "鬼車鬼車鬼", 0, 12);
    // test case 692
    x2("(?:鬼車)*{0}", "鬼車鬼車鬼", 0, 0);
    // test case 693
    x2("(?:鬼車){3,}", "鬼車鬼車鬼車鬼車", 0, 24);
    // test case 694
    n("(?:鬼車){3,}", "鬼車鬼車");
    // test case 695
    x2("(?:鬼車){2,4}", "鬼車鬼車鬼車", 0, 18);
    // test case 696
    x2("(?:鬼車){2,4}", "鬼車鬼車鬼車鬼車鬼車", 0, 24);
    // test case 697
    x2("(?:鬼車){2,4}?", "鬼車鬼車鬼車鬼車鬼車", 0, 12);
    // test case 698
    x2("(?:鬼車){,}", "鬼車{,}", 0, 9);
    // test case 699
    x2("(?:かきく)+?{2}", "かきくかきくかきく", 0, 18);
    // test case 700
    x3("(火)", "火", 0, 3, 1);
    // test case 701
    x3("(火水)", "火水", 0, 6, 1);
    // test case 702
    x2("((時間))", "時間", 0, 6);
    // test case 703
    x3("((風水))", "風水", 0, 6, 1);
    // test case 704
    x3("((昨日))", "昨日", 0, 6, 2);
    // test case 705
    x3("((((((((((((((((((((量子))))))))))))))))))))", "量子", 0, 6, 20);
    // test case 706
    x3("(あい)(うえ)", "あいうえ", 0, 6, 1);
    // test case 707
    x3("(あい)(うえ)", "あいうえ", 6, 12, 2);
    // test case 708
    x3("()(あ)いう(えおか)きくけこ", "あいうえおかきくけこ", 9, 18, 3);
    // test case 709
    x3("(()(あ)いう(えおか)きくけこ)", "あいうえおかきくけこ", 9, 18, 4);
    // test case 710
    x3(".*(フォ)ン・マ(ン()シュタ)イン", "フォン・マンシュタイン", 15, 27, 2);
    // test case 711
    x2("(^あ)", "あ", 0, 3);
    // test case 712
    x3("(あ)|(あ)", "いあ", 3, 6, 1);
    // test case 713
    x3("(^あ)|(あ)", "いあ", 3, 6, 2);
    // test case 714
    x3("(あ?)", "あああ", 0, 3, 1);
    // x3("(ま*)", "ままま", 0, 9, 1);
    // x3("(と*)", "", 0, 0, 1);
    // x3("(る+)", "るるるるるるる", 0, 21, 1);
    // x3("(ふ+|へ*)", "ふふふへへ", 0, 9, 1);
    // test case 715
    x3("(あ+|い?)", "いいいああ", 0, 3, 1);
    // test case 716
    x3("(あいう)?", "あいう", 0, 9, 1);
    // test case 717
    x3("(あいう)*", "あいう", 0, 9, 1);
    // test case 718
    x3("(あいう)+", "あいう", 0, 9, 1);
    // test case 719
    x3("(さしす|あいう)+", "あいう", 0, 9, 1);
    // x3("([なにぬ][かきく]|かきく)+", "かきく", 0, 9, 1);
    // test case 720
    x3("((?i:あいう))", "あいう", 0, 9, 1);
    // test case 721
    x3("((?m:あ.う))", "あ\nう", 0, 7, 1);
    // test case 722
    x3("((?=あん)あ)", "あんい", 0, 3, 1);
    // x3("あいう|(.あいえ)", "んあいえ", 0, 12, 1);
    // x3("あ*(.)", "ああああん", 12, 15, 1);
    // x3("あ*?(.)", "ああああん", 0, 3, 1);
    // test case 723
    x3("あ*?(ん)", "ああああん", 12, 15, 1);
    // x3("[いうえ]あ*(.)", "えああああん", 15, 18, 1);
    // test case 724
    x3("(\\Aいい)うう", "いいうう", 0, 6, 1);
    // test case 725
    n("(\\Aいい)うう", "んいいうう");
    // test case 726
    x3("(^いい)うう", "いいうう", 0, 6, 1);
    // test case 727
    n("(^いい)うう", "んいいうう");
    // test case 728
    x3("ろろ(るる$)", "ろろるる", 6, 12, 1);
    // test case 729
    n("ろろ(るる$)", "ろろるるる");
    // test case 730
    x2("(無)\\1", "無無", 0, 6);
    // test case 731
    n("(無)\\1", "無武");
    // test case 732
    x2("(空?)\\1", "空空", 0, 6);
    // x2("(空?\?)\\1", "空空", 0, 0);
    // x2("(空*)\\1", "空空空空空", 0, 12);
    // x3("(空*)\\1", "空空空空空", 0, 6, 1);
    // x2("あ(い*)\\1", "あいいいい", 0, 15);
    // x2("あ(い*)\\1", "あい", 0, 3);
    // x2("(あ*)(い*)\\1\\2", "あああいいあああいい", 0, 30);
    // x2("(あ*)(い*)\\2", "あああいいいい", 0, 21);
    // x3("(あ*)(い*)\\2", "あああいいいい", 9, 15, 2);
    // x2("(((((((ぽ*)ぺ))))))ぴ\\7", "ぽぽぽぺぴぽぽぽ", 0, 24);
    // x3("(((((((ぽ*)ぺ))))))ぴ\\7", "ぽぽぽぺぴぽぽぽ", 0, 9, 7);
    // test case 733
    x2("(は)(ひ)(ふ)\\2\\1\\3", "はひふひはふ", 0, 18);
    // x2("([き-け])\\1", "くく", 0, 6);
    // x2("(\\w\\d\\s)\\1", "あ5 あ5 ", 0, 10);
    // test case 734
    n("(\\w\\d\\s)\\1", "あ5 あ5");
    // test case 735
    x2("(誰？|[あ-う]{3})\\1", "誰？誰？", 0, 12);
    // x2("...(誰？|[あ-う]{3})\\1", "あaあ誰？誰？", 0, 19);
    // x2("(誰？|[あ-う]{3})\\1", "ういうういう", 0, 18);
    // test case 736
    x2("(^こ)\\1", "ここ", 0, 6);
    // test case 737
    n("(^む)\\1", "めむむ");
    // test case 738
    n("(あ$)\\1", "ああ");
    // test case 739
    n("(あい\\Z)\\1", "あい");
    // x2("(あ*\\Z)\\1", "あ", 3, 3);
    // x2(".(あ*\\Z)\\1", "いあ", 3, 6);
    // test case 740
    x3("(.(やいゆ)\\2)", "zやいゆやいゆ", 0, 19, 1);
    // x3("(.(..\\d.)\\2)", "あ12341234", 0, 11, 1);
    // test case 741
    x2("((?i:あvず))\\1", "あvずあvず", 0, 14);
    // x2("(?<愚か>変|\\(\\g<愚か>\\))", "((((((変))))))", 0, 15);
    // x2("\\A(?:\\g<阿_1>|\\g<云_2>|\\z終了  (?<阿_1>観|自\\g<云_2>自)(?<云_2>在|菩薩\\g<阿_1>菩薩))$",
    //   "菩薩自菩薩自在自菩薩自菩薩", 0, 39);
    // x2("[[ひふ]]", "ふ", 0, 3);
    // x2("[[いおう]か]", "か", 0, 3);
    // test case 742
    n("[[^あ]]", "あ");
    // test case 743
    n("[^[あ]]", "あ");
    // x2("[^[^あ]]", "あ", 0, 3);
    // x2("[[かきく]&&きく]", "く", 0, 3);
    // n("[[かきく]&&きく]", "か");
    // n("[[かきく]&&きく]", "け");
    // x2("[あ-ん&&い-を&&う-ゑ]", "ゑ", 0, 3);
    // test case 744
    n("[^あ-ん&&い-を&&う-ゑ]", "ゑ");
    // x2("[[^あ&&あ]&&あ-ん]", "い", 0, 3);
    // test case 745
    n("[[^あ&&あ]&&あ-ん]", "あ");
    // x2("[[^あ-ん&&いうえお]&&[^う-か]]", "き", 0, 3);
    // test case 746
    n("[[^あ-ん&&いうえお]&&[^う-か]]", "い");
    // x2("[^[^あいう]&&[^うえお]]", "う", 0, 3);
    // x2("[^[^あいう]&&[^うえお]]", "え", 0, 3);
    // n("[^[^あいう]&&[^うえお]]", "か");
    // test case 747
    x2("[あ-&&-あ]", "-", 0, 1);
    // x2("[^[^a-zあいう]&&[^bcdefgうえお]q-w]", "え", 0, 3);
    // test case 748
    x2("[^[^a-zあいう]&&[^bcdefgうえお]g-w]", "f", 0, 1);
    // test case 749
    x2("[^[^a-zあいう]&&[^bcdefgうえお]g-w]", "g", 0, 1);
    // test case 750
    n("[^[^a-zあいう]&&[^bcdefgうえお]g-w]", "2");
    // test case 751
    x2("a<b>バージョンのダウンロード<\\/b>", "a<b>バージョンのダウンロード</b>", 0, 44);
    // test case 752
    x2(".<b>バージョンのダウンロード<\\/b>", "a<b>バージョンのダウンロード</b>", 0, 44);
    // test case 753
    x2("\\n?\\z", "こんにちは", 15, 15);
    // test case 754
    x2("(?m).*", "青赤黄", 0, 9);
    // test case 755
    x2("(?m).*a", "青赤黄a", 0, 10);

    // x2("\\p{Hiragana}", "ぴ", 0, 3);
    // n("\\P{Hiragana}", "ぴ");
    // x2("\\p{Emoji}", "\xE2\xAD\x90", 0, 3);
    // x2("\\p{^Emoji}", "\xEF\xBC\x93", 0, 3);
    // x2("\\p{Extended_Pictographic}", "\xE2\x9A\xA1", 0, 3);
    // n("\\p{Extended_Pictographic}", "\xE3\x81\x82");

    // x2("\\p{Word}", "こ", 0, 3);
    // n("\\p{^Word}", "こ");
    // x2("[\\p{Word}]", "こ", 0, 3);
    // n("[\\p{^Word}]", "こ");
    // n("[^\\p{Word}]", "こ");
    // x2("[^\\p{^Word}]", "こ", 0, 3);
    // x2("[^\\p{^Word}&&\\p{ASCII}]", "こ", 0, 3);
    // test case 756
    x2("[^\\p{^Word}&&\\p{ASCII}]", "a", 0, 1);
    // test case 757
    n("[^\\p{^Word}&&\\p{ASCII}]", "#");
    // x2("[^[\\p{^Word}]&&[\\p{ASCII}]]", "こ", 0, 3);
    // x2("[^[\\p{ASCII}]&&[^\\p{Word}]]", "こ", 0, 3);
    // test case 758
    n("[[\\p{ASCII}]&&[^\\p{Word}]]", "こ");
    // x2("[^[\\p{^Word}]&&[^\\p{ASCII}]]", "こ", 0, 3);
    // x2("[^\\x{104a}]", "こ", 0, 3);
    // x2("[^\\p{^Word}&&[^\\x{104a}]]", "こ", 0, 3);
    // x2("[^[\\p{^Word}]&&[^\\x{104a}]]", "こ", 0, 3);
    // n("[^\\p{Word}||[^\\x{104a}]]", "こ");

    // x2("\\p{^Cntrl}", "こ", 0, 3);
    // test case 759
    n("\\p{Cntrl}", "こ");
    // x2("[\\p{^Cntrl}]", "こ", 0, 3);
    // test case 760
    n("[\\p{Cntrl}]", "こ");
    // test case 761
    n("[^\\p{^Cntrl}]", "こ");
    // x2("[^\\p{Cntrl}]", "こ", 0, 3);
    // x2("[^\\p{Cntrl}&&\\p{ASCII}]", "こ", 0, 3);
    // test case 762
    x2("[^\\p{Cntrl}&&\\p{ASCII}]", "a", 0, 1);
    // test case 763
    n("[^\\p{^Cntrl}&&\\p{ASCII}]", "#");
    // x2("[^[\\p{^Cntrl}]&&[\\p{ASCII}]]", "こ", 0, 3);
    // x2("[^[\\p{ASCII}]&&[^\\p{Cntrl}]]", "こ", 0, 3);
    // test case 764
    n("[[\\p{ASCII}]&&[^\\p{Cntrl}]]", "こ");
    // test case 765
    n("[^[\\p{^Cntrl}]&&[^\\p{ASCII}]]", "こ");
    // n("[^\\p{^Cntrl}&&[^\\x{104a}]]", "こ");
    // n("[^[\\p{^Cntrl}]&&[^\\x{104a}]]", "こ");
    // n("[^\\p{Cntrl}||[^\\x{104a}]]", "こ");

    // x2("(?-W:\\p{Word})", "こ", 0, 3);
    // test case 766
    n("(?W:\\p{Word})", "こ");
    // test case 767
    x2("(?W:\\p{Word})", "k", 0, 1);
    // x2("(?-W:[[:word:]])", "こ", 0, 3);
    // test case 768
    n("(?W:[[:word:]])", "こ");
    // x2("(?-D:\\p{Digit})", "３", 0, 3);
    // test case 769
    n("(?D:\\p{Digit})", "３");
    // x2("(?-S:\\p{Space})", "\xc2\x85", 0, 2);
    // test case 770
    n("(?S:\\p{Space})", "\xc2\x85");
    // x2("(?-P:\\p{Word})", "こ", 0, 3);
    // test case 771
    n("(?P:\\p{Word})", "こ");
    // x2("(?-W:\\w)", "こ", 0, 3);
    // test case 772
    n("(?W:\\w)", "こ");
    // test case 773
    x2("(?-W:\\w)", "k", 0, 1);
    // test case 774
    x2("(?W:\\w)", "k", 0, 1);
    // n("(?-W:\\W)", "こ");
    // x2("(?W:\\W)", "こ", 0, 3);
    // test case 775
    n("(?-W:\\W)", "k");
    // test case 776
    n("(?W:\\W)", "k");

    // x2("(?-W:\\b)", "こ", 0, 0);
    // test case 777
    n("(?W:\\b)", "こ");
    // test case 778
    x2("(?-W:\\b)", "h", 0, 0);
    // test case 779
    x2("(?W:\\b)", "h", 0, 0);
    // n("(?-W:\\B)", "こ");
    // test case 780
    x2("(?W:\\B)", "こ", 0, 0);
    // test case 781
    n("(?-W:\\B)", "h");
    // test case 782
    n("(?W:\\B)", "h");
    // x2("(?-P:\\b)", "こ", 0, 0);
    // test case 783
    n("(?P:\\b)", "こ");
    // test case 784
    x2("(?-P:\\b)", "h", 0, 0);
    // test case 785
    x2("(?P:\\b)", "h", 0, 0);
    // n("(?-P:\\B)", "こ");
    // test case 786
    x2("(?P:\\B)", "こ", 0, 0);
    // test case 787
    n("(?-P:\\B)", "h");
    // test case 788
    n("(?P:\\B)", "h");

    // x2("\\p{InBasicLatin}", "\x41", 0, 1);
    // x2("\\p{Grapheme_Cluster_Break_Regional_Indicator}", "\xF0\x9F\x87\xA9", 0, 4);
    // n("\\p{Grapheme_Cluster_Break_Regional_Indicator}",  "\xF0\x9F\x87\xA5");

    // extended grapheme cluster

    // CR + LF
    // test case 789
    n(".\\y\\O", "\x0d\x0a");
    // test case 790
    x2(".\\Y\\O", "\x0d\x0a", 0, 2);

    // LATIN SMALL LETTER G, COMBINING DIAERESIS
    // test case 791
    n("^.\\y.$", "\x67\xCC\x88");
    // x2(".\\Y.", "\x67\xCC\x88", 0, 3);
    // x2("\\y.\\Y.\\y", "\x67\xCC\x88", 0, 3);
    // HANGUL SYLLABLE GAG
    // x2("\\y.\\y", "\xEA\xB0\x81", 0, 3);
    // HANGUL CHOSEONG KIYEOK, HANGUL JUNGSEONG A, HANGUL JONGSEONG KIYEOK
    // x2("^.\\Y.\\Y.$", "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", 0, 9);
    // test case 792
    n("^.\\y.\\Y.$", "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8");
    // TAMIL LETTER NA, TAMIL VOWEL SIGN I,
    // x2(".\\Y.", "\xE0\xAE\xA8\xE0\xAE\xBF", 0, 6);
    // n(".\\y.", "\xE0\xAE\xA8\xE0\xAE\xBF");
    // THAI CHARACTER KO KAI, THAI CHARACTER SARA AM
    // x2(".\\Y.", "\xE0\xB8\x81\xE0\xB8\xB3", 0, 6);
    // n(".\\y.", "\xE0\xB8\x81\xE0\xB8\xB3");
    // DEVANAGARI LETTER SSA, DEVANAGARI VOWEL SIGN I
    // x2(".\\Y.", "\xE0\xA4\xB7\xE0\xA4\xBF", 0, 6);
    // n(".\\y.", "\xE0\xA4\xB7\xE0\xA4\xBF");

    // {Extended_Pictographic} Extend* ZWJ x {Extended_Pictographic}
    // x2("..\\Y.", "\xE3\x80\xB0\xE2\x80\x8D\xE2\xAD\x95", 0, 9);
    // x2("...\\Y.", "\xE3\x80\xB0\xCC\x82\xE2\x80\x8D\xE2\xAD\x95", 0, 11);
    // test case 793
    n("...\\Y.", "\xE3\x80\xB0\xCD\xB0\xE2\x80\x8D\xE2\xAD\x95");

    // CR + LF
    // test case 794
    n("^\\X\\X$", "\x0d\x0a");
    // test case 795
    x2("^\\X$", "\x0d\x0a", 0, 2);
    // LATIN SMALL LETTER G, COMBINING DIAERESIS
    // n("^\\X\\X.$", "\x67\xCC\x88");
    // x2("^\\X$", "\x67\xCC\x88", 0, 3);
    // HANGUL CHOSEONG KIYEOK, HANGUL JUNGSEONG A, HANGUL JONGSEONG KIYEOK
    // x2("^\\X$", "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", 0, 9);
    // test case 796
    n("^\\X\\X\\X$", "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8");
    // TAMIL LETTER NA, TAMIL VOWEL SIGN I,
    // x2("^\\X$", "\xE0\xAE\xA8\xE0\xAE\xBF", 0, 6);
    // n("\\X\\X", "\xE0\xAE\xA8\xE0\xAE\xBF");
    // THAI CHARACTER KO KAI, THAI CHARACTER SARA AM
    // x2("^\\X$", "\xE0\xB8\x81\xE0\xB8\xB3", 0, 6);
    // n("\\X\\X", "\xE0\xB8\x81\xE0\xB8\xB3");
    // DEVANAGARI LETTER SSA, DEVANAGARI VOWEL SIGN I
    // x2("^\\X$", "\xE0\xA4\xB7\xE0\xA4\xBF", 0, 6);
    // n("\\X\\X", "\xE0\xA4\xB7\xE0\xA4\xBF");

    // test case 797
    n("^\\X.$", "\xE0\xAE\xA8\xE0\xAE\xBF");

    // a + COMBINING GRAVE ACCENT (U+0300)
    // x2("h\\Xllo", "ha\xCC\x80llo", 0, 7);

    // Text Segment: Extended Grapheme Cluster <-> Word Boundary
    // x2("(?y{g})\\yabc\\y", "abc", 0, 3);
    // x2("(?y{g})\\y\\X\\y", "abc", 0, 1);
    // x2("(?y{w})\\yabc\\y", "abc", 0, 3);                  // WB1, WB2
    // x2("(?y{w})\\X", "\r\n", 0, 2);                       // WB3
    // x2("(?y{w})\\X", "\x0cz", 0, 1);                      // WB3a
    // x2("(?y{w})\\X", "q\x0c", 0, 1);                      // WB3b
    // x2("(?y{w})\\X", "\xE2\x80\x8D\xE2\x9D\x87", 0, 6);   // WB3c
    // x2("(?y{w})\\X", "\x20\x20", 0, 2);                   // WB3d
    // x2("(?y{w})\\X", "a\xE2\x80\x8D", 0, 4);              // WB4
    // x2("(?y{w})\\y\\X\\y", "abc", 0, 3);                  // WB5
    // x2("(?y{w})\\y\\X\\y", "v\xCE\x87w", 0, 4);           // WB6, WB7
    // x2("(?y{w})\\y\\X\\y", "\xD7\x93\x27", 0, 3);         // WB7a
    // x2("(?y{w})\\y\\X\\y", "\xD7\x93\x22\xD7\x93", 0, 5); // WB7b, WB7c
    // x2("(?y{w})\\X", "14 45", 0, 2);                      // WB8
    // x2("(?y{w})\\X", "a14", 0, 3);                        // WB9
    // x2("(?y{w})\\X", "832e", 0, 4);                       // WB10
    // x2("(?y{w})\\X", "8\xEF\xBC\x8C\xDB\xB0", 0, 6);      // WB11, WB12
    // x2("(?y{w})\\y\\X\\y", "ケン", 0, 6);                 // WB13
    // x2("(?y{w})\\y\\X\\y", "ケン\xE2\x80\xAFタ", 0, 12);  // WB13a, WB13b
    // x2("(?y{w})\\y\\X\\y", "\x21\x23", 0, 1);             // WB999
    // x2("(?y{w})\\y\\X\\y", "山ア", 0, 3);
    // x2("(?y{w})\\X", "3.14", 0, 4);
    // x2("(?y{w})\\X", "3 14", 0, 1);

    // test case 798
    x2("\\x40", "@", 0, 1);
    // test case 799
    x2("\\x1", "\x01", 0, 1);
    // test case 800
    x2("\\x{1}", "\x01", 0, 1);
    // x2("\\x{4E38}", "\xE4\xB8\xB8", 0, 3);
    // x2("\\u4E38", "\xE4\xB8\xB8", 0, 3);
    // test case 801
    x2("\\u0040", "@", 0, 1);

    // test case 802
    x2("c.*\\b", "abc", 2, 3);
    // test case 803
    x2("\\b.*abc.*\\b", "abc", 0, 3);
    // test case 804
    x2("((?()0+)+++(((0\\g<0>)0)|())++++((?(1)(0\\g<0>))++++++0*())++++((?(1)(0\\g<1>)+)++++++++++*())++++((?(1)(("
       "0)"
       "\\g<0>)+)++())+0++*+++(((0\\g<0>))*())++++((?(1)(0\\g<0>)+)++++++++++*|)++++*+++((?(1)((0)\\g<0>)+)++++++++"
       "+())"
       "++*|)++++((?()0))|",
       "abcde", 0, 0); // #139

    // test case 805
    n("(*FAIL)", "abcdefg");
    // test case 806
    n("abcd(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*"
      "FAIL)(*"
      "FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*"
      "FAIL)(*"
      "FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*"
      "FAIL)(*"
      "FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)(*FAIL)",
      "abcdefg");
    // test case 807
    x2("(?:[ab]|(*MAX{2}).)*", "abcbaaccaaa", 0, 7);
    // test case 808
    x2("(?:(*COUNT[AB]{X})[ab]|(*COUNT[CD]{X})[cd])*(*CMP{AB,<,CD})", "abababcdab", 5, 8);
    // test case 809
    x2("(?(?{....})123|456)", "123", 0, 3);
    // test case 810
    x2("(?(*FAIL)123|456)", "456", 0, 3);

    // test case 811
    x2("\\g'0'++{,0}", "abcdefgh", 0, 0);
    // test case 812
    x2("\\g'0'++{,0}?", "abcdefgh", 0, 0);
    // test case 813
    x2("\\g'0'++{,0}b", "abcdefgh", 1, 2);
    // test case 814
    x2("\\g'0'++{,0}?def", "abcdefgh", 3, 6);
    // test case 815
    x2("a{1,3}?", "aaa", 0, 1);
    // test case 816
    x2("a{3}", "aaa", 0, 3);
    // test case 817
    x2("a{3}?", "aaa", 0, 3);
    // test case 818
    x2("a{3}?", "aa", 0, 0);
    // test case 819
    x2("a{3,3}?", "aaa", 0, 3);
    // test case 820
    n("a{3,3}?", "aa");
    // test case 821
    x2("a{1,3}+", "aaaaaa", 0, 6);
    // test case 822
    x2("a{3}+", "aaaaaa", 0, 6);
    // test case 823
    x2("a{3,3}+", "aaaaaa", 0, 6);
    // test case 824
    n("a{2,3}?", "a");
    // test case 825
    n("a{3,2}a", "aaa");
    // test case 826
    x2("a{3,2}b", "aaab", 0, 4);
    // test case 827
    x2("a{3,2}b", "aaaab", 1, 5);
    // test case 828
    x2("a{3,2}b", "aab", 0, 3);
    // test case 829
    x2("a{3,2}?", "", 0, 0); /* == (?:a{3,2})?*/
    // test case 830
    x2("a{2,3}+a", "aaa", 0, 3); /* == (?:a{2,3})+*/
    // x2("[\\x{0}-\\x{7fffffff}]", "a", 0, 1);
    // x2("[\\x{7f}-\\x{7fffffff}]", "\xe5\xae\xb6", 0, 3);
    // test case 831
    x2("[a[cdef]]", "a", 0, 1);
    // test case 832
    n("[a[xyz]-c]", "b");
    // test case 833
    x2("[a[xyz]-c]", "a", 0, 1);
    // test case 834
    x2("[a[xyz]-c]", "-", 0, 1);
    // test case 835
    x2("[a[xyz]-c]", "c", 0, 1);
    // test case 836
    x2("(a.c|def)(.{4})(?<=\\1)", "abcdabc", 0, 7);
    // test case 837
    x2("(a.c|de)(.{4})(?<=\\1)", "abcdabc", 0, 7);
    // test case 838
    x2("(a.c|def)(.{5})(?<=d\\1e)", "abcdabce", 0, 8);
    // test case 839
    x2("(a.c|.)d(?<=\\k<1>d)", "zzzzzabcdabc", 5, 9);
    // test case 840
    x2("(?<=az*)abc", "azzzzzzzzzzabcdabcabc", 11, 14);
    // test case 841
    x2("(?<=ab|abc|abcd)ef", "abcdef", 4, 6);
    // test case 842
    x2("(?<=ta+|tb+|tc+|td+)zz", "tcccccccccczz", 11, 13);
    // test case 843
    x2("(?<=t.{7}|t.{5}|t.{2}|t.)zz", "tczz", 2, 4);
    // test case 844
    x2("(?<=t.{7}|t.{5}|t.{2})zz", "tczzzz", 3, 5);
    // test case 845
    x2("(?<=t.{7}|t.{5}|t.{3})zz", "tczzazzbzz", 8, 10);
    // test case 846
    n("(?<=t.{7}|t.{5}|t.{3})zz", "tczzazzbczz");
    // test case 847
    x2("(?<=(ab|abc|abcd))ef", "abcdef", 4, 6);
    // test case 848
    x2("(?<=(ta+|tb+|tc+|td+))zz", "tcccccccccczz", 11, 13);
    // test case 849
    x2("(?<=(t.{7}|t.{5}|t.{2}|t.))zz", "tczz", 2, 4);
    // test case 850
    x2("(?<=(t.{7}|t.{5}|t.{2}))zz", "tczzzz", 3, 5);
    // test case 851
    x2("(?<=(t.{7}|t.{5}|t.{3}))zz", "tczzazzbzz", 8, 10);
    // test case 852
    n("(?<=(t.{7}|t.{5}|t.{3}))zz", "tczzazzbczz");
    // test case 853
    x2("(.{1,4})(.{1,4})(?<=\\2\\1)", "abaaba", 0, 6);
    // test case 854
    x2("(.{1,4})(.{1,4})(?<=\\2\\1)", "ababab", 0, 6);
    // test case 855
    n("(.{1,4})(.{1,4})(?<=\\2\\1)", "abcdabce");
    // test case 856
    x2("(.{1,4})(.{1,4})(?<=\\2\\1)", "abcdabceabce", 4, 12);
    // test case 857
    x2("(?<=a)", "a", 1, 1);
    // test case 858
    x2("(?<=a.*\\w)z", "abbbz", 4, 5);
    // test case 859
    n("(?<=a.*\\w)z", "abb z");
    // test case 860
    x2("(?<=a.*\\W)z", "abb z", 4, 5);
    // test case 861
    x2("(?<=a.*\\b)z", "abb z", 4, 5);
    // test case 862
    x2("(?<=(?>abc))", "abc", 3, 3);
    // test case 863
    x2("(?<=a\\Xz)", "abz", 3, 3);
    // test case 864
    n("(?<=^a*)bc", "zabc");
    // test case 865
    n("(?<=a*\\b)b", "abc");
    // test case 866
    x2("(?<=a+.*[efg])z", "abcdfz", 5, 6);
    // test case 867
    x2("(?<=a+.*[efg])z", "abcdfgz", 6, 7);
    // test case 868
    n("(?<=a+.*[efg])z", "bcdfz");
    // test case 869
    x2("(?<=a*.*[efg])z", "bcdfz", 4, 5);
    // test case 870
    n("(?<=a+.*[efg])z", "abcdz");
    // test case 871
    x2("(?<=v|t|a+.*[efg])z", "abcdfz", 5, 6);
    // test case 872
    x2("(?<=v|t|^a+.*[efg])z", "abcdfz", 5, 6);
    // test case 873
    x2("(?<=^(?:v|t|a+.*[efg]))z", "abcdfz", 5, 6);
    // test case 874
    x2("(?<=v|^t|a+.*[efg])z", "uabcdfz", 6, 7);
    // test case 875
    n("^..(?<=(a{,2}))\\1z", "aaaaz"); // !!! look-behind is shortest priority
    // test case 876
    x2("^..(?<=(a{,2}))\\1z", "aaz", 0, 3); // shortest priority
    // test case 877
    e("(?<=(?~|zoo)a.*z)", "abcdefz", ONIGERR_INVALID_LOOK_BEHIND_PATTERN);
    // test case 878
    e("(?<=(?~|)a.*z)", "abcdefz", ONIGERR_INVALID_LOOK_BEHIND_PATTERN);
    // test case 879
    e("(a(?~|boo)z){0}(?<=\\g<1>)", "abcdefz", ONIGERR_INVALID_LOOK_BEHIND_PATTERN);
    // test case 880
    x2("(?<=(?<= )| )", "abcde fg", 6, 6); // #173
    // test case 881
    x2("(?<=D|)(?<=@!nnnnnnnnnIIIIn;{1}D?()|<x@x*xxxD|)(?<=@xxx|xxxxx\\g<1>;{1}x)",
       "(?<=D|)(?<=@!nnnnnnnnnIIIIn;{1}D?()|<x@x*xxxD|)(?<=@xxx|xxxxx\\g<1>;{1}x)", 55, 55); // #173
    // test case 882
    x2("(?<=;()|)\\g<1>", "", 0, 0); // reduced #173
    // test case 883
    x2("(?<=;()|)\\k<1>", ";", 1, 1);
    // test case 884
    x2("(())\\g<3>{0}(?<=|())", "abc", 0, 0); // #175
    // test case 885
    x2("(?<=()|)\\1{0}", "abc", 0, 0);
    // test case 886
    e("(?<!xxxxxxxxxxxxxxxxxxxxxxx{32774}{65521}xxxxxxxx{65521}xxxxxxxxxxxxxx{32774}"
      "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)",
      "", ONIGERR_INVALID_LOOK_BEHIND_PATTERN); // #177
    // test case 887
    x2("(?<=(?<=abc))def", "abcdef", 3, 6);
    // test case 888
    x2("(?<=ab(?<=.+b)c)def", "abcdef", 3, 6);
    // test case 889
    n("(?<=ab(?<=a+)c)def", "abcdef");
    // test case 890
    n("(?<=abc)(?<!abc)def", "abcdef");
    // test case 891
    n("(?<!ab.)(?<=.bc)def", "abcdef");
    // test case 892
    x2("(?<!ab.)(?<=.bc)def", "abcdefcbcdef", 9, 12);
    // test case 893
    n("(?<!abc)def", "abcdef");
    // test case 894
    n("(?<!xxx|abc)def", "abcdef");
    // test case 895
    n("(?<!xxxxx|abc)def", "abcdef");
    // test case 896
    n("(?<!xxxxx|abc)def", "xxxxxxdef");
    // test case 897
    n("(?<!x+|abc)def", "abcdef");
    // test case 898
    n("(?<!x+|abc)def", "xxxxxxxxxdef");
    // test case 899
    x2("(?<!x+|abc)def", "xxxxxxxxzdef", 9, 12);
    // test case 900
    n("(?<!a.*z|a)def", "axxxxxxxzdef");
    // test case 901
    n("(?<!a.*z|a)def", "bxxxxxxxadef");
    // test case 902
    x2("(?<!a.*z|a)def", "axxxxxxxzdefxxdef", 14, 17);
    // test case 903
    x2("(?<!a.*z|a)def", "bxxxxxxxadefxxdef", 14, 17);
    // test case 904
    x2("(?<!a.*z|a)def", "bxxxxxxxzdef", 9, 12);
    // test case 905
    x2("(?<!x+|y+)\\d+", "xxx572", 4, 6);
    // test case 906
    x2("(?<!3+|4+)\\d+", "33334444", 0, 8);
    // test case 907
    n(".(?<!3+|4+)\\d+", "33334444");
    // test case 908
    n("(.{,3})..(?<!\\1)", "aaaaa");
    // test case 909
    x2("(.{,3})..(?<!\\1)", "abcde", 0, 5);
    // test case 910
    x2("(.{,3})...(?<!\\1)", "abcde", 0, 5);
    // test case 911
    x2("(a.c)(.{3,}?)(?<!\\1)", "abcabcd", 0, 7);
    // test case 912
    x2("(a*)(.{3,}?)(?<!\\1)", "abcabcd", 0, 5);
    // test case 913
    x2("(?:(a.*b)|c.*d)(?<!(?(1))azzzb)", "azzzzb", 0, 6);
    // test case 914
    n("(?:(a.*b)|c.*d)(?<!(?(1))azzzb)", "azzzb");
    // test case 915
    x2("<(?<!NT{+}abcd)", "<(?<!NT{+}abcd)", 0, 1);
    // test case 916
    x2("(?<!a.*c)def", "abbbbdef", 5, 8);
    // test case 917
    n("(?<!a.*c)def", "abbbcdef");
    // test case 918
    x2("(?<!a.*X\\b)def", "abbbbbXdef", 7, 10);
    // test case 919
    n("(?<!a.*X\\B)def", "abbbbbXdef");
    // test case 920
    x2("(?<!a.*[uvw])def", "abbbbbXdef", 7, 10);
    // test case 921
    n("(?<!a.*[uvw])def", "abbbbbwdef");
    // test case 922
    x2("(?<!ab*\\S+)def", "abbbbb   def", 9, 12);
    // test case 923
    x2("(?<!a.*\\S)def", "abbbbb def", 7, 10);
    // test case 924
    n("(?<!ab*\\s+)def", "abbbbb   def");
    // test case 925
    x2("(?<!ab*\\s+\\B)def", "abbbbb   def", 9, 12);
    // test case 926
    n("(?<!v|t|a+.*[efg])z", "abcdfz");
    // test case 927
    x2("(?<!v|t|a+.*[efg])z", "abcdfzavzuz", 10, 11);
    // test case 928
    n("(?<!v|t|^a+.*[efg])z", "abcdfz");
    // test case 929
    n("(?<!^(?:v|t|a+.*[efg]))z", "abcdfz");
    // test case 930
    x2("(?<!v|^t|^a+.*[efg])z", "uabcdfz", 6, 7);
    // test case 931
    n("(\\k<2>)|(?<=(\\k<1>))", "");
    // test case 932
    x2("(a|\\k<2>)|(?<=(\\k<1>))", "a", 0, 1);
    // test case 933
    x2("(a|\\k<2>)|(?<=b(\\k<1>))", "ba", 1, 2);
    // test case 934
    x2("((?(a)\\g<1>|b))", "aab", 0, 3);
    // test case 935
    x2("((?(a)\\g<1>))", "aab", 0, 2);
    // test case 936
    x2("(b(?(a)|\\g<1>))", "bba", 0, 3);
    // test case 937
    e("(()(?(2)\\g<1>))", "", ONIGERR_NEVER_ENDING_RECURSION);
    // test case 938
    x2("(?(a)(?:b|c))", "ac", 0, 2);
    // test case 939
    n("^(?(a)b|c)", "ac");
    // test case 940
    x2("(?i)a|b", "B", 0, 1);
    // test case 941
    n("((?i)a|b.)|c", "C");
    // test case 942
    n("c(?i)a.|b.", "Caz");
    // test case 943
    x2("c(?i)a|b", "cB", 0, 2); /* == c(?i:a|b) */
    // test case 944
    x2("c(?i)a.|b.", "cBb", 0, 3);
    // test case 945
    x2("(?i)st", "st", 0, 2);
    // test case 946
    x2("(?i)st", "St", 0, 2);
    // test case 947
    x2("(?i)st", "sT", 0, 2);
    // x2("(?i)st", "\xC5\xBFt", 0, 3);    // U+017F
    // x2("(?i)st", "\xEF\xAC\x85", 0, 3); // U+FB05
    // x2("(?i)st", "\xEF\xAC\x86", 0, 3); // U+FB06
    // test case 948
    x2("(?i)ast", "Ast", 0, 3);
    // test case 949
    x2("(?i)ast", "ASt", 0, 3);
    // test case 950
    x2("(?i)ast", "AsT", 0, 3);
    // x2("(?i)ast", "A\xC5\xBFt", 0, 4);    // U+017F
    // x2("(?i)ast", "A\xEF\xAC\x85", 0, 4); // U+FB05
    // x2("(?i)ast", "A\xEF\xAC\x86", 0, 4); // U+FB06
    // test case 951
    x2("(?i)stZ", "stz", 0, 3);
    // test case 952
    x2("(?i)stZ", "Stz", 0, 3);
    // test case 953
    x2("(?i)stZ", "sTz", 0, 3);
    // x2("(?i)stZ", "\xC5\xBFtz", 0, 4);    // U+017F
    // x2("(?i)stZ", "\xEF\xAC\x85z", 0, 4); // U+FB05
    // x2("(?i)stZ", "\xEF\xAC\x86z", 0, 4); // U+FB06
    // test case 954
    x2("(?i)BstZ", "bstz", 0, 4);
    // test case 955
    x2("(?i)BstZ", "bStz", 0, 4);
    // test case 956
    x2("(?i)BstZ", "bsTz", 0, 4);
    // x2("(?i)BstZ", "b\xC5\xBFtz", 0, 5);                     // U+017F
    // x2("(?i)BstZ", "b\xEF\xAC\x85z", 0, 5);                  // U+FB05
    // x2("(?i)BstZ", "b\xEF\xAC\x86z", 0, 5);                  // U+FB06
    // x2("(?i).*st\\z", "tttssss\xC5\xBFt", 0, 10);            // U+017F
    // x2("(?i).*st\\z", "tttssss\xEF\xAC\x85", 0, 10);         // U+FB05
    // x2("(?i).*st\\z", "tttssss\xEF\xAC\x86", 0, 10);         // U+FB06
    // x2("(?i).*あstい\\z", "tttssssあ\xC5\xBFtい", 0, 16);    // U+017F
    // x2("(?i).*あstい\\z", "tttssssあ\xEF\xAC\x85い", 0, 16); // U+FB05
    // x2("(?i).*あstい\\z", "tttssssあ\xEF\xAC\x86い", 0, 16); // U+FB06
    // x2("(?i).*\xC5\xBFt\\z", "tttssssst", 0, 9);             // U+017F
    // x2("(?i).*\xEF\xAC\x85\\z", "tttssssあst", 0, 12);       // U+FB05
    // x2("(?i).*\xEF\xAC\x86い\\z", "tttssssstい", 0, 12);     // U+FB06
    // test case 957
    x2("(?i).*\xEF\xAC\x85\\z", "tttssssあ\xEF\xAC\x85", 0, 13);

    // x2("(?i).*ss", "abcdefghijklmnopqrstuvwxyz\xc3\x9f", 0, 28);      // U+00DF
    // x2("(?i).*ss.*", "abcdefghijklmnopqrstuvwxyz\xc3\x9fxyz", 0, 31); // U+00DF
    // x2("(?i).*\xc3\x9f", "abcdefghijklmnopqrstuvwxyzss", 0, 28);      // U+00DF
    // test case 958
    x2("(?i).*ss.*", "abcdefghijklmnopqrstuvwxyzSSxyz", 0, 31);

    // x2("(?i)ssv", "\xc3\x9fv", 0, 3); // U+00DF
    // test case 959
    x2("(?i)(?<=ss)v", "SSv", 2, 3);
    // test case 960
    x2("(?i)(?<=\xc3\x9f)v", "\xc3\x9fv", 2, 3);
    // x2("(?i)(?<=\xc3\x9f)v", "ssv", 2, 3);
    // x2("(?i)(?<=ss)v", "\xc3\x9fv", 2, 3);

    /* #156 U+01F0 (UTF-8: C7 B0) */
    // test case 961
    x2("(?i).+Isssǰ", ".+Isssǰ", 0, 8);
    // test case 962
    x2(".+Isssǰ", ".+Isssǰ", 0, 8);
    // test case 963
    x2("(?i)ǰ", "ǰ", 0, 2);
    // x2("(?i)ǰ", "j\xcc\x8c", 0, 3);
    // x2("(?i)j\xcc\x8c", "ǰ", 0, 2);
    // test case 964
    x2("(?i)5ǰ", "5ǰ", 0, 3);
    // x2("(?i)5ǰ", "5j\xcc\x8c", 0, 4);
    // x2("(?i)5j\xcc\x8c", "5ǰ", 0, 3);
    // test case 965
    x2("(?i)ǰv", "ǰV", 0, 3);
    // x2("(?i)ǰv", "j\xcc\x8cV", 0, 4);
    // x2("(?i)j\xcc\x8cv", "ǰV", 0, 3);
    // x2("(?i)[ǰ]", "ǰ", 0, 2);
    // x2("(?i)[ǰ]", "j\xcc\x8c", 0, 3);
    // x2("(?i)[j]\xcc\x8c", "ǰ", 0, 2);
    // x2("(?i)\ufb00a", "ffa", 0, 3);
    // x2("(?i)ffz", "\xef\xac\x80z", 0, 4);
    // x2("(?i)\u2126", "\xcf\x89", 0, 2);
    // x2("a(?i)\u2126", "a\xcf\x89", 0, 3);
    // x2("(?i)A\u2126", "a\xcf\x89", 0, 3);
    // x2("(?i)A\u2126=", "a\xcf\x89=", 0, 4);
    // x2("(?i:ss)=1234567890", "\xc5\xbf\xc5\xbf=1234567890", 0, 15);

    // test case 966
    x2("\\x{000A}", "\x0a", 0, 1);
    // test case 967
    x2("\\x{000A 002f}", "\x0a\x2f", 0, 2);
    // test case 968
    x2("\\x{000A 002f }", "\x0a\x2f", 0, 2);
    // test case 969
    x2("\\x{007C     001b}", "\x7c\x1b", 0, 2);
    // test case 970
    x2("\\x{1 2 3 4 5 6 7 8 9 a b c d e f}", "\x01\x02\x3\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 0, 15);
    // test case 971
    x2("a\\x{000A 002f}@", "a\x0a\x2f@", 0, 4);
    // test case 972
    x2("a\\x{0060\n0063}@", "a\x60\x63@", 0, 4);
    // test case 973
    e("\\x{00000001 000000012}", "", ONIGERR_TOO_LONG_WIDE_CHAR_VALUE);
    // test case 974
    e("\\x{000A 00000002f}", "", ONIGERR_TOO_LONG_WIDE_CHAR_VALUE);
    // test case 975
    e("\\x{000A 002f/", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 976
    e("\\x{000A 002f /", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 977
    e("\\x{000A", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 978
    e("\\x{000A ", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 979
    e("\\x{000A 002f ", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 980
    x2("\\o{102}", "B", 0, 1);
    // test case 981
    x2("\\o{102 103}", "BC", 0, 2);
    // test case 982
    x2("\\o{0160 0000161}", "pq", 0, 2);
    // test case 983
    x2("\\o{1 2 3 4 5 6 7 10 11 12 13 14 15 16 17}", "\x01\x02\x3\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 0,
       15);
    // test case 984
    x2("\\o{0007 0010 }", "\x07\x08", 0, 2);
    // test case 985
    e("\\o{0000 0015/", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 986
    e("\\o{0000 0015 /", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 987
    e("\\o{0015", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 988
    e("\\o{0015 ", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 989
    e("\\o{0007 002f}", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 990
    x2("[\\x{000A}]", "\x0a", 0, 1);
    // test case 991
    x2("[\\x{000A 002f}]+", "\x0a\x2f\x2e", 0, 2);
    // test case 992
    x2("[\\x{01 0F 1A 2c 4B}]+", "\x20\x01\x0f\x1a\x2c\x4b\x1b", 1, 6);
    // test case 993
    x2("[\\x{0020 0024}-\\x{0026}]+", "\x25\x24\x26\x23", 0, 3);
    // test case 994
    x2("[\\x{0030}-\\x{0033 005a}]+", "\x30\x31\x32\x33\x5a\34", 0, 5);
    // test case 995
    e("[\\x{000A]", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 996
    e("[\\x{000A ]", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 997
    e("[\\x{000A }]", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 998
    x2("[\\o{102}]", "B", 0, 1);
    // test case 999
    x2("[\\o{102 103}]*", "BC", 0, 2);
    // test case 1000
    e("[a\\o{002  003]bcde|zzz", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1001
    x2("[\\x{0030-0039}]+", "abc0123456789def", 3, 13);
    // test case 1002
    x2("[\\x{0030 - 0039 }]+", "abc0123456789def", 3, 13);
    // test case 1003
    x2("[\\x{0030 - 0039 0063 0064}]+", "abc0123456789def", 2, 14);
    // test case 1004
    x2("[\\x{0030 - 0039 0063-0065}]+", "acde019b", 1, 7);
    // test case 1005
    e("[\\x{0030 - 0039-0063 0064}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1006
    e("[\\x{0030 - }]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1007
    e("[\\x{0030 -- 0040}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1008
    e("[\\x{0030--0040}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1009
    e("[\\x{0030 - - 0040}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1010
    e("[\\x{0030 0044 - }]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1011
    e("[a-\\x{0070 - 0039}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1012
    x2("[a-\\x{0063 0071}]+", "dabcqz", 1, 5);
    // test case 1013
    x2("[-\\x{0063-0065}]+", "ace-df", 1, 5);
    // test case 1014
    x2("[\\x61-\\x{0063 0065}]+", "abced", 0, 4);
    // test case 1015
    e("[\\x61-\\x{0063-0065}]+", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1016
    x2("[t\\x{0063 0071}]+", "tcqb", 0, 3);
    // test case 1017
    x2("[\\W\\x{0063 0071}]+", "*cqa", 0, 3);

    // test case 1018
    n("a(b|)+d", "abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcd"); /* https://www.haijin-boys.com/discussions/5079 */
    // test case 1019
    n("   \xfd", ""); /* https://bugs.php.net/bug.php?id=77370 */
    /* can't use \xfc00.. because compiler error: hex escape sequence out of range */
    // test case 1020
    n("()0\\xfc00000\\xfc00000\\xfc00000\xfc", ""); /* https://bugs.php.net/bug.php?id=77371 */
    // test case 1021
    x2("000||0\xfa", "0", 0, 0); /* https://bugs.php.net/bug.php?id=77381 */
    // e("(?i)000000000000000000000\xf0", "",
    //  ONIGERR_INVALID_CODE_POINT_VALUE);   /* https://bugs.php.net/bug.php?id=77382 */
    // test case 1022
    n("0000\\\xf5", "0"); /* https://bugs.php.net/bug.php?id=77385 */
    // test case 1023
    n("(?i)FFF00000000000000000\xfd", ""); /* https://bugs.php.net/bug.php?id=77394 */
    // test case 1024
    n("(?x)\n  "
      "(?<!\\+\\+|--)(?<=[({\\[,?=>:*]|&&|\\|\\||\\?|\\*\\/"
      "|^await|[^\\._$[:alnum:]]await|^return|[^\\._$[:alnum:]]return|^default|[^\\._$[:alnum:]]default|^yield|[^"
      "\\._$["
      ":alnum:]]yield|^)\\s*\n  (?!<\\s*[_$[:alpha:]][_$[:alnum:]]*((\\s+extends\\s+[^=>])|,)) # look ahead is not "
      "type parameter of arrow\n  "
      "(?=(<)\\s*(?:([_$[:alpha:]][-_$[:alnum:].]*)(?<!\\.|-)(:))?((?:[a-z][a-z0-9]*|([_$[:alpha:]][-_$[:alnum:].]*"
      "))(?"
      "<!\\.|-))(?=((<\\s*)|(\\s+))(?!\\?)|\\/?>))",
      "    while (i < len && f(array[i]))"); /* Issue #192 */

    // test case 1025
    e("x{55380}{77590}", "", ONIGERR_TOO_BIG_NUMBER_FOR_REPEAT_RANGE);
    // test case 1026
    e("(xyz){40000}{99999}(?<name>vv)", "", ONIGERR_TOO_BIG_NUMBER_FOR_REPEAT_RANGE);
    // test case 1027
    e("f{90000,90000}{80000,80000}", "", ONIGERR_TOO_BIG_NUMBER_FOR_REPEAT_RANGE);
    // test case 1028
    n("f{90000,90000}{80000,80001}", "");

    // x2("\\p{Common}", "\xe3\x8b\xbf", 0, 3);                             /* U+32FF */
    // x2("\\p{In_Enclosed_CJK_Letters_and_Months}", "\xe3\x8b\xbf", 0, 3); /* U+32FF */

    // test case 1029
    e("\\x{7fffffff}", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1030
    e("[\\x{7fffffff}]", "", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1031
    e("\\u040", "@", ONIGERR_INVALID_CODE_POINT_VALUE);
    // test case 1032
    e("(?<abc>\\g<abc>)", "zzzz", ONIGERR_NEVER_ENDING_RECURSION);
    // test case 1033
    e("(*FOO)", "abcdefg", ONIGERR_UNDEFINED_CALLOUT_NAME);
    // test case 1034
    e("*", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_NOT_SPECIFIED);
    // test case 1035
    e("|*", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_NOT_SPECIFIED);
    // test case 1036
    e("(?i)*", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_NOT_SPECIFIED);
    // test case 1037
    e("(?:*)", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_NOT_SPECIFIED);
    // test case 1038
    e("(?m:*)", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_NOT_SPECIFIED);
    // test case 1039
    x2("(?:)*", "abc", 0, 0);
    // test case 1040
    e("^*", "abc", ONIGERR_TARGET_OF_REPEAT_OPERATOR_INVALID);
    // test case 1041
    x2(".*", "abc", 0, 4);
    // test case 1042
    x2("a{1,100}b{1,100}c{1,100}", "aaaaabbbbbccccc", 0, 15);
    // test case 1043
    x2("^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \\[(?<time>[^\\]]*)\\] \"(?<method>\\S+)(?: "
       "+(?<path>[^\\\"]*?)(?: +\\S*)?)?\" (?<code>[^ ]*) (?<size>[^ ]*)(?: \"(?<referer>[^\\\"]*)\" "
       "\"(?<agent>[^\\\"]*)\"(?:\\s+(?<http_x_forwarded_for>[^ ]+))?)?$",
       "192.168.0.1 - - [28/Feb/2013:12:00:00 +0900] \"GET / HTTP/1.1\" 200 777 \"-\" \"Opera/12.O\"", 0, 86);
    // test case 1044
    x2("^\\[[^ ]* (?<time>[^\\\]]*)\\] \\[(?<level>[^\\\]]*)\\](?: \\[pid(?<pid>[^\\\]]*)\\])? \\[client "
       "(?<client>[^\\\]]*)\\] (?<message>.*)$",
       "[Wed Oct 11 14:32:52 2000] [error] [client 127.0.0.1] client denied by server configuration ", 0, 92);
    // test case 1045
    x2("^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \\[(?<time>[^\\\]]*)\\] \"(?<method>\\S+)(?: "
       "+(?<path>[^\\\"]*?)(    ?: +\\S*)?)?\" (?<code>[^ ]*) (?<size>[^ ]*)(?: \"(?<referer>[^\\\"]*)\" "
       "\"(?<agent>[^\\\"]*)\"(?:\\s+(?<http_x_forwarded_for>[^ ]+))?)?$",
       "127.0.0.1 192.168.0.1 - [28/Feb/2013:12:00:00 +0900] \"GET / HTTP/1.1\" 200 777 \"-\" \"Opera/12.0\" -", 0, 96);
    // test case 1046
    x2("^\\<(?<pri>[0-9]+)\\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) "
       "(?<ident>[^:\\[]*)(?:\\[(?<pid>[0-9]+)\\])?(    ?:[^\\:]*\\:)? *(?<message>.*)$",
       "<6>Feb 28 12:00:00 192.168.0.1 fluentd[11111]: [error] Syslog test", 0, 66);
    // test case 1047
    x2("\\A\\<(?<pri>[0-9]{1,3})\\>[1-9]\\d{0,2} (?<time>[^ ]+) (?<host>[!-~]{1,255}) (?<ident>[!-~]{1,48}) "
       "(?<pid>[!-~]{1,128}) (?<msgid>[!-~]{1,32}) (?<extradata>(?:-|(?:\\[.*?(?<!\\\\)\\])+))(?: (?<message>.+))?\\z",
       "<16>1 2013-02-28T12:00:00.003Z 192.168.0.1 fluentd 11111 ID24224 [exampleSDID@20224 iut=\"3\" "
       "eventSource=\"Application\" eventID=\"11211\"] Hi, from Fluentd!",
       0, 152);

    fprintf(stdout, "\nRESULT   SUCC: %4d,   FAIL: %d,   ERROR: %d,   UNSUPPORT: %d\n", nsucc, nfail, nerror, nsupport);

    onig_region_free(region, 1);
    onig_end();

    return ((nfail == 0 && nerror == 0) ? 0 : -1);
}
