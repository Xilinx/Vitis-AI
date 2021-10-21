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

/**
 *
 * @file xf_re_compile.c
 * @brief Compiler based on Oniguruma for parsing pattern to RE.
 *
 */

#include <stdio.h>
#include <string.h>
#include "oniguruma.h"
#include "regint.h"
#include "st.h"
#include <assert.h>
#include <stdbool.h>

#include "xf_data_analytics/text/xf_re_compile.h"

#ifdef USE_DIRECT_THREADED_CODE
#define GET_OPCODE(reg, index) (reg)->ocs[index]
#else
#define GET_OPCODE(reg, index) (reg)->ops[index].opcode
#endif
typedef struct {
    unsigned int* data;
    unsigned int size;
    unsigned int count;
} dyn_buff;

typedef struct st_table_entry st_table_entry;

struct st_table_entry {
    unsigned int hash;
    unsigned long key;
    unsigned long record;
    st_table_entry* next;
};
typedef struct {
    UChar* name;
    int name_len; /* byte length */
    int back_num; /* number of backrefs */
    int back_alloc;
    int back_ref1;
    int* back_refs;
} NameEntry;
void init_buff(dyn_buff* tb) {
    tb->data = NULL;
    tb->size = 0;
    tb->count = 0;
}
void buff_add(dyn_buff* tb, unsigned int data) {
    if (tb->size == 0) {
        tb->size = 128;
        tb->data = malloc(sizeof(unsigned int) * tb->size);
        memset(tb->data, 0, sizeof(unsigned int) * tb->size);
    }
    if (tb->size == tb->count) {
        tb->size += 128;
        tb->data = realloc(tb->data, tb->size * sizeof(unsigned int));
    }
    tb->data[tb->count] = data;
    tb->count++;
}

unsigned int buff_get(dyn_buff* tb, unsigned int index) {
    if (index >= tb->count) {
        printf("ERROR: could not found\n");
    }
    return tb->data[index];
}
void buff_free(dyn_buff* tb) {
    free(tb->data);
}
extern int xf_re_compile(const char* pattern,
                         unsigned int* bitset,
                         uint64_t* instructions,
                         unsigned int* instr_num,
                         unsigned int* cclass_num,
                         unsigned int* cpgp_nm,
                         uint8_t* cpgp_name_val,
                         uint32_t* cpgp_name_oft) {
    int r;
    OnigErrorInfo einfo;
    regex_t* reg;

    UChar* pattern_c = (UChar*)pattern;

    r = onig_new(&reg, pattern_c, pattern_c + strlen((char*)pattern_c), ONIG_OPTION_DEFAULT, ONIG_ENCODING_ASCII,
                 ONIG_SYNTAX_DEFAULT, &einfo);

    if (r != ONIG_NORMAL) {
        return r;
    }
    *cpgp_nm = reg->num_mem;
    // get the name of each capture group
    if (cpgp_name_val != NULL && cpgp_name_oft != NULL) {
        unsigned int* len_buff = (unsigned int*)malloc((*cpgp_nm) * sizeof(unsigned int));

        st_table_entry* ptr;
        st_table* table = (st_table*)reg->name_table;
        for (int i = 0; i < table->num_bins; i++) {
            for (ptr = table->bins[i]; ptr != 0;) {
                NameEntry* e = (NameEntry*)ptr->record;
                if (e->back_num == 1) {
                    len_buff[e->back_ref1 - 1] = e->name_len;
                } else {
                    fprintf(stderr, "ERROR: capture name is wrong\n");
                }
                ptr = ptr->next;
            }
        }
        unsigned int offt = 0;
        for (int i = 0; i < table->num_bins; i++) {
            cpgp_name_oft[i] = offt;
            offt += len_buff[i];
        }
        cpgp_name_oft[table->num_bins] = offt;
        for (int i = 0; i < table->num_bins; i++) {
            for (ptr = table->bins[i]; ptr != 0;) {
                NameEntry* e = (NameEntry*)ptr->record;
                memcpy(cpgp_name_val + cpgp_name_oft[e->back_ref1 - 1], e->name, len_buff[e->back_ref1 - 1]);
                ptr = ptr->next;
            }
        }
        free(len_buff);
    }

    Operation* bp;
    Operation* start = reg->ops;
    Operation* end = reg->ops + reg->ops_used;

    // printf("code-length: %d\n", reg->ops_used);
    bp = start;
    dyn_buff abs_add_tb;
    init_buff(&abs_add_tb);
    dyn_buff str_n_len_tb;
    init_buff(&str_n_len_tb);
    dyn_buff str_n_addr_tb;
    init_buff(&str_n_addr_tb);
    while (bp < end) {
        int pos = bp - start;
        enum OpCode opcode = GET_OPCODE(reg, pos);

        switch (opcode) {
            // no operand
            case OP_END:
            case OP_ANYCHAR:
            case OP_ANYCHAR_STAR:
            case OP_BEGIN_BUF:
            case OP_END_BUF:
            case OP_BEGIN_LINE:
            case OP_END_LINE:
            case OP_FAIL:
            case OP_POP:
                break;
            // for str_N
            case OP_STR_1:
            case OP_STR_2:
            case OP_STR_3:
            case OP_STR_4:
                break;
            case OP_STR_5: {
                // when the len more than 4, separate into multiple instructions
                int add_inst_nm = 2;
                buff_add(&str_n_addr_tb, pos);
                buff_add(&str_n_len_tb, add_inst_nm - 1);
            } break;
            case OP_STR_N: {
                if (bp->exact_n.n > 4) {
                    // when the len more than 4, separate into multiple instructions
                    int add_inst_nm = (bp->exact_n.n + 3) / 4;
                    buff_add(&str_n_addr_tb, pos);
                    buff_add(&str_n_len_tb, add_inst_nm - 1);
                }
            } break;
            // character class, OP_CCLASS_NOT is merged into OP_CCLASS
            case OP_CCLASS:
            case OP_CCLASS_NOT:
            // 1 operand for number of capture group
            case OP_MEM_START:
            case OP_MEM_START_PUSH:
            // case OP_MEM_END_PUSH:
            case OP_MEM_END:
                break;
            // 1 operand for absolute address
            case OP_JUMP:
            case OP_PUSH: {
                buff_add(&abs_add_tb, pos + bp->jump.addr);
            } break;
            // 3 operand, 1 for absolute address, 2 for repetation lower and upper bound
            case OP_REPEAT: {
                buff_add(&abs_add_tb, pos + bp->repeat.addr);
            } break;
            case OP_REPEAT_INC: {
                buff_add(&abs_add_tb, reg->repeat_range[bp->repeat_inc.id].u.pcode - start);
            } break;
            // 2 operand, 1 for absolute address, 1 for compared char
            case OP_PUSH_OR_JUMP_EXACT1: {
                buff_add(&abs_add_tb, pos + bp->push_or_jump_exact1.addr);
            } break;
            // 2 operand, 1 for mark id, 1 for save position flag
            case OP_MARK: {
            } break;
            // 3 operand, 1 for absolute address, 1 for inital step, 1 for remaining step
            case OP_STEP_BACK_START: {
                buff_add(&abs_add_tb, pos + bp->step_back_start.addr);
            } break;
            // 1 operand, 1 for mark-id
            case OP_POP_TO_MARK: {
            } break;
            default:
                break;
        }
        bp++;
    }
    for (unsigned int i = 0; i < abs_add_tb.count; ++i) {
        unsigned int rel_add = 0;
        for (unsigned int j = 0; j < str_n_addr_tb.count; ++j) {
            if (abs_add_tb.data[i] > str_n_addr_tb.data[j]) {
                rel_add += str_n_len_tb.data[j];
            }
        }
        abs_add_tb.data[i] += rel_add;
    }
    // first scan to prepare for the address calculation
    // second scan
    bp = start;

    *cclass_num = 0;
    *instr_num = 0; // reg->ops_used;
    unsigned int index = 0;
    while (bp < end) {
        int pos = bp - start;

        // printf("%4d: ", pos);
        xf_instruction instr;

        // 8-bit opcode
        enum OpCode opcode = GET_OPCODE(reg, pos);
        // 4-bit mode, 1 for using jump address, 0 for using predict address
        unsigned int mode = 0;

        uint16_t oprand_1 = 0;
        uint16_t oprand_2 = 0;
        uint16_t oprand_3 = 0;
        uint8_t len = 0;

        switch (opcode) {
            // no operand
            case OP_END:
            case OP_ANYCHAR_STAR:
            case OP_BEGIN_BUF:
            case OP_END_BUF:
            case OP_BEGIN_LINE:
            case OP_END_LINE:
            case OP_FAIL:
            case OP_POP:
                mode = 0;
                break;
            case OP_ANYCHAR:
                mode = 0;
                len = 1;
                break;
            // variable operand
            // for str_N
            case OP_STR_1:
            case OP_STR_2:
            case OP_STR_3:
            case OP_STR_4:
            case OP_STR_5:
            case OP_STR_N: {
                mode = 0;
                UChar* s;
                int st_pos = 0;
                if (opcode == OP_STR_1) {
                    len = 1;
                } else if (opcode == OP_STR_2) {
                    len = 2;
                } else if (opcode == OP_STR_3) {
                    len = 3;
                } else if (opcode == OP_STR_4) {
                    len = 4;
                } else if (opcode == OP_STR_5) {
                    len = 4;
                    s = &bp->exact.s[st_pos];
                    st_pos += 4;
                    oprand_1 = (s[0] << 8) + s[1];
                    oprand_2 = (s[2] << 8) + s[3];
                    oprand_3 = 0; //(s[4] << 8) + s[5];

                    instr.inst_format.opcode = (unsigned int)OP_STR_N;
                    instr.inst_format.mode_len = (mode << 4) + len;
                    instr.inst_format.oprand_0 = oprand_1;
                    instr.inst_format.oprand_1 = oprand_2;
                    instr.inst_format.oprand_2 = oprand_3;
                    instructions[(*instr_num)++] = instr.d;
                    len = 1;
                } else if (opcode == OP_STR_N) {
                    if (bp->exact_n.n > 4) {
                        // when the len more than 4, separate into multiple instructions
                        int add_inst_nm = (bp->exact_n.n + 3) / 4;
                        for (int i = 0; i < add_inst_nm - 1; ++i) {
                            len = 4;
                            s = &bp->exact_n.s[st_pos];
                            st_pos += 4;
                            oprand_1 = (s[0] << 8) + s[1];
                            oprand_2 = (s[2] << 8) + s[3];
                            oprand_3 = 0; //(s[4] << 8) + s[5];

                            instr.inst_format.opcode = (unsigned int)opcode;
                            instr.inst_format.mode_len = (mode << 4) + len;
                            instr.inst_format.oprand_0 = oprand_1;
                            instr.inst_format.oprand_1 = oprand_2;
                            instr.inst_format.oprand_2 = oprand_3;
                            instructions[(*instr_num)++] = instr.d;
                        }
                        len = bp->exact_n.n - (add_inst_nm - 1) * 4;
#ifdef XF_DEBUG
                        printf("str_n with length %d more than 4\n", bp->exact_n.n);
#endif
                    } else {
                        len = bp->exact_n.n;
                    }
                }
                if (opcode != OP_STR_N) {
                    s = &bp->exact.s[st_pos];
                } else {
                    s = &bp->exact_n.s[st_pos];
                }
                opcode = OP_STR_N;
                oprand_1 = (s[0] << 8);
                if (len > 1) oprand_1 += s[1];
                if (len > 2) oprand_2 = (s[2] << 8);
                if (len > 3) oprand_2 += s[3];
                // if (len > 4) oprand_3 = (s[4] << 8);
                // if (len > 5) oprand_3 += s[5];
            } break;
            // 1 operand for offset of bitset.
            // character class, OP_CCLASS_NOT is merged into OP_CCLASS
            case OP_CCLASS:
            case OP_CCLASS_NOT: {
                mode = 0;
                len = 1;
                oprand_1 = *cclass_num * 8;
                for (int i = 0; i < 8; ++i) {
                    if (opcode == OP_CCLASS_NOT)
                        bitset[oprand_1 + i] = ~bp->cclass.bsp[i];
                    else
                        bitset[oprand_1 + i] = bp->cclass.bsp[i];
                }
                if (opcode == OP_CCLASS_NOT) opcode = OP_CCLASS;
                (*cclass_num)++;
            } break;
            // 1 operand for number of capture group
            case OP_MEM_START:
            case OP_MEM_START_PUSH:
            case OP_MEM_END: {
                mode = 0;
                if (opcode == OP_MEM_START || opcode == OP_MEM_START_PUSH) {
                    oprand_1 = bp->memory_start.num * 2;
                } else {
                    oprand_1 = bp->memory_end.num * 2 + 1;
                }
            } break;
            // 1 operand for absolute address
            case OP_JUMP:
            case OP_PUSH: {
                mode = 0;
                if (opcode == OP_JUMP) {
                    // oprand_1 = bp->jump.addr;
                    oprand_1 = buff_get(&abs_add_tb, index++);
                } else {
                    mode = 1;
                    // oprand_1 = bp->push.addr;
                    oprand_1 = buff_get(&abs_add_tb, index++);
                }
            } break;
            // 3 operand, 1 for absolute address, 2 for repetation lower and upper bound
            case OP_REPEAT:
            case OP_REPEAT_INC: {
                mode = 0;
                // oprand_1 = bp->repeat.addr;
                oprand_1 = buff_get(&abs_add_tb, index++);
                oprand_2 = reg->repeat_range[bp->repeat_inc.id].lower;
                oprand_3 = reg->repeat_range[bp->repeat_inc.id].upper;
            } break;
            // 2 operand, 1 for absolute address, 1 for compared char
            case OP_PUSH_OR_JUMP_EXACT1: {
                mode = 1;
                // len = 1;
                // oprand_1 = bp->push_or_jump_exact1.addr;
                oprand_1 = buff_get(&abs_add_tb, index++);
                oprand_2 = bp->push_or_jump_exact1.c << 8;
            } break;
            // 2 operand, 1 for mark id, 1 for save position flag
            case OP_MARK: {
                mode = 0;
                oprand_1 = bp->mark.id * 2;
                oprand_2 = bp->mark.save_pos;
            } break;
            // 3 operand, 1 for absolute address, 1 for inital step, 1 for remaining step
            case OP_STEP_BACK_START: {
                mode = 0;
                oprand_1 = buff_get(&abs_add_tb, index++);
                oprand_2 = bp->step_back_start.initial;
                oprand_3 = bp->step_back_start.remaining * 2;
            } break;
            // 1 operand, 1 for mark-id
            case OP_POP_TO_MARK: {
                mode = 0;
                oprand_1 = bp->pop_to_mark.id * 2;
            } break;
            default:
#ifdef XF_DEBUG
                fprintf(stderr, "ERROR: undefined code %d\n", (unsigned int)opcode);
#endif
                r = XF_UNSUPPORTED_OPCODE;
                break;
        }
        instr.inst_format.opcode = (unsigned int)opcode;
        instr.inst_format.mode_len = (mode << 4) + len;
        instr.inst_format.oprand_0 = oprand_1;
        instr.inst_format.oprand_1 = oprand_2;
        instr.inst_format.oprand_2 = oprand_3;
        instructions[(*instr_num)++] = instr.d;
        // printf("opcode = %d, mode = %d, len = %d, oprand_1 = %d, oprand_2 = %d, oprand_3 = %d\n", (unsigned int)
        // opcode, mode, len, oprand_1, oprand_2, oprand_3);
        bp++;
    }
    onig_free(reg);
    buff_free(&abs_add_tb);
    buff_free(&str_n_addr_tb);
    buff_free(&str_n_len_tb);
    return r;
}
