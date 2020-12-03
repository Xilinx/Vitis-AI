/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _ELF_H_
#define _ELF_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <linux/elf.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include "dpu_types.h"

#define ELF_CHECK_RET(condition, error_id) \
    if (!(condition)) { \
        return (error_id); \
    }

#define ELF_CHECK_RET_2(condition, errorcode, format, ...) \
    if (!(condition)) { \
        return (errorcode); \
    }

#ifndef UNIT_TEST
#define INTERNAL           static
#else
#define INTERNAL
#endif

#define DPU_SEGMENT_NUM        (9)
#define DPU_SEGMENT_NAMES \
    {\
        { ".deephi.metadata.",    "METADATA"},\
        { ".deephi.code.",        "CODE"},\
        { ".deephi.bias.",        "BIAS"},\
        { ".deephi.weights.",     "WEIGHT"},\
        { ".deephi.tensor.",      "TENSOR"},\
        { ".deephi.strtab.",      "STRTAB"},\
        { ".deephi.parameter.",   "PARAMETER"},\
        { ".deephi.node.",        "NODE"},\
        { ".deephi.configurable.", "CONFIGURABLE"}\
    }

/*
* Environmental variable name for setting compilation mode
* value list:
*   0: hybrid compilation mode (default)
*   1: separate mode
*   others: not support now
*/
#define ENV_COMPILATIONMODE "DPU_COMPILATIONMODE"
#define ELF_METADATA_SEGMENT_STR ".deephi.metadata."
#define COMPILATIONMODE_HYBRID 0
#define COMPILATIONMODE_SEPARATE 1

/* variable and macros for ELF CLASS/ENDIAN identification */
extern int elf_class_32;
extern int elf_endian_lsb;

/*path for loading kenrel specified by vitis spec */
extern char vitisKernelPath[];

#define ELF_CLASS_32()    (elf_class_32 == 1)
#define ELF_CLASS_64()    (elf_class_32 == 0)
#define ELF_ENDIAN_LSB()  (elf_endian_lsb == 1)
#define ELF_ENDIAN_MSB()  (elf_endian_lsb == 0)

typedef struct elf {
    char elf_name[MAX_NAME_LEN];
    FILE *elf_fd;
    unsigned char  e_ident[16];
    char  *elf_data;
    size_t elf_length;
    union {
        Elf32_Ehdr* header_32;  /* ELF header for 32-bit ARCH */
        Elf64_Ehdr* header_64;  /* ELF header for 64-bit ARCH */
    } header;

    union {
        Elf32_Shdr* shdr_32;
        Elf64_Shdr* shdr_64;
    } shdr;

    union {
        Elf32_Phdr* phdr_32;
        Elf64_Phdr* phdr_64;
    } phdr;

    union {
        Elf32_Sym* symtab_32;
        Elf64_Sym* symtab_64;
    } symtab;

    int num_segments;    /* number of segments in hybrid executable */
    int num_sections;    /* number of sections in hybrid executable */
    int num_symbols;     /* number of symbols in hybrid executable */

    int sec_strtab;      /* index for ".strtab" string name section */
    int sec_shstrtab;    /* index for string table of section name */
} elf_t;

typedef struct dpu_segment {
    char seg_name[MAX_NAME_LEN];
    char log_name[30];
    elf_segment_t elf_seg;
} dpu_segment_t;
void elf_segment_init(dpu_segment_t *dpu_seg);
void elf_free(elf_t* elf);
int elf_get_myself_name(kernel_t *, int);
int elf_segment_read(elf_t *elf, dpu_segment_t *dpu_seg);
int elf_read(kernel_t *kernel, elf_t* elf);
INTERNAL int elf_read_header(elf_t* elf);
INTERNAL int elf_read_shdr(elf_t* elf);
INTERNAL int elf_read_phdr(elf_t* elf);
int elf_read_symtab(elf_t* elf, char *name);
int elf_get_section_name(elf_t* elf, int section, char* out);
int elf_get_strtab_entry(elf_t* elf, int strtab, int ofs, char* out);
int elf_get_section_data(elf_t* elf, int section, char** out);
void elf_print_sections(elf_t* elf);
void elf_print_symbols(elf_t* elf);
void elf_print_segments(elf_t* elf);
int elf_get_section_by_name(elf_t* elf, const char* name);
int elf_get_section_faddr(elf_t* elf, int section, uint64_t* faddr);
int elf_get_section_len(elf_t* elf, int section, uint64_t* len);
int elf_get_symbol_faddr(elf_t* elf, int symbol, uint64_t* out);
int elf_get_symbol_section(elf_t* elf, int symidx, int* section);
int elf_get_symbol_by_name(elf_t* elf, char* name);
int elf_get_symbol_name(elf_t* elf, int symbol, char* out);
int elf_map_vaddr_arch32(elf_t* elf, Elf32_Addr vaddr, uint64_t* faddr);
int elf_map_vaddr_section_arch32(elf_t* elf, Elf32_Addr vaddr, int* section);
int elf_map_vaddr_section_arch64(elf_t* elf, Elf64_Addr vaddr, int* section);
void elf_copy_segments(elf_segment_t *src, elf_segment_t *dst);
int elf_read_configurable(kernel_t *kernel, elf_t *elf);
void elf_copy_segment_kernel(dpu_segment_t *dpu_seg, kernel_t *kernel);

#ifdef __cplusplus
}
#endif
#endif /* _DPU_ELF_H_ */
