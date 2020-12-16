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
/**
 * elf.c
 *
 * Read various sections for DPU kernel from hybrid CPU-DPU ELF executable
 */
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "elf.h"
#include "err.h"

#ifndef PATH_MAX
	#define PATH_MAX 1024
#endif
extern char vitisKernelPath[PATH_MAX];

/*
 * Entry offset for metadata section in unit of 32-bit length
 *
 * Format of Section ".deephi.metadeta.netName" (totally 16*4 bytes)
 *
 * 0 - 32bit: Version ID of DPU target
 * 1 - 32bit: Version ID of DNNC
 * 2 - 32bit: Mode (0x0000 0001 means debug mode)
 * 3 - 32bit: Number of kernel's nodes (include virtual nodes)
 * 4 - 32bit: Size of each node (in byte)
 * 5 - 32bit: Size of kernel's IO memory space (in byte)
 * 6 - 32bit: Mean value of channel 1
 * 7 - 32bit: Mean value of channel 2
 * 8 - 32bit: Mean value of channel 3
 * 9 - 32bit: ABI version
 * 10 - 32bit: Reserved
 * 11 - 32bit: Reserved
 * 12 - 32bit: Reserved
 * 13 - 32bit: Reserved
 * 14 - 32bit: Reserved
 * 15 - 32bit: Reserved
 *
 */
const int OFF_META_DPU_ARCH         = 0;
const int OFF_META_VER_DNNC         = 1;
const int OFF_META_MODE             = 2;
const int OFF_META_NODE_CNT         = 3;
const int OFF_META_TENSOR_SIZE      = 4;
const int OFF_META_KERNEL_IO_SIZE   = 5;
const int OFF_META_KERNEL_MEAN_C1   = 6;
const int OFF_META_KERNEL_MEAN_C2   = 7;
const int OFF_META_KERNEL_MEAN_C3   = 8;
const int OFF_META_ABI_VER          = 9;   //LSB(0~15): minor verion   MSB(16~31): major version
const int OFF_META_DPU_VER          = 10;  //LSB(0~15): dpu target   MSB(16~31): dpu arch type
const int OFF_META_MEM_MODE         = 11;

int elf_class_32   = -1;
int elf_endian_lsb = -1;

char vitisKernelPath[PATH_MAX];

/*
 * copy the dpu_seg info to the kernel`s segment
 */
void elf_copy_segment_kernel(dpu_segment_t *dpu_seg, kernel_t *kernel) {
    elf_copy_segments(&dpu_seg[0].elf_seg, &kernel->elf_meta);
    elf_copy_segments(&dpu_seg[1].elf_seg, &kernel->elf_code);
    elf_copy_segments(&dpu_seg[2].elf_seg, &kernel->elf_bias);
    elf_copy_segments(&dpu_seg[3].elf_seg, &kernel->elf_weight);
    elf_copy_segments(&dpu_seg[4].elf_seg, &kernel->elf_tensor);
    elf_copy_segments(&dpu_seg[5].elf_seg, &kernel->elf_strtab);
    elf_copy_segments(&dpu_seg[6].elf_seg, &kernel->elf_param);
    elf_copy_segments(&dpu_seg[7].elf_seg, &kernel->elf_node_pool);
    elf_copy_segments(&dpu_seg[8].elf_seg, &kernel->elf_conf);
    return;
}

void elf_copy_segments(elf_segment_t *src, elf_segment_t *dst) {
    dst->link = src->link;
    dst->offset = src->offset;
    dst->size = src->size;
    return;
}

/**
 * initial the dpu_segment information with defined names.
 */
void elf_segment_init(dpu_segment_t *dpu_seg) {
    char temp_seg[DPU_SEGMENT_NUM][2][MAX_NAME_LEN] = DPU_SEGMENT_NAMES;
    int seg_idx = 0;
    for (seg_idx = 0; seg_idx < DPU_SEGMENT_NUM; seg_idx++) {
        strcpy(dpu_seg[seg_idx].seg_name, &temp_seg[seg_idx][0][0]);
        strcpy(dpu_seg[seg_idx].log_name, &temp_seg[seg_idx][1][0]);
    }
    return;
}

/**
 * Frees up the internal structures in an elf_t
 */
void elf_free(elf_t* elf)
{
    /* free memory buffer for holding ELF data */
    free(elf->elf_data);
}

/**
 * Gets the hybrid CPU-DPU executable name
 */
int elf_get_myself_name(kernel_t *kernel, int hb)
{
    int size;
    char elf_name[MAX_NAME_LEN];

    if(hb) {
        if((size = readlink("/proc/self/exe", elf_name, MAX_NAME_LEN)) !=-1 ) {
            elf_name[size] = '\0';

            /* hybrid ELF binary name*/
            strcpy(kernel->elf_name, elf_name);

            return N2CUBE_SUCCESS;
        }

        //DPU_FAIL_ON_MSG("Fail to locate hybrid ELF.");
        return ERR_LOCATE_HYBRID_ELF;
    } else {
        int i,len;
        char tmp_name[MAX_NAME_LEN];
        char so_name[MAX_NAME_LEN];
        char* path[4] = {
                "./",
                "/usr/local/lib/",
                "/usr/lib/",
                "/lib/"
        };

        strcpy(tmp_name,kernel->name);
        len = strlen(tmp_name);
        if((len > 2) && (tmp_name[len-2] == '_') && isdigit(tmp_name[len-1])) {
            tmp_name[len-2] = '\0';
        }
        sprintf(so_name,"libdpumodel%s.so",tmp_name);

        sprintf(kernel->elf_name, "%s%s", vitisKernelPath, so_name);
        if(!access(kernel->elf_name,0)) {
            return N2CUBE_SUCCESS;
        }

        for(i=0;i<4;i++) {
            sprintf(kernel->elf_name,"%s%s",path[i],so_name);
            if(!access(kernel->elf_name,0)) {
                return N2CUBE_SUCCESS;
            }
        }
        sprintf(kernel->elf_name,"%s",so_name);
        //DPU_FAIL_ON_MSG("Fail to locate library for DPU kernel: %s.",name);
        return ERR_LOCATE_LIBRARY_DPU_KERNEL;
    }
}

/**
 * Gets the contents of a section by its index
 */
int elf_get_section_data(elf_t* elf, int section, char** out)
{
    uint64_t file_ofs;

    ELF_CHECK_RET(section < elf->num_sections, ERR_ELF_INVALID_SECTION);

    if (ELF_CLASS_32()) {
        file_ofs = elf->shdr.shdr_32[section].sh_offset;
    } else {
        file_ofs = elf->shdr.shdr_64[section].sh_offset;
    }

    *out = elf->elf_data + file_ofs;

    return N2CUBE_SUCCESS;
}

/**
 * Reads a strtab entry
 *  - strtab: The section index of the strtab to read
 *  - ofs: The offset to read into the strtab
 */
int elf_get_strtab_entry(elf_t* elf, int strtab, int ofs, char* out)
{
    char* shstrtab;
    int ret = elf_get_section_data(elf, strtab, &shstrtab);

    ELF_CHECK_RET(ret == 0, ERR_ELF_NO_SHSTRTAB);

    strcpy(out, shstrtab+ofs);

    return N2CUBE_SUCCESS;
}

/**
 * Looks up a section's name in shstrtab by it's index
 */
int elf_get_section_name(elf_t* elf, int section, char* out)
{
    int ret, strtab_ofs;

    ELF_CHECK_RET(section < elf->num_sections, ERR_ELF_INVALID_SECTION);

    if (ELF_CLASS_32()) {
        strtab_ofs = elf->shdr.shdr_32[section].sh_name;
    } else {
        strtab_ofs = elf->shdr.shdr_64[section].sh_name;
    }

    ret = elf_get_strtab_entry(elf, elf->sec_shstrtab, strtab_ofs, out);
    return ret;
}

/*
 * Get DPU elf and segments info from CPU-DPU mixed hybrid binary and fill
 * them into dpu_segment_t and elf_t.
 */
int elf_segment_read(elf_t *elf, dpu_segment_t *dpu_seg) {
    char sec_name[MAX_NAME_LEN] = {0};
    uint64_t sh_size, sh_offset, sh_link;
    int ret = N2CUBE_SUCCESS;
    int i = 0, j = 0;

    for (i = 1; i < elf->num_sections; i++) {
        /* search kernel's specific section name from symbol table */
        /* Report error and exit in case failure while reading section symbols */
        ret = elf_get_section_name(elf, i, sec_name);
        /* Report error and exit in case failure while reading section symbols */
        if (!(ret == N2CUBE_SUCCESS)) {
            return N2CUBE_ERR_KERNEL_LOAD_SECTION;
        }

        sh_link = i;
        if (ELF_CLASS_32()) {
            sh_offset = elf->shdr.shdr_32[i].sh_offset;
            sh_size = elf->shdr.shdr_32[i].sh_size;
        } else {
            sh_offset = elf->shdr.shdr_64[i].sh_offset;
            sh_size = elf->shdr.shdr_64[i].sh_size;
        }

        for(j = 0; j < DPU_SEGMENT_NUM; j++) {
            if(!strcmp(sec_name, dpu_seg[j].seg_name)) {
                dpu_seg[j].elf_seg.offset = (uint32_t)sh_offset;
                dpu_seg[j].elf_seg.size   = (uint32_t)sh_size;
                dpu_seg[j].elf_seg.link   = sh_link;
            }
        }
    }
    return N2CUBE_SUCCESS;
}

/**
 * Reads the elf program headers
 */
INTERNAL int elf_read_phdr(elf_t* elf)
{
    if (ELF_CLASS_32()) {
        elf->phdr.phdr_32 = (Elf32_Phdr*)(elf->elf_data + elf->header.header_32->e_phoff);
        elf->num_segments = elf->header.header_32->e_phnum;
    } else {
        elf->phdr.phdr_64 = (Elf64_Phdr*)(elf->elf_data + elf->header.header_64->e_phoff);
        elf->num_segments = elf->header.header_64->e_phnum;
    }

    return N2CUBE_SUCCESS;
}

/**
 * Reads an elf executable
 */
int elf_read(kernel_t *kernel, elf_t* elf)
{
    struct stat fstat;
    int ret = N2CUBE_SUCCESS;

    ELF_CHECK_RET(!ret, ret);
    /* get the elf file size */
    ret = stat(kernel->elf_name, &fstat);
    ELF_CHECK_RET(ret == 0, ERR_ELF_NO_FILE);
    strcpy(elf->elf_name, kernel->elf_name);

    /* allocate space for storing ELF data */
    elf->elf_data = (char*)malloc(fstat.st_size);
    elf->elf_length = fstat.st_size;

    /* read it from disk */
    elf->elf_fd = fopen(kernel->elf_name, "r");
    ELF_CHECK_RET(elf->elf_fd != 0, ERR_ELF_NO_FILE);
    ret = fread(elf->elf_data, 1, elf->elf_length, elf->elf_fd);
    ELF_CHECK_RET(ret == elf->elf_length, ERR_ELF_NO_FILE);
    /* parse it */
    ret = elf_read_header(elf);
    ELF_CHECK_RET(!ret, ret);
    ret = elf_read_shdr(elf);
    ELF_CHECK_RET(!ret, ret);
    ret = elf_read_phdr(elf);
    ELF_CHECK_RET(!ret, ret);
    fclose(elf->elf_fd);
    return ret;
}

/**
 * Prints the name of each section
 */
void elf_print_sections(elf_t* elf)
{
    int i, ret = 0;
    char sh_name[MAX_NAME_LEN];

    for (i = 1; i < elf->num_sections; i++) {
        ret = elf_get_section_name(elf, i, sh_name);

        if (ret == 0) {
            if (ELF_CLASS_32()) {
                printf("section %d: %s %x\n", i, sh_name, (elf->shdr.shdr_32[i]).sh_offset);
            } else {
                printf("section %d: %s %llx\n", i, sh_name, (elf->shdr.shdr_64[i]).sh_offset);
            }
        } else {
            printf("section %d: error\n", i);
        }
    }
}

/**
 * Prints each segment and their virtual memory mapping
 */
void elf_print_segments(elf_t* elf)
{
    int i;

    for (i = 0; i < elf->num_segments; i++) {
        if (ELF_CLASS_32()) {
            uint32_t end_addr = elf->phdr.phdr_32[i].p_vaddr +
                elf->phdr.phdr_32[i].p_memsz;

            printf("segment %2d: 0x%lx-0x%lx\n", i,
                (unsigned long)(elf->phdr.phdr_32[i].p_vaddr), (unsigned long)end_addr);
        } else {
            uint64_t end_addr = elf->phdr.phdr_64[i].p_vaddr +
                elf->phdr.phdr_64[i].p_memsz;

            printf("segment %4d: 0x%llx-0x%llx\n", i,
                (unsigned long long)(elf->phdr.phdr_64[i].p_vaddr), (unsigned long long)end_addr);
        }
    }
}

/**
 * Prints the name, type, vaddr, and file addr of each symbol
 */
#if 0
void elf_print_symbols(elf_t* elf)
{
    int i, ret = 0;
    char sym_name[MAX_NAME_LEN];
    char sym_type[MAX_NAME_LEN];

    for (i = 1; i < elf->num_symbols; i++) {
        ret = elf_get_symbol_name(elf, i, sym_name);
        ELF_CHECK_RET(!ret, ret);

        Elf32_Sym* symbol = &elf->symtab.symtab_32[i];

        /* fill in a blank name */
        if (strcmp(sym_name, "") == 0) {
            sprintf(sym_name, "no name");
        }

        /* convert the type to a string */
        switch (ELF64_ST_TYPE(symbol->st_info)) {
        case STT_NOTYPE:
            sprintf(sym_type, "no type");
            break;
        case STT_OBJECT:
            sprintf(sym_type, "object");
            break;
        case STT_FUNC:
            sprintf(sym_type, "function");
            break;
        case STT_SECTION:
            sprintf(sym_type, "section");
            break;
        case STT_FILE:
            sprintf(sym_type, "file");
            break;
        }

        /* find it's virtual and file addr */
        uint64_t vaddr = (unsigned long)(symbol->st_value);
        uint64_t faddr = 0;
        elf_map_vaddr_arch32(elf, vaddr, &faddr); /* ignore mapping errors - dynamic linking etc. */

        if (ret == 0) {
            printf("symbol %3d: %-10s %-10s vaddr: %lx faddr: %lx\n", i, sym_type, sym_name, (unsigned long)vaddr, (unsigned long)faddr);
        } else {
            printf("symbol %d: error\n", i);
        }
    }
}
#endif


/**
 * Maps a virtual address to an offset in the elf file
 */
int elf_map_vaddr_arch32(elf_t* elf, Elf32_Addr vaddr, uint64_t* faddr)
{
    int i;

    /* try to find the segment which contains the virtual address */
    for (i = 1; i < elf->num_segments; i++) {
        Elf32_Phdr* segment = &elf->phdr.phdr_32[i];
        Elf32_Addr base_addr = segment->p_vaddr;
        Elf32_Addr lim_addr = segment->p_vaddr + elf->phdr.phdr_32[i].p_memsz;
        if (vaddr > base_addr && vaddr < lim_addr) {
            /* Success! */
            Elf32_Addr offset_addr = vaddr - base_addr;
            *faddr = segment->p_offset + offset_addr;

            return N2CUBE_SUCCESS;
        }
    }

    return ERR_ELF_UNMAPPED;
}

int elf_map_vaddr_arch64(elf_t* elf, Elf64_Addr vaddr, uint64_t* faddr)
{
    int i;

    /* try to find the segment which contains the virtual address */
    for (i = 1; i < elf->num_segments; i++) {
        Elf64_Phdr* segment = &elf->phdr.phdr_64[i];
        Elf64_Addr base_addr = segment->p_vaddr;
        Elf64_Addr lim_addr = segment->p_vaddr + elf->phdr.phdr_64[i].p_memsz;

        if (vaddr > base_addr && vaddr < lim_addr) {
            /* Success! */
            Elf64_Addr offset_addr = vaddr - base_addr;
            *faddr = segment->p_offset + offset_addr;

            return N2CUBE_SUCCESS;
        }
    }

    return ERR_ELF_UNMAPPED;
}

/**
 * Maps a virtual address to a section
 */
int elf_map_vaddr_section_arch32(elf_t* elf, Elf32_Addr vaddr, int* out)
{
    int i;

    /* try to find the section which contains the virtual address */
    for (i = 1; i < elf->num_sections; i++) {
        Elf32_Shdr* section = &elf->shdr.shdr_32[i];
        Elf32_Addr base_addr = section->sh_addr;
        Elf32_Addr lim_addr = section->sh_addr + section->sh_size;

        if (vaddr > base_addr && vaddr < lim_addr) {
            *out = i;  /* sucessfully find it */
            return N2CUBE_SUCCESS;
        }
    }

    return ERR_ELF_UNMAPPED;
}

int elf_map_vaddr_section_arch64(elf_t* elf, Elf64_Addr vaddr, int* out)
{
    int i;

    /* try to find the section which contains the virtual address */
    for (i = 1; i < elf->num_sections; i++) {
        Elf64_Shdr* section = &elf->shdr.shdr_64[i];
        Elf64_Addr base_addr = section->sh_addr;
        Elf64_Addr lim_addr = section->sh_addr + section->sh_size;

        if (vaddr > base_addr && vaddr < lim_addr) {
            *out = i;  /* sucessfully find it */
            return N2CUBE_SUCCESS;
        }
    }

    return ERR_ELF_UNMAPPED;
}


/**
 * Reads the elf header
 */
INTERNAL int elf_read_header(elf_t* elf)
{
    Elf32_Ehdr *header;

    elf->header.header_32 = (Elf32_Ehdr*)elf->elf_data;
    header = (Elf32_Ehdr*)(elf->header.header_32);

    ELF_CHECK_RET(header->e_ident[EI_MAG0] == ELFMAG0, ERR_ELF_INVALID);
    ELF_CHECK_RET(header->e_ident[EI_MAG1] == ELFMAG1, ERR_ELF_INVALID);
    ELF_CHECK_RET(header->e_ident[EI_MAG2] == ELFMAG2, ERR_ELF_INVALID);
    ELF_CHECK_RET(header->e_ident[EI_MAG3] == ELFMAG3, ERR_ELF_INVALID);

    if (header->e_ident[EI_DATA] == ELFDATA2LSB) {
        elf_endian_lsb = 1;
    } else if (header->e_ident[EI_CLASS] == ELFDATA2MSB) {
        elf_endian_lsb = 0;
    } else {
        //DPU_FAIL_ON_MSG("Invalid Hybrid ELF file: %s", elf->elf_name);
        return ERR_INVALID_HYBRID_ELFFILE;
    }

    /* at present only LSB format hybrid binary is supporrted */
    if (ELF_ENDIAN_MSB()) {
        //DPU_FAIL_ON_MSG("MSB format ELF executable is not supported.");
        return ERR_MSB_FORMAT_NOT_SUPPORT;
    }

    if (header->e_ident[EI_CLASS] == ELFCLASS32) {
        elf_class_32 = 1;
    } else if (header->e_ident[EI_CLASS] == ELFCLASS64) {
        elf_class_32 = 0;
        elf->header.header_64 = (Elf64_Ehdr*)elf->elf_data;
    } else {
        //DPU_FAIL_ON_MSG("Invalid Hybrid ELF file: %s", elf->elf_name);
        return ERR_INVALID_HYBRID_ELFFILE;
    }

    return N2CUBE_SUCCESS;
}

/**
 * Reads the elf section headers
 */
INTERNAL int elf_read_shdr(elf_t* elf)
{
    int ret = 0;

    if (ELF_CLASS_32()) {
        elf->shdr.shdr_32 = (Elf32_Shdr*)(elf->elf_data + elf->header.header_32->e_shoff);
        elf->num_sections = elf->header.header_32->e_shnum;
        /* locate the common section indices */
        elf->sec_shstrtab = elf->header.header_32->e_shstrndx;
    } else {
        elf->shdr.shdr_64 = (Elf64_Shdr*)(elf->elf_data + elf->header.header_64->e_shoff);
        elf->num_sections = elf->header.header_64->e_shnum;
        /* locate the common section indices */
        elf->sec_shstrtab = elf->header.header_64->e_shstrndx;
    }

    return N2CUBE_SUCCESS;
}

/**
 * Locates the elf symtab
 */
int elf_read_symtab(elf_t* elf, char *name)
{
    int symtab_sec;
    ELF_CHECK_RET(elf, ERR); // TODO DPU_CHECK_RET
    ELF_CHECK_RET(name, ERR);

    symtab_sec = elf_get_section_by_name(elf, ".symtab");
    ELF_CHECK_RET(symtab_sec >= 0, N2CUBE_ERR_KERNEL_LOAD_SECTION);

    if (ELF_CLASS_32()) {
        elf->symtab.symtab_32 = (Elf32_Sym*)(elf->elf_data +
            elf->shdr.shdr_32[symtab_sec].sh_offset);
        elf->num_symbols = elf->shdr.shdr_32[symtab_sec].sh_size/sizeof(Elf32_Sym);
    } else {
        elf->symtab.symtab_64 = (Elf64_Sym*)(elf->elf_data +
            elf->shdr.shdr_64[symtab_sec].sh_offset);
        elf->num_symbols = elf->shdr.shdr_64[symtab_sec].sh_size/sizeof(Elf64_Sym);
    }

    return N2CUBE_SUCCESS;
}

/**
 * Locates a section's index by name
 */
int elf_get_section_by_name(elf_t* elf, const char* name)
{
    int i;
    char sec_name[MAX_NAME_LEN];
    int ret = 0;

    for (i = 1; i < elf->num_sections; i++) {
        ret = elf_get_section_name(elf, i, sec_name);
        ELF_CHECK_RET(ret == 0, ret);

        if (!strcmp(sec_name, name)) {
            return i;
        }
    }

    return N2CUBE_FAILURE;
}

/**
 * Looks up the file address of a symbol
 */
int elf_get_symbol_faddr(elf_t* elf, int symidx, uint64_t* out)
{
    int ret;

    if (ELF_CLASS_32()) {
        Elf32_Sym* symbol = &elf->symtab.symtab_32[symidx];
        Elf32_Addr vaddr = symbol->st_value;
        ret = elf_map_vaddr_arch32(elf, vaddr, out);
    } else {
        Elf64_Sym* symbol = &elf->symtab.symtab_64[symidx];
        Elf64_Addr vaddr = symbol->st_value;
        ret = elf_map_vaddr_arch64(elf, vaddr, out);
    }

    ELF_CHECK_RET(!ret, ERR_ELF_UNMAPPED);

    return N2CUBE_SUCCESS;
}

/**
 * Looks up the section of a symbol
 */
int elf_get_symbol_section(elf_t* elf, int symidx, int* section)
{
    int ret;

    if (ELF_CLASS_32()) {
        Elf32_Sym* symbol = &elf->symtab.symtab_32[symidx];
        Elf32_Addr vaddr = symbol->st_value;
        ret = elf_map_vaddr_section_arch32(elf, vaddr, section);
    } else {
        Elf64_Sym* symbol = &elf->symtab.symtab_64[symidx];
        Elf64_Addr vaddr = symbol->st_value;
        ret = elf_map_vaddr_section_arch64(elf, vaddr, section);
    }

    ELF_CHECK_RET(ret == 0, ERR_ELF_UNMAPPED);

    return N2CUBE_SUCCESS;
}

/**
 * Looks up a symbol's index by name
 */
int elf_get_symbol_by_name(elf_t* elf, char* name)
{
    char sym_name[MAX_NAME_LEN];
    int i, ret = 0;
    for (i = 0; i < elf->num_symbols; i++) {
        elf_get_symbol_name(elf, i, sym_name); /* ignore errors */
        if (strcmp(sym_name, name) == 0) {
            return i;
        }
    }

    return ERR_ELF_INVALID_SYM_NAME;
}

#ifndef UNIT_TEST
/**
 * Looks up a symbol's name in strtab by it's index
 */
int elf_get_symbol_name(elf_t* elf, int symbol, char* out)
{
    int strtab_ofs, ret;

    ELF_CHECK_RET(symbol < elf->num_symbols, ERR_ELF_INVALID_SYMBOL);

    if (ELF_CLASS_32()) {
       strtab_ofs = elf->symtab.symtab_32[symbol].st_name;
    } else {
       strtab_ofs = elf->symtab.symtab_64[symbol].st_name;
    }

    ret = elf_get_strtab_entry(elf, elf->sec_strtab, strtab_ofs, out);
    ELF_CHECK_RET(ret == 0, ret);

    return N2CUBE_SUCCESS;
}
#endif

/**
 * Gets the faddr of a section by its index
 */
int elf_get_section_faddr(elf_t* elf, int section, uint64_t* faddr)
{
    ELF_CHECK_RET(section < elf->num_sections, ERR_ELF_INVALID_SECTION);

    if (ELF_CLASS_32()) {
        *faddr = elf->shdr.shdr_32[section].sh_offset;
    } else {
        *faddr = elf->shdr.shdr_64[section].sh_offset;
    }

    return N2CUBE_SUCCESS;
}

/**
 * Gets the length of a section by its index
 */
int elf_get_section_len(elf_t* elf, int section, uint64_t* len)
{
    ELF_CHECK_RET(section < elf->num_sections, ERR_ELF_INVALID_SECTION);

    if (ELF_CLASS_32()) {
        *len = elf->shdr.shdr_32[section].sh_size;
    } else {
        *len = elf->shdr.shdr_64[section].sh_size;
    }

    return N2CUBE_SUCCESS;
}


int elf_read_configurable(kernel_t *kernel, elf_t *elf) {
    uint32_t *readBase;

    ELF_CHECK_RET(kernel != NULL && elf != NULL, ERR_ELF_READ_CONFIGURABLE);

    readBase = (uint32_t*)(elf->elf_data + kernel->elf_conf.offset);
    dpu_configurable_t *conf = &(kernel->dpu_conf);
    //basic info, reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->sys.sys_ip_type      = *readBase++;
    conf->sys.sys_regmap_ver   = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->sub_version.ver_target = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->arch.arch_hp_bw     = *readBase++;
    conf->arch.arch_data_bw   = *readBase++;
    conf->arch.arch_img_bkgrp = *readBase++;
    conf->arch.arch_pp        = *readBase++;
    conf->arch.arch_icp       = *readBase++;
    conf->arch.arch_ocp       = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->ram.ram_depth_mean = *readBase++;
    conf->ram.ram_depth_bias = *readBase++;
    conf->ram.ram_depth_wgt  = *readBase++;
    conf->ram.ram_depth_img  = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->load.load_augm_enable       = *readBase++;
    conf->load.load_img_mean_enable   = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->conv.conv_leakyrelu_enable  = *readBase++;
    conf->conv.conv_relu6_enable      = *readBase++;
    conf->conv.conv_wr_parallel       = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->pool.pool_average_enable    = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->elew.elew_parallel          = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->dwcv.dwcv_alu_mode_enable   = *readBase++;
    conf->dwcv.dwcv_relu6_enable      = *readBase++;
    conf->dwcv.dwcv_parallel          = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    conf->misc.misc_wr_parallel = *readBase++;
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved
    readBase++;  //reserved

    return N2CUBE_SUCCESS;
}
