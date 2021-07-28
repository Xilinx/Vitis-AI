/*
 * Copyright (c) 2019 Mujib Haider
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "jfif.h"

void help(void) {
    printf("Usage: jfif [options] <file.jpeg>\n");
    printf("Options:\n");
    printf("  --verbose      Print JSON string to console.\n");
    printf("  --vverbose     Print JSON string to console with scan data.\n");
    printf("  --dump         Dump binary scan data to file.\n");
}

int main(int argc, char *argv[])
{
    uint32_t status; 
    jfif_t *jfif_parse_inst;
    jfif_t *jfif_compose_inst;
    int i, _length;
    FILE *fh;
    //char str[100];
    char zero = 0;

    // Options
    char *filename;
    int verbose  = 0;
    int dump     = 0;

    if (argc < 2) {
        help();
        return EXIT_FAILURE;
    }

    // Parse args
    for (i=1; i<argc; i++) {
        if (strcmp("--verbose", argv[i]) == 0) {
            verbose = 1;
        } else if (strcmp("--vverbose", argv[i]) == 0) {
            verbose = 2;
        } else if (strcmp("--dump", argv[i]) == 0) {
            dump = 1;
        } else if (strcmp("--help", argv[i]) == 0) {
            help();
            return EXIT_SUCCESS;
        } else {
            filename = argv[i];
        }
    }

    //-----------------------------
    // Parser
    //-----------------------------

    // Initialize context
    jfif_parse_inst = jfif_init(NULL);
    if (jfif_parse_inst == 0) {
        printf("Error: Problem initializing parser context\n");
        return EXIT_FAILURE;
    }

    // Read file
    status = jfif_file_read(jfif_parse_inst, filename);
    if (status == 0) {
        printf("Error: Problem reading file: %s\n", filename);
        return EXIT_FAILURE;
    }

    // Run parser
    status = jfif_parse(jfif_parse_inst);
    if (status != 0) {
        printf("Error: Problem parsing file: %s\n", filename);
        return EXIT_FAILURE;
    }

    // Print jfif to console
    jfif_dump_file(jfif_parse_inst, verbose);

    // Dump scan/table data to file
    if (dump) {
        // Scan data
        //sprintf(str, "scan.bin", i);
        fh = fopen("scan.bin", "wb");
        for (i=0; i<jfif_get_scan_num(jfif_parse_inst); i++) {
            //sprintf(str, "scan_%d.bin", i);
            //fh = fopen(str, "wb");
            fwrite(jfif_get_scan_data(jfif_parse_inst, i), jfif_get_scan_size(jfif_parse_inst,i), 1, fh);
            //fclose(fh);
        }
        fclose(fh);
        // DQT table
        fh = fopen("table.bin", "wb");
        for (i=0; i < jfif_get_q_num(jfif_parse_inst); i++) {
            _length = jfif_get_q_length(jfif_parse_inst, i) + 2;

            fwrite(jfif_get_q_offset(jfif_parse_inst, i), _length, 1, fh);

            if (_length%2)
                fwrite(&zero, 1, 1, fh);
        }
        // DHT Table
        for (i=0; i < jfif_get_h_num(jfif_parse_inst); i++) {
            _length = jfif_get_h_length(jfif_parse_inst, i) + 2;

            fwrite(jfif_get_h_offset(jfif_parse_inst, i), _length, 1, fh);

            if (_length%2)
                fwrite(&zero, 1, 1, fh);
        }
        fclose(fh);
    }

    //-----------------------------
    // Composer
    //-----------------------------

    // Initialize context
    jfif_compose_inst = jfif_init(NULL);
    if (jfif_compose_inst == 0) {
        printf("Error: Problem initializing composer context\n");
        return EXIT_FAILURE;
    }

    // Copy the parser to composer context
    jfif_copy(jfif_compose_inst, jfif_parse_inst);

    // Run composer
    status = jfif_compose(jfif_compose_inst);

    if (dump) {
        jfif_file_write(jfif_compose_inst, "output.jpg");
    }


    //-----------------------------
    // Clean-Up
    //-----------------------------
    jfif_destroy(jfif_parse_inst);
    jfif_destroy(jfif_compose_inst);

    return EXIT_SUCCESS;
}


