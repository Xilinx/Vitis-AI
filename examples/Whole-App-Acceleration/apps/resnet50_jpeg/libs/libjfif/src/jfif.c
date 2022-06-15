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

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "jfif.h"


jfif_t *jfif_init(unsigned char *FilePtr)
{
    jfif_t *ctx;

    ctx = (jfif_t *)malloc(sizeof(jfif_t));
    memset(ctx, 0, sizeof(jfif_t));

    if (FilePtr != NULL) {

        ctx->file_ptr = FilePtr;

        ctx->parser_full_file = 0;

    } else {

        ctx->file_ptr = (unsigned char *)malloc(JFIF_MAX_FILE_SIZE);

        ctx->parser_full_file = 1;
    }

    assert(ctx != NULL);

    return ctx;
}


void jfif_reset(jfif_t *ctx)
{
    assert(ctx != NULL);

    memset(ctx, 0, sizeof(jfif_t));
}


void jfif_destroy(jfif_t *ctx)
{
    assert(ctx != NULL);

    free(ctx->file_ptr);
    free(ctx);
}


void jfif_copy(jfif_t *dst, jfif_t *src)
{
    assert(src != NULL);
    assert(dst != NULL);

    unsigned char *file_ptr;
    int i;

    // Save the dst file pointer first
    file_ptr = dst->file_ptr;

    // Shallow copy
    memcpy(dst, src, sizeof(jfif_t));

    // Copy the scan pointer
    for (i=0; i<src->sos_num; i++) {
        dst->scan_ptr[i] = src->scan_ptr[i];
        dst->scan_size[i] = src->scan_size[i];
    }

    // Assign back the file pointer
    dst->file_ptr = file_ptr;
    dst->file_size = 0;
}


unsigned int jfif_file_read(jfif_t *ctx, const char *filename)
{
    assert(ctx != NULL);

    FILE *fh;

    fh = fopen(filename, "rb");
    if (fh == NULL)
        return 0;

    fseek(fh, 0, SEEK_END);
    ctx->file_size = ftell(fh);

    if (ctx->file_size > JFIF_MAX_FILE_SIZE)
        return 0;

    fseek(fh, 0, SEEK_SET);
    fread(ctx->file_ptr, ctx->file_size, 1, fh);
    fclose(fh);

    return ctx->file_size;
}


unsigned int jfif_file_write(jfif_t *ctx, const char *filename)
{
    assert(ctx != NULL);

    FILE *fh;

    fh = fopen(filename, "wb");
    if (fh == NULL)
        return 0;

    fwrite(ctx->file_ptr, ctx->file_size, 1, fh);
    fclose(fh);

    return ctx->file_size;
}
