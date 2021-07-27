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
#include <assert.h>
#include "jfif.h"

static int jfif_compose_app_header(jfif_t *ctx);
static int jfif_compose_q_table(jfif_t *ctx, int id);
static int jfif_compose_frame_header(jfif_t *ctx);
static int jfif_compose_h_table(jfif_t *ctx, int id);
static int jfif_compose_scan_header(jfif_t *ctx, int id);

int jfif_compose(jfif_t *ctx)
{
    assert(ctx != NULL);

    ctx->file_size = 0;

    // Step 1: SOI Marker
    ctx->soi_ptr = ctx->file_ptr;
    ctx->file_ptr[0] = 0xFF;
    ctx->file_ptr[1] = 0xD8;
    ctx->file_size += 2;

    // Step 2: JFIF App Header
    ctx->file_size += jfif_compose_app_header(ctx);

    // Step 3: Quantization Table (Luma)
    ctx->file_size += jfif_compose_q_table(ctx, 0);

    // Step 4: Quantization Table (Chroma)
    if (ctx->frame_header.Nf > 1) {
        ctx->file_size += jfif_compose_q_table(ctx, 1);
    }

    // Step 5: Frame Header (SOF0)
    ctx->file_size += jfif_compose_frame_header(ctx);

    // Step 6: DC Huffman Table (Luma)
    ctx->file_size += jfif_compose_h_table(ctx, 0);

    // Step 7: AC Huffman Table (Luma)
    ctx->file_size += jfif_compose_h_table(ctx, 1);

    // Step 8: Y Scan - Skip for interleaved scan
    if (ctx->sos_num > 1 || ctx->frame_header.Nf == 1) {
        ctx->file_size += jfif_compose_scan_header(ctx, 0);
    }

    // Step 9: DC Huffman Table (Chroma) - Skip for grayscale
    if (ctx->frame_header.Nf > 1) {
        ctx->file_size += jfif_compose_h_table(ctx, 2);
    }

    // Step 10: AC Huffman Table (Chroma) - Skip for grayscale
    if (ctx->frame_header.Nf > 1) {
        ctx->file_size += jfif_compose_h_table(ctx, 3);
    }

    // Step 11: Cb Scan - Skip for interleaved scan
    if (ctx->sos_num > 1) {
        ctx->file_size += jfif_compose_scan_header(ctx, 1);
    }

    // Step 12: Cr Scan - Skip for interleaved scan
    if (ctx->sos_num > 1) {
        ctx->file_size += jfif_compose_scan_header(ctx, 2);
    }

    // Step 13: Interleavd Scan - Skip for non-interleaved scan
    if (ctx->sos_num == 1 && ctx->frame_header.Nf > 1) {
        ctx->file_size += jfif_compose_scan_header(ctx, 0);
    }    

    // Step 14: EOI Marker
    ctx->file_ptr[ctx->file_size + 0] = 0xFF;
    ctx->file_ptr[ctx->file_size + 1] = 0xD9;
    ctx->file_size += 2;
    ctx->eoi_ptr = ctx->file_ptr + ctx->file_size;

    return ctx->file_size;
}


int jfif_compose_defaults(jfif_t *ctx, jfif_format_t format, int x, int y, int qf)
{
    assert(ctx != NULL);
    assert(format > JFIF_FORMAT_INVALID && format < JFIF_FORMAT_UNDEF);

    // App header
    ctx->app_header = (jfif_app_header_t) {
        .APP = 0xFFE0,
        .Lp = 16,
        .identifier = {0x4A, 0x46, 0x49, 0x46, 0x00},
        .version = 257,
        .units = 0,
        .Hdensity = 1,
        .Vdensity = 1,
        .HthumbnailA = 0,
        .VthumbnailA = 0,
        .RGB = NULL
    };

    // Frame header
    ctx->frame_header.SOF = 0xFFC0;
    ctx->frame_header.P = 8;
    ctx->frame_header.X = x;
    ctx->frame_header.Y = y;

    switch (format) {
    case JFIF_FORMAT_YUV400_N:
        ctx->frame_header.Lf = 11;
        ctx->frame_header.Nf = 1;
        ctx->frame_header.c[0] = (jfif_frame_component_t) {
            .C = 1, .H = 1, .V = 1, .Tq = 0
        };
        break;
    case JFIF_FORMAT_YUV420_N:
    case JFIF_FORMAT_YUV420_I:
        ctx->frame_header.Lf = 17;
        ctx->frame_header.Nf = 3;
        ctx->frame_header.c[0] = (jfif_frame_component_t) {
            .C = 1, .H = 2, .V = 2, .Tq = 0
        };
        ctx->frame_header.c[1] = (jfif_frame_component_t) {
            .C = 2, .H = 1, .V = 1, .Tq = 1
        };
        ctx->frame_header.c[2] = (jfif_frame_component_t) {
            .C = 3, .H = 1, .V = 1, .Tq = 1
        };
        break;
    case JFIF_FORMAT_YUV422_N:
    case JFIF_FORMAT_YUV422_I:
        ctx->frame_header.Lf = 17;
        ctx->frame_header.Nf = 3;
        ctx->frame_header.c[0] = (jfif_frame_component_t) {
            .C = 1, .H = 2, .V = 2, .Tq = 0
        };
        ctx->frame_header.c[1] = (jfif_frame_component_t) {
            .C = 2, .H = 1, .V = 2, .Tq = 1
        };
        ctx->frame_header.c[2] = (jfif_frame_component_t) {
            .C = 3, .H = 1, .V = 2, .Tq = 1
        };        
        break;
    case JFIF_FORMAT_YUV444_N:
    case JFIF_FORMAT_YUV444_I:    
        ctx->frame_header.Lf = 17;
        ctx->frame_header.Nf = 3;
        ctx->frame_header.c[0] = (jfif_frame_component_t) {
            .C = 1, .H = 1, .V = 1, .Tq = 0
        };
        ctx->frame_header.c[1] = (jfif_frame_component_t) {
            .C = 2, .H = 1, .V = 1, .Tq = 1
        };
        ctx->frame_header.c[2] = (jfif_frame_component_t) {
            .C = 3, .H = 1, .V = 1, .Tq = 1
        };        
        break;
    default:
        return -1;
    }

    // Quantization tables (Luma)
    ctx->dqt_num = 1;
    memcpy(&ctx->q_table[0], &jfif_default_q_table[0],
            sizeof(jfif_quantization_table_t));

    // Quantization tables (Chroma)
    if (ctx->frame_header.Nf > 1) {
        ctx->dqt_num = 2;
        memcpy(&ctx->q_table[1], &jfif_default_q_table[1],
                sizeof(jfif_quantization_table_t));
    }
        
    // Huffman tables (Luma)
    ctx->dht_num = 2;
    memcpy(&ctx->h_table[0], &jfif_default_h_table[0],
            sizeof(jfif_huffman_table_t));
    memcpy(&ctx->h_table[1], &jfif_default_h_table[1],
            sizeof(jfif_huffman_table_t));

    // Huffman tables (Chroma)
    if (ctx->frame_header.Nf > 1) {
        ctx->dht_num = 4;
        memcpy(&ctx->h_table[2], &jfif_default_h_table[2],
                sizeof(jfif_huffman_table_t));
        memcpy(&ctx->h_table[3], &jfif_default_h_table[3],
                sizeof(jfif_huffman_table_t));
    }

    return 0;
}


int jfif_compose_scan(jfif_t *ctx, int cmps, unsigned int size, unsigned char *scan)
{
    assert(ctx != NULL);
    assert(scan != NULL);
    assert(cmps > 0);
    assert(size > 0);

    // Scan header
    if (cmps == 1) {
        ctx->scan_header[ctx->sos_num] = (jfif_scan_header_t) {
            .SOS = 0xFFDA,
            .Ls = 8,
            .Ns = cmps,
            .c[0] = (jfif_scan_component_t) {
               .Cs = ctx->sos_num + 1,
               .Td = (ctx->sos_num == 0) ? 0 : 1,
               .Ta = (ctx->sos_num == 0) ? 0 : 1 
            },
            .Ss = 0,
            .Se = 63,
            .Ah = 0,
            .Al = 0
        };
    } else {
        ctx->scan_header[ctx->sos_num] = (jfif_scan_header_t) {
            .SOS = 0xFFDA,
            .Ls = 12,
            .Ns = cmps,
            .c[0] = (jfif_scan_component_t) {
               .Cs = 1,
               .Td = 0,
               .Ta = 0
            },
            .c[1] = (jfif_scan_component_t) {
               .Cs = 2,
               .Td = 1,
               .Ta = 1
            },
            .c[2] = (jfif_scan_component_t) {
               .Cs = 3,
               .Td = 1,
               .Ta = 1
            },
            .Ss = 0,
            .Se = 63,
            .Ah = 0,
            .Al = 0
        };        
    }

    // Scan data
    ctx->scan_size[ctx->sos_num] = size;
    ctx->scan_ptr[ctx->sos_num] = scan;

    // Increment number of scans
    ctx->sos_num++;

    return 0;
}


int jfif_set_scan_num(jfif_t *ctx, int num)
{
    assert(ctx != NULL);

    ctx->sos_num = num;

    return 0;
}


int jfif_set_scan_components(jfif_t *ctx, int id, int num)
{
    assert(ctx != NULL);
    assert(id < ctx->sos_num);

    ctx->scan_header[id].Ns = num;

    return 0;
}


int jfif_set_scan_size(jfif_t *ctx, int id, unsigned int size)
{
    assert(ctx != NULL);
    assert(id < ctx->sos_num);

    ctx->scan_size[id] = size;

    return 0;
}


int jfif_set_scan_data(jfif_t *ctx, int id, unsigned char *scan)
{
    assert(ctx != NULL);
    assert(scan != NULL);
    assert(id < ctx->sos_num);

    ctx->scan_ptr[id] = scan;

    return 0;
}


static int jfif_compose_app_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;
    int tn_size = 3*ctx->app_header.HthumbnailA*ctx->app_header.VthumbnailA;

    ctx->app_ptr = ctx->file_ptr + ctx->file_size;

    SHORT_SWAP(*((uint16_t *)(ctx->app_ptr + offset)), ctx->app_header.APP);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->app_ptr + offset)), ctx->app_header.Lp);
    offset += 2;
    for (i=0; i<5; i++) {
        *(ctx->app_ptr + offset) = ctx->app_header.identifier[i];
        offset += 1;
    }
    SHORT_SWAP(*((uint16_t *)(ctx->app_ptr + offset)), ctx->app_header.version);
    offset += 2;
    *(ctx->app_ptr + offset) = ctx->app_header.units;
    offset += 1;
    SHORT_SWAP(*((uint16_t *)(ctx->app_ptr + offset)), ctx->app_header.Hdensity);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->app_ptr + offset)), ctx->app_header.Vdensity);
    offset += 2;
    *(ctx->app_ptr + offset) = ctx->app_header.HthumbnailA;
    offset += 1;
    *(ctx->app_ptr + offset) = ctx->app_header.VthumbnailA;
    offset += 1;
    memcpy(ctx->app_ptr + offset, ctx->app_header.RGB, tn_size);
    offset += tn_size;

    return offset;
}


static int jfif_compose_q_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;

    ctx->dqt_ptr[id] = ctx->file_ptr + ctx->file_size;

    SHORT_SWAP(*((uint16_t *)(ctx->dqt_ptr[id] + offset)), ctx->q_table[id].DQT);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->dqt_ptr[id] + offset)), ctx->q_table[id].Lq);
    offset += 2;
    *(ctx->dqt_ptr[id] + offset) = ctx->q_table[id].Pq << 4;
    *(ctx->dqt_ptr[id] + offset) |= ctx->q_table[id].Tq;
    offset += 1;
    if (!ctx->q_table[id].Pq) {
        memcpy(ctx->dqt_ptr[id] + offset, ctx->q_table[id].Qk, 64);
        offset += 64;
    } else {
        memcpy(ctx->dqt_ptr[id] + offset, ctx->q_table[id].Qk, 128);
        offset += 128;
    }

    return offset;
}


static int jfif_compose_frame_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;

    ctx->sof_ptr = ctx->file_ptr + ctx->file_size;

    SHORT_SWAP(*((uint16_t *)(ctx->sof_ptr + offset)), ctx->frame_header.SOF);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->sof_ptr + offset)), ctx->frame_header.Lf);
    offset += 2;
    *(ctx->sof_ptr + offset) = ctx->frame_header.P;
    offset += 1;
    SHORT_SWAP(*((uint16_t *)(ctx->sof_ptr + offset)), ctx->frame_header.Y);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->sof_ptr + offset)), ctx->frame_header.X);
    offset += 2;
    *(ctx->sof_ptr + offset) = ctx->frame_header.Nf;
    offset += 1;

    for (i=0; i<ctx->frame_header.Nf; i++) {
        *(ctx->sof_ptr + offset) = ctx->frame_header.c[i].C;
        offset += 1;
        *(ctx->sof_ptr + offset) = ctx->frame_header.c[i].H << 4;
        *(ctx->sof_ptr + offset) |= ctx->frame_header.c[i].V;
        offset += 1;
        *(ctx->sof_ptr + offset) = ctx->frame_header.c[i].Tq;
        offset += 1;
    }

    return offset;
}


static int jfif_compose_h_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;
    int i, j;

    ctx->dht_ptr[id] = ctx->file_ptr + ctx->file_size;

    SHORT_SWAP(*((uint16_t *)(ctx->dht_ptr[id] + offset)), ctx->h_table[id].DHT);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->dht_ptr[id] + offset)), ctx->h_table[id].Lh);
    offset += 2;
    *(ctx->dht_ptr[id] + offset) = ctx->h_table[id].Tc << 4;
    *(ctx->dht_ptr[id] + offset) |= ctx->h_table[id].Th;
    offset += 1;
    for (i=0; i<16; i++) {
        *(ctx->dht_ptr[id] + offset) = ctx->h_table[id].L[i];
        offset++;
    }
    for (i=0; i<16; i++) {
        for (j=0; j<ctx->h_table[id].L[i]; j++) {
            *(ctx->dht_ptr[id] + offset) = ctx->h_table[id].V[i][j];
            offset++;
        }
    }

    return offset;
}


static int jfif_compose_scan_header(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;

    ctx->sos_ptr[id] = ctx->file_ptr + ctx->file_size;

    SHORT_SWAP(*((uint16_t *)(ctx->sos_ptr[id] + offset)), ctx->scan_header[id].SOS);
    offset += 2;
    SHORT_SWAP(*((uint16_t *)(ctx->sos_ptr[id] + offset)), ctx->scan_header[id].Ls);
    offset += 2;
    *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].Ns;
    offset += 1;

    for (i=0; i<ctx->scan_header[id].Ns; i++) {
        *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].c[i].Cs;
        offset += 1;
        *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].c[i].Td << 4;
        *(ctx->sos_ptr[id] + offset) |= ctx->scan_header[id].c[i].Ta;
        offset += 1;
    }

    *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].Ss;
    offset += 1;
    *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].Se;
    offset += 1;
    *(ctx->sos_ptr[id] + offset) = ctx->scan_header[id].Ah << 4;
    *(ctx->sos_ptr[id] + offset) |= ctx->scan_header[id].Al;
    offset += 1;

    memcpy(ctx->sos_ptr[id] + offset, ctx->scan_ptr[id], ctx->scan_size[id]);
    offset += ctx->scan_size[id];

    return offset;
}
