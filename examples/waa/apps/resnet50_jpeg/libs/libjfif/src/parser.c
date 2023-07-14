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


static int jfif_parse_app_header(jfif_t *ctx);
static int jfif_parse_frame_header(jfif_t *ctx);
static int jfif_parse_scan_header(jfif_t *ctx, int id);
static int jfif_parse_q_table(jfif_t *ctx, int id);
static int jfif_parse_h_table(jfif_t *ctx, int id);
static int jfif_round(int value, int round);


int jfif_parse(jfif_t *ctx)
{
    assert(ctx != NULL);

    uint32_t offset=0;
    uint16_t word = 0;
    int      pmarker;
    uint16_t APPx_Lp;

    while ((offset < ctx->file_size) && ((word != JFIF_MARKER_SOF0) || (ctx->parser_full_file == 1))) {

        word = *(uint16_t *)(ctx->file_ptr + offset);

        switch(word) {

        case JFIF_MARKER_SOI:
            ctx->soi_ptr = ctx->file_ptr + offset;
            offset += 2;
            pmarker = JFIF_MARKER_SOI;
            // printf("SOI done\n");
            break;

        case JFIF_MARKER_APP0:
            ctx->app_ptr = ctx->file_ptr + offset;
            offset += jfif_parse_app_header(ctx);
            pmarker = JFIF_MARKER_APP0;
            // printf("APP0 done\n");
            break;

        case JFIF_MARKER_DQT:
            if (pmarker == JFIF_MARKER_SOS) {
                ctx->scan_size[ctx->sos_num-1] = (ctx->file_ptr + offset) - ctx->scan_ptr[ctx->sos_num-1];
            }

            ctx->dqt_ptr[ctx->dqt_num] = ctx->file_ptr + offset;
            offset += jfif_parse_q_table(ctx, ctx->dqt_num);
            ctx->dqt_num++;
            pmarker = JFIF_MARKER_DQT;
            // printf("DQT done\n");
            break;

        case JFIF_MARKER_DHT:
            if (pmarker == JFIF_MARKER_SOS) {
                ctx->scan_size[ctx->sos_num-1] = (ctx->file_ptr + offset) - ctx->scan_ptr[ctx->sos_num-1];
            }

            ctx->dht_ptr[ctx->dht_num] = ctx->file_ptr + offset;
            offset += jfif_parse_h_table(ctx, ctx->dht_num);
            ctx->dht_num++;
            pmarker = JFIF_MARKER_DHT;
            // printf("DHT done\n");
            break;

        case JFIF_MARKER_SOF0:
            ctx->sof_ptr = ctx->file_ptr + offset;
            offset += jfif_parse_frame_header(ctx);
            pmarker = JFIF_MARKER_SOF0;
            // printf("SOF0 done\n");
            break;

        case JFIF_MARKER_SOF1:
        case JFIF_MARKER_SOF2:
        case JFIF_MARKER_SOF3:
        case JFIF_MARKER_SOF5:
        case JFIF_MARKER_SOF6:
        case JFIF_MARKER_SOF7:
        case JFIF_MARKER_SOF9:
        case JFIF_MARKER_SOF10:
        case JFIF_MARKER_SOF11:
        case JFIF_MARKER_SOF13:
        case JFIF_MARKER_SOF14:
        case JFIF_MARKER_SOF15:
            // printf("SOF* done\n");
            printf("[Parser][  ERROR] : Found unsupported SOF marker (%04X).\n", word);
            return -1;

        case JFIF_MARKER_APP1:
        case JFIF_MARKER_APP2:
        case JFIF_MARKER_APP3:
        case JFIF_MARKER_APP4:
        case JFIF_MARKER_APP5:
        case JFIF_MARKER_APP6:
        case JFIF_MARKER_APP7:
        case JFIF_MARKER_APP8:
        case JFIF_MARKER_APP9:
        case JFIF_MARKER_APP10:
        case JFIF_MARKER_APP11:
        case JFIF_MARKER_APP12:
        case JFIF_MARKER_APP13:
        case JFIF_MARKER_APP14:
        case JFIF_MARKER_APP15:
            offset += 2;
            SHORT_SWAP(APPx_Lp, *(uint16_t *)(ctx->file_ptr + offset));
            //printf("[Parser][WARNING] : Found non APP0 marker (skipped %d bytes by reading next word).\n", APPx_Lp);
            offset += APPx_Lp;
            break;

        case JFIF_MARKER_SOS:
            if (pmarker == JFIF_MARKER_SOS) {
                ctx->scan_size[ctx->sos_num-1] = (ctx->file_ptr + offset) - ctx->scan_ptr[ctx->sos_num-1];
            }

            ctx->sos_ptr[ctx->sos_num] = ctx->file_ptr + offset;
            offset += jfif_parse_scan_header(ctx, ctx->sos_num);
            ctx->sos_num++;
            pmarker = JFIF_MARKER_SOS;
            // printf("SOS done\n");
            break;

        case JFIF_MARKER_EOI:
            if (pmarker == JFIF_MARKER_SOS) {
                ctx->scan_size[ctx->sos_num-1] = (ctx->file_ptr + offset) - ctx->scan_ptr[ctx->sos_num-1];
            }

            ctx->eoi_ptr = ctx->file_ptr + offset;
            offset += 2;
            pmarker = JFIF_MARKER_EOI;
            // printf("EOI done\n");
            break;

        default:
            offset += 1;
            // printf("Found Unknown Marker (%04X)\n", word);
            break;
        }
    }

    // Error checking
    // Check that APP0 marker is set
    if ((ctx->app_ptr == 0) && (ctx->parser_full_file == 1))
        return -1;

    return 0;
}


int jfif_get_frame_components(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->frame_header.Nf;
}


int jfif_get_frame_h_factor(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->frame_header.Nf);

    return ctx->frame_header.c[id].H;
}


int jfif_get_frame_v_factor(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->frame_header.Nf);

    return ctx->frame_header.c[id].V;
}

int jfif_get_frame_x(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->frame_header.X;
}


int jfif_get_frame_y(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->frame_header.Y;
}


int jfif_get_frame_width(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->frame_header.Nf);

    int i;
    int width    = 0;
    int max      = 0;
    int ratio    = 0;
    int scale    = 0;

    for (i=0; i<ctx->frame_header.Nf; i++) {
        if (max < ctx->frame_header.c[i].H) {
            max = ctx->frame_header.c[i].H;
        }
    }

    ratio   = max / ctx->frame_header.c[id].H;

    // Interleaved scan
    //if (jfif_get_scan_components(ctx, 0) > 1) {
        width   = jfif_round(ctx->frame_header.X, ctx->frame_header.c[id].H*8);
        scale   = jfif_round(width / ratio, ctx->frame_header.c[id].H*8);
    //}

    //// Non-Interleaved scans
    //else {
    //    width   = jfif_round(ctx->frame_header.X, 8);
    //    scale   = jfif_round(width / ratio, 8);
    //}

    return scale;
}


int jfif_get_frame_height(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->frame_header.Nf);

    int i;
    int height  = 0;
    int max     = 0;
    int ratio   = 0;
    int scale   = 0;

    for (i=0; i<ctx->frame_header.Nf; i++) {
        if (max < ctx->frame_header.c[i].V) {
            max = ctx->frame_header.c[i].V;
        }
    }

    ratio   = max / ctx->frame_header.c[id].V;

    //// Interleaved scan
    //if (jfif_get_scan_components(ctx, 0) > 1) {
        height  = jfif_round(ctx->frame_header.Y, ctx->frame_header.c[id].V*8);
        scale   = jfif_round(height / ratio, ctx->frame_header.c[id].V*8);
    //}

    //// Non-Interleaved scans
    //else {
    //    height  = jfif_round(ctx->frame_header.Y, 8);
    //    scale   = jfif_round(height / ratio, 8);
    //}

    return scale;
}


int jfif_get_scan_num(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->sos_num;
}


int jfif_get_scan_components(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->sos_num);

    return ctx->scan_header[id].Ns;
}


int jfif_get_scan_start_component(jfif_t *ctx, int id)
{
    return ctx->scan_header[id].c[0].Cs;
}


int jfif_get_scan_end_component(jfif_t *ctx, int id)
{
    return ctx->scan_header[id].c[ctx->scan_header[id].Ns-1].Cs;
}


unsigned int jfif_get_scan_size(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->sos_num);

    return ctx->scan_size[id];
}


unsigned char *jfif_get_scan_data(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->sos_num);

    return ctx->scan_ptr[id];
}


int jfif_get_q_num(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->dqt_num;
}


unsigned char *jfif_get_q_offset(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dqt_num);

    return ctx->dqt_ptr[id];
}


int jfif_get_q_length(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dqt_num);

    return ctx->q_table[id].Lq;
}


int jfif_get_q_precision(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dqt_num);

    if (!ctx->q_table[id].Pq)
        return 8;
    else
        return 16;
}


unsigned char *jfif_get_q_data(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dqt_num);

    return ctx->q_table[id].Qk;
}


int jfif_get_h_num(jfif_t *ctx)
{
    assert(ctx != NULL);

    return ctx->dht_num;
}


unsigned char *jfif_get_h_offset(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dht_num);

    return ctx->dht_ptr[id];
}


int jfif_get_h_length(jfif_t *ctx, int id)
{
    assert(ctx != NULL);
    assert(id < ctx->dht_num);

    return ctx->h_table[id].Lh;
}


void jfif_dump_app_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int i;

    printf("\"app_header\": {\n");
    printf("    \"APP\": %d,\n", ctx->app_header.APP);
    printf("    \"Lp\": %d,\n", ctx->app_header.Lp);
    printf("    \"identifier\": [");
    for (i=0; i<5; i++) {
        if (i) printf(",");
        printf(" %d", ctx->app_header.identifier[i]);
    }
    printf(" ],\n");
    printf("    \"version\": %d,\n", ctx->app_header.version);
    printf("    \"units\": %d,\n", ctx->app_header.units);
    printf("    \"Hdensity\": %d,\n", ctx->app_header.Hdensity);
    printf("    \"Vdensity\": %d,\n", ctx->app_header.Vdensity);
    printf("    \"HthumbnailA\": %d,\n", ctx->app_header.HthumbnailA);
    printf("    \"VthumbnailA\": %d\n", ctx->app_header.VthumbnailA);
    printf("}\n");
}


void jfif_dump_frame_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int i;

    printf("\"frame_header\": {\n");
    printf("    \"SOF\": %d,\n", ctx->frame_header.SOF);
    printf("    \"Lf\": %d,\n", ctx->frame_header.Lf);
    printf("    \"P\": %d,\n", ctx->frame_header.P);
    printf("    \"Y\": %d,\n", ctx->frame_header.Y);
    printf("    \"X\": %d,\n", ctx->frame_header.X);
    printf("    \"Nf\": %d,\n", ctx->frame_header.Nf);
    printf("    \"Component\": [\n");
    for (i=0; i<ctx->frame_header.Nf; i++) {
        if (i) printf(",\n");
        printf("        { ");
        printf(" \"C\": %d,", ctx->frame_header.c[i].C);
        printf(" \"H\": %d,", ctx->frame_header.c[i].H);
        printf(" \"V\": %d,", ctx->frame_header.c[i].V);
        printf(" \"Tq\": %d", ctx->frame_header.c[i].Tq);
        printf(" }");
    }
    printf("\n    ],\n");
    printf("    \"User\": [\n");
    for (i=0; i<ctx->frame_header.Nf; i++) {
        if (i) printf(",\n");
        printf("        { ");
        printf(" \"Y\": %d,", jfif_get_frame_height(ctx, i));
        printf(" \"X\": %d", jfif_get_frame_width(ctx, i));
        printf(" }");
    }
    printf("\n    ]\n");
    printf("}\n");
}


void jfif_dump_scan_header(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int i;

    printf("\"scan_header\": {\n");
    printf("    \"SOS\": %d,\n", ctx->scan_header[id].SOS);
    printf("    \"Ls\": %d,\n", ctx->scan_header[id].Ls);
    printf("    \"Ns\": %d,\n", ctx->scan_header[id].Ns);
    printf("    \"Ss\": %d,\n", ctx->scan_header[id].Ss);
    printf("    \"Se\": %d,\n", ctx->scan_header[id].Se);
    printf("    \"Ah\": %d,\n", ctx->scan_header[id].Ah);
    printf("    \"Al\": %d,\n", ctx->scan_header[id].Al);
    printf("    \"Component\": [\n");
    for (i=0; i<ctx->scan_header[id].Ns; i++) {
        if (i) printf(",\n");
        printf("        {");
        printf(" \"Cs\": %d,", ctx->scan_header[id].c[i].Cs);
        printf(" \"Td\": %d,", ctx->scan_header[id].c[i].Td);
        printf(" \"Ta\": %d",  ctx->scan_header[id].c[i].Ta);
        printf(" }");
    }
    printf("\n    ]\n");
    printf("}\n");
}


void jfif_dump_scan_data(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int i;

    printf("\"scan_data\": {\n");
    printf("    \"size\": %d,\n", ctx->scan_size[id]);
    printf("    \"data\": [");
    for (i=0; i<ctx->scan_size[id]; i++) {
        if (i) printf(",");
        printf(" %d", *(ctx->scan_ptr[id]+i));
    }
    printf(" ]\n");
    printf("}\n");
}


void jfif_dump_q_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int row;
    int col;
    int val;

    printf("\"quantization_table_%d\": {\n", id);
    printf("    \"DQT\": %d,\n", ctx->q_table[id].DQT);
    printf("    \"Lq\": %d,\n", ctx->q_table[id].Lq);
    printf("    \"Pq\": %d,\n", ctx->q_table[id].Pq);
    printf("    \"Tq\": %d,\n", ctx->q_table[id].Tq);
    printf("    \"Qk\": [");
    for (row=0; row<8; row++) {
        for (col=0; col<8; col++) {
            if (row || col) printf(", ");
            if (!ctx->q_table[id].Pq)
                val = ctx->q_table[id].Qk[row*8+col];
            else
                SHORT_SWAP(val, *(uint16_t *)(ctx->q_table[id].Qk + (row*16+col*2)));
            printf("%d", val);
        }
    }
    printf("]\n");
    printf("}\n");
}

void jfif_dump_h_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int i,j;

    printf("\"huffman_table_%d\": {\n", id);
    printf("    \"DHT\": %d,\n", ctx->h_table[id].DHT);
    printf("    \"Lh\": %d,\n", ctx->h_table[id].Lh);
    printf("    \"Tc\": %d,\n", ctx->h_table[id].Tc);
    printf("    \"Th\": %d,\n", ctx->h_table[id].Th);
    printf("    \"Code\": [\n");
    for (i=0; i<16; i++) {
        if (i) printf(",\n");
        printf("        { ");
        printf("\"L\": %d, ", ctx->h_table[id].L[i]);
        printf("\"V\": [");
        for (j=0; j<ctx->h_table[id].L[i]; j++) {
            if (j) printf(", ");
            printf("%d", ctx->h_table[id].V[i][j]);
        }
        printf("]");
        printf(" }");
    }
    printf("\n    ]\n");
    printf("}\n");
}

void jfif_dump_file(jfif_t *ctx, int loglevel)
{
    assert(ctx != NULL);

    int i;

    if (loglevel > 0) {
        printf("{\n");
        jfif_dump_app_header(ctx);
        printf(",\n");
        jfif_dump_frame_header(ctx);

        if (ctx->parser_full_file) {
            if (loglevel > 1) {
                for (i=0; i<jfif_get_q_num(ctx); i++) {
                    printf(",\n");
                    jfif_dump_q_table(ctx, i);
                }

                for (i=0; i<jfif_get_h_num(ctx); i++) {
                    printf(",\n");
                    jfif_dump_h_table(ctx, i);
                }
            }

            for (i=0; i<jfif_get_scan_num(ctx); i++) {
                printf(",\n");
                jfif_dump_scan_header(ctx, i);

                if (loglevel > 1) {
                    printf(",\n");
                    jfif_dump_scan_data(ctx, i);
                }
            }
        }
        printf("\n}\n");
    }
}

static int jfif_parse_app_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;
    uint8_t jfif_id[5] = {0x4A, 0x46, 0x49, 0x46, 0x00};

    // If this is an extension header, skip it.
    if (memcmp(jfif_id, ctx->app_ptr+4, 5) != 0) {
        SHORT_SWAP(offset, *(uint16_t *)(ctx->app_ptr + 2));
        return (offset + 2);
    }

    SHORT_SWAP(ctx->app_header.APP, *((uint16_t *)(ctx->app_ptr + offset)));
    offset += 2;
    SHORT_SWAP(ctx->app_header.Lp, *((uint16_t *)(ctx->app_ptr + offset)));
    offset += 2;
    for (i=0; i<5; i++) {
        ctx->app_header.identifier[i] = *(ctx->app_ptr + offset);
        offset += 1;
    }
    SHORT_SWAP(ctx->app_header.version, *((uint16_t *)(ctx->app_ptr + offset)));
    offset += 2;
    ctx->app_header.units = *(ctx->app_ptr + offset);
    offset += 1;
    SHORT_SWAP(ctx->app_header.Hdensity, *((uint16_t *)(ctx->app_ptr + offset)));
    offset += 2;
    SHORT_SWAP(ctx->app_header.Vdensity, *((uint16_t *)(ctx->app_ptr + offset)));
    offset += 2;
    ctx->app_header.HthumbnailA = *(ctx->app_ptr + offset);
    offset += 1;
    ctx->app_header.VthumbnailA = *(ctx->app_ptr + offset);
    offset += 1;
    ctx->app_header.RGB = ctx->app_ptr + offset;
    offset += 3*ctx->app_header.HthumbnailA*ctx->app_header.VthumbnailA;

    return offset;
}


static int jfif_parse_frame_header(jfif_t *ctx)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;

    SHORT_SWAP(ctx->frame_header.SOF, *((uint16_t *)(ctx->sof_ptr + offset)));
    offset += 2;
    SHORT_SWAP(ctx->frame_header.Lf,  *((uint16_t *)(ctx->sof_ptr + offset)));
    offset += 2;
    ctx->frame_header.P = *(ctx->sof_ptr + offset);
    offset += 1;
    SHORT_SWAP(ctx->frame_header.Y,  *((uint16_t *)(ctx->sof_ptr + offset)));
    offset += 2;
    SHORT_SWAP(ctx->frame_header.X,  *((uint16_t *)(ctx->sof_ptr + offset)));
    offset += 2;
    ctx->frame_header.Nf = *(ctx->sof_ptr + offset);
    offset += 1;

    for (i=0; i<ctx->frame_header.Nf; i++) {
        ctx->frame_header.c[i].C  = *(ctx->sof_ptr + offset);
        offset += 1;
        ctx->frame_header.c[i].H  = *(ctx->sof_ptr + offset) >> 4;
        ctx->frame_header.c[i].V  = *(ctx->sof_ptr + offset);
        offset += 1;
        ctx->frame_header.c[i].Tq = *(ctx->sof_ptr + offset);
        offset += 1;
    }

    return offset;
}


static int jfif_parse_scan_header(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;
    int i;

    SHORT_SWAP(ctx->scan_header[id].SOS, *((uint16_t *)(ctx->sos_ptr[id] + offset)));
    offset += 2;
    SHORT_SWAP(ctx->scan_header[id].Ls, *((uint16_t *)(ctx->sos_ptr[id] + offset)));
    offset += 2;
    ctx->scan_header[id].Ns = *(ctx->sos_ptr[id] + offset);
    offset += 1;

    for (i=0; i<ctx->scan_header[id].Ns; i++) {
        ctx->scan_header[id].c[i].Cs = *(ctx->sos_ptr[id] + offset);
        offset += 1;
        ctx->scan_header[id].c[i].Td = *(ctx->sos_ptr[id] + offset) >> 4;
        ctx->scan_header[id].c[i].Ta = *(ctx->sos_ptr[id] + offset);
        offset += 1;
    }

    ctx->scan_header[id].Ss = *(ctx->sos_ptr[id] + offset);
    offset += 1;
    ctx->scan_header[id].Se = *(ctx->sos_ptr[id] + offset);
    offset += 1;
    ctx->scan_header[id].Ah = *(ctx->sos_ptr[id] + offset) >> 4;
    ctx->scan_header[id].Al = *(ctx->sos_ptr[id] + offset);
    offset += 1;

    ctx->scan_ptr[id] = (ctx->sos_ptr[id] + offset);

    return offset;
}


static int jfif_parse_q_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;

    SHORT_SWAP(ctx->q_table[id].DQT, *((uint16_t *)(ctx->dqt_ptr[id] + offset)));
    offset += 2;
    SHORT_SWAP(ctx->q_table[id].Lq, *((uint16_t *)(ctx->dqt_ptr[id] + offset)));
    offset += 2;
    ctx->q_table[id].Pq  = *(ctx->dqt_ptr[id] + offset) >> 4;
    ctx->q_table[id].Tq  = *(ctx->dqt_ptr[id] + offset);
    offset += 1;
    if (!ctx->q_table[id].Pq) {
        memcpy(ctx->q_table[id].Qk, (uint8_t *)(ctx->dqt_ptr[id] + offset), 64);
        offset += 64;
    } else {
        memcpy(ctx->q_table[id].Qk, (uint8_t *)(ctx->dqt_ptr[id] + offset), 128);
        offset += 128;
    }
    offset = ctx->q_table[id].Lq + 2;

    return offset;
}


static int jfif_parse_h_table(jfif_t *ctx, int id)
{
    assert(ctx != NULL);

    int offset = 0;
    int i, j;

    SHORT_SWAP(ctx->h_table[id].DHT, *((uint16_t *)(ctx->dht_ptr[id] + offset)));
    offset += 2;
    SHORT_SWAP(ctx->h_table[id].Lh, *((uint16_t *)(ctx->dht_ptr[id] + offset)));
    offset += 2;
    ctx->h_table[id].Tc = *(ctx->dht_ptr[id] + offset) >> 4;
    ctx->h_table[id].Th = *(ctx->dht_ptr[id] + offset);
    offset += 1;
    for (i=0; i<16; i++) {
        ctx->h_table[id].L[i] = *(ctx->dht_ptr[id] + offset);
        offset++;
    }
    for (i=0; i<16; i++) {
        for (j=0; j<ctx->h_table[id].L[i]; j++) {
            ctx->h_table[id].V[i][j] = *(ctx->dht_ptr[id] + offset);
            offset++;
        }
    }

    offset = ctx->h_table[id].Lh + 2;

    return offset;
}


static int jfif_round(int value, int round)
{
    int residue;

    residue = value % round;

    // Round up to integer multiple of 8
    if (residue > 0)
        residue = round - residue;

    return (value + residue);
}
